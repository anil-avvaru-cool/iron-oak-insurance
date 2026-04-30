"""
Risk Scoring — Two-Stage Hurdle Model (XGBoost).

Stage 1 (Classifier): predicts P(loss > 0) — binary, trained on all rows.
Stage 2 (Regressor):  predicts expected loss amount, trained only on rows
                      where total_claim_amount > 0.

Final risk_score = normalize(P(loss > 0) × predicted_loss_amount)
Final risk_tier  = "low" | "medium" | "high"

Why Hurdle over plain XGBRegressor:
  - ~77% of policies have zero claims. MSE regression collapses toward zero
    to minimize loss on the majority, making it a poor severity predictor.
  - Hurdle decomposes cleanly: Stage 1 = "will this customer cost us anything?"
    Stage 2 = "how much, given they do?" Two interpretable feature importance charts.
  - Industry standard for zero-inflated insurance loss modeling.
    (So & Valdez 2024; ASTIN Best Paper Award)

Why total_claim_amount as Stage 2 target (not a ratio):
  - premium_annual is intentionally excluded from risk_features() to prevent
    target leakage: our own pricing formula would become a feature.
  - Stage 2 regresses directly on log1p(total_claim_amount) for variance
    stabilization on this right-skewed, zero-inflated distribution. Predictions
    are back-transformed with expm1() before combining with Stage 1.
  - Scale normalization to 0–100 happens on the combined hurdle score at the end,
    so the absolute dollar scale does not affect tier ranking.

Evaluation:
  - Gini coefficient (2 * AUC - 1) replaces R². R² is misleading on skewed,
    zero-inflated distributions. Gini measures how well the model discriminates
    high-risk from low-risk policies — the actuarially relevant question.

  - Combined Gini is computed as: rank_corr(P(claim) × severity_pred, actual_loss)
    Concretely: gini_coefficient(has_claim, clf_proba * expm1(reg_pred))
    where gini_coefficient = 2 * roc_auc_score(actual_positive, combined_score) - 1.
    This is the standard insurance Lorenz-curve Gini, not R²-derived.

Feature interactions (added to improve Stage 1 AUC):
  - age_x_violations:       driver_age_bucket_num × active_violation_count
  - miles_x_night:          annual_miles × avg_night_driving_pct
  - violation_recency_score: active_violation_points / (months_since_last_violation + 1)
  These are computed before preprocess() in both train() and predict() so
  encoding and prediction paths are identical.

Leakage watch — has_lapse:
  has_lapse is included in risk_features() as a policyholder behavior signal.
  It is a legitimate pre-incident feature IF it reflects lapse history prior to
  the current policy period, not a lapse that occurred after a claim.
  If has_lapse importance exceeds 0.20 at train time, a warning is emitted —
  investigate whether the feature encodes post-event state in the feature
  engineer before deploying to production.

Module run:  uv run python -m ai.models.risk_scoring.model
Library use: from ai.models.risk_scoring.model import train, predict

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ai.utils.log import get_logger

log = get_logger(__name__)

MODEL_PATH_CLF   = Path("ai/models/risk_scoring/risk_model_stage1_clf.json")
MODEL_PATH_REG   = Path("ai/models/risk_scoring/risk_model_stage2_reg.json")
BOUNDS_PATH      = Path("ai/models/risk_scoring/risk_model.bounds.json")
CALIBRATION_PATH = Path("ai/models/risk_scoring/risk_model.calibration.json")

# driver_age_bucket added: string categorical ('under25', 'standard', 'senior65plus')
# without this it silently becomes 0 via fillna — wrong behavior
CATEGORICAL = ["state", "vehicle_make", "driver_age_bucket"]

# Columns excluded from model features.
# zip_prefix: retained in DataFrame for fairness audit slicing but excluded
#   from model inputs to avoid geographic proxy leakage until calibration
#   offsets are validated.
# total_claim_amount / loss_score / has_claim: targets — never features.
# state_raw: calibration lookup key, not a model input.
# Interaction columns are added at runtime; no exclusion needed.
EXCLUDE_COLS = {
    "policy_number",
    "zip_prefix",
    "total_claim_amount",
    "loss_score",
    "has_claim",
    "state_raw",
    "total_claims",
    "claims_last_12m",
    "claims_last_90d",
    "amount_last_12m",
}

# Risk tier thresholds applied to normalized 0–100 score.
# Calibrated lower than the old model (was 40/70) because the hurdle score
# distribution is right-skewed: most policies score low, high-risk tail is sparse.
TIER_THRESHOLDS = {"low": 35, "medium": 65}

# Leakage guard: warn if has_lapse dominates Stage 1 importance.
# Real leakage threshold: importance > 0.20 suggests post-event encoding.
_LAPSE_IMPORTANCE_WARN_THRESHOLD = 0.20


# ── Feature Interactions ──────────────────────────────────────────────────────

def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered interaction features before label encoding.

    These are computed on the raw (pre-encoded) DataFrame so the arithmetic
    operates on numeric values, not label-encoded integers.

    age_x_violations:
        Multiplies driver age bucket (numeric proxy) by active violation count.
        Young drivers with violations are disproportionately high-risk. Without
        the interaction, the model must discover this multiplicative effect
        independently from two separate features — harder in a tree model with
        limited depth.

    miles_x_night:
        Multiplies annual miles by night driving fraction. High mileage at night
        is a stronger risk signal than either feature alone. A high-mileage
        commuter driving mostly in daylight is meaningfully different from a
        lower-mileage driver who drives primarily at night.

    violation_recency_score:
        Decays violation points by months since last violation. A driver with
        5 points 2 months ago is materially higher risk than the same points
        3 years ago. Neither active_violation_points nor months_since_last_violation
        captures this decay on its own.
    """
    df = df.copy()

    # age bucket → ordinal numeric for interaction arithmetic
    age_map = {"under25": 3, "standard": 1, "senior65plus": 2}
    age_num = df.get("driver_age_bucket", pd.Series(["standard"] * len(df), index=df.index))
    age_num = age_num.map(age_map).fillna(1)

    viol_count = df.get("active_violation_count", pd.Series(0, index=df.index)).fillna(0)
    annual_miles = df.get("annual_miles", pd.Series(0, index=df.index)).fillna(0)
    night_pct = df.get("avg_night_driving_pct", pd.Series(0, index=df.index)).fillna(0)
    viol_points = df.get("active_violation_points", pd.Series(0, index=df.index)).fillna(0)
    months_since = df.get("months_since_last_violation", pd.Series(12, index=df.index)).fillna(12)

    df["age_x_violations"]      = age_num * viol_count
    df["miles_x_night"]         = annual_miles * night_pct
    df["violation_recency_score"] = viol_points / (months_since + 1)

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode all CATEGORICAL columns, fill remaining NaN with 0.
    Called identically at train time and predict time so encoding is consistent.
    NOTE: call _add_interactions() BEFORE preprocess() — interactions depend on
    the raw string value of driver_age_bucket, not its encoded integer.
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    df = df.fillna(0)
    return df, encoders


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive model targets from total_claim_amount.

    loss_score = total_claim_amount (raw loss dollars, stored for Stage 2 training).
        Stage 2 trains on log1p(loss_score) for variance stabilization.
        Predictions are back-transformed with expm1() before combining with Stage 1.
        premium_annual is intentionally absent from risk_features() to prevent
        target leakage via the pricing formula.

    has_claim = 1 if total_claim_amount > 0 else 0
        Stage 1 binary classification target.
    """
    df = df.copy()
    df["loss_score"] = df["total_claim_amount"].clip(lower=0.0)
    df["has_claim"]  = (df["total_claim_amount"] > 0).astype(int)
    return df


# ── Gini Coefficient ──────────────────────────────────────────────────────────

def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Gini coefficient = 2 * AUC - 1.
    Industry-standard evaluation metric for insurance risk models.
    Range: [-1, 1]. Random model = 0. Perfect model = 1.

    Computation:
        AUC = roc_auc_score(binarize(y_true), y_pred)
        Gini = 2 * AUC - 1
    This is the standard insurance Lorenz-curve Gini derived from AUC, NOT R²-derived.

    For the combined hurdle score, y_pred = P(claim) × E[loss | claim], which
    is the expected loss. y_true is binarized to has_claim (0/1). The Gini
    measures how well expected loss discriminates claimants from non-claimants —
    the actuarially relevant ordering question.

    R² is not used: it is misleading on zero-inflated, right-skewed distributions
    and can be negative even for reasonable models — common in insurance.
    """
    auc = roc_auc_score((y_true > 0).astype(int), y_pred)
    return round(2 * auc - 1, 4)


# ── Optimal Threshold ─────────────────────────────────────────────────────────

def _best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Sweep probability thresholds [0.05, 0.95] and return (threshold, f1) at peak F1.

    The default 0.5 threshold is rarely optimal for imbalanced data. For demo:
    show how precision/recall tradeoffs shift with threshold — the gap between
    best-F1 threshold and 0.5 is the single most vivid illustration of
    imbalance handling.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), round(best_f1, 4)


# ── Leakage Guard ─────────────────────────────────────────────────────────────

def _check_lapse_leakage(model: xgb.XGBClassifier, feature_cols: list[str]) -> None:
    """
    Warn if has_lapse dominates Stage 1 feature importance.

    has_lapse is a legitimate pre-incident signal (prior lapse history) but can
    encode post-event state if the feature engineer derives it from the same policy
    period as the claim. An importance > _LAPSE_IMPORTANCE_WARN_THRESHOLD suggests
    the feature may be leaking — investigate feature_engineer.risk_features() before
    production deployment.

    This does not block training; it surfaces the diagnostic at train time when the
    signal is easiest to act on.
    """
    importance = dict(zip(feature_cols, model.feature_importances_))
    lapse_imp = importance.get("has_lapse", 0.0)
    if lapse_imp > _LAPSE_IMPORTANCE_WARN_THRESHOLD:
        log.warning(
            "risk_lapse_leakage_suspect",
            has_lapse_importance=round(lapse_imp, 4),
            threshold=_LAPSE_IMPORTANCE_WARN_THRESHOLD,
            note=(
                "has_lapse importance exceeds leakage threshold. "
                "Verify feature_engineer.risk_features() derives has_lapse from "
                "history prior to current policy period only. "
                "See model.py module docstring — 'Leakage watch' section."
            ),
        )
    else:
        log.info(
            "risk_lapse_leakage_ok",
            has_lapse_importance=round(lapse_imp, 4),
            threshold=_LAPSE_IMPORTANCE_WARN_THRESHOLD,
        )


# ── State Calibration ─────────────────────────────────────────────────────────

def compute_state_calibration(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    """
    Compute per-state calibration offsets to correct systematic geographic bias.

    For each state: offset = overall_mean_score - state_mean_score
    Applied additively at predict time.

    Only computed for states with >= 30 policies (matches MIN_SLICE_SIZE in
    fairness_audit.py — slices smaller than this are high-variance).

    This directly addresses fairness flags like AR/DE/MN where the model's
    predicted rate deviates from the actual labeled rate.
    """
    overall_mean = df[score_col].mean()
    offsets: dict[str, float] = {}
    for state, grp in df.groupby("state_raw"):
        if len(grp) >= 30:
            state_mean = grp[score_col].mean()
            offsets[str(state)] = round(float(overall_mean - state_mean), 6)
    return offsets


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    """
    Train the two-stage hurdle model.

    Returns metadata dict with Gini, MAE, stage sample sizes, and calibration offsets.
    Saves four artifacts:
      risk_model_stage1_clf.json   — XGBClassifier  (P(has_claim))
      risk_model_stage2_reg.json   — XGBRegressor   (log1p(loss_score) | has_claim=1)
      risk_model.bounds.json       — normalization bounds for combined score
      risk_model.calibration.json  — per-state calibration offsets
    """
    t0 = time.time()

    df = _build_target(df)

    # Add feature interactions before encoding — arithmetic must run on raw values
    df = _add_interactions(df)

    # Preserve raw (pre-encoding) state for calibration groupby
    df["state_raw"] = df["state"].copy()

    n_total      = len(df)
    n_with_claim = int(df["has_claim"].sum())
    n_zero       = n_total - n_with_claim
    log.info(
        "risk_data_profile",
        total=n_total,
        with_claim=n_with_claim,
        zero_claim=n_zero,
        zero_pct=round(n_zero / n_total * 100, 1),
    )

    df_proc, _ = preprocess(df)
    feature_cols = _feature_cols(df_proc)

    # ── Stage 1: Classifier (all rows) ───────────────────────────────────────
    # Predict P(total_claim_amount > 0). Imbalanced: use scale_pos_weight.
    X_all = df_proc[feature_cols]
    y_clf = df_proc["has_claim"]

    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X_all, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    pos_weight = float(n_zero) / max(float(n_with_claim), 1)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_tr1, y_tr1, eval_set=[(X_te1, y_te1)], verbose=False)

    clf_proba_test = clf.predict_proba(X_te1)[:, 1]
    clf_auc        = roc_auc_score(y_te1, clf_proba_test)
    clf_gini       = round(2 * clf_auc - 1, 4)

    # Optimal F1 threshold — shows imbalance tradeoff at demo time
    best_threshold, best_f1 = _best_f1_threshold(y_te1.values, clf_proba_test)
    f1_at_default = f1_score(y_te1, (clf_proba_test >= 0.5).astype(int), zero_division=0)

    print(f"\nStage 1 (Classifier)  AUC={clf_auc:.4f}  Gini={clf_gini:.4f}")
    print(
        f"  Threshold 0.50 → F1={f1_at_default:.4f}  |  "
        f"Best threshold {best_threshold:.2f} → F1={best_f1:.4f}"
    )
    _print_top_features(clf, X_all.columns, "Stage 1")

    # Leakage guard — warn if has_lapse is suspiciously dominant
    _check_lapse_leakage(clf, list(X_all.columns))

    MODEL_PATH_CLF.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(MODEL_PATH_CLF)

    # ── Stage 2: Regressor (non-zero rows only) ───────────────────────────────
    # Predict log1p(loss_score) given a claim occurred.
    # Log-transform stabilizes variance on the right-skewed claim severity
    # distribution. Back-transform with expm1() at predict time.
    # Trained only on rows where total_claim_amount > 0 — the 'claim' population.
    # Up-weight top-quartile loss events so the model doesn't ignore the tail.
    # NOTE: severity_weights are computed on the ORIGINAL scale (raw dollars),
    # not the log scale — we want to up-weight high-dollar events, not high-log events.
    mask_nonzero = df_proc["has_claim"] == 1
    df_nonzero   = df_proc[mask_nonzero]

    if len(df_nonzero) < 50:
        log.warning(
            "risk_stage2_insufficient_data",
            nonzero_rows=len(df_nonzero),
            note="Stage 2 regressor may be unreliable with < 50 non-zero rows",
        )

    X_pos = df_nonzero[feature_cols]
    y_reg_raw = df_nonzero["loss_score"]                   # raw dollars (for weights, reporting)
    y_reg     = np.log1p(y_reg_raw)                        # log-transformed target for training

    # Severity weights on raw dollar scale — penalize the model more for
    # missing high-dollar claims than low-dollar ones
    severity_weights = np.where(y_reg_raw >= y_reg_raw.quantile(0.75), 3.0, 1.0)

    X_tr2, X_te2, y_tr2, y_te2, y_raw_tr2, y_raw_te2, w_tr2, _ = train_test_split(
        X_pos, y_reg, y_reg_raw, severity_weights, test_size=0.2, random_state=42
    )

    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        objective="reg:squarederror",
        eval_metric="mae",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    reg.fit(
        X_tr2, y_tr2,
        sample_weight=w_tr2,
        eval_set=[(X_te2, y_te2)],
        verbose=False,
    )

    # Back-transform predictions to dollars for MAE reporting
    reg_preds_dollars = np.expm1(reg.predict(X_te2))
    reg_mae           = mean_absolute_error(y_raw_te2, reg_preds_dollars)

    # Contextual MAE — report as % of mean severity so it's interpretable
    mean_severity     = float(y_reg_raw.mean())
    mae_pct           = round(reg_mae / mean_severity * 100, 1) if mean_severity > 0 else None

    print(
        f"\nStage 2 (Regressor)   MAE=${reg_mae:,.2f}  "
        f"({mae_pct}% of mean severity ${mean_severity:,.0f})  "
        f"(non-zero rows: {len(df_nonzero):,})"
    )
    _print_top_features(reg, X_pos.columns, "Stage 2")

    MODEL_PATH_REG.parent.mkdir(parents=True, exist_ok=True)
    reg.save_model(MODEL_PATH_REG)

    # ── Combined score: P(claim) × E[loss | claim] ───────────────────────────
    # Combined Gini = 2 * roc_auc_score(has_claim, clf_proba * expm1(reg_pred)) - 1
    # Measures how well expected loss discriminates claimants from non-claimants.
    # This is the standard insurance Lorenz-curve Gini, NOT R²-derived.
    clf_all_proba     = clf.predict_proba(X_all)[:, 1]
    reg_all_preds_log = reg.predict(X_all)
    reg_all_preds     = np.maximum(np.expm1(reg_all_preds_log), 0.0)  # back-transform, clamp negatives
    combined          = clf_all_proba * reg_all_preds

    gini_combined = gini_coefficient(df_proc["has_claim"].values, combined)
    print(f"\nCombined Hurdle Score  Gini={gini_combined:.4f}")

    min_c, max_c = float(combined.min()), float(combined.max())
    BOUNDS_PATH.write_text(json.dumps({"min": min_c, "max": max_c}))

    # ── State calibration ─────────────────────────────────────────────────────
    df["_normalized_score"] = _normalize(combined, min_c, max_c)
    calibration = compute_state_calibration(df, "_normalized_score")
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))

    n_offset_flags = sum(1 for v in calibration.values() if abs(v) > 5.0)
    print(
        f"\nState calibration: {len(calibration)} states computed, "
        f"{n_offset_flags} with offset > 5 pts (review in fairness audit)"
    )

    elapsed = round(time.time() - t0, 2)
    metadata = {
        "stage1_auc":           round(clf_auc, 4),
        "stage1_gini":          clf_gini,
        "stage1_best_threshold": best_threshold,
        "stage1_f1_at_best":    best_f1,
        "stage1_f1_at_default": round(float(f1_at_default), 4),
        "stage2_mae":           round(reg_mae, 2),
        "stage2_mean_severity": round(mean_severity, 2),
        "stage2_mae_pct":       mae_pct,
        "combined_gini":        gini_combined,
        "n_total":              n_total,
        "n_with_claim":         n_with_claim,
        "elapsed_s":            elapsed,
    }
    log.info("model_trained", model="risk", **metadata)
    return metadata


def _normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == lo:
        return np.full_like(values, 50.0, dtype=float)
    return np.clip((values - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def _tier(score: float) -> str:
    if score < TIER_THRESHOLDS["low"]:
        return "low"
    if score < TIER_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def _print_top_features(model, columns, label: str, top: int = 10) -> None:
    importance = dict(zip(columns, model.feature_importances_))
    top_n = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top]
    print("Top features:")
    for feat, imp in top_n:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<45} {imp:.4f}  {bar}")


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(records: list[dict]) -> list[dict]:
    """
    Score a batch of policy dicts.

    Returns records enriched with:
      risk_score        float 0–100  (normalized hurdle score, state-calibrated)
      risk_tier         str          "low" | "medium" | "high"
      claim_probability float 0–1   (Stage 1 output — interpretable standalone)
    """
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH_CLF)
    reg = xgb.XGBRegressor()
    reg.load_model(MODEL_PATH_REG)

    bounds      = json.loads(BOUNDS_PATH.read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())

    df = pd.DataFrame(records)

    # Preserve raw state before encoding for calibration lookup
    if "state" in df.columns:
        df["state_raw"] = df["state"].copy()

    # Add interactions before encoding — mirrors train() path exactly
    df = _add_interactions(df)

    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)

    clf_proba         = clf.predict_proba(df[feature_cols])[:, 1]
    reg_preds_log     = reg.predict(df[feature_cols])
    reg_preds         = np.maximum(np.expm1(reg_preds_log), 0.0)
    combined          = clf_proba * reg_preds

    scores = _normalize(combined, bounds["min"], bounds["max"])

    for i, rec in enumerate(records):
        state_key  = str(rec.get("state", ""))
        offset     = calibration.get(state_key, 0.0)
        calibrated = float(np.clip(scores[i] + offset, 0.0, 100.0))

        rec["risk_score"]        = round(calibrated, 2)
        rec["risk_tier"]         = _tier(calibrated)
        rec["claim_probability"] = round(float(clf_proba[i]), 4)

    return records


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import risk_features
    from ai.models.fairness_audit import run_audit

    df = risk_features()
    log.info("risk_train_start", rows=len(df))

    train(df)

    # Re-score full dataset using saved models for fairness audit.
    # Re-load from disk rather than reusing in-memory objects so this
    # path is identical to the inference path in predict().
    df = _build_target(df)
    df = _add_interactions(df)
    df["state_raw"] = df["state"].copy()
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)

    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH_CLF)

    reg = xgb.XGBRegressor()
    reg.load_model(MODEL_PATH_REG)

    bounds = json.loads(BOUNDS_PATH.read_text())

    clf_proba     = clf.predict_proba(df_proc[feature_cols])[:, 1]
    reg_preds_log = reg.predict(df_proc[feature_cols])
    reg_preds     = np.maximum(np.expm1(reg_preds_log), 0.0)
    combined      = clf_proba * reg_preds

    df["predicted_score"] = _normalize(combined, bounds["min"], bounds["max"])

    # Pass Stage 1 classifier probability (0–1) to the fairness audit, not the
    # combined hurdle score (0–100). The audit's _positive_rate() uses a fixed
    # >= 0.5 threshold, which only makes sense for a probability score.
    #
    # Passing predicted_score (0–100) caused the audit to divide by 100
    # (because max > 1.5) and then count score >= 0.5 — meaning only policies
    # with raw risk_score >= 50. The hurdle distribution is right-skewed so
    # almost nobody clears that bar, producing a 0.12% predicted positive rate
    # vs 25% actual and 10 false-positive fairness flags amplified by tiny
    # absolute differences over a near-zero baseline.
    #
    # clf_proba is already on the [0, 1] scale, asks the same question
    # (will this policy generate a claim?), and aligns correctly with actual_label.
    df["clf_proba"]     = clf_proba
    df["actual_label"]  = (df["total_claim_amount"] > 0).astype(int)
    run_audit(df, model_name="risk", score_col="clf_proba", label_col="actual_label")


if __name__ == "__main__":
    main()
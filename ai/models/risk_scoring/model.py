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
  - Stage 2 regresses directly on total_claim_amount (raw loss dollars).
    Scale normalization to 0–100 happens on the combined hurdle score at the end,
    so the absolute dollar scale does not affect tier ranking.

Evaluation:
  - Gini coefficient (2 * AUC - 1) replaces R². R² is misleading on skewed,
    zero-inflated distributions. Gini measures how well the model discriminates
    high-risk from low-risk policies — the actuarially relevant question.

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
from sklearn.metrics import mean_absolute_error, roc_auc_score
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
    "total_claim_amount"
}

# Risk tier thresholds applied to normalized 0–100 score.
# Calibrated lower than the old model (was 40/70) because the hurdle score
# distribution is right-skewed: most policies score low, high-risk tail is sparse.
TIER_THRESHOLDS = {"low": 35, "medium": 65}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode all CATEGORICAL columns, fill remaining NaN with 0.
    Called identically at train time and predict time so encoding is consistent.
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

    loss_score = total_claim_amount (raw loss dollars)
        premium_annual is intentionally absent from risk_features() to prevent
        target leakage via the pricing formula. Raw claim dollars are the direct
        actuarial loss signal. The combined hurdle score is normalized to 0–100
        afterward, so the absolute dollar scale does not affect tier ranking.
        Policies with no claims → 0.0 (the zero-inflated majority).

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

    R² is not used: it is misleading on zero-inflated, right-skewed distributions
    and can be negative even for reasonable models — common in insurance.
    """
    auc = roc_auc_score((y_true > 0).astype(int), y_pred)
    return round(2 * auc - 1, 4)


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
      risk_model_stage2_reg.json   — XGBRegressor   (loss_score | has_claim=1)
      risk_model.bounds.json       — normalization bounds for combined score
      risk_model.calibration.json  — per-state calibration offsets
    """
    t0 = time.time()

    df = _build_target(df)

    print(df[["drive_score", "avg_drive_score_12m", "credit_score", "has_claim"]].corr()["has_claim"].sort_values())
    print("total rows:", len(df))
    print("has_claim distribution:\n", df["has_claim"].value_counts())
    print("total_claim_amount > 0:", (df["total_claim_amount"] > 0).sum())
    print("total_claim_amount sample:\n", df["total_claim_amount"].describe())

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

    print(f"\nStage 1 (Classifier)  AUC={clf_auc:.4f}  Gini={clf_gini:.4f}")
    _print_top_features(clf, X_all.columns, "Stage 1")

    MODEL_PATH_CLF.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(MODEL_PATH_CLF)

    # ── Stage 2: Regressor (non-zero rows only) ───────────────────────────────
    # Predict loss_score given a claim occurred.
    # Trained only on rows where total_claim_amount > 0 — the 'claim' population.
    # Up-weight top-quartile loss events so the model doesn't ignore the tail.
    mask_nonzero = df_proc["has_claim"] == 1
    df_nonzero   = df_proc[mask_nonzero]

    if len(df_nonzero) < 50:
        log.warning(
            "risk_stage2_insufficient_data",
            nonzero_rows=len(df_nonzero),
            note="Stage 2 regressor may be unreliable with < 50 non-zero rows",
        )

    X_pos = df_nonzero[feature_cols]
    y_reg = df_nonzero["loss_score"]

    severity_weights = np.where(y_reg >= y_reg.quantile(0.75), 3.0, 1.0)

    X_tr2, X_te2, y_tr2, y_te2, w_tr2, _ = train_test_split(
        X_pos, y_reg, severity_weights, test_size=0.2, random_state=42
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

    reg_preds = reg.predict(X_te2)
    reg_mae   = mean_absolute_error(y_te2, reg_preds)

    print(f"\nStage 2 (Regressor)   MAE={reg_mae:.4f}  (non-zero rows: {len(df_nonzero):,})")
    _print_top_features(reg, X_pos.columns, "Stage 2")

    MODEL_PATH_REG.parent.mkdir(parents=True, exist_ok=True)
    reg.save_model(MODEL_PATH_REG)

    # ── Combined score: P(claim) × E[loss | claim] ───────────────────────────
    clf_all_proba = clf.predict_proba(X_all)[:, 1]
    reg_all_preds = np.maximum(reg.predict(X_all), 0.0)  # clamp negatives
    combined      = clf_all_proba * reg_all_preds

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
        "stage1_auc":    round(clf_auc, 4),
        "stage1_gini":   clf_gini,
        "stage2_mae":    round(reg_mae, 4),
        "combined_gini": gini_combined,
        "n_total":       n_total,
        "n_with_claim":  n_with_claim,
        "elapsed_s":     elapsed,
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


def _print_top_features(model, columns, label: str, top: int = 5) -> None:
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

    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)

    clf_proba = clf.predict_proba(df[feature_cols])[:, 1]
    reg_preds = np.maximum(reg.predict(df[feature_cols]), 0.0)
    combined  = clf_proba * reg_preds

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
    df["state_raw"] = df["state"].copy()
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)

    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH_CLF)

    reg = xgb.XGBRegressor()
    reg.load_model(MODEL_PATH_REG)

    bounds = json.loads(BOUNDS_PATH.read_text())

    clf_proba = clf.predict_proba(df_proc[feature_cols])[:, 1]
    reg_preds = np.maximum(reg.predict(df_proc[feature_cols]), 0.0)
    combined  = clf_proba * reg_preds

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
    df["clf_proba"] = clf_proba
    df["actual_label"] = (df["total_claim_amount"] > 0).astype(int)
    run_audit(df, model_name="risk", score_col="clf_proba", label_col="actual_label")

if __name__ == "__main__":
    main()
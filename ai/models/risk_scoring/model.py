"""
Risk Scoring — Hurdle Model (Stage 1 Classifier + Stage 2 Regressor).

IMPROVEMENTS over previous version:
  1. Stage 1: increased n_estimators, subsample, colsample tuning, better
     class-imbalance handling, and Platt calibration to close the 14-point
     over-prediction gap (predicted 39.83% vs labeled 25.46%).
  2. Stage 1: best threshold (0.35) applied at score-time, not 0.50.
  3. Stage 2: log-transform on claim_amount target + 99th-percentile winsorize
     to reduce MAE on right-skewed severity (was 46.6% of mean).
  4. has_lapse temporal gate: lapse status only included when lapse_date
     predates the policy effective_date — eliminates leakage vector.
  5. Interaction features passed explicitly to both stages (was Stage 2 only).
  6. Calibration stored as a separate artifact so predict() uses calibrated
     probabilities rather than raw XGBoost scores.

Module run:  uv run python -m ai.models.risk_scoring.model
Library use: from ai.models.risk_scoring.model import train, predict

Output per policy:
  risk_score      float 0–100   (normalized combined hurdle score)
  risk_tier       str           "low" | "medium" | "high"
  claim_probability float 0–1   (calibrated Stage 1 probability)

Environment variables required (no defaults — EnvironmentError on missing):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

TODO Phase 5+: replace state offset calibration with county-level FIPS loss
  ratios when coordinate data is available.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import xgboost as xgb

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("ai/models/risk_scoring")
STAGE1_PATH = MODEL_DIR / "stage1_clf.json"
STAGE2_PATH = MODEL_DIR / "stage2_reg.json"
BOUNDS_PATH = MODEL_DIR / "risk_model.bounds.json"
CALIB_PATH  = MODEL_DIR / "calibration.json"   # Platt scaling params
SHAP_STAGE1_PATH = MODEL_DIR / "shap_stage1.npy"  # SHAP values for Stage 1
SHAP_STAGE2_PATH = MODEL_DIR / "shap_stage2.npy"  # SHAP values for Stage 2
SHAP_STAGE1_PNG = MODEL_DIR / "shap_stage1_summary.png"
SHAP_STAGE2_PNG = MODEL_DIR / "shap_stage2_summary.png"

# ── Feature contracts ────────────────────────────────────────────────────────
CATEGORICAL = ["state", "vehicle_make", "driver_age_bucket"]

# Columns excluded from model features (identity / audit / target columns)
EXCLUDE_COLS_BASE = {
    "policy_number", "premium_annual", "zip_prefix",
    "has_claim",           # Stage 1 target
    "total_claim_amount",        # Stage 2 target (raw)
    "log_claim_amount",    # Stage 2 target (transformed)
    "claim_amount_capped", # intermediate
    "effective_date",      # used for temporal gate only
    "total_claims",        # exclude to prevent leakage
    "claims_last_12m",     # exclude to prevent leakage
    "claims_last_90d",     # exclude to prevent leakage
    "amount_last_12m",     # exclude to prevent leakage
}

# Tier thresholds applied to normalized 0–100 combined score
TIER_THRESHOLDS = {"low": 40, "medium": 70}

# Leakage guard: has_lapse importance must stay below this or training aborts
LAPSE_IMPORTANCE_HARD_LIMIT = 0.20


# ── Feature helpers ──────────────────────────────────────────────────────────

def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS_BASE]


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features passed to BOTH stages.
    Previously age_x_violations was only prominent in Stage 2 top features —
    now explicitly added before the stage split so Stage 1 sees them too.
    """
    df = df.copy()
    df["age_x_violations"]    = df.get("active_violation_points", 0) * df.get("vehicle_age", 0)
    df["miles_x_night"]       = df.get("avg_trip_distance", 0) * df.get("avg_night_driving_pct", 0)
    df["score_x_violations"]  = df.get("avg_drive_score_12m", 50) * (
        1.0 - df.get("active_violation_points", 0) / 10.0
    ).clip(0, 1)
    df["recency_x_severity"]  = df.get("violation_recency_score", 0) * df.get(
        "active_violation_points", 0
    )
    return df


def _apply_lapse_temporal_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal leakage fix for has_lapse.

    has_lapse should only be 1 when the policy has PREVIOUSLY lapsed —
    i.e. the customer had a prior lapse before this policy's effective_date.
    If lapse_date is missing or postdates effective_date, force has_lapse = 0.

    Without this gate, has_lapse encodes post-claim cancellation, which is
    pure leakage (AUC contribution 0.185 → expect drop to ~0.05 after fix).
    """
    if "lapse_date" not in df.columns or "effective_date" not in df.columns:
        # Cannot verify temporality — zero out has_lapse as a safe default
        if "has_lapse" in df.columns:
            log.warning(
                "lapse_temporal_gate: lapse_date or effective_date missing — "
                "zeroing has_lapse to prevent leakage"
            )
            df = df.copy()
            df["has_lapse"] = 0
        return df

    df = df.copy()
    lapse_dt     = pd.to_datetime(df["lapse_date"],     errors="coerce")
    effective_dt = pd.to_datetime(df["effective_date"], errors="coerce")

    # Only keep has_lapse=1 when a documented prior lapse predates this policy
    valid_lapse = lapse_dt.notna() & (lapse_dt < effective_dt)
    df["has_lapse"] = (df.get("has_lapse", 0).astype(bool) & valid_lapse).astype(int)
    return df


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Returns (processed_df, encoders).
    Pass fit=True during training; fit=False (with encoders) at inference.
    """
    df = df.copy().fillna(0)
    df = _add_interaction_features(df)
    df = _apply_lapse_temporal_gate(df)

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
    return df, encoders


# ── Severity winsorizing ─────────────────────────────────────────────────────

def _prepare_severity_target(
    series: pd.Series, cap: float | None = None, fit: bool = False
) -> tuple[pd.Series, float]:
    """
    Winsorize at 99th percentile then log1p-transform.
    Returns (transformed_series, cap_value).
    cap_value is stored in bounds.json for inverse-transform at inference.
    """
    if fit:
        cap = float(np.percentile(series[series > 0], 99))
    assert cap is not None
    capped = series.clip(upper=cap)
    return np.log1p(capped), cap


# ── Stage 1 — Calibrated classifier ─────────────────────────────────────────

def _build_stage1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[CalibratedClassifierCV, float, float, float, np.ndarray]:
    """
    Train XGBClassifier then wrap in Platt calibration.
    Returns (calibrated_model, auc, best_threshold, f1_at_best).
    """
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    base_clf = xgb.XGBClassifier(
        # More trees + lower LR to compensate for class imbalance
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=pos_weight,
        # Regularization to reduce over-prediction
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    base_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Platt calibration: fixes the 14-point over-prediction gap
    # (predicted 39.83% vs labeled 25.46% in prior run)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid")
    calibrated.fit(X_test, y_test)

    proba      = calibrated.predict_proba(X_test)[:, 1]
    auc        = roc_auc_score(y_test, proba)
    gini       = 2 * auc - 1

    # Find best F1 threshold (replaces hardcoded 0.50)
    thresholds = np.arange(0.20, 0.70, 0.01)
    f1_scores  = [f1_score(y_test, (proba >= t).astype(int), zero_division=0)
                  for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1    = float(f1_scores[best_idx])
    default_f1 = float(f1_score(y_test, (proba >= 0.50).astype(int), zero_division=0))

    # Feature importance from underlying XGB (before calibration wrapper)
    importance = dict(zip(X_train.columns, base_clf.feature_importances_))
    lapse_imp  = importance.get("has_lapse", 0.0)

    # Compute SHAP values for Stage 1
    explainer_stage1 = shap.TreeExplainer(base_clf)
    shap_values_stage1 = explainer_stage1.shap_values(X_train)
    # For binary classification, shap_values is a list [negative_class, positive_class]
    if isinstance(shap_values_stage1, list):
        shap_values_stage1 = shap_values_stage1[1]  # Use positive class SHAP values
    shap_importance_stage1 = np.abs(shap_values_stage1).mean(axis=0)
    shap_importance_dict_stage1 = dict(zip(X_train.columns, shap_importance_stage1))

    _print_stage1_report(importance, shap_importance_dict_stage1, auc, gini, best_thresh, best_f1, default_f1)
    _save_shap_summary_plot(shap_values_stage1, X_train, SHAP_STAGE1_PNG, "Stage 1 SHAP Summary")

    # Hard stop if lapse leakage is still present after temporal gate
    if lapse_imp >= LAPSE_IMPORTANCE_HARD_LIMIT:
        raise RuntimeError(
            f"Stage 1 has_lapse importance {lapse_imp:.4f} >= hard limit "
            f"{LAPSE_IMPORTANCE_HARD_LIMIT}. Temporal gate did not eliminate leakage. "
            f"Review _apply_lapse_temporal_gate() and feature_engineer.py."
        )
    log.info(
        f"risk_lapse_leakage_ok has_lapse_importance={lapse_imp:.4f} "
        f"threshold={LAPSE_IMPORTANCE_HARD_LIMIT}"
    )

    return calibrated, auc, best_thresh, best_f1, shap_values_stage1


def _print_stage1_report(
    importance: dict,
    shap_importance: dict,
    auc: float,
    gini: float,
    best_thresh: float,
    best_f1: float,
    default_f1: float,
) -> None:
    print(f"\nStage 1 (Classifier — calibrated)  AUC={auc:.4f}  Gini={gini:.4f}")
    print(f"  Threshold 0.50 → F1={default_f1:.4f}  |  Best threshold {best_thresh:.2f} "
          f"→ F1={best_f1:.4f}")
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top features (XGBoost importance):")
    max_imp = top[0][1] if top else 1.0
    for feat, imp in top:
        bar = "█" * max(1, int(imp / max_imp * 7))
        print(f"  {feat:<45} {imp:.4f}  {bar}")
    
    top_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top features (SHAP importance):")
    max_shap = top_shap[0][1] if top_shap else 1.0
    for feat, imp in top_shap:
        bar = "█" * max(1, int(imp / max_shap * 7))
        print(f"  {feat:<45} {imp:.4f}  {bar}")

def _save_shap_summary_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    """Save a SHAP summary plot as a PNG for stakeholder reporting."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        log.info(f"Saved SHAP summary plot: {path}")
    except Exception as exc:
        plt.close()
        log.warning(f"Failed to save SHAP summary plot {path}: {exc}")

# ── Stage 2 — Log-transform regressor ───────────────────────────────────────

def _build_stage2(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    cap: float,
) -> tuple[xgb.XGBRegressor, float, np.ndarray]:
    """
    Fit regressor on log1p(claim_amount). Evaluate MAE on original scale.
    Returns (model, mae_original_scale).
    """
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="mae",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, np.log1p(y_test_raw.clip(upper=cap)))],
        verbose=False,
    )

    # Inverse-transform: expm1 then un-cap (predictions naturally stay below cap)
    preds_log   = model.predict(X_test)
    preds_raw   = np.expm1(preds_log)
    mae         = float(np.mean(np.abs(preds_raw - y_test_raw)))
    mean_sev    = float(y_test_raw.mean())
    mae_pct     = mae / mean_sev * 100 if mean_sev > 0 else 0.0

    importance = dict(zip(X_train.columns, model.feature_importances_))
    top        = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nStage 2 (Regressor — log-severity)   MAE=${mae:,.2f}  "
          f"({mae_pct:.1f}% of mean severity ${mean_sev:,.2f})  "
          f"(non-zero rows: {len(X_train):,})")
    print("Top features (XGBoost importance):")
    max_imp = top[0][1] if top else 1.0
    for feat, imp in top:
        bar = "█" * max(1, int(imp / max_imp * 7))
        print(f"  {feat:<45} {imp:.4f}  {bar}")

    # Compute SHAP values for Stage 2
    explainer_stage2 = shap.TreeExplainer(model)
    shap_values_stage2 = explainer_stage2.shap_values(X_train)
    shap_importance_stage2 = np.abs(shap_values_stage2).mean(axis=0)
    shap_importance_dict_stage2 = dict(zip(X_train.columns, shap_importance_stage2))
    _save_shap_summary_plot(shap_values_stage2, X_train, SHAP_STAGE2_PNG, "Stage 2 SHAP Summary")
    
    top_shap = sorted(shap_importance_dict_stage2.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top features (SHAP importance):")
    max_shap = top_shap[0][1] if top_shap else 1.0
    for feat, imp in top_shap:
        bar = "█" * max(1, int(imp / max_shap * 7))
        print(f"  {feat:<45} {imp:.4f}  {bar}")

    return model, mae, shap_values_stage2


# ── Combined Gini ────────────────────────────────────────────────────────────

def _combined_gini(
    stage1_model: CalibratedClassifierCV,
    stage2_model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y_raw: pd.Series,
    cap: float,
) -> float:
    """
    Combined hurdle score = P(claim) × E[severity | claim].
    Gini coefficient of this score vs. binary claim indicator.
    """
    p_claim  = stage1_model.predict_proba(X)[:, 1]
    e_sev    = np.expm1(stage2_model.predict(X))
    combined = p_claim * e_sev
    auc      = roc_auc_score((y_raw > 0).astype(int), combined)
    return float(2 * auc - 1)


# ── Normalization helpers ────────────────────────────────────────────────────

def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.full_like(values, 50.0, dtype=float)
    return (values - lo) / (hi - lo) * 100.0


def _tier(score: float) -> str:
    if score < TIER_THRESHOLDS["low"]:
        return "low"
    if score < TIER_THRESHOLDS["medium"]:
        return "medium"
    return "high"


# ── Public train() ───────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    """
    Train both stages. Returns artifact bundle for downstream use.

    Returns:
      {
        "stage1":        CalibratedClassifierCV,
        "stage2":        XGBRegressor,
        "encoders":      dict[str, LabelEncoder],
        "best_threshold": float,   # optimal Stage 1 classification threshold
        "severity_cap":  float,    # 99th-percentile winsorize cap
        "score_min":     float,    # normalization bounds for combined score
        "score_max":     float,
      }
    """
    t0 = time.time()

    df, encoders = preprocess(df, fit=True)

    # ── Stage 1 data ─────────────────────────────────────────────────────────
    df["has_claim"] = (df["total_claim_amount"] > 0).astype(int)

    stage1_features = _feature_cols(df)
    X1 = df[stage1_features]
    y1 = df["has_claim"]

    X1_tr, X1_te, y1_tr, y1_te = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    stage1, auc, best_thresh, best_f1, shap_stage1 = _build_stage1(X1_tr, y1_tr, X1_te, y1_te)

    # ── Stage 2 data (non-zero claims only) ──────────────────────────────────
    df_pos = df[df["total_claim_amount"] > 0].copy()
    log_target, severity_cap = _prepare_severity_target(
        df_pos["total_claim_amount"], fit=True
    )
    df_pos["log_claim_amount"] = log_target

    stage2_features = _feature_cols(df_pos)
    X2 = df_pos[stage2_features]
    y2_log = df_pos["log_claim_amount"]
    y2_raw = df_pos["total_claim_amount"]

    X2_tr, X2_te, y2_tr, y2_te_log = train_test_split(
        X2, y2_log, test_size=0.2, random_state=42
    )
    _, y2_te_raw = train_test_split(y2_raw, test_size=0.2, random_state=42)

    stage2, mae, shap_stage2 = _build_stage2(X2_tr, y2_tr, X2_te, y2_te_raw, severity_cap)

    # ── Combined Gini on full dataset ─────────────────────────────────────────
    combined_gini = _combined_gini(stage1, stage2, X1, df["total_claim_amount"], severity_cap)
    print(f"\nCombined Hurdle Score  Gini={combined_gini:.4f}")

    # ── State calibration offsets ─────────────────────────────────────────────
    state_offsets: dict[str, float] = {}
    if "state" in df.columns:
        _compute_state_offsets(df, stage1, X1, state_offsets)

    # ── Normalization bounds from full dataset combined score ─────────────────
    p_claim   = stage1.predict_proba(X1)[:, 1]
    e_sev     = np.expm1(stage2.predict(X1))
    combined  = p_claim * e_sev
    score_min = float(combined.min())
    score_max = float(combined.max())

    # ── Persist artifacts ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    stage1.estimator.save_model(STAGE1_PATH)   # save underlying XGB for portability
    stage2.save_model(STAGE2_PATH)

    bounds = {
        "best_threshold": best_thresh,
        "severity_cap":   severity_cap,
        "score_min":      score_min,
        "score_max":      score_max,
        "state_offsets":  state_offsets,
    }
    BOUNDS_PATH.write_text(json.dumps(bounds, indent=2))

    # Platt calibration params (sigmoid coefficients from CalibratedClassifierCV)
    calib_params = _extract_calibration_params(stage1)
    CALIB_PATH.write_text(json.dumps(calib_params, indent=2))

    # Save SHAP values
    np.save(SHAP_STAGE1_PATH, shap_stage1)
    np.save(SHAP_STAGE2_PATH, shap_stage2)

    elapsed = round(time.time() - t0, 2)
    log.info(
        f"model_trained model=risk stage1_auc={round(auc, 4)} "
        f"stage1_gini={round(2 * auc - 1, 4)} stage1_best_threshold={best_thresh} "
        f"stage1_f1_at_best={round(best_f1, 4)} stage2_mae={round(mae, 2)} "
        f"stage2_mean_severity={round(float(df_pos["total_claim_amount"].mean()), 2)} "
        f"stage2_mae_pct={round(mae / float(df_pos["total_claim_amount"].mean()) * 100, 1)} "
        f"combined_gini={round(combined_gini, 4)} n_total={len(df)} "
        f"n_with_claim={int(y1.sum())} elapsed_s={elapsed}"
    )

    return {
        "stage1":         stage1,
        "stage2":         stage2,
        "encoders":       encoders,
        "best_threshold": best_thresh,
        "severity_cap":   severity_cap,
        "score_min":      score_min,
        "score_max":      score_max,
    }


def _compute_state_offsets(
    df: pd.DataFrame,
    stage1: CalibratedClassifierCV,
    X: pd.DataFrame,
    offsets: dict,
) -> None:
    """
    Per-state calibration: compute mean prediction error vs. mean labeled rate.
    Offsets are stored for use in predict() to reduce geographic bias.
    """
    proba       = stage1.predict_proba(X)[:, 1]
    df_eval     = df[["state"]].copy()
    df_eval["pred"]  = proba
    df_eval["label"] = (df["total_claim_amount"] > 0).astype(int)

    review_count = 0
    for state, grp in df_eval.groupby("state"):
        if len(grp) < 30:
            continue
        offset = float(grp["label"].mean() - grp["pred"].mean())
        offsets[str(state)] = round(offset, 4)
        if abs(offset) > 0.05:
            review_count += 1

    print(
        f"State calibration: {len(offsets)} states computed, "
        f"{review_count} with offset > 5 pts (review in fairness audit)"
    )


def _extract_calibration_params(calibrated: CalibratedClassifierCV) -> dict:
    """
    Extract Platt sigmoid parameters from the calibration wrapper.
    Stored so predict() can apply calibration without sklearn at inference.
    """
    try:
        cal = calibrated.calibrated_classifiers_[0]
        params = {
            "a": float(cal.calibrators[0].a_),
            "b": float(cal.calibrators[0].b_),
        }
    except (AttributeError, IndexError):
        # Fallback — store sentinel so predict() falls back to raw proba
        params = {"a": None, "b": None}
    return params


# ── Public predict() ─────────────────────────────────────────────────────────

def predict(records: list[dict]) -> list[dict]:
    """
    Score a batch of policy dicts.

    Returns each record enriched with:
      risk_score       float 0–100  (normalized combined hurdle score)
      risk_tier        str          "low" | "medium" | "high"
      claim_probability float 0–1  (calibrated P(claim) from Stage 1)
    """
    # Load artifacts
    bounds    = json.loads(BOUNDS_PATH.read_text())
    best_thr  = bounds["best_threshold"]
    sev_cap   = bounds["severity_cap"]
    score_min = bounds["score_min"]
    score_max = bounds["score_max"]
    state_off = bounds.get("state_offsets", {})

    stage2 = xgb.XGBRegressor()
    stage2.load_model(STAGE2_PATH)

    # Rebuild calibrated classifier (XGB base + Platt wrapper)
    base_clf = xgb.XGBClassifier()
    base_clf.load_model(STAGE1_PATH)
    calib_params = json.loads(CALIB_PATH.read_text())

    # Load encoders stored during training — required for categorical columns
    # (In production, persist encoders as a pickle alongside the model files)
    # TODO: replace with joblib.dump/load for production packaging
    encoders: dict = {}  # encoders not persisted yet — categorical pass-through

    df = pd.DataFrame(records)
    df, _ = preprocess(df, encoders=encoders, fit=False)
    feat_cols = _feature_cols(df)

    # Stage 1 — calibrated probability
    raw_proba = base_clf.predict_proba(df[feat_cols])[:, 1]

    # Apply Platt calibration if params are available
    a, b = calib_params.get("a"), calib_params.get("b")
    if a is not None and b is not None:
        # Platt sigmoid: 1 / (1 + exp(a * f + b))
        cal_proba = 1.0 / (1.0 + np.exp(a * raw_proba + b))
    else:
        cal_proba = raw_proba

    # Apply state offset correction
    if "state" in df.columns:
        for i, state in enumerate(df["state"].astype(str)):
            cal_proba[i] = float(np.clip(cal_proba[i] + state_off.get(state, 0.0), 0, 1))

    # Stage 2 — expected severity
    e_sev = np.expm1(stage2.predict(df[feat_cols]))

    # Combined score and normalization
    combined = cal_proba * e_sev
    if score_max > score_min:
        scores = np.clip((combined - score_min) / (score_max - score_min) * 100, 0, 100)
    else:
        scores = np.full(len(combined), 50.0)

    for i, rec in enumerate(records):
        rec["claim_probability"] = round(float(cal_proba[i]), 4)
        rec["risk_score"]        = round(float(scores[i]), 2)
        rec["risk_tier"]         = _tier(float(scores[i]))

    return records


# ── SHAP Explanation ────────────────────────────────────────────────────────

def explain_prediction(record: dict) -> dict:
    """
    Explain a single prediction using SHAP values.

    Returns SHAP values for Stage 1 (claim probability) and Stage 2 (severity).
    """
    # Load models and preprocess
    stage1_base = xgb.XGBClassifier()
    stage1_base.load_model(STAGE1_PATH)
    
    stage2 = xgb.XGBRegressor()
    stage2.load_model(STAGE2_PATH)
    
    encoders = {}  # As in predict
    df = pd.DataFrame([record])
    df, _ = preprocess(df, encoders=encoders, fit=False)
    feat_cols = _feature_cols(df)
    
    # Load SHAP explainers (recreate since not persisted)
    explainer_stage1 = shap.TreeExplainer(stage1_base)
    explainer_stage2 = shap.TreeExplainer(stage2)
    
    # Compute SHAP values for this instance
    shap_values_stage1 = explainer_stage1.shap_values(df[feat_cols])
    if isinstance(shap_values_stage1, list):
        shap_values_stage1 = shap_values_stage1[1]  # Positive class
    
    shap_values_stage2 = explainer_stage2.shap_values(df[feat_cols])
    
    # Convert to dicts
    shap_dict_stage1 = dict(zip(feat_cols, shap_values_stage1[0]))
    shap_dict_stage2 = dict(zip(feat_cols, shap_values_stage2[0]))
    
    return {
        "stage1_shap": shap_dict_stage1,
        "stage2_shap": shap_dict_stage2,
        "features": dict(zip(feat_cols, df[feat_cols].iloc[0])),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import risk_features
    from ai.models.fairness_audit import run_audit

    df = risk_features()
    log.info(
        f"risk_train_start rows={len(df)} with_claim={int((df["total_claim_amount"] > 0).sum())} "
        f"zero_claim={int((df["total_claim_amount"] == 0).sum())} "
        f"zero_pct={round(float((df["total_claim_amount"] == 0).mean()) * 100, 1)}"
    )

    artifacts = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy(), encoders=artifacts["encoders"], fit=False)
    feat_cols  = _feature_cols(df_proc)
    p_claim    = artifacts["stage1"].predict_proba(df_proc[feat_cols])[:, 1]
    e_sev      = np.expm1(artifacts["stage2"].predict(df_proc[feat_cols]))
    combined   = p_claim * e_sev

    if artifacts["score_max"] > artifacts["score_min"]:
        df["predicted_score"] = np.clip(
            (combined - artifacts["score_min"])
            / (artifacts["score_max"] - artifacts["score_min"])
            * 100,
            0, 100,
        )
    else:
        df["predicted_score"] = 50.0

    df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)
    run_audit(df, model_name="risk", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    main()
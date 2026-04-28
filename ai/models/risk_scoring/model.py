"""
Risk Scoring — XGBoost regressor.

Target: loss_score — a weighted composite of claim frequency, severity, and
recency. Built from claims data directly in build_loss_score(). premium_annual
is fully removed to prevent target leakage via the synthetic pricing formula.

loss_score = (0.50 × freq_score) + (0.35 × severity_score) + (0.15 × recency_score)
Scaled 0–100. Zero-claim policies score 0 on frequency and severity components.

Module run:  uv run python -m ai.models.risk_scoring.model
Library use: from ai.models.risk_scoring.model import train, predict
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ai.utils.log import get_logger

log = get_logger(__name__)

MODEL_PATH = Path("ai/models/risk_scoring/risk_model.json")

CATEGORICAL = ["state", "vehicle_make", "zip_prefix", "driver_age_bucket"]

# Identity only — loss_score target components are excluded separately
# inside build_loss_score() before the feature matrix is assembled.
EXCLUDE_COLS = {
    "policy_number",        # identity — never a feature
    "total_claim_amount",   # loss_score component — exclude to prevent leakage
    "claims_last_12m",      # loss_score component — exclude to prevent leakage
    "claims_last_90d",      # loss_score component — exclude to prevent leakage
    "amount_last_12m",      # loss_score component — exclude to prevent leakage
    "total_claims",         # all-time proxy for loss_score — soft leakage
}

# loss_score weights — must sum to 1.0
_FREQ_WEIGHT     = 0.50
_SEVERITY_WEIGHT = 0.35
_RECENCY_WEIGHT  = 0.15

# Tier thresholds applied to the normalized 0–100 risk score.
# Calibrated to loss_score distribution: most policies score 0 (no claims),
# so thresholds are intentionally asymmetric.
#   low    : loss_score < 10  (zero or minimal claim history)
#   medium : loss_score 10–40 (one or more claims, moderate severity)
#   high   : loss_score >= 40 (frequent or severe claims)
TIER_THRESHOLDS = {"low": 10, "medium": 40}


# ── Target engineering ─────────────────────────────────────────────────────

def build_loss_score(df: pd.DataFrame) -> pd.Series:
    """
    Construct the behavioral loss target from claims components.

    Inputs (columns must be present in df):
        claims_last_12m   — claim count in last 12 months
        amount_last_12m   — total claim amount in last 12 months
        claims_last_90d   — claim count in last 90 days (recency signal)

    Returns:
        pd.Series of float, range 0–100, named 'loss_score'.
        Zero-claim policies score 0.

    Design notes:
        - Normalization uses dataset max so scores are relative within the
          training cohort, not absolute dollar amounts. This keeps the target
          stable across datasets of different sizes.
        - Recency component is binary (any claim in last 90 days = full weight)
          rather than continuous. A single recent claim is a stronger signal
          than a large old claim for near-term risk prediction.
        - Clipped to [0, 100] — floating point arithmetic can produce values
          marginally outside this range on edge cases.
    """
    max_claims = max(df["claims_last_12m"].max(), 1)
    max_amount = max(df["amount_last_12m"].max(), 1)

    freq_score     = df["claims_last_12m"] / max_claims
    severity_score = df["amount_last_12m"]  / max_amount
    recency_score  = (df["claims_last_90d"] > 0).astype(float)

    raw = (
        _FREQ_WEIGHT     * freq_score
      + _SEVERITY_WEIGHT * severity_score
      + _RECENCY_WEIGHT  * recency_score
    ) * 100

    return raw.clip(0, 100).rename("loss_score")


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy().fillna(0)
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


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


# ── Training ───────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float, float]:
    """
    Train and save model.

    Returns:
        (model, min_pred, max_pred) — bounds written to risk_model.bounds.json
        for use by predict() at inference time.
    """
    t0 = time.time()

    # Build target before preprocessing so loss_score components
    # are present in df when build_loss_score() reads them.
    loss_score = build_loss_score(df)

    # Print target distribution — expected to be heavily zero-inflated.
    nonzero = (loss_score > 0).sum()
    print(f"\nloss_score distribution (n={len(loss_score):,}):")
    print(f"  zero-claim policies : {len(loss_score) - nonzero:,} "
          f"({(len(loss_score) - nonzero) / len(loss_score):.1%})")
    print(f"  nonzero policies    : {nonzero:,} ({nonzero / len(loss_score):.1%})")
    print(f"  p50={loss_score.quantile(0.50):.1f}  "
          f"p75={loss_score.quantile(0.75):.1f}  "
          f"p90={loss_score.quantile(0.90):.1f}  "
          f"p99={loss_score.quantile(0.99):.1f}  "
          f"max={loss_score.max():.1f}\n")

    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    X, y = df[feature_cols], loss_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds      = model.predict(X_test)
    mae        = mean_absolute_error(y_test, preds)
    r2         = r2_score(y_test, preds)
    full_preds = model.predict(X)
    min_pred   = float(full_preds.min())
    max_pred   = float(full_preds.max())

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    top8 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"MAE: {mae:.4f}  |  R²: {r2:.4f}")
    print("Top features:")
    for feat, imp in top8:
        print(f"  {feat:<35} {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    bounds_path = MODEL_PATH.with_suffix(".bounds.json")
    bounds_path.write_text(json.dumps({"min": min_pred, "max": max_pred}))

    log.info(
        "model_trained",
        model="risk",
        rows=len(df),
        target="loss_score",
        nonzero_rows=int(nonzero),
        mae=round(mae, 4),
        r2=round(r2, 4),
        elapsed_s=round(time.time() - t0, 2),
    )
    return model, min_pred, max_pred


# ── Inference ──────────────────────────────────────────────────────────────

def predict(records: list[dict]) -> list[dict]:
    """
    Score a batch of policy dicts.

    Input records must contain the behavioral feature columns produced by
    risk_features() in feature_engineer.py. premium_annual is not required
    and is ignored if present.

    Returns records enriched with:
        risk_score  float 0–100  (normalized predicted loss score)
        risk_tier   str          "low" | "medium" | "high"
    """
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    bounds = json.loads(MODEL_PATH.with_suffix(".bounds.json").read_text())
    min_pred, max_pred = bounds["min"], bounds["max"]

    df = pd.DataFrame(records)
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    raw = model.predict(df[feature_cols])

    raw = np.clip(raw, min_pred, max_pred)
    scores = (
        (raw - min_pred) / (max_pred - min_pred) * 100.0
        if max_pred > min_pred
        else np.full_like(raw, 50.0, dtype=float)
    )

    for i, rec in enumerate(records):
        score = round(float(scores[i]), 2)
        rec["risk_score"] = score
        rec["risk_tier"]  = _tier(score)
    return records


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import risk_features
    from ai.models.fairness_audit import run_audit

    df = risk_features()
    log.info("risk_train_start", rows=len(df))

    model, min_pred, max_pred = train(df)

    # Score full dataset for fairness audit using normalized risk score
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    raw    = model.predict(df_proc[feature_cols])
    raw    = np.clip(raw, min_pred, max_pred)
    scores = _normalize(raw) if max_pred > min_pred else raw
    df["predicted_score"] = scores

    # Fairness audit: high-risk tier = positive outcome
    df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)

    # Derive violation_tier for fairness audit slicing — added after training, not before
    df["violation_tier"] = pd.cut(
        df["active_violation_points"].fillna(0),
        bins=[-1, 0, 3, 7, float("inf")],
        labels=["none", "minor", "moderate", "major"]
    )

    run_audit(df, model_name="risk", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
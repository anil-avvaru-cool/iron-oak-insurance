"""
Risk Scoring — XGBoost regressor.

Target: premium_annual (actuarially-informed proxy for risk)
Features: drive score, credit score, vehicle year, state, coverage elections, telematics trend

Module run:  uv run python -m ai.models.risk_scoring.model
Library use: from ai.models.risk_scoring.model import train, predict
"""
from __future__ import annotations

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

# Columns excluded from model features — audit/identity/target only
EXCLUDE_COLS = {
    "policy_number",
    "premium_annual",   # this IS the target    
}

# Tier thresholds applied to normalized 0–100 risk score
# Tuned to premium_annual distribution: low < p33, medium < p67, high >= p67
TIER_THRESHOLDS = {"low": 33, "medium": 67}


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


def train(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float, float]:
    """Train and save model. Returns (model, min_pred, max_pred) for normalization."""
    t0 = time.time()
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    X, y = df[feature_cols], df["premium_annual"]

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

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Normalization bounds from full training set predictions
    full_preds = model.predict(X)
    min_pred, max_pred = float(full_preds.min()), float(full_preds.max())

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    top8 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"MAE: ${mae:,.2f}  |  R²: {r2:.4f}")
    print("Top features:")
    for feat, imp in top8:
        print(f"  {feat:<30} {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    import json
    bounds_path = MODEL_PATH.with_suffix(".bounds.json")
    bounds_path.write_text(json.dumps({"min": min_pred, "max": max_pred}))

    log.info(
        "model_trained",
        model="risk",
        rows=len(df),
        target="premium_annual",
        mae=round(mae, 2),
        r2=round(r2, 4),
        elapsed_s=round(time.time() - t0, 2),
    )
    return model, min_pred, max_pred


def predict(records: list[dict]) -> list[dict]:
    """Score a batch of policy dicts. Returns records with risk_score and risk_tier."""
    import json

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    bounds_path = MODEL_PATH.with_suffix(".bounds.json")
    bounds = json.loads(bounds_path.read_text())
    min_pred, max_pred = bounds["min"], bounds["max"]

    df = pd.DataFrame(records)
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    raw = model.predict(df[feature_cols])

    raw = np.clip(raw, min_pred, max_pred)
    if max_pred > min_pred:
        scores = (raw - min_pred) / (max_pred - min_pred) * 100.0
    else:
        scores = np.full_like(raw, 50.0, dtype=float)

    for i, rec in enumerate(records):
        score = round(float(scores[i]), 2)
        rec["risk_score"] = score
        rec["risk_tier"] = _tier(score)
    return records


def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import risk_features
    from ai.models.fairness_audit import run_audit

    df = risk_features()

    # print("\nFeature sample (20 rows):")
    # print(df.sample(min(20, len(df))).to_string())
    # print(f"\npremium_annual stats:\n{df['premium_annual'].describe().round(2)}")

    log.info("risk_train_start", rows=len(df))
    model, min_pred, max_pred = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    raw = model.predict(df_proc[feature_cols])
    raw = np.clip(raw, min_pred, max_pred)
    scores = _normalize(raw) if max_pred > min_pred else raw
    df["predicted_score"] = scores

    # High-risk tier = positive outcome for fairness audit
    df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)
    run_audit(df, model_name="risk", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
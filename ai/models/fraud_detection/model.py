"""
Fraud Detection — XGBoost binary classifier.

Module run:  uv run python -m ai.models.fraud_detection.model
Library use: from ai.models.fraud_detection.model import train, predict
Lambda use:  handler wraps predict()

Design changes (v2):
  - days_to_file added to EXCLUDE_COLS; days_to_file_log used instead.
    Raw days_to_file had 0.82 feature importance due to synthetic artifact
    (fraud claims filed in 0-2 days by claim_gen.py). The log-transformed
    version is kept as a legitimate signal but no longer dominates.
  - fraud_signal_count removed from feature set — replaced by 11 binary
    sig_* columns produced by feature_engineer.py's unnest pivot.
  - CATEGORICAL updated to reflect actual string columns in the query.
  - scale_pos_weight computed from training split, not full dataset,
    to prevent mild data leakage in the class weight calculation.

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ai.utils.log import get_logger

log = get_logger(__name__)

MODEL_PATH = Path("ai/models/fraud_detection/fraud_model.json")

CATEGORICAL = ["state", "claim_type", "vehicle_make"]

# Columns excluded from model input features:
#   claim_id, label       — identity / target
#   zip_prefix            — kept for fairness audit only
#   days_to_file          — raw value excluded; log-transformed version used instead
#   fraud_signal_count    — replaced by individual sig_* binary columns
EXCLUDE_COLS = {
    # Identity / target
    "claim_id",
    "label",
    # Fairness audit only
    "zip_prefix",
    # Raw value replaced by log-transformed version
    "days_to_file",
    # Legacy guard
    "fraud_signal_count",
    # Fraud signals — fetched for explainability (Phase 5 fraud agent) but
    # must NOT be model features: they are derived directly from is_fraud=True
    # in the generator, so including them guarantees a perfect separator and
    # produces an unrealistically high ROC-AUC on synthetic data.
    "sig_claim_delta_high",
    "sig_telematics_anomaly",
    "sig_staged_accident",
    "sig_frequency_spike",
    "sig_location_mismatch",
    "sig_multiple_claimants",
    "sig_no_police_report",
    "sig_attorney_early",
    "sig_lapse_reinstatement",
    "sig_rapid_refiling",
    "sig_recent_reinstatement",
}

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


def train(df: pd.DataFrame) -> xgb.XGBClassifier:
    t0 = time.time()
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    X, y = df[feature_cols], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute scale_pos_weight from training split only (not full dataset)
    # to avoid mild leakage from knowing the full label distribution.
    neg_train = float((y_train == 0).sum())
    pos_train = float((y_train == 1).sum())
    pos_weight = neg_train / max(pos_train, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)

    print(classification_report(y_test, preds, target_names=["clean", "fraud"]))

    importance = dict(zip(X.columns, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"ROC-AUC: {roc:.4f}")
    print("Top features:")
    for feat, imp in top10:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<45} {imp:.4f}  {bar}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    log.info(
        "model_trained",
        model="fraud",
        rows=len(df),
        fraud_rows=int(y.sum()),
        feature_count=len(feature_cols),
        features=feature_cols,
        roc_auc=round(roc, 4),
        elapsed_s=round(time.time() - t0, 2),
    )
    return model


def predict(records: list[dict]) -> list[dict]:
    """Score a batch of claim dicts. Returns records enriched with fraud_score."""
    import numpy as np

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df = pd.DataFrame(records)

    # Apply the same log1p transform applied during training
    if "days_to_file" in df.columns and "days_to_file_log" not in df.columns:
        df["days_to_file_log"] = np.log1p(df["days_to_file"].clip(lower=0))

    # Cast bool signal columns to int if present
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    if sig_cols:
        df[sig_cols] = df[sig_cols].astype("Int64")

    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    proba = model.predict_proba(df[feature_cols])[:, 1]
    for i, rec in enumerate(records):
        rec["fraud_score"] = round(float(proba[i]), 4)
        rec["is_fraud_predicted"] = bool(proba[i] >= 0.5)
    return records


def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import fraud_features
    from ai.models.fairness_audit import run_audit

    df = fraud_features()
    log.info(
        "fraud_train_start",
        rows=len(df),
        fraud_rows=int(df["label"].sum()),
        fraud_pct=round(df["label"].mean() * 100, 2),
    )

    # Print feature column list so it's visible in training output
    from ai.models.fraud_detection.model import EXCLUDE_COLS
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"  {col}")
    print()

    model = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy())
    feature_cols_proc = _feature_cols(df_proc)
    df["predicted_score"] = model.predict_proba(df_proc[feature_cols_proc])[:, 1]
    run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
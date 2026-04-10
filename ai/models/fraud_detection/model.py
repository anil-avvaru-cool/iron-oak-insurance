"""
Fraud Detection — XGBoost binary classifier.

Module run:  uv run python -m ai.models.fraud_detection.model
Library use: from ai.models.fraud_detection.model import train, predict
Lambda use:  handler wraps predict()

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
# Columns excluded from model features (used for audit only)
EXCLUDE_COLS = {"claim_id", "label", "zip_prefix"}


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

    pos_weight = float((y == 0).sum()) / float((y == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    print("preds:", preds[:10])
    proba = model.predict_proba(X_test)[:, 1]
    print("probas:", proba[:10])
    roc = roc_auc_score(y_test, proba)

    # Print report for meetup demo
    print(classification_report(y_test, preds, target_names=["clean", "fraud"]))

    importance = dict(zip(X.columns, model.feature_importances_))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"ROC-AUC: {roc:.4f}")
    print("Top features:", top5)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    log.info(
        "model_trained",
        model="fraud",
        rows=len(df),
        fraud_rows=int(y.sum()),
        roc_auc=round(roc, 4),
        elapsed_s=round(time.time() - t0, 2),
    )
    return model


def predict(records: list[dict]) -> list[dict]:
    """Score a batch of claim dicts. Returns records enriched with fraud_score."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df = pd.DataFrame(records)
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
    log.info("fraud_train_start", rows=len(df), fraud_rows=int(df["label"].sum()))
    print(df.head(10))    
    
    model = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    df["predicted_score"] = model.predict_proba(df_proc[feature_cols])[:, 1]
    run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
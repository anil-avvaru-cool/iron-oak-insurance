"""
Fraud Detection — XGBoost binary classifier.
Runnable: uv run python ai/models/fraud_detection/model.py
Lambda-compatible: import train, predict as library functions.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

MODEL_PATH = Path("ai/models/fraud_detection/fraud_model.json")
CATEGORICAL = ["state", "claim_type"]

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[LabelEncoder]]:
    df = df.copy().fillna(0)
    encoders = []
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders.append(le)
    return df, encoders

def train(df: pd.DataFrame) -> xgb.XGBClassifier:
    df, _ = preprocess(df)
    feature_cols = [c for c in df.columns if c not in ("claim_id", "label")]
    X, y = df[feature_cols], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # handle class imbalance
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds, target_names=["clean", "fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    # Feature importance for demo
    importance = dict(zip(X.columns, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop features:", top)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    return model

def predict(records: list[dict]) -> list[dict]:
    """Score a batch of claim dicts. Returns records with fraud_score and is_fraud_predicted."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df = pd.DataFrame(records)
    df, _ = preprocess(df)
    feature_cols = [c for c in df.columns if c not in ("claim_id", "label")]
    proba = model.predict_proba(df[feature_cols])[:, 1]
    for i, rec in enumerate(records):
        rec["fraud_score"] = round(float(proba[i]), 4)
        rec["is_fraud_predicted"] = bool(proba[i] >= 0.5)
    return records

def main():
    from ai.pipelines.ingestion.feature_engineer import fraud_features
    df = fraud_features()
    print(f"Training on {len(df):,} claims ({df['label'].sum()} fraud)")
    print(df.head(10))
    #train(df)
    # TODO: run fairness_audit.py after training.
    # Slice fraud_score distribution by state, ZIP prefix, vehicle make.
    # Flag if any slice deviates > ±2× from overall rate without matching label deviation.
    # See strategy Section 11.3 and CROSS_PHASE.md §9.3.

if __name__ == "__main__":
    main()
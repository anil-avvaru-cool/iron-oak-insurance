"""
Churn Prediction — XGBoost binary classifier.

Module run:  uv run python -m ai.models.churn_prediction.model
Library use: from ai.models.churn_prediction.model import train, predict

Output per customer:
  churn_probability  float 0–1
  churn_predicted    bool

Key feature: drive_score_delta (3m avg − 12m avg)
  Negative delta = deteriorating driving → higher churn risk.

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ai.utils.log import get_logger

log = get_logger(__name__)

MODEL_PATH = Path("ai/models/churn_prediction/churn_model.json")
CATEGORICAL = ["state"]
# Audit/identity columns excluded from model features
EXCLUDE_COLS = {"customer_id", "label", "zip_prefix"}


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

    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)

    print(classification_report(y_test, preds, target_names=["retained", "churned"]))

    importance = dict(zip(X.columns, model.feature_importances_))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"ROC-AUC: {roc:.4f}")
    print("Top features (note drive_score_delta):", top5)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    log.info(
        "model_trained",
        model="churn",
        rows=len(df),
        churn_rows=int(y.sum()),
        roc_auc=round(roc, 4),
        elapsed_s=round(time.time() - t0, 2),
    )
    return model


def predict(records: list[dict]) -> list[dict]:
    """Score a batch of customer dicts. Returns records with churn_probability."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df = pd.DataFrame(records)
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    proba = model.predict_proba(df[feature_cols])[:, 1]
    for i, rec in enumerate(records):
        rec["churn_probability"] = round(float(proba[i]), 4)
        rec["churn_predicted"] = bool(proba[i] >= 0.5)
    return records


def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import churn_features
    from ai.models.fairness_audit import run_audit

    df = churn_features()
    log.info("churn_train_start", rows=len(df), churn_rows=int(df["label"].sum()))
    model = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    df["predicted_score"] = model.predict_proba(df_proc[feature_cols])[:, 1]
    run_audit(df, model_name="churn", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
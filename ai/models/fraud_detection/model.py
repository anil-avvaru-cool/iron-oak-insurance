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

import json
import os
from dotenv import load_dotenv
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
# Find the threshold that maximizes F1 or hits a recall target
from sklearn.metrics import precision_recall_curve
import numpy as np
from ai.utils.log import get_logger
from dataclasses import dataclass
from sklearn.metrics import recall_score, precision_score, f1_score

log = get_logger(__name__)
load_dotenv()

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
    # Fairness audit / chart column — added by main() after training
    "zip_prefix",
    "predicted_score",          # ← ADD THIS — prevents feature mismatch in predict()
    # Raw value replaced by log-transformed version
    "days_to_file",
    # Legacy guard
    "fraud_signal_count",
    # Lifetime claim count is too strong a separator on small synthetic datasets
    "customer_claim_count",
    # Fraud signals — explainability only, must NOT be model features
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

_FRAUD_THRESHOLD_DEFAULT = 0.50

@dataclass
class FraudTrainResult:
    """All outputs from a single training run. Keeps main() clean."""
    model:             xgb.XGBClassifier
    roc_auc:           float
    recall:            float   # at chosen_threshold — used as detection_rate in waterfall
    precision:         float   # at chosen_threshold
    f1:                float   # at chosen_threshold
    threshold:         float   # chosen_threshold persisted to .threshold.json
    fraud_rows:        int
    total_rows:        int

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

def train(df: pd.DataFrame) -> FraudTrainResult:
    t0 = time.time()
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    X, y = df[feature_cols], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_test, proba)

    # Threshold selection — computed from data, not a hardcoded default
    chosen_threshold = _select_threshold(proba, y_test.values)
    preds = (proba >= chosen_threshold).astype(int)

    # Extract metrics at chosen_threshold — recall is the detection_rate for waterfall
    recall    = recall_score(y_test,    preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test,        preds, zero_division=0)

    print(f"\nClassification report at threshold={chosen_threshold:.2f} (computed)")
    print(classification_report(
        y_test, preds,
        target_names=["clean", "fraud"],
        zero_division=0,
    ))

    # Default threshold kept for reference only — not used downstream
    preds_default = (proba >= 0.50).astype(int)
    print("Classification report at threshold=0.50 (default, for comparison)")
    print(classification_report(
        y_test, preds_default,
        target_names=["clean", "fraud"],
        zero_division=0,
    ))

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"Recall:    {recall:.4f}  ← detection_rate used in waterfall")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Top 10 features:")
    for feat, imp in top10:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<45} {imp:.4f}  {bar}")

    # Confusion matrix chart
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["clean", "fraud"])
    disp.plot(cmap="Blues")
    plt.title(
        f"Fraud Detection — Confusion Matrix\n"
        f"threshold={chosen_threshold:.2f}  |  "
        f"ROC-AUC={roc:.4f}  |  Recall={recall:.4f}"
    )
    plt.tight_layout()
    plt.savefig("ai/models/fraud_detection/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved → ai/models/fraud_detection/confusion_matrix.png")

    # Persist model and threshold
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    threshold_path = MODEL_PATH.with_suffix(".threshold.json")
    threshold_path.write_text(json.dumps({
        "fraud_threshold": chosen_threshold,
        "roc_auc":         round(roc, 4),
        "recall":          round(recall, 4),
        "precision":       round(precision, 4),
        "f1":              round(f1, 4),
    }))
    print(f"Threshold + metrics saved → {threshold_path}")

    log.info(
        "model_trained",
        model="fraud",
        rows=len(df),
        fraud_rows=int(y.sum()),
        threshold=chosen_threshold,
        roc_auc=round(roc, 4),
        recall=round(recall, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        elapsed_s=round(time.time() - t0, 2),
    )

    return FraudTrainResult(
        model=model,
        roc_auc=roc,
        recall=recall,
        precision=precision,
        f1=f1,
        threshold=chosen_threshold,
        fraud_rows=int(y.sum()),
        total_rows=len(df),
    )

def predict(records: list[dict]) -> list[dict]:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Threshold priority: saved file → env var override → hardcoded default
    threshold_path = MODEL_PATH.with_suffix(".threshold.json")
    
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text())["fraud_threshold"])
    else:
        # Env var as operational override — optional, no default in os.environ call
        threshold = float(os.environ.get("FRAUD_THRESHOLD", str(_FRAUD_THRESHOLD_DEFAULT)))

    df = pd.DataFrame(records)
    df, _ = preprocess(df)
    feature_cols = _feature_cols(df)
    proba = model.predict_proba(df[feature_cols])[:, 1]

    for i, rec in enumerate(records):
        rec["fraud_score"]        = round(float(proba[i]), 4)
        rec["is_fraud_predicted"] = bool(proba[i] >= threshold)
        rec["threshold_used"]     = threshold
    return records

def _select_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    Select the threshold that maximizes F1 on the fraud class,
    subject to a minimum precision floor of 0.20.

    A precision floor prevents the degenerate case where the optimizer
    picks a threshold so low that every claim is flagged as fraud
    (100% recall, ~5% precision). SIU investigators need at least 1-in-5
    referrals to be real fraud to make the queue actionable.

    Falls back to the F1-optimal threshold without the floor if no
    threshold satisfies the constraint (rare on very small test sets).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    # The last element of precisions/recalls has no corresponding threshold
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    MIN_PRECISION = 0.20

    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )

    # Constrained: best F1 where precision >= floor
    eligible = precisions >= MIN_PRECISION
    if eligible.any():
        best_idx = np.argmax(np.where(eligible, f1_scores, 0.0))
    else:
        # Floor not achievable — fall back to unconstrained F1 max
        best_idx = np.argmax(f1_scores)

    chosen = float(thresholds[best_idx])
    # Round to 2 decimal places — avoids spurious precision in saved file
    chosen = round(chosen, 2)

    print(f"\nThreshold selection:")
    print(f"  Candidates evaluated: {len(thresholds)}")
    print(f"  Chosen threshold:     {chosen:.2f}")
    print(f"  At this threshold → precision={precisions[best_idx]:.2f}  "
          f"recall={recalls[best_idx]:.2f}  f1={f1_scores[best_idx]:.2f}")
    print(f"  Min precision floor:  {MIN_PRECISION}")
    return chosen

def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import fraud_features
    from ai.models.fairness_audit import run_audit
    from ai.models.fraud_detection.waterfall_params import extract_params
    from ai.models.fraud_detection.annual_loss_waterfall import build_waterfall
    from ai.models.fraud_detection.fraud_pie_chart import from_scored_records

    df = fraud_features()

    log.info(
        "fraud_train_start",
        rows=len(df),
        fraud_rows=int(df["label"].sum()),
        fraud_pct=round(df["label"].mean() * 100, 2),
    )

    # Print feature columns so they're visible in training output
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"  {col}")
    print()

    # Train — result carries model + all metrics
    result = train(df)

    # ── FIX 2: snapshot BEFORE df is mutated ──────────────────────────────
    # df["predicted_score"] below adds a column to df.
    # predict() calls _feature_cols(df) internally — if predicted_score is
    # present in the records dict, XGBoost rejects the feature set.
    # Snapshot now while df is still clean.
    raw_records = df.to_dict("records")

    # Score full dataset for fairness audit — this mutates df
    df_proc, _ = preprocess(df.copy())
    feature_cols_proc = _feature_cols(df_proc)
    df["predicted_score"] = result.model.predict_proba(df_proc[feature_cols_proc])[:, 1]

    # Fairness audit — needs predicted_score column on df
    run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")

    # Pie chart — uses raw_records (no predicted_score column)
    scored = predict(raw_records)
    from_scored_records(scored, roc_auc=result.roc_auc)

    # Waterfall — recall = TP / (TP + FN), most defensible detection_rate
    params = extract_params(detection_rate=result.recall)
    build_waterfall(**params)

    log.info(
        "fraud_charts_saved",
        model="fraud",
        recall_used_as_detection_rate=round(result.recall, 4),
        roc_auc=round(result.roc_auc, 4),
    )


if __name__ == "__main__":
    main()
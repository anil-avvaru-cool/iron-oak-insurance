# Phase 3 — ML Models

**Git tag:** `v0.3.0`  
**Deliverable:** Fraud, risk, and churn models served via FastAPI at `http://localhost:8000`.

**Meetup demo:** Score a batch of claims for fraud live, show feature importance, compare risk scores for the same driver across TX vs. MI.

---

## Table of Contents

1. [Feature Engineering Pipeline](#1-feature-engineering-pipeline)
2. [Fraud Detection Model](#2-fraud-detection-model)
3. [Risk Scoring & Churn Prediction](#3-risk-scoring--churn-prediction)
4. [FastAPI Application](#4-fastapi-application)
5. [Verification & Git Tag](#5-verification--git-tag)

---

## 1. Feature Engineering Pipeline

**`ai/pipelines/ingestion/feature_engineer.py`**

All three models share a common feature extraction step. Pull from Postgres, return a `pd.DataFrame`.

```python
"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER','aioi')}:{os.getenv('DB_PASSWORD','aioi_local')}"
        f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT',5432)}/{os.getenv('DB_NAME','aioi')}"
    )
    return create_engine(url)

def fraud_features(engine=None) -> pd.DataFrame:
    """Returns one row per claim with fraud label and signal features."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            c.claim_id,
            c.is_fraud                                              AS label,
            c.claim_amount,
            c.claim_amount / NULLIF(p.premium_annual, 0)           AS claim_to_premium_ratio,
            EXTRACT(DAY FROM c.filed_date::date - c.incident_date::date) AS days_to_file,
            COUNT(c2.claim_id) OVER (PARTITION BY c.customer_id)   AS customer_claim_count,
            COALESCE(t.avg_drive_score, 50)                        AS avg_drive_score,
            COALESCE(t.hard_brakes_90d, 0)                         AS hard_brakes_90d,
            p.state,
            c.claim_type,
            ARRAY_LENGTH(c.fraud_signals, 1)                       AS fraud_signal_count
        FROM claims c
        JOIN policies p ON p.policy_number = c.policy_number
        LEFT JOIN (
            SELECT policy_number,
                   AVG(drive_score)   AS avg_drive_score,
                   SUM(hard_brakes)   AS hard_brakes_90d
            FROM telematics
            WHERE trip_date >= NOW() - INTERVAL '90 days'
            GROUP BY policy_number
        ) t ON t.policy_number = c.policy_number
        LEFT JOIN claims c2 ON c2.customer_id = c.customer_id
    """)
    return pd.read_sql(sql, engine)

def risk_features(engine=None) -> pd.DataFrame:
    """Returns one row per policy for risk scoring."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            p.policy_number,
            p.state,
            p.premium_annual,
            p.drive_score,
            cust.credit_score,
            (p.vehicle->>'year')::int                              AS vehicle_year,
            p.vehicle->>'make'                                     AS vehicle_make,
            COUNT(c.claim_id)                                      AS total_claims,
            COALESCE(SUM(c.claim_amount), 0)                       AS total_claim_amount,
            COALESCE(AVG(t.drive_score), 50)                       AS avg_drive_score_12m
        FROM policies p
        JOIN customers cust ON cust.customer_id = p.customer_id
        LEFT JOIN claims c ON c.policy_number = p.policy_number
        LEFT JOIN telematics t ON t.policy_number = p.policy_number
            AND t.trip_date >= NOW() - INTERVAL '365 days'
        GROUP BY p.policy_number, p.state, p.premium_annual,
                 p.drive_score, cust.credit_score,
                 p.vehicle->>'year', p.vehicle->>'make'
    """)
    return pd.read_sql(sql, engine)

def churn_features(engine=None) -> pd.DataFrame:
    """Returns one row per customer with churn label (lapsed/cancelled = 1)."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            cust.customer_id,
            (MAX(p.status) IN ('lapsed','cancelled'))::int         AS label,
            cust.credit_score,
            COUNT(p.policy_number)                                 AS policy_count,
            AVG(p.premium_annual)                                  AS avg_premium,
            AVG(p.drive_score)                                     AS avg_drive_score,
            COUNT(c.claim_id)                                      AS total_claims,
            MAX(CASE WHEN p.status = 'active' THEN 1 ELSE 0 END)  AS has_active_policy
        FROM customers cust
        LEFT JOIN policies p ON p.customer_id = cust.customer_id
        LEFT JOIN claims c ON c.customer_id = cust.customer_id
        GROUP BY cust.customer_id, cust.credit_score
    """)
    return pd.read_sql(sql, engine)
```

---

## 2. Fraud Detection Model

**`ai/models/fraud_detection/model.py`**

```python
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
    train(df)
    # TODO: run fairness_audit.py after training.
    # Slice fraud_score distribution by state, ZIP prefix, vehicle make.
    # Flag if any slice deviates > ±2× from overall rate without matching label deviation.
    # See strategy Section 11.3 and CROSS_PHASE.md §9.3.

if __name__ == "__main__":
    main()
```

---

## 3. Risk Scoring & Churn Prediction

Both follow the same pattern as fraud detection. Key differences:

**Risk Scoring** (`ai/models/risk_scoring/model.py`)
- Regression variant (`XGBRegressor`) predicting `premium_annual` as a proxy risk score
- Output: `risk_score` (0–100 normalized), `risk_tier` (`low` / `medium` / `high`)
- Feature importance demo shows how `drive_score` and `state` dominate the prediction

```python
# TODO: run fairness_audit.py after training — check premium_risk_score
# distribution consistency across demographic proxies.
# See CROSS_PHASE.md §9.3.
```

> **Placeholder:** Geospatial features (county-level loss ratios) are marked `# TODO: Phase 3+` in `risk_scoring/model.py`. Add county lookup from FIPS codes when the dataset is expanded to include coordinates.

**Churn Prediction** (`ai/models/churn_prediction/model.py`)
- Binary classifier identical in structure to fraud; target is `label` (lapsed/cancelled)
- Key feature: rolling drive score trend (12-month avg vs 3-month avg delta)
- Output: `churn_probability`, `churn_predicted`

**Fairness audit stub** — create at project start:

`ai/models/fairness_audit.py`:
```python
"""
fairness_audit.py — post-training fairness check for fraud and risk models.

Usage:
  uv run python ai/models/fairness_audit.py --model fraud
  uv run python ai/models/fairness_audit.py --model risk

# TODO: implement disparate impact analysis per strategy Section 11.3.
# Slices: state, ZIP prefix (first 3 digits), vehicle make.
# Threshold: flag if slice predicted-positive rate deviates > 2× from overall rate
#            without corresponding deviation in labeled positive rate.
"""

def main():
    raise NotImplementedError("# TODO: implement — see strategy Section 11.3")

if __name__ == "__main__":
    main()
```

---

## 4. FastAPI Application

**`ai/api/routers/models_router.py`**
```python
from fastapi import APIRouter
from pydantic import BaseModel
from ai.models.fraud_detection.model import predict as fraud_predict

router = APIRouter(prefix="/models", tags=["models"])

class ClaimBatch(BaseModel):
    claims: list[dict]

class FraudResponse(BaseModel):
    results: list[dict]

@router.post("/fraud/score", response_model=FraudResponse)
async def score_fraud(batch: ClaimBatch):
    results = fraud_predict(batch.claims)
    return FraudResponse(results=results)

@router.get("/fraud/health")
async def fraud_health():
    return {"status": "ok", "model": "xgboost-fraud-v1"}
```

**`ai/api/handlers/main.py`** — Lambda + local entry point
```python
from fastapi import FastAPI
from mangum import Mangum
from ai.api.routers.models_router import router as models_router

app = FastAPI(title="AIOI AI API", version="0.3.0")
app.include_router(models_router)

# TODO: add structured JSON logging middleware.
# Log per-request: request_id, endpoint, strategy, policy_number, customer_id,
# model_id, latency_ms, input_tokens, output_tokens, chunks_retrieved, error.
# NEVER log chunk_text, adjuster_notes, incident_narrative, or any free-text field.
# See CROSS_PHASE.md §9.4.

# Lambda handler
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 5. Verification & Git Tag

### Verification

```bash
# Train all models
uv run python ai/models/fraud_detection/model.py
uv run python ai/models/risk_scoring/model.py
uv run python ai/models/churn_prediction/model.py

# Start API
uv run python ai/api/handlers/main.py

# Test (separate terminal)
curl -X POST http://localhost:8000/models/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claims": [{"claim_id":"CLM-00001","claim_amount":12000,"claim_to_premium_ratio":3.2,"days_to_file":1,"customer_claim_count":4,"avg_drive_score":31,"hard_brakes_90d":45,"state":"TX","claim_type":"collision","fraud_signal_count":3}]}'
```

### Phase Gate Checklist

- [ ] All 3 models train without errors
- [ ] `fairness_audit.py` stub exists (implementation is a Phase 3+ TODO)
- [ ] Fraud API returns `fraud_score` and `is_fraud_predicted` in response
- [ ] ROC-AUC reported during training (expect > 0.85 on synthetic data)
- [ ] FastAPI `/fraud/health` returns `{"status": "ok"}`

### Git Tag

```bash
git add -A
git commit -m "Phase 3: ML models — fraud, risk, churn + FastAPI"
git tag v0.3.0
```

---

*Previous: [PHASE_2_DATABASE.md](./PHASE_2_DATABASE.md) · Next: [PHASE_4_RAG.md](./PHASE_4_RAG.md)*

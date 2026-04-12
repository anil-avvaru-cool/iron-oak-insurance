# Phase 3 — ML Models

**Git tag:** `v0.3.0`  
**Deliverable:** Fraud, risk, and churn models served via FastAPI at `http://localhost:8000`.

**Meetup demo:** Score a batch of claims for fraud live, show feature importance, compare risk scores for the same driver across TX vs. MI.

---

## Table of Contents

1. [Feature Engineering Pipeline](#1-feature-engineering-pipeline)
2. [Logging Utility](#2-logging-utility)
3. [Fraud Detection Model](#3-fraud-detection-model)
4. [Risk Scoring Model](#4-risk-scoring-model)
5. [Churn Prediction Model](#5-churn-prediction-model)
6. [Fairness Audit](#6-fairness-audit)
7. [FastAPI Application](#7-fastapi-application)
8. [Verification & Git Tag](#8-verification--git-tag)

---

## Implementation Plan

### What changed from the stub

| Area | Old stub | This phase |
|------|----------|------------|
| Risk scoring | Description only | Full `XGBRegressor`, normalized score, risk tier |
| Churn prediction | Description only | Full `XGBClassifier`, drive score delta feature |
| Fairness audit | `raise NotImplementedError` | Implemented disparate impact analysis |
| Logging | TODO comment | Minimal structured JSON logger, shared across all modules |
| Env vars | Had default values | **No defaults** — fail fast on missing config |
| Run command | `python model.py` | `uv run python -m ai.models.<module>.model` |
| FastAPI | Fraud only | All 3 models + health endpoints |

### Module invocation pattern

All runnable modules use the `-m` flag from the **repo root**:

```bash
# Linux / macOS
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model
uv run python -m ai.models.fairness_audit --model fraud
uv run python -m ai.models.fairness_audit --model risk
uv run python -m ai.models.fairness_audit --model churn
uv run python -m ai.api.handlers.main
```

```powershell
# Windows
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model
uv run python -m ai.models.fairness_audit --model fraud
uv run python -m ai.models.fairness_audit --model risk
uv run python -m ai.models.fairness_audit --model churn
uv run python -m ai.api.handlers.main
```

This requires `__init__.py` files at every package level. Section 1 covers the full structure.

---

## 1. Feature Engineering Pipeline

### 1.1 Package `__init__.py` files

Create empty `__init__.py` at every level so the `-m` module flag resolves correctly:

```bash
# Linux / macOS
touch ai/__init__.py
touch ai/models/__init__.py
touch ai/models/fraud_detection/__init__.py
touch ai/models/risk_scoring/__init__.py
touch ai/models/churn_prediction/__init__.py
touch ai/pipelines/__init__.py
touch ai/pipelines/ingestion/__init__.py
touch ai/api/__init__.py
touch ai/api/routers/__init__.py
touch ai/api/handlers/__init__.py
```

```powershell
# Windows
$pkgs = @(
  "ai","ai\models","ai\models\fraud_detection",
  "ai\models\risk_scoring","ai\models\churn_prediction",
  "ai\pipelines","ai\pipelines\ingestion",
  "ai\api","ai\api\routers","ai\api\handlers"
)
$pkgs | ForEach-Object { New-Item -ItemType File -Force -Path "$_\__init__.py" }
```

### 1.2 `ai/pipelines/ingestion/feature_engineer.py`

All three models share this extraction step. Environment variables have **no defaults** — missing config raises immediately rather than silently using wrong credentials.

```python
"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return val


def get_engine():
    url = (
        f"postgresql+psycopg2://{_require_env('DB_USER')}:{_require_env('DB_PASSWORD')}"
        f"@{_require_env('DB_HOST')}:{_require_env('DB_PORT')}/{_require_env('DB_NAME')}"
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
            c.filed_date::date - c.incident_date::date             AS days_to_file,
            COUNT(c2.claim_id) OVER (PARTITION BY c.customer_id)   AS customer_claim_count,
            COALESCE(t.avg_drive_score, 50)                        AS avg_drive_score,
            COALESCE(t.hard_brakes_90d, 0)                         AS hard_brakes_90d,
            p.state,
            p.vehicle->>'make'                                     AS vehicle_make,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            c.claim_type,
            COALESCE(ARRAY_LENGTH(c.fraud_signals, 1), 0)          AS fraud_signal_count
        FROM claims c
        JOIN policies p ON p.policy_number = c.policy_number
        JOIN customers cust ON cust.customer_id = c.customer_id
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
    """Returns one row per policy for risk scoring.

    Extra columns included for fairness audit (state, vehicle_make, zip_prefix)
    but excluded from model features via RISK_EXCLUDE in risk model.
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            p.policy_number,
            p.state,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            p.premium_annual,
            COALESCE(p.drive_score, 50)                            AS drive_score,
            COALESCE(cust.credit_score, 650)                       AS credit_score,
            COALESCE((p.vehicle->>'year')::int, 2015)              AS vehicle_year,
            p.vehicle->>'make'                                     AS vehicle_make,
            COUNT(c.claim_id)                                      AS total_claims,
            COALESCE(SUM(c.claim_amount), 0)                       AS total_claim_amount,
            COALESCE(AVG(t12.drive_score), 50)                     AS avg_drive_score_12m,
            COALESCE(AVG(t3.drive_score), 50)                      AS avg_drive_score_3m
        FROM policies p
        JOIN customers cust ON cust.customer_id = p.customer_id
        LEFT JOIN claims c ON c.policy_number = p.policy_number
        LEFT JOIN telematics t12 ON t12.policy_number = p.policy_number
            AND t12.trip_date >= NOW() - INTERVAL '365 days'
        LEFT JOIN telematics t3 ON t3.policy_number = p.policy_number
            AND t3.trip_date >= NOW() - INTERVAL '90 days'
        GROUP BY p.policy_number, p.state, cust.zip,
                 p.premium_annual, p.drive_score, cust.credit_score,
                 p.vehicle->>'year', p.vehicle->>'make'
    """)
    return pd.read_sql(sql, engine)


def churn_features(engine=None) -> pd.DataFrame:
    """Returns one row per customer with churn label (lapsed/cancelled = 1).

    Extra columns included for fairness audit (state, zip_prefix).
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            cust.customer_id,
            cust.state,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            (MAX(p.status) IN ('lapsed','cancelled'))::int         AS label,
            COALESCE(cust.credit_score, 650)                       AS credit_score,
            COUNT(p.policy_number)                                 AS policy_count,
            COALESCE(AVG(p.premium_annual), 0)                     AS avg_premium,
            COALESCE(AVG(p.drive_score), 50)                       AS avg_drive_score,
            COALESCE(AVG(t12.drive_score), 50)                     AS avg_drive_score_12m,
            COALESCE(AVG(t3.drive_score), 50)                      AS avg_drive_score_3m,
            -- Drive score delta: negative = deteriorating driving, churn signal
            COALESCE(AVG(t3.drive_score), 50)
              - COALESCE(AVG(t12.drive_score), 50)                 AS drive_score_delta,
            COUNT(c.claim_id)                                      AS total_claims,
            MAX(CASE WHEN p.status = 'active' THEN 1 ELSE 0 END)  AS has_active_policy
        FROM customers cust
        LEFT JOIN policies p ON p.customer_id = cust.customer_id
        LEFT JOIN claims c ON c.customer_id = cust.customer_id
        LEFT JOIN telematics t12 ON t12.policy_number = p.policy_number
            AND t12.trip_date >= NOW() - INTERVAL '365 days'
        LEFT JOIN telematics t3 ON t3.policy_number = p.policy_number
            AND t3.trip_date >= NOW() - INTERVAL '90 days'
        GROUP BY cust.customer_id, cust.state, cust.zip, cust.credit_score
    """)
    return pd.read_sql(sql, engine)
```

---

## 2. Logging Utility

A single shared logger used by models, the fairness audit, and API handlers. Emits structured JSON to stdout — CloudWatch Logs ingests it without additional parsing. Minimal: one line per event, no sensitive fields.

**`ai/utils/__init__.py`** — empty

**`ai/utils/log.py`**

```python
"""
log.py — minimal structured JSON logger for AIOI.

Design goals:
  - One import, one call: log.info("event", key=value)
  - Outputs newline-delimited JSON to stdout (CloudWatch-compatible)
  - Never logs free-text fields (chunk_text, adjuster_notes, narratives)
  - No external dependencies beyond stdlib

Usage:
  from ai.utils.log import get_logger
  log = get_logger(__name__)
  log.info("model_trained", model="fraud", roc_auc=0.91, rows=5000)
  log.error("db_connect_failed", error=str(e))
"""
import json
import logging
import os
import sys
import time
from typing import Any


class _JsonHandler(logging.Handler):
    """Emits one JSON object per log record to stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra kwargs passed via log.info("msg", key=val)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        print(json.dumps(payload, default=str), file=sys.stdout, flush=True)


class _StructuredLogger(logging.Logger):
    """Extends Logger to accept keyword arguments as structured fields."""

    def _log_structured(self, level: int, msg: str, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown)", 0, msg, (), None
            )
            record.extra = kwargs  # type: ignore[attr-defined]
            self.handle(record)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_structured(logging.DEBUG, msg, **kwargs)


_initialized = False


def get_logger(name: str) -> _StructuredLogger:
    """Return a named structured logger. Safe to call multiple times."""
    global _initialized
    if not _initialized:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.setLoggerClass(_StructuredLogger)
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(_JsonHandler())
        root.setLevel(level)
        _initialized = True
    logging.setLoggerClass(_StructuredLogger)
    return logging.getLogger(name)  # type: ignore[return-value]
```

Add `LOG_LEVEL` to `.env.example`:

```dotenv
# Logging
LOG_LEVEL=INFO   # DEBUG | INFO | WARNING | ERROR
```

---

## 3. Fraud Detection Model

**`ai/models/fraud_detection/model.py`**

```python
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
    proba = model.predict_proba(X_test)[:, 1]
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
    model = train(df)

    # Score full dataset for fairness audit
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    df["predicted_score"] = model.predict_proba(df_proc[feature_cols])[:, 1]
    run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")


if __name__ == "__main__":
    main()
```

---

## 4. Risk Scoring Model

**`ai/models/risk_scoring/model.py`**

Risk scoring uses `XGBRegressor` to predict `premium_annual` as a proxy for actuarial risk. The raw prediction is normalized to a 0–100 scale and bucketed into three tiers.

```python
"""
Risk Scoring — XGBoost regressor predicting premium as a risk proxy.

Module run:  uv run python -m ai.models.risk_scoring.model
Library use: from ai.models.risk_scoring.model import train, predict

Output per policy:
  risk_score      float 0–100   (normalized predicted premium)
  risk_tier       str           "low" | "medium" | "high"

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

TODO Phase 3+: add county-level loss ratios via FIPS code lookup
  when the dataset includes coordinate data.
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
CATEGORICAL = ["state", "vehicle_make"]
# Audit/identity columns excluded from model features
EXCLUDE_COLS = {"policy_number", "premium_annual", "zip_prefix"}
# Tier thresholds (applied to normalized 0–100 score)
TIER_THRESHOLDS = {"low": 40, "medium": 70}  # low < 40 ≤ medium < 70 ≤ high


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
    """Min-max normalize to [0, 100]."""
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
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
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

    # Feature importance for demo
    importance = dict(zip(X.columns, model.feature_importances_))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"MAE: ${mae:,.2f}  |  R²: {r2:.4f}")
    print("Top features:", top5)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    # Save normalization bounds alongside model
    import json
    bounds_path = MODEL_PATH.with_suffix(".bounds.json")
    bounds_path.write_text(json.dumps({"min": min_pred, "max": max_pred}))

    log.info(
        "model_trained",
        model="risk",
        rows=len(df),
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

    # Clamp and normalize using training bounds
    raw = np.clip(raw, min_pred, max_pred)
    scores = _normalize(raw) if max_pred > min_pred else raw

    for i, rec in enumerate(records):
        score = round(float(scores[i]), 2)
        rec["risk_score"] = score
        rec["risk_tier"] = _tier(score)
    return records


def main() -> None:
    from ai.pipelines.ingestion.feature_engineer import risk_features
    from ai.models.fairness_audit import run_audit

    df = risk_features()
    log.info("risk_train_start", rows=len(df))
    model, min_pred, max_pred = train(df)

    # Score full dataset for fairness audit using normalized risk score
    df_proc, _ = preprocess(df.copy())
    feature_cols = _feature_cols(df_proc)
    raw = model.predict(df_proc[feature_cols])
    raw = np.clip(raw, min_pred, max_pred)
    scores = _normalize(raw) if max_pred > min_pred else raw
    df["predicted_score"] = scores

    # Fairness audit treats high-risk tier as "positive" outcome
    df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)
    run_audit(df, model_name="risk", score_col="predicted_score", label_col="label")

    # TODO Phase 3+: add county-level loss ratios via FIPS code lookup.


if __name__ == "__main__":
    main()
```

---

## 5. Churn Prediction Model

**`ai/models/churn_prediction/model.py`**

Churn uses the same binary classifier structure as fraud. The distinguishing feature is `drive_score_delta` (3-month minus 12-month average) — customers whose driving is deteriorating churn at higher rates.

```python
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
```

---

## 6. Fairness Audit

**`ai/models/fairness_audit.py`**

Implements disparate impact analysis sliced by `state`, `zip_prefix`, and `vehicle_make`. Flags any slice whose model-predicted positive rate deviates more than ±2× from the overall rate without a matching deviation in the labeled rate.

```python
"""
Fairness Audit — post-training disparate impact analysis.

Module run:
  uv run python -m ai.models.fairness_audit --model fraud
  uv run python -m ai.models.fairness_audit --model risk
  uv run python -m ai.models.fairness_audit --model churn

Called programmatically after each model trains:
  from ai.models.fairness_audit import run_audit
  run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")

Slice dimensions:
  state         — 50 states + DC
  zip_prefix    — first 3 digits of ZIP (geographic proxy)
  vehicle_make  — vehicle manufacturer (proxy for owner demographics in fraud/risk)

Threshold: flag if a slice's predicted positive rate deviates > 2× from the
overall rate without a corresponding deviation in the labeled positive rate.
Minimum slice size: MIN_SLICE_SIZE rows (slices smaller than this are skipped
to avoid high-variance flags on tiny populations).

Output: prints a report to stdout; logs summary via structured logger.
Writes a JSON report to ai/models/fairness_reports/<model>_<timestamp>.json
for record-keeping.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from ai.utils.log import get_logger

log = get_logger(__name__)

MIN_SLICE_SIZE = 50       # skip slices smaller than this
DEVIATION_THRESHOLD = 2.0  # flag if predicted rate / overall rate > threshold
REPORT_DIR = Path("ai/models/fairness_reports")

# Slice columns available per model
SLICE_COLS: dict[str, list[str]] = {
    "fraud":  ["state", "zip_prefix", "vehicle_make"],
    "risk":   ["state", "zip_prefix", "vehicle_make"],
    "churn":  ["state", "zip_prefix"],
}


def _positive_rate(series: pd.Series) -> float:
    """Fraction of values >= 0.5 (predicted positive rate)."""
    return float((series >= 0.5).mean())


def _analyze_slice(
    df: pd.DataFrame,
    slice_col: str,
    score_col: str,
    label_col: str,
    overall_pred_rate: float,
    overall_label_rate: float,
) -> list[dict[str, Any]]:
    """Return flagged slice records for one dimension."""
    flags = []
    for val, grp in df.groupby(slice_col):
        if len(grp) < MIN_SLICE_SIZE:
            continue
        pred_rate = _positive_rate(grp[score_col])
        label_rate = float(grp[label_col].mean())

        ratio = pred_rate / overall_pred_rate if overall_pred_rate > 0 else 0.0
        label_ratio = label_rate / overall_label_rate if overall_label_rate > 0 else 0.0

        # Flag: predicted rate deviates > threshold from overall,
        # but label rate does NOT — suggests model bias, not ground truth
        predicted_deviation = abs(ratio - 1.0) > (DEVIATION_THRESHOLD - 1.0)
        label_explains_it = abs(label_ratio - 1.0) > (DEVIATION_THRESHOLD - 1.0)
        flagged = predicted_deviation and not label_explains_it

        if flagged:
            flags.append({
                "slice_col": slice_col,
                "slice_val": str(val),
                "n": len(grp),
                "pred_rate": round(pred_rate, 4),
                "label_rate": round(label_rate, 4),
                "overall_pred_rate": round(overall_pred_rate, 4),
                "overall_label_rate": round(overall_label_rate, 4),
                "pred_ratio": round(ratio, 3),
                "label_ratio": round(label_ratio, 3),
                "flagged": True,
            })
    return flags


def run_audit(
    df: pd.DataFrame,
    model_name: str,
    score_col: str,
    label_col: str,
) -> dict[str, Any]:
    """
    Run disparate impact analysis. Prints report, logs summary, writes JSON.

    Args:
        df:          DataFrame containing score_col, label_col, and slice columns.
        model_name:  "fraud" | "risk" | "churn"
        score_col:   Column with model predicted probability / score (0–1 or 0–100).
        label_col:   Column with ground truth binary label (0/1).

    Returns:
        Report dict (same structure written to JSON).
    """
    t0 = time.time()
    slice_cols = SLICE_COLS.get(model_name, ["state"])
    available_slices = [c for c in slice_cols if c in df.columns]

    if score_col not in df.columns or label_col not in df.columns:
        log.warning(
            "fairness_audit_skipped",
            model=model_name,
            reason="score_col or label_col missing from dataframe",
        )
        return {}

    # Normalize score to 0-1 if it looks like 0-100 range (risk model)
    if df[score_col].max() > 1.5:
        df = df.copy()
        df[score_col] = df[score_col] / 100.0

    overall_pred_rate = _positive_rate(df[score_col])
    overall_label_rate = float(df[label_col].mean())
    total_rows = len(df)

    all_flags: list[dict[str, Any]] = []
    slice_summaries: list[dict[str, Any]] = []

    for col in available_slices:
        flags = _analyze_slice(
            df, col, score_col, label_col, overall_pred_rate, overall_label_rate
        )
        all_flags.extend(flags)
        slice_summaries.append({"col": col, "flags": len(flags)})

    # Print report
    print(f"\n{'='*60}")
    print(f"FAIRNESS AUDIT — {model_name.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Rows: {total_rows:,}  |  Overall predicted positive rate: {overall_pred_rate:.2%}")
    print(f"Overall labeled positive rate: {overall_label_rate:.2%}")
    print(f"Deviation threshold: {DEVIATION_THRESHOLD}×  |  Min slice size: {MIN_SLICE_SIZE}")
    print()

    if not all_flags:
        print("✓ No disparate impact flags found.")
    else:
        print(f"⚠  {len(all_flags)} slice(s) flagged for review:\n")
        for f in all_flags:
            print(
                f"  [{f['slice_col']}={f['slice_val']}] n={f['n']}  "
                f"pred_rate={f['pred_rate']:.2%} ({f['pred_ratio']:.2f}× overall)  "
                f"label_rate={f['label_rate']:.2%} ({f['label_ratio']:.2f}× overall)"
            )
        print(
            "\n  Action: review flagged slices before production deployment.",
        )
    print(f"{'='*60}\n")

    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_rows": total_rows,
        "overall_pred_rate": round(overall_pred_rate, 4),
        "overall_label_rate": round(overall_label_rate, 4),
        "threshold": DEVIATION_THRESHOLD,
        "min_slice_size": MIN_SLICE_SIZE,
        "slices_analyzed": available_slices,
        "total_flags": len(all_flags),
        "flags": all_flags,
        "elapsed_s": round(time.time() - t0, 3),
    }

    # Write JSON report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    report_path = REPORT_DIR / f"{model_name}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))

    log.info(
        "fairness_audit_complete",
        model=model_name,
        total_rows=total_rows,
        total_flags=len(all_flags),
        report_path=str(report_path),
        elapsed_s=report["elapsed_s"],
    )

    return report


def _load_and_score(model_name: str) -> tuple[pd.DataFrame, str, str]:
    """Load features and a saved model, return scored DataFrame."""
    if model_name == "fraud":
        from ai.pipelines.ingestion.feature_engineer import fraud_features
        from ai.models.fraud_detection.model import preprocess, _feature_cols, MODEL_PATH
        import xgboost as xgb

        df = fraud_features()
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        df_proc, _ = preprocess(df.copy())
        df["predicted_score"] = model.predict_proba(df_proc[_feature_cols(df_proc)])[:, 1]
        return df, "predicted_score", "label"

    elif model_name == "risk":
        from ai.pipelines.ingestion.feature_engineer import risk_features
        from ai.models.risk_scoring.model import preprocess, _feature_cols, MODEL_PATH, TIER_THRESHOLDS
        import xgboost as xgb
        import json as _json
        import numpy as np

        df = risk_features()
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        bounds = _json.loads(MODEL_PATH.with_suffix(".bounds.json").read_text())
        df_proc, _ = preprocess(df.copy())
        raw = model.predict(df_proc[_feature_cols(df_proc)])
        raw = np.clip(raw, bounds["min"], bounds["max"])
        df["predicted_score"] = (raw - bounds["min"]) / max(bounds["max"] - bounds["min"], 1) * 100
        df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)
        return df, "predicted_score", "label"

    elif model_name == "churn":
        from ai.pipelines.ingestion.feature_engineer import churn_features
        from ai.models.churn_prediction.model import preprocess, _feature_cols, MODEL_PATH
        import xgboost as xgb

        df = churn_features()
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        df_proc, _ = preprocess(df.copy())
        df["predicted_score"] = model.predict_proba(df_proc[_feature_cols(df_proc)])[:, 1]
        return df, "predicted_score", "label"

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: fraud | risk | churn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fairness audit on a trained model.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["fraud", "risk", "churn"],
        help="Model to audit.",
    )
    args = parser.parse_args()
    df, score_col, label_col = _load_and_score(args.model)
    run_audit(df, model_name=args.model, score_col=score_col, label_col=label_col)


if __name__ == "__main__":
    main()
```

---

## 7. FastAPI Application

### 7.1 `ai/api/routers/models_router.py`

All three models exposed with consistent request/response contracts.

```python
"""
models_router.py — FastAPI routes for fraud, risk, and churn model endpoints.
"""
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.utils.log import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


# ── Request / Response models ──────────────────────────────────────────────

class ClaimBatch(BaseModel):
    claims: list[dict]

class PolicyBatch(BaseModel):
    policies: list[dict]

class CustomerBatch(BaseModel):
    customers: list[dict]

class ScoredResponse(BaseModel):
    results: list[dict]
    request_id: str
    latency_ms: int


# ── Fraud ──────────────────────────────────────────────────────────────────

@router.post("/fraud/score", response_model=ScoredResponse)
async def score_fraud(batch: ClaimBatch):
    from ai.models.fraud_detection.model import predict
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        results = predict(batch.claims)
        latency = int((time.time() - t0) * 1000)
        log.info("fraud_scored", request_id=request_id, n=len(results), latency_ms=latency)
        return ScoredResponse(results=results, request_id=request_id, latency_ms=latency)
    except Exception as e:
        log.error("fraud_score_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud/health")
async def fraud_health():
    from ai.models.fraud_detection.model import MODEL_PATH
    return {"status": "ok", "model": "xgboost-fraud-v1", "model_exists": MODEL_PATH.exists()}


# ── Risk ───────────────────────────────────────────────────────────────────

@router.post("/risk/score", response_model=ScoredResponse)
async def score_risk(batch: PolicyBatch):
    from ai.models.risk_scoring.model import predict
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        results = predict(batch.policies)
        latency = int((time.time() - t0) * 1000)
        log.info("risk_scored", request_id=request_id, n=len(results), latency_ms=latency)
        return ScoredResponse(results=results, request_id=request_id, latency_ms=latency)
    except Exception as e:
        log.error("risk_score_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/health")
async def risk_health():
    from ai.models.risk_scoring.model import MODEL_PATH
    return {"status": "ok", "model": "xgboost-risk-v1", "model_exists": MODEL_PATH.exists()}


# ── Churn ──────────────────────────────────────────────────────────────────

@router.post("/churn/score", response_model=ScoredResponse)
async def score_churn(batch: CustomerBatch):
    from ai.models.churn_prediction.model import predict
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        results = predict(batch.customers)
        latency = int((time.time() - t0) * 1000)
        log.info("churn_scored", request_id=request_id, n=len(results), latency_ms=latency)
        return ScoredResponse(results=results, request_id=request_id, latency_ms=latency)
    except Exception as e:
        log.error("churn_score_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/churn/health")
async def churn_health():
    from ai.models.churn_prediction.model import MODEL_PATH
    return {"status": "ok", "model": "xgboost-churn-v1", "model_exists": MODEL_PATH.exists()}
```

### 7.2 `ai/api/handlers/main.py`

```python
"""
main.py — FastAPI entry point, Lambda-compatible via Mangum.

Module run: uv run python -m ai.api.handlers.main
Lambda:     handler = Mangum(app)

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
  LOG_LEVEL (optional, defaults to INFO)
"""
import time
import uuid

from fastapi import FastAPI, Request, Response
from mangum import Mangum

from ai.api.routers.models_router import router as models_router
from ai.utils.log import get_logger

log = get_logger(__name__)

app = FastAPI(title="AIOI AI API", version="0.3.0")
app.include_router(models_router)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next) -> Response:
    """Log one structured line per request. Never logs body content."""
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    response = await call_next(request)
    latency = int((time.time() - t0) * 1000)
    log.info(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency,
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


# Lambda handler — used by Mangum in Phase 5
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.api.handlers.main:app", host="0.0.0.0", port=8000, reload=True)
```

> **Note on `reload=True`:** The `uvicorn.run` call uses the module path string `"ai.api.handlers.main:app"` (not `"main:app"`) because we run from the repo root with `-m`. This is required for hot-reload to resolve imports correctly.

---

## 8. Verification & Git Tag

### 8.1 Full build sequence

```bash
# Linux / macOS — run from repo root

# 1. Ensure Docker is running with data loaded
docker compose up -d postgres
uv run python -m db.load_json --truncate  # if not already loaded

# 2. Create __init__.py files (one-time)
touch ai/__init__.py ai/models/__init__.py ai/utils/__init__.py \
      ai/models/fraud_detection/__init__.py \
      ai/models/risk_scoring/__init__.py \
      ai/models/churn_prediction/__init__.py \
      ai/pipelines/__init__.py ai/pipelines/ingestion/__init__.py \
      ai/api/__init__.py ai/api/routers/__init__.py ai/api/handlers/__init__.py

# 3. Train all models (each also runs fairness audit)
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model

# 4. Run standalone fairness audits (re-runs on saved models without retraining)
uv run python -m ai.models.fairness_audit --model fraud
uv run python -m ai.models.fairness_audit --model risk
uv run python -m ai.models.fairness_audit --model churn

# 5. Start API
uv run python -m ai.api.handlers.main
```

```powershell
# Windows — run from repo root

# 1. Ensure Docker is running
docker compose up -d postgres
uv run python -m db.load_json

# 2. Create __init__.py files
$pkgs = @(
  "ai","ai\models","ai\utils",
  "ai\models\fraud_detection","ai\models\risk_scoring","ai\models\churn_prediction",
  "ai\pipelines","ai\pipelines\ingestion",
  "ai\api","ai\api\routers","ai\api\handlers"
)
$pkgs | ForEach-Object { New-Item -ItemType File -Force -Path "$_\__init__.py" }

# 3. Train
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model

# 4. Fairness audits
uv run python -m ai.models.fairness_audit --model fraud
uv run python -m ai.models.fairness_audit --model risk
uv run python -m ai.models.fairness_audit --model churn

# 5. Start API
uv run python -m ai.api.handlers.main
```

### 8.2 API smoke tests

```bash
# Linux / macOS — in a second terminal

# Health check
curl http://localhost:8000/health

# Fraud score
# Positive example
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99001",
    "claim_amount": 18500,
    "claim_to_premium_ratio": 4.8,
    "days_to_file": 0,
    "customer_claim_count": 5,
    "avg_drive_score": 22,
    "hard_brakes_90d": 67,
    "state": "FL",
    "vehicle_make": "BMW",
    "zip_prefix": "331",
    "claim_type": "collision",
    "fraud_signal_count": 4
  }]
}' | python -m json.tool

# Neutral example:
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99003",
    "claim_amount": 6200,
    "claim_to_premium_ratio": 1.8,
    "days_to_file": 3,
    "customer_claim_count": 2,
    "avg_drive_score": 54,
    "hard_brakes_90d": 18,
    "state": "TX",
    "vehicle_make": "Ford",
    "zip_prefix": "750",
    "claim_type": "collision",
    "fraud_signal_count": 1
  }]
}' | python -m json.tool

# Negative example
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99002",
    "claim_amount": 1800,
    "claim_to_premium_ratio": 0.6,
    "days_to_file": 5,
    "customer_claim_count": 1,
    "avg_drive_score": 84,
    "hard_brakes_90d": 3,
    "state": "VT",
    "vehicle_make": "Toyota",
    "zip_prefix": "054",
    "claim_type": "comprehensive",
    "fraud_signal_count": 0
  }]
}' | python -m json.tool

# Risk score
curl -s -X POST http://localhost:8000/models/risk/score `
  -H "Content-Type: application/json" `
  -d '{
  "policies": [{
    "policy_number": "TX-00142",
    "state": "TX",
    "is_telematics_enrolled": 1,
    "zip_prefix": "750",
    "drive_score": 38,
    "credit_score": 540,
    "vehicle_year": 2009,
    "vehicle_make": "Dodge",
    "avg_drive_score_12m": 45,
    "avg_drive_score_3m": 36,    
    "coverage_count": 5,
    "has_collision": 1,
    "has_comprehensive": 1,
    "has_pip": 0,
    "drive_score_delta": -9
  }]
}' | python -m json.tool

# low tier example
curl -s -X POST http://localhost:8000/models/risk/score `
  -H "Content-Type: application/json" `
  -d '{
  "policies": [{
    "policy_number": "VT-00021",
    "state": "VT",
    "is_telematics_enrolled": 1,
    "zip_prefix": "056",
    "drive_score": 97,
    "credit_score": 820,
    "vehicle_year": 2015,
    "vehicle_make": "Toyota",
    "avg_drive_score_12m": 93,
    "avg_drive_score_3m": 96,    
    "coverage_count": 2,
    "has_collision": 0,
    "has_comprehensive": 0,
    "has_pip": 0,
    "drive_score_delta": 3
  }]
}' | python -m json.tool

# Churn score
curl -s -X POST http://localhost:8000/models/churn/score \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "customer_id": "CUST-08821",
      "state": "TX",
      "zip_prefix": "750",
      "credit_score": 680,
      "policy_count": 1,
      "avg_premium": 1450,
      "avg_drive_score": 72,
      "avg_drive_score_12m": 75,
      "avg_drive_score_3m": 61,
      "drive_score_delta": -14,
      "total_claims": 2,
      "has_active_policy": 1
    }]
  }' | python -m json.tool
```

```powershell
# Windows — use Invoke-RestMethod

# Health check
Invoke-RestMethod http://localhost:8000/health

# Fraud score
$body = @{
  claims = @(@{
    claim_id = "CLM-00001"; claim_amount = 12000; claim_to_premium_ratio = 3.2
    days_to_file = 1; customer_claim_count = 4; avg_drive_score = 31
    hard_brakes_90d = 45; state = "TX"; vehicle_make = "Toyota"
    zip_prefix = "750"; claim_type = "collision"; fraud_signal_count = 3
  })
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri http://localhost:8000/models/fraud/score `
  -Method POST -ContentType "application/json" -Body $body
```

### 8.3 Expected log output

Training a model emits one structured line per phase:

```json
{"ts":"2026-03-15T14:22:01Z","level":"INFO","logger":"ai.models.fraud_detection.model","msg":"fraud_train_start","rows":3200,"fraud_rows":128}
{"ts":"2026-03-15T14:22:04Z","level":"INFO","logger":"ai.models.fraud_detection.model","msg":"model_trained","model":"fraud","rows":3200,"fraud_rows":128,"roc_auc":0.9312,"elapsed_s":3.14}
{"ts":"2026-03-15T14:22:05Z","level":"INFO","logger":"ai.models.fairness_audit","msg":"fairness_audit_complete","model":"fraud","total_rows":3200,"total_flags":0,"report_path":"ai/models/fairness_reports/fraud_20260315_142205.json","elapsed_s":0.21}
```

An HTTP request logs one line:

```json
{"ts":"2026-03-15T14:23:10Z","level":"INFO","logger":"ai.api.handlers.main","msg":"http_request","request_id":"a3f9c1b2","method":"POST","path":"/models/fraud/score","status":200,"latency_ms":12}
```

### 8.4 Phase Gate Checklist

- [ ] All `__init__.py` files created — `uv run python -m ai.models.fraud_detection.model` resolves without `ModuleNotFoundError`
- [ ] All 3 models train without errors; ROC-AUC > 0.85 reported for fraud and churn
- [ ] Risk model writes `risk_model.json` and `risk_model.bounds.json`
- [ ] Fairness audit runs for all 3 models; JSON reports written to `ai/models/fairness_reports/`
- [ ] No flags for clean synthetic data (optional — flags are expected if intentional bias was injected)
- [ ] Fraud API returns `fraud_score`, `is_fraud_predicted`, `request_id`, `latency_ms`
- [ ] Risk API returns `risk_score`, `risk_tier`, `request_id`, `latency_ms`
- [ ] Churn API returns `churn_probability`, `churn_predicted`, `request_id`, `latency_ms`
- [ ] Log output is newline-delimited JSON (no plain-text lines)
- [ ] `LOG_LEVEL=DEBUG uv run python -m ai.models.fraud_detection.model` produces debug output
- [ ] Missing env var causes `EnvironmentError` with the variable name, not a silent default
- [ ] `uv run ruff check .` passes

### 8.5 Git Tag

```bash
git add -A
git commit -m "Phase 3: ML models — fraud, risk, churn, fairness audit, structured logging"
git tag v0.3.0
```

---

*Previous: [PHASE_2_DATABASE.md](./PHASE_2_DATABASE.md) · Next: [PHASE_4_RAG.md](./PHASE_4_RAG.md)*

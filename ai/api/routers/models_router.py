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
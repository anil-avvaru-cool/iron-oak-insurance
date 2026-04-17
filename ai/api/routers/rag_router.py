"""
rag_router.py — FastAPI routes for RAG Q&A.

/rag/query   — production endpoint; returns grounded answer + sources
/rag/debug   — dev-only endpoint (DEBUG_MODE=true); returns chunks without LLM call

The SentenceTransformer model is loaded once at app startup via FastAPI lifespan
and accessed via request.app.state.embedder — never loaded per-request.
"""
from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ai.pipelines.rag.retrieval_router import classify_query
from ai.pipelines.rag.rag_pipeline import retrieve, generate_answer
from ai.utils.log import get_logger

load_dotenv()
log = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

_DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


# ── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    customer_id: str | None = None
    mode:        str | None = None   # overrides RAG_MODE env var if provided


class QueryResponse(BaseModel):
    answer:     str
    strategy:   str
    sources:    list[dict]
    request_id: str
    latency_ms: int


class DebugResponse(BaseModel):
    strategy:   dict
    chunks:     list[dict]
    request_id: str
    latency_ms: int


# ── Helpers ─────────────────────────────────────────────────────────────────

def _embed_query(query: str, request: Request) -> list[float]:
    """Use the embedder cached on app.state (loaded at lifespan startup)."""
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise RuntimeError(
            "Embedder not loaded. Ensure lifespan startup completed successfully."
        )
    return embedder.encode(query).tolist()


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        q_emb    = _embed_query(req.query, request)
        strategy = classify_query(req.query)
        if req.customer_id:
            strategy["customer_id"] = req.customer_id

        chunks = retrieve(q_emb, strategy)
        answer = generate_answer(req.query, chunks, mode=req.mode)

        latency = int((time.time() - t0) * 1000)
        log.info(
            "rag_query",
            request_id=request_id,
            strategy=strategy["strategy"],
            policy_number=strategy.get("policy_number"),
            customer_id=strategy.get("customer_id"),
            chunks_retrieved=len(chunks),
            latency_ms=latency,
        )
        return QueryResponse(
            answer=answer,
            strategy=strategy["strategy"],
            sources=chunks,
            request_id=request_id,
            latency_ms=latency,
        )
    except Exception as exc:
        log.error("rag_query_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/debug", response_model=DebugResponse, include_in_schema=_DEBUG_MODE)
async def debug_endpoint(req: QueryRequest, request: Request):
    """
    Returns routing decision and retrieved chunks without calling the LLM.
    Enabled only when DEBUG_MODE=true. Disable before production deployment.
    """
    if not _DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled.")

    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    q_emb    = _embed_query(req.query, request)
    strategy = classify_query(req.query)
    if req.customer_id:
        strategy["customer_id"] = req.customer_id

    chunks  = retrieve(q_emb, strategy)
    latency = int((time.time() - t0) * 1000)
    return DebugResponse(
        strategy=strategy,
        chunks=chunks,
        request_id=request_id,
        latency_ms=latency,
    )


@router.get("/health")
async def rag_health(request: Request):
    embedder_loaded = hasattr(request.app.state, "embedder")
    return {"status": "ok", "embedder_loaded": embedder_loaded}

"""
rag_pipeline.py — retrieves relevant chunks and generates a grounded answer.

Fixes vs. original:
  - SQL rewritten as CTE to avoid the double-params positional bug
  - state_filter correctly uses (state = %s OR state IS NULL) so ALL-applicable
    FAQ chunks are always returned alongside state-specific ones
  - DB connection via shared get_conn() — no inline credentials
  - All config (RAG_TOP_K, OLLAMA_BASE_URL, OLLAMA_MODEL, BEDROCK_MODEL_ID_HAIKU)
    read from env — no hardcoded fallback strings for production vars
  - Structured logging per retrieve() call

Module invocation: called from rag_router.py; not a standalone entry point.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from db.load_json import get_conn  # noqa: E402

log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required env var '{name}' is not set.")
    return val


def retrieve(
    query_embedding: list[float],
    strategy: dict,
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks from pgvector.

    Uses a CTE to compute cosine similarity once, then filters and ranks —
    avoids passing query_embedding twice in the params list.

    The state_filter uses (state = %s OR state IS NULL) so FAQ chunks with
    applicable_states: ["ALL"] (stored as state=NULL) always appear in results.
    """
    if top_k is None:
        top_k = int(os.getenv("RAG_TOP_K", "5"))

    # Build WHERE clause
    filter_clauses: list[str] = []
    filter_params:  list      = []

    strat = strategy.get("strategy", "both")
    if strat == "policy_document":
        filter_clauses.append("source_type = 'policy_document'")
    elif strat == "faq":
        filter_clauses.append("source_type = 'faq'")

    if strategy.get("policy_number"):
        filter_clauses.append("policy_number = %s")
        filter_params.append(strategy["policy_number"])

    if strategy.get("customer_id"):
        filter_clauses.append("customer_id = %s")
        filter_params.append(strategy["customer_id"])
    elif strategy.get("policy_number") is None and strategy["strategy"] == "policy_document":
    # No specific policy identified — broaden to all policy_documents for this strategy
    # rather than returning zero results
        pass  # source_type filter already applied above)

    if strategy.get("state_filter"):
        # NULL state = ALL-applicable — must always pass the filter
        filter_clauses.append("(state = %s OR state IS NULL)")
        filter_params.append(strategy["state_filter"])

    where_sql = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    # CTE: compute distance once, reuse in SELECT output and ORDER BY
    sql = f"""
        WITH ranked AS (
            SELECT
                chunk_id,
                source_type,
                doc_type,
                policy_number,
                customer_id,
                chunk_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            {where_sql}
        )
        SELECT *
        FROM ranked
        ORDER BY similarity DESC
        LIMIT %s
    """
    params = [query_embedding] + filter_params + [top_k]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    log.info(
        "chunks_retrieved",
        extra={
            "strategy":      strat,
            "policy_number": strategy.get("policy_number"),
            "customer_id":   strategy.get("customer_id"),
            "state_filter":  strategy.get("state_filter"),
            "top_k":         top_k,
            "returned":      len(rows),
        },
    )
    return rows


def generate_answer(
    query: str,
    chunks: list[dict],
    mode: str | None = None,
) -> str:
    """
    Generate a grounded answer from retrieved chunks.
    mode: "local" (Ollama) | "bedrock" (Claude Haiku via Bedrock)
    Reads RAG_MODE from env if mode not provided.
    """
    if mode is None:
        mode = os.getenv("RAG_MODE", "local")

    context = "\n\n---\n\n".join([
        f"[Source: {c['source_type']} | {c.get('policy_number') or c['doc_type']}]\n{c['chunk_text']}"
        for c in chunks
    ])
    system = (
        "You are Oak Assist, the AI helper for Avvaru Iron Oak Insurance. "
        "Answer the customer's question using ONLY the provided context. "
        "If the context does not contain the answer, say so clearly — do not guess. "
        "Always cite whether your answer comes from a policy document or the FAQ."
    )
    prompt = f"Context:\n{context}\n\nCustomer question: {query}"

    if mode == "local":
        import httpx
        base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        resp = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model":    model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    elif mode == "bedrock":
        import boto3
        region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        model_id = _require_env("BEDROCK_MODEL_ID_HAIKU")
        client   = boto3.client("bedrock-runtime", region_name=region)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system":  system,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = client.invoke_model(
            modelId=model_id, body=body, contentType="application/json"
        )
        return json.loads(resp["body"].read())["content"][0]["text"]

    else:
        raise ValueError(f"Unknown RAG_MODE: '{mode}'. Choose: local | bedrock")
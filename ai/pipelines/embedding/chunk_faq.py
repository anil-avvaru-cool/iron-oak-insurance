"""
chunk_faq.py — one chunk per Q&A pair, no splitting, no overlap.

The question text is prepended to the answer so semantic search on
either surface retrieves the correct pair.

Token counting uses tiktoken (cl100k_base) for accuracy. Falls back
to word-count estimate if tiktoken is not installed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_MAX_TOKENS = 200  # warn if exceeded — FAQ answers should be concise

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    log.warning("tiktoken not installed — using word-count approximation for token counts")
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)  # rough correction factor


def chunk_faq_records(faq_path: Path) -> list[dict]:
    """
    Convert FAQ JSON corpus into pgvector-ready chunk dicts.

    Each record becomes exactly one chunk:
      chunk_text = "Q: <question>\\nA: <answer>"
      state      = first applicable_state if not ALL; else NULL
    """
    records: list[dict] = json.loads(faq_path.read_text())
    chunks: list[dict] = []

    for rec in records:
        chunk_text = f"Q: {rec['question']}\nA: {rec['answer']}"
        token_count = _count_tokens(chunk_text)

        if token_count > _MAX_TOKENS:
            log.warning(
                "faq_chunk_oversized",
                extra={
                    "faq_id": rec["faq_id"],
                    "token_count": token_count,
                    "max": _MAX_TOKENS,
                },
            )

        # State: NULL for ALL-applicable FAQs so state-filtered queries still return them
        states = rec.get("applicable_states", ["ALL"])
        state = states[0] if states != ["ALL"] and len(states) == 1 else None

        chunks.append({
            "chunk_id":      rec["faq_id"],
            "source_type":   "faq",
            "doc_type":      "faq",
            "policy_number": None,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       rec["category"],
            "chunk_index":   0,
            "token_count":   token_count,
            "chunk_text":    chunk_text,
        })

    log.info("faq_chunks_built", extra={"count": len(chunks)})
    return chunks
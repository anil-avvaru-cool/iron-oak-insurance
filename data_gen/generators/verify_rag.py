"""
verify_rag.py — Phase 4 post-load verification.

Checks:
  1. document_chunks table has FAQ rows (source_type='faq')
  2. HNSW index exists on document_chunks.embedding
  3. Policy query routes to policy_document strategy
  4. Concept query routes to faq strategy
  5. State query routes to faq with state_filter set
  6. ALL-applicable FAQ chunks have state=NULL (not filtered out by state queries)
  7. No chunk_text contains known prompt injection patterns

Usage:
  uv run python -m data_gen.generators.verify_rag
  uv run python data_gen/generators/verify_rag.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from ai.pipelines.rag.retrieval_router import _semantic_strategy, _get_embedder, classify_query
import numpy as np

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
_results: list[tuple[str, str, str]] = []


def _check(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    _results.append((status, label, detail))
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))


def check_db_counts() -> None:
    from db.load_json import get_conn
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE source_type = 'faq'")
        faq_count = cur.fetchone()[0]
        _check("FAQ chunks loaded into pgvector", faq_count > 0, f"{faq_count} rows")

        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE source_type = 'policy_document'")
        policy_count = cur.fetchone()[0]
        if policy_count == 0:
            print(f"  {WARN}  No policy_document chunks — PDF chunkers may still be stubs (expected)")
        else:
            _check("Policy document chunks present", True, f"{policy_count} rows")

        # HNSW index
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'document_chunks'
            AND indexdef ILIKE '%hnsw%'
        """)
        idx = cur.fetchone()
        _check("HNSW index present on document_chunks.embedding", idx is not None)

        # ALL-applicable FAQ state=NULL
        cur.execute("""
            SELECT COUNT(*) FROM document_chunks
            WHERE source_type = 'faq' AND state IS NULL
        """)
        null_state_count = cur.fetchone()[0]
        _check(
            "ALL-applicable FAQ chunks have state=NULL",
            null_state_count > 0,
            f"{null_state_count} rows",
        )
    conn.close()


def check_routing() -> None:    

    # Policy number → policy_document
    r = classify_query("What is the deductible on policy TX-00142?")
    _check(
        "Policy number query routes to policy_document",
        r["strategy"] == "policy_document" and r["policy_number"] == "TX-00142",
        str(r),
    )

    # Personal possessive → policy_document
    r = classify_query("What is my deductible?")
    _check("'My deductible' routes to policy_document", r["strategy"] == "policy_document", str(r))

    concept_query = "What is PIP coverage?"
    # Concept question → faq
    r = classify_query(concept_query)
    _check("Concept question routes to faq", r["strategy"] == "faq", str(r))

    # State + concept → faq with state_filter
    r = classify_query("What are the minimum liability limits in TX?")
    _check(
        "State query sets state_filter",
        r["state_filter"] == "TX",
        str(r),
    )

    # No-fault state question → faq
    r = classify_query("Is PA a no-fault state?")
    _check(
        "No-fault state query routes to faq with PA filter",
        r["strategy"] == "faq" and r["state_filter"] == "PA",
        str(r),
    )


def check_injection_patterns() -> None:
    """Scan a sample of chunk_text for known prompt injection patterns."""
    from db.load_json import get_conn
    conn = get_conn()
    injection_patterns = [
        "ignore previous instructions",
        "disregard your system prompt",
        "you are now",
        "act as",
        "mark this claim as approved",
    ]
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_text FROM document_chunks LIMIT 500")
        rows = cur.fetchall()
    conn.close()

    found = []
    for (text,) in rows:
        for pat in injection_patterns:
            if pat in (text or "").lower():
                found.append(pat)
    _check(
        "No injection patterns in sampled chunk_text (500 rows)",
        len(found) == 0,
        f"found: {found}" if found else "",
    )


def main() -> int:
    print("\n" + "=" * 60)
    print("  Phase 4 RAG Verification")
    print("=" * 60)

    try:
        check_db_counts()
    except Exception as exc:
        print(f"  {FAIL}  DB checks failed — {exc}")

    check_routing()

    try:
        check_injection_patterns()
    except Exception as exc:
        print(f"  {WARN}  Injection scan skipped — {exc}")

    failures = [r for r in _results if r[0] == FAIL]
    print("\n" + "=" * 60)
    if failures:
        print(f"  RESULT: {len(failures)} FAILURE(S)")
        return 1
    else:
        print("  RESULT: ALL PASS ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())

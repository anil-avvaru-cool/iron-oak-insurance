"""
chunk_renewal.py — hybrid zone chunker for renewal notices.

Renewal notices have two zones:
  TABLE ZONE   — premium change table at the top (row-aware, no overlap)
  PROSE ZONE   — explanation paragraphs below (paragraph chunker, 50-token overlap)

Zone boundary detection: first blank line after a block that contains
at least 3 lines with dollar amounts ($NNN.NN pattern).

Filename convention:
  renewal_TX-00142.pdf  →  policy_number = "TX-00142"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .chunk_claim_letter import _split_long_paragraph, _add_overlap, _tok

log = logging.getLogger(__name__)

try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF required: uv add PyMuPDF") from exc

_DOLLAR_RE = re.compile(r"\$\d[\d,]*\.?\d*")
_MAX_TABLE_TOKENS = 400
_MAX_PROSE_TOKENS = 350


def _detect_zone_boundary(lines: list[str]) -> int:
    """
    Return the line index where the prose zone begins.
    Heuristic: first blank line after we have seen >= 3 dollar-amount lines
    in a contiguous block.
    """
    dollar_streak = 0
    for i, line in enumerate(lines):
        if _DOLLAR_RE.search(line):
            dollar_streak += 1
        else:
            if dollar_streak >= 3 and not line.strip():
                return i + 1  # first line of prose zone
            if not line.strip():
                dollar_streak = 0
    return len(lines)  # fallback: entire document is table zone


def _chunk_table_zone(lines: list[str], policy_number: str, state: str | None) -> list[dict]:
    """
    Chunk the table zone. Each non-blank line (or short group) becomes a chunk.
    Dollar-amount lines that belong to the same row are grouped together.
    """
    chunks: list[dict] = []
    idx = 0
    buffer: list[str] = []

    def flush(buf: list[str]) -> None:
        nonlocal idx
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append({
                "chunk_id":      f"renewal-{policy_number}-table-{idx:03d}",
                "source_type":   "policy_document",
                "doc_type":      "renewal_notice",
                "policy_number": policy_number,
                "customer_id":   None,
                "state":         state,
                "page_number":   None,
                "section":       "premium_table",
                "chunk_index":   idx,
                "token_count":   _tok(text),
                "chunk_text":    text,
            })
            idx += 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush(buffer)
            buffer = []
            continue
        buffer.append(stripped)
        if _tok("\n".join(buffer)) > _MAX_TABLE_TOKENS:
            flush(buffer)
            buffer = []

    flush(buffer)
    return chunks


def _chunk_prose_zone(
    lines: list[str], policy_number: str, state: str | None, start_idx: int
) -> list[dict]:
    """
    Chunk the prose zone using paragraph splitting with 50-token overlap.
    """
    full_text = "\n".join(lines)
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]

    para_chunks: list[str] = []
    for para in raw_paragraphs:
        if _tok(para) > _MAX_PROSE_TOKENS:
            para_chunks.extend(_split_long_paragraph(para))
        else:
            para_chunks.append(para)

    para_chunks = _add_overlap(para_chunks)

    chunks = []
    for i, text in enumerate(para_chunks):
        chunks.append({
            "chunk_id":      f"renewal-{policy_number}-prose-{start_idx + i:03d}",
            "source_type":   "policy_document",
            "doc_type":      "renewal_notice",
            "policy_number": policy_number,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       "prose_explanation",
            "chunk_index":   start_idx + i,
            "token_count":   _tok(text),
            "chunk_text":    text,
        })
    return chunks


def chunk_renewal_notice(path: Path) -> list[dict]:
    """
    Chunk a renewal notice PDF using hybrid zone detection.
    """
    stem = path.stem  # "renewal_TX-00142"
    prefix = "renewal_"
    policy_number = stem[len(prefix):] if stem.startswith(prefix) else stem
    state = policy_number[:2] if len(policy_number) >= 2 else None

    doc = fitz.open(path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    lines = full_text.split("\n")
    boundary = _detect_zone_boundary(lines)

    table_lines = lines[:boundary]
    prose_lines = lines[boundary:]

    table_chunks = _chunk_table_zone(table_lines, policy_number, state)
    prose_chunks = _chunk_prose_zone(
        prose_lines, policy_number, state, start_idx=len(table_chunks)
    )

    all_chunks = table_chunks + prose_chunks
    log.info(
        "renewal_chunked",
        extra={
            "policy_number": policy_number,
            "table_chunks": len(table_chunks),
            "prose_chunks": len(prose_chunks),
            "file": path.name,
        },
    )
    return all_chunks
"""
chunk_claim_letter.py — paragraph chunker with 50-token overlap for claim letters.

Strategy:
  - Split on paragraph breaks (\\n\\n).
  - Max 350 tokens per chunk; split at sentence boundary if exceeded.
  - 50-token overlap between consecutive chunks for cross-boundary context.
  - claim_id and policy_number injected into every chunk's metadata
    (extracted from filename — do NOT rely on them surviving into chunk text).

Filename convention:
  claim_letter_CLM-00001.pdf  →  claim_id = "CLM-00001"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF required: uv add PyMuPDF") from exc

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tok(text: str) -> int:
        return len(_enc.encode(text))
    def _detok(tokens: list) -> str:
        return _enc.decode(tokens)
except ImportError:
    def _tok(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)
    def _detok(tokens: list) -> str:  # type: ignore[misc]
        return " ".join(str(t) for t in tokens)

_MAX_TOKENS   = 350
_OVERLAP_TOKENS = 50
_SENTENCE_RE  = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_long_paragraph(text: str) -> list[str]:
    """Split a paragraph that exceeds _MAX_TOKENS at sentence boundaries."""
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = _tok(sent)
        if current_tokens + t > _MAX_TOKENS and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def _add_overlap(chunks: list[str]) -> list[str]:
    """
    Prepend the last _OVERLAP_TOKENS tokens of the previous chunk
    to each chunk (except the first).
    """
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        try:
            prev_tokens = _enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-_OVERLAP_TOKENS:]
            overlap_text = _enc.decode(overlap_tokens)
        except Exception:
            # Fallback if tiktoken not available
            words = chunks[i - 1].split()
            overlap_text = " ".join(words[-15:])  # ~50 token approximation
        result.append(overlap_text + " " + chunks[i])
    return result


def chunk_claim_letter(path: Path) -> list[dict]:
    """
    Chunk a claim letter PDF into paragraph-level chunks with overlap.
    """
    stem = path.stem  # "claim_letter_CLM-00001"
    prefix = "claim_letter_"
    claim_id = stem[len(prefix):] if stem.startswith(prefix) else stem

    # Extract policy_number from letter text heuristic (pattern XX-NNNNN)
    doc = fitz.open(path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    policy_match = re.search(r"\b([A-Z]{2}-\d{5})\b", full_text)
    policy_number = policy_match.group(1) if policy_match else None
    state = policy_number[:2] if policy_number else None

    # Split into paragraphs
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]

    # Split oversized paragraphs at sentence boundaries
    para_chunks: list[str] = []
    for para in raw_paragraphs:
        if _tok(para) > _MAX_TOKENS:
            para_chunks.extend(_split_long_paragraph(para))
        else:
            para_chunks.append(para)

    # Add overlap
    para_chunks = _add_overlap(para_chunks)

    chunks = []
    for i, text in enumerate(para_chunks):
        chunks.append({
            "chunk_id":      f"claim-{claim_id}-{i:03d}",
            "source_type":   "policy_document",
            "doc_type":      "claim_letter",
            "policy_number": policy_number,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       "body",
            "chunk_index":   i,
            "token_count":   _tok(text),
            "chunk_text":    text,
        })

    log.info(
        "claim_letter_chunked",
        extra={"claim_id": claim_id, "chunks": len(chunks), "file": path.name},
    )
    return chunks
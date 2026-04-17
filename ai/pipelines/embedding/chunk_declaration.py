"""
chunk_declaration.py — table-aware section chunker for declaration pages.

Strategy:
  1. Extract text blocks from PDF using PyMuPDF, preserving font metadata.
  2. Detect section boundaries by bold headers (font flags) or font size > body.
  3. Split into logical sections: named_insured_block, vehicle_details,
     coverage_table, endorsements.
  4. Coverage table: each row (coverage type + limit + deductible) is one chunk.
  5. Target: 200–400 tokens per chunk; no overlap at section boundaries.

Filename convention (load-bearing for metadata extraction):
  decl_TX-00142.pdf  →  policy_number = "TX-00142"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required for declaration page chunking. "
        "Install with: uv add PyMuPDF"
    ) from exc

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tok(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _tok(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)

# Pattern to identify coverage table rows:
# e.g. "Collision   $500   ACV" or "Liability   30/60/25   —"
_COVERAGE_ROW_RE = re.compile(
    r"(liability|collision|comprehensive|pip|uninsured|gap|roadside)",
    re.IGNORECASE,
)
# Bold flag in PyMuPDF span flags bitmask
_BOLD_FLAG = 1 << 4


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & _BOLD_FLAG)


def _extract_text_blocks(page: "fitz.Page") -> list[dict]:
    """Return list of {text, is_bold, font_size, bbox} for each span on the page."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                blocks.append({
                    "text": text,
                    "is_bold": _is_bold(span),
                    "font_size": span.get("size", 10),
                    "bbox": span.get("bbox"),
                })
    return blocks


def _detect_section(text: str, is_bold: bool) -> str | None:
    """Return section name if this block is a section header, else None."""
    t = text.lower().strip()
    if is_bold:
        if any(k in t for k in ["named insured", "policyholder", "insured:"]):
            return "named_insured_block"
        if any(k in t for k in ["vehicle", "automobile", "auto description"]):
            return "vehicle_details"
        if any(k in t for k in ["coverage", "coverages", "schedule of"]):
            return "coverage_table"
        if any(k in t for k in ["endorsement", "addendum"]):
            return "endorsements"
    return None


def _build_chunk(
    policy_number: str,
    section: str,
    text_lines: list[str],
    chunk_index: int,
) -> dict:
    chunk_text = "\n".join(text_lines).strip()
    return {
        "chunk_id":      f"decl-{policy_number}-{section}-{chunk_index:03d}",
        "source_type":   "policy_document",
        "doc_type":      "declaration_page",
        "policy_number": policy_number,
        "customer_id":   None,  # resolved downstream from policy lookup if needed
        "state":         policy_number[:2] if policy_number else None,
        "page_number":   None,
        "section":       section,
        "chunk_index":   chunk_index,
        "token_count":   _tok(chunk_text),
        "chunk_text":    chunk_text,
    }


def chunk_declaration_page(path: Path) -> list[dict]:
    """
    Chunk a declaration page PDF into section-aware chunks.
    Returns list of chunk dicts ready for embedding.
    """
    # Extract policy number from filename: decl_TX-00142.pdf → TX-00142
    stem = path.stem  # "decl_TX-00142"
    policy_number = stem[len("decl_"):] if stem.startswith("decl_") else stem

    doc = fitz.open(path)
    all_blocks: list[dict] = []
    for page_num, page in enumerate(doc):
        for block in _extract_text_blocks(page):
            block["page_number"] = page_num + 1
            all_blocks.append(block)
    doc.close()

    chunks: list[dict] = []
    current_section = "named_insured_block"
    current_lines: list[str] = []
    chunk_index = 0
    in_coverage_table = False

    for block in all_blocks:
        text = block["text"]
        new_section = _detect_section(text, block["is_bold"])

        if new_section:
            # Flush current section before starting new one
            if current_lines:
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []
            current_section = new_section
            in_coverage_table = (new_section == "coverage_table")
            continue  # section header is not included in chunk text

        if in_coverage_table and _COVERAGE_ROW_RE.search(text):
            # Each coverage row is its own chunk to keep limit+deductible together
            if current_lines:
                # Flush any buffered non-row lines in the table section
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []
            # Collect the full row — gather adjacent non-header lines as the row continues
            row_lines = [text]
            chunks.append(
                _build_chunk(policy_number, "coverage_table_row", row_lines, chunk_index)
            )
            chunk_index += 1
        else:
            current_lines.append(text)
            # Flush if chunk is getting large (>400 tokens)
            if _tok("\n".join(current_lines)) > 400:
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []

    # Flush final section
    if current_lines:
        chunks.append(
            _build_chunk(policy_number, current_section, current_lines, chunk_index)
        )

    log.info(
        "declaration_chunked",
        extra={"policy_number": policy_number, "chunks": len(chunks), "file": path.name},
    )
    return chunks
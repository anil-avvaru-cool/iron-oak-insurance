"""
chunk_declaration.py — table-aware section chunker for declaration pages.

Strategy:
  1. Extract text blocks from PDF using PyMuPDF, preserving font metadata.
  2. Detect section boundaries by bold headers (font flags) or font size > body.
  3. Split into logical sections: named_insured_block, vehicle_details,
     coverage_table, endorsements.
  4. Coverage table: accumulate consecutive spans per row so that the coverage
     label, status, limit, and deductible all land in the SAME chunk.
     A new row starts when a span matches _COVERAGE_ROW_RE.
     Orphaned header/footer lines (no coverage label) are grouped as one chunk.
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


_FOOTER_RE = re.compile(
    r"(avvaru|iron oak|confidential|governed by the laws|policyholder use only|synthetic-v1|www\.)",
    re.IGNORECASE,
)

# Matches the start of a coverage row (the coverage-type label cell)
_COVERAGE_ROW_RE = re.compile(
    r"^(liability|collision|comprehensive|pip|personal injury|uninsured|underinsured|gap|roadside)",
    re.IGNORECASE,
)
# Table column headers — skip these as standalone chunks
_TABLE_HEADER_RE = re.compile(
    r"^(coverage|status|limit|deductible|type|included|not included|see policy|n/?a)$",
    re.IGNORECASE,
)
_BOLD_FLAG = 1 << 4


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & _BOLD_FLAG)


def _extract_text_blocks(page: "fitz.Page") -> list[dict]:
    """Return list of {text, is_bold, font_size, bbox} for each span on the page."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
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
        "customer_id":   None,
        "state":         policy_number[:2] if policy_number else None,
        "page_number":   None,
        "section":       section,
        "chunk_index":   chunk_index,
        "token_count":   _tok(chunk_text),
        "chunk_text":    chunk_text,
    }


def _chunk_coverage_table(
    table_blocks: list[dict],
    policy_number: str,
    start_index: int,
) -> list[dict]:
    chunks: list[dict] = []
    chunk_index = start_index

    current_row_lines: list[str] = []
    current_coverage_label: str | None = None
    header_footer_lines: list[str] = []

    def flush_row():
        nonlocal chunk_index, current_row_lines, current_coverage_label
        if current_row_lines and current_coverage_label:
            chunks.append(
                _build_chunk(policy_number, "coverage_table_row", current_row_lines, chunk_index)
            )
            chunk_index += 1
        current_row_lines = []
        current_coverage_label = None

    for block in table_blocks:
        text = block["text"].strip()
        if not text:
            continue

        # Stop absorbing into coverage rows if this is a footer line
        if _FOOTER_RE.search(text):
            flush_row()
            header_footer_lines.append(text)
            continue

        if _COVERAGE_ROW_RE.match(text):
            flush_row()
            current_coverage_label = text
            current_row_lines = [text]
        elif current_coverage_label is not None:
            current_row_lines.append(text)
        else:
            if not _TABLE_HEADER_RE.match(text):
                header_footer_lines.append(text)

    flush_row()

    if header_footer_lines:
        chunks.append(
            _build_chunk(policy_number, "coverage_table_header", header_footer_lines, chunk_index)
        )

    return chunks


def chunk_declaration_page(path: Path) -> list[dict]:
    """
    Chunk a declaration page PDF into section-aware chunks.
    Returns list of chunk dicts ready for embedding.
    """
    stem = path.stem  # "decl_TX-00142"
    policy_number = stem[len("decl_"):] if stem.startswith("decl_") else stem

    doc = fitz.open(path)
    all_blocks: list[dict] = []
    for page_num, page in enumerate(doc):
        for block in _extract_text_blocks(page):
            block["page_number"] = page_num + 1
            all_blocks.append(block)
    doc.close()

    # ── Pass 1: split all blocks into named sections ──────────────────────
    sections: dict[str, list[dict]] = {
        "named_insured_block": [],
        "vehicle_details":     [],
        "coverage_table":      [],
        "endorsements":        [],
        "other":               [],
    }
    current_section = "named_insured_block"

    for block in all_blocks:
        new_section = _detect_section(block["text"], block["is_bold"])
        if new_section:
            current_section = new_section
            continue  # section header itself is not content
        sections[current_section].append(block)

    # ── Pass 2: chunk each section appropriately ──────────────────────────
    chunks: list[dict] = []
    chunk_index = 0

    # Non-table sections: simple line accumulation with 400-token flush
    for section_name in ("named_insured_block", "vehicle_details", "endorsements", "other"):
        current_lines: list[str] = []
        for block in sections[section_name]:
            current_lines.append(block["text"])
            if _tok("\n".join(current_lines)) > 400:
                chunks.append(_build_chunk(policy_number, section_name, current_lines, chunk_index))
                chunk_index += 1
                current_lines = []
        if current_lines:
            chunks.append(_build_chunk(policy_number, section_name, current_lines, chunk_index))
            chunk_index += 1

    # Coverage table: row-aware chunking
    coverage_chunks = _chunk_coverage_table(
        sections["coverage_table"], policy_number, chunk_index
    )
    chunks.extend(coverage_chunks)
    chunk_index += len(coverage_chunks)

    log.info(
        "declaration_chunked",
        extra={"policy_number": policy_number, "chunks": len(chunks), "file": path.name},
    )
    return chunks
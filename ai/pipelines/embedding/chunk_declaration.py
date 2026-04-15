"""
chunk_declaration.py — table-aware section chunker for declaration pages.

Sections extracted:
  - named_insured_block  → 1 chunk
  - vehicle_details      → 1 chunk per vehicle
  - coverage_table_rows  → 1 chunk per coverage type (limit+deductible kept together)
  - endorsements         → 1 chunk per endorsement

Uses PyMuPDF (fitz) to extract text, then heuristic section detection
based on bold headers and table row patterns.
"""
import fitz  # PyMuPDF
from pathlib import Path

def chunk_declaration_page(path: Path) -> list[dict]:
    # TODO: implement section parser
    # Heuristic: bold text → section header; tabular rows → coverage entries
    # Each coverage_table row: "Coverage Type · Limit · Deductible" kept as one chunk
    raise NotImplementedError("Placeholder — implement in Phase 4 build")
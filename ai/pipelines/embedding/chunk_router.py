"""
chunk_router.py — detects document type from filename, dispatches to the correct chunker.

Relies on the filename convention established in document_gen.py:
  decl_<policy_number>.pdf         → chunk_declaration_page()
  claim_letter_<claim_id>.pdf      → chunk_claim_letter()
  renewal_<policy_number>.pdf      → chunk_renewal_notice()

No ML classification needed — filename prefix is load-bearing.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .chunk_declaration import chunk_declaration_page
from .chunk_claim_letter import chunk_claim_letter
from .chunk_renewal import chunk_renewal_notice

log = logging.getLogger(__name__)


def route(path: Path) -> list[dict]:
    """Dispatch a PDF to the appropriate chunker. Returns list of chunk dicts."""
    name = path.name.lower()
    if name.startswith("decl_"):
        log.debug("route", extra={"file": path.name, "chunker": "declaration"})
        return chunk_declaration_page(path)
    elif name.startswith("claim_letter_"):
        log.debug("route", extra={"file": path.name, "chunker": "claim_letter"})
        return chunk_claim_letter(path)
    elif name.startswith("renewal_"):
        log.debug("route", extra={"file": path.name, "chunker": "renewal"})
        return chunk_renewal_notice(path)
    else:
        raise ValueError(
            f"Unknown document type for file '{path.name}'. "
            f"Expected prefix: decl_ | claim_letter_ | renewal_"
        )
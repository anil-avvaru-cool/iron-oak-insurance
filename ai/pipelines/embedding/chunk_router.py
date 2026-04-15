"""
chunk_router.py — detects document type from filename and dispatches to the correct chunker.
No ML classification — relies on the filename convention from document_gen.py.
"""
from pathlib import Path
from .chunk_declaration import chunk_declaration_page
from .chunk_claim_letter import chunk_claim_letter
from .chunk_renewal import chunk_renewal_notice
from .chunk_faq import chunk_faq_records

def route(path: Path) -> list[dict]:
    name = path.name.lower()
    if name.startswith("decl_"):
        return chunk_declaration_page(path)
    elif name.startswith("claim_letter_"):
        return chunk_claim_letter(path)
    elif name.startswith("renewal_"):
        return chunk_renewal_notice(path)
    else:
        raise ValueError(f"Unknown document type: {path.name}")
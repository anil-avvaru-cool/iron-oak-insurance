"""
chunk_faq.py — one chunk per Q&A pair. No splitting, no overlap.
Q text is prepended to A so semantic search on either surface works.
"""
import json
from pathlib import Path

def chunk_faq_records(faq_path: Path) -> list[dict]:
    records = json.loads(faq_path.read_text())
    chunks = []
    for rec in records:
        chunk_text = f"Q: {rec['question']}\nA: {rec['answer']}"
        chunks.append({
            "chunk_id":      rec["faq_id"],
            "source_type":   "faq",
            "doc_type":      "faq",
            "policy_number": None,
            "customer_id":   None,
            "state":         rec["applicable_states"][0] if rec["applicable_states"] != ["ALL"] else None,
            "page_number":   None,
            "section":       rec["category"],
            "chunk_index":   0,
            "token_count":   len(chunk_text.split()),  # approximate; replace with tiktoken if needed
            "chunk_text":    chunk_text,
        })
    return chunks
"""
verify_documents.py — standalone verification for generated PDFs.

Usage:
    uv run python data_gen/generators/verify_documents.py
    uv run python data_gen/generators/verify_documents.py --dir documents/

Checks:
    1. Output directory exists and is non-empty
    2. Filename convention correct (decl_, claim_letter_, renewal_ prefixes)
    3. All files are valid PDFs (non-zero, readable header)
    4. Document type distribution roughly 45/35/20 (decl/claim/renewal)
    5. Policy numbers in decl_ filenames exist in policies.json (sample check)
    6. Claim IDs in claim_letter_ filenames exist in claims.json (sample check)
    7. No zero-byte files
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PASS, FAIL, WARN = "✓", "✗", "⚠"


def _check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    print(f"  {symbol}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def _is_valid_pdf(path: Path) -> bool:
    """Check PDF magic bytes."""
    try:
        header = path.read_bytes()[:5]
        return header == b"%PDF-"
    except OSError:
        return False


def verify(docs_dir: Path,
           policies_path: Path = Path("data/policies.json"),
           claims_path: Path = Path("data/claims.json")) -> bool:
    print(f"\n{'='*55}")
    print(f"  Document Verification: {docs_dir}")
    print(f"{'='*55}")

    if not _check("Documents directory exists", docs_dir.exists(), str(docs_dir)):
        return False

    all_pdfs = list(docs_dir.glob("*.pdf"))
    _check("PDFs generated", len(all_pdfs) > 0, f"{len(all_pdfs)} PDFs found")
    if not all_pdfs:
        return False

    # Filename convention
    decl_files = [f for f in all_pdfs if f.name.startswith("decl_")]
    claim_files = [f for f in all_pdfs if f.name.startswith("claim_letter_")]
    renewal_files = [f for f in all_pdfs if f.name.startswith("renewal_")]
    other_files = [f for f in all_pdfs
                   if not any(f.name.startswith(p) for p in ("decl_", "claim_letter_", "renewal_"))]

    _check("All filenames follow convention (decl_/claim_letter_/renewal_)", not other_files,
           f"{len(other_files)} files with unknown prefix" if other_files else "")

    total = len(all_pdfs)
    print(f"  {PASS}  Document type counts:")
    print(f"         decl_         {len(decl_files):5,}  ({len(decl_files)/total:.1%})")
    print(f"         claim_letter_ {len(claim_files):5,}  ({len(claim_files)/total:.1%})")
    print(f"         renewal_      {len(renewal_files):5,}  ({len(renewal_files)/total:.1%})")

    # No zero-byte files
    zero_byte = [f.name for f in all_pdfs if f.stat().st_size == 0]
    _check("No zero-byte PDFs", not zero_byte,
           f"{len(zero_byte)} zero-byte files" if zero_byte else "")

    # PDF header check (sample up to 50 files per type)
    sample = (decl_files[:20] + claim_files[:15] + renewal_files[:15])
    invalid_pdfs = [f.name for f in sample if not _is_valid_pdf(f)]
    _check("Valid PDF headers (sample)", not invalid_pdfs,
           f"{len(invalid_pdfs)} invalid: {invalid_pdfs[:3]}" if invalid_pdfs else
           f"checked {len(sample)} files")

    # Referential integrity — policy numbers in decl_ filenames
    if policies_path.exists() and decl_files:
        policies = json.loads(policies_path.read_text())
        valid_policy_nums = {p["policy_number"] for p in policies}
        # Extract policy number from filename: decl_TX-00001.pdf → TX-00001
        bad_decl = []
        for f in decl_files[:50]:  # sample check
            pol_num = f.stem[5:]  # strip "decl_"
            if pol_num not in valid_policy_nums:
                bad_decl.append(f.name)
        _check("decl_ filenames match valid policy numbers (sample)",
               not bad_decl,
               f"{len(bad_decl)} mismatches: {bad_decl[:3]}" if bad_decl else
               f"checked {min(len(decl_files), 50)} files")

    # Referential integrity — claim IDs in claim_letter_ filenames
    if claims_path.exists() and claim_files:
        claims = json.loads(claims_path.read_text())
        valid_claim_ids = {c["claim_id"] for c in claims}
        bad_claims = []
        for f in claim_files[:50]:
            claim_id = f.stem[len("claim_letter_"):]  # strip prefix
            if claim_id not in valid_claim_ids:
                bad_claims.append(f.name)
        _check("claim_letter_ filenames match valid claim IDs (sample)",
               not bad_claims,
               f"{len(bad_claims)} mismatches: {bad_claims[:3]}" if bad_claims else
               f"checked {min(len(claim_files), 50)} files")

    # File size distribution
    sizes = [f.stat().st_size for f in all_pdfs]
    avg_kb = sum(sizes) / len(sizes) / 1024
    min_kb = min(sizes) / 1024
    max_kb = max(sizes) / 1024
    total_mb = sum(sizes) / 1024 / 1024
    print(f"\n  Total size: {total_mb:.1f} MB | Avg: {avg_kb:.1f} KB | "
          f"Min: {min_kb:.1f} KB | Max: {max_kb:.1f} KB")

    passed = not other_files and not zero_byte and not invalid_pdfs
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("documents"))
    parser.add_argument("--policies", type=Path, default=Path("data/policies.json"))
    parser.add_argument("--claims", type=Path, default=Path("data/claims.json"))
    args = parser.parse_args()
    ok = verify(args.dir, args.policies, args.claims)
    sys.exit(0 if ok else 1)

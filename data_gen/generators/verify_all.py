"""
verify_all.py — runs all Phase 1 verifications in sequence.

Usage:
    uv run python data-gen/generators/verify_all.py
    uv run python data-gen/generators/verify_all.py --skip-pdfs

Returns exit code 0 if all pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

import verify_customers
import verify_policies
import verify_claims
import verify_telematics
import verify_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Phase 1 data verifications")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--docs-dir", type=Path, default=Path("documents"))
    parser.add_argument("--skip-pdfs", action="store_true")
    args = parser.parse_args()

    d = args.data_dir
    results = {}

    results["customers"] = verify_customers.verify(d / "customers.json")
    results["policies"] = verify_policies.verify(d / "policies.json", d / "customers.json")
    results["claims"] = verify_claims.verify(d / "claims.json", d / "policies.json")
    results["telematics"] = verify_telematics.verify(d / "telematics.json", d / "policies.json")

    if not args.skip_pdfs:
        results["documents"] = verify_documents.verify(args.docs_dir, d / "policies.json", d / "claims.json")

    print(f"\n{'='*55}")
    print("  Phase 1 Verification Summary")
    print(f"{'='*55}")
    all_passed = True
    for name, passed in results.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol}  {name}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'ALL PASS ✓' if all_passed else 'FAILURES DETECTED ✗'}")
    print()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

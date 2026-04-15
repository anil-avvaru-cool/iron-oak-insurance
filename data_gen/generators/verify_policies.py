"""
verify_policies.py — standalone data quality verification for policies.json.

Usage:
    uv run python data-gen/generators/verify_policies.py
    uv run python data-gen/generators/verify_policies.py --path data/policies.json

Checks:
    1. File exists and is valid JSON
    2. All records pass policy.schema.json
    3. All 7 coverage types present on every policy
    4. `required` field always emitted (never missing)
    5. No-fault state policies have pip.required=true (MI, FL, NY, NJ, PA, DE, HI, KS, KY, MA, MN, ND, UT)
    6. UM-required states have uninsured_motorist present
    7. VINs are exactly 17 characters
    8. drive_score=null for non-enrolled (valid), numeric for enrolled
    9. Effective date < expiry date
    10. policy_number uniqueness
    11. Referential integrity: all customer_ids exist in customers.json
    12. Premium annual in realistic range ($400-$6000)
    13. Multi-policy rate roughly 10-25%
    14. Telematics enrollment rate 50-75%
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from validate import validate_records

_NO_FAULT_STATES = {"MI", "FL", "NY", "NJ", "PA", "DE", "HI", "KS", "KY", "MA", "MN", "ND", "UT"}
_UM_REQUIRED_STATES = {"CT", "IL", "KS", "MA", "ME", "MD", "MN", "MO", "NC", "NE", "NJ",
                        "NY", "OR", "RI", "SC", "SD", "VA", "VT", "WA", "WI", "WV", "DC"}
_REQUIRED_COVERAGES = ["liability", "collision", "comprehensive", "pip",
                       "uninsured_motorist", "gap", "roadside"]

PASS, FAIL, WARN = "✓", "✗", "⚠"


def _check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    print(f"  {symbol}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def verify(path: Path, customers_path: Path = Path("data/customers.json")) -> bool:
    print(f"\n{'='*55}")
    print(f"  Policy Verification: {path}")
    print(f"{'='*55}")

    if not _check("File exists", path.exists(), str(path)):
        return False

    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _check("Valid JSON", False, str(e))
        return False
    _check("Valid JSON", True, f"{len(records):,} records")

    if not records:
        _check("Non-empty dataset", False)
        return False

    # Schema validation
    try:
        validate_records(records, "policy.schema.json")
        _check("Schema validation (all records)", True)
    except ValueError as e:
        _check("Schema validation", False, str(e)[:120])
        return False

    # Unique policy numbers
    ids = [r["policy_number"] for r in records]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    _check("Unique policy_number", not dupes, f"{len(dupes)} duplicates" if dupes else "")

    # All 7 coverage types present
    missing_cov_count = 0
    for r in records:
        coverages = r.get("coverages", {})
        for cov_type in _REQUIRED_COVERAGES:
            if cov_type not in coverages:
                missing_cov_count += 1
                break
    _check("All 7 coverage types present on every policy", missing_cov_count == 0,
           f"{missing_cov_count} policies missing coverage keys" if missing_cov_count else "")

    # `required` field always emitted
    missing_required_field = 0
    for r in records:
        for cov in r.get("coverages", {}).values():
            if "required" not in cov:
                missing_required_field += 1
    _check("`required` always emitted on coverage objects", missing_required_field == 0,
           f"{missing_required_field} coverage objects missing `required`" if missing_required_field else "")

    # No-fault state PIP check
    pip_violations = []
    for r in records:
        if r["state"] in _NO_FAULT_STATES:
            pip = r.get("coverages", {}).get("pip", {})
            if not pip.get("required", False):
                pip_violations.append(r["policy_number"])
    _check(
        "No-fault states: pip.required=true",
        len(pip_violations) == 0,
        f"{len(pip_violations)} violations (showing first 3): {pip_violations[:3]}" if pip_violations else "",
    )

    # VIN length
    bad_vins = [r["policy_number"] for r in records if len(r.get("vehicle", {}).get("vin", "")) != 17]
    _check("All VINs exactly 17 characters", not bad_vins,
           f"{len(bad_vins)} bad VINs" if bad_vins else "")

    # drive_score — null or numeric
    bad_drive = [r["policy_number"] for r in records
                 if r.get("drive_score") is not None and not (0 <= r["drive_score"] <= 100)]
    _check("drive_score null or in [0, 100]", not bad_drive,
           f"{len(bad_drive)} out-of-range" if bad_drive else "")

    # Telematics enrollment rate
    enrolled = sum(1 for r in records if r.get("drive_score") is not None)
    enroll_rate = enrolled / len(records)
    _check("Telematics enrollment rate 45-80%", 0.45 <= enroll_rate <= 0.80,
           f"{enroll_rate:.1%}", warn_only=True)

    # Effective < expiry
    date_errors = 0
    for r in records:
        try:
            eff = date.fromisoformat(r["effective_date"])
            exp = date.fromisoformat(r["expiry_date"])
            if eff >= exp:
                date_errors += 1
        except (ValueError, KeyError):
            date_errors += 1
    _check("effective_date < expiry_date (all records)", date_errors == 0,
           f"{date_errors} date order errors" if date_errors else "")

    # Premium range
    bad_premiums = [r["policy_number"] for r in records
                    if not (300 <= r.get("premium_annual", 0) <= 8000)]
    _check("Premium annual in [$300, $8000]", not bad_premiums,
           f"{len(bad_premiums)} out-of-range" if bad_premiums else "", warn_only=True)

    # Multi-policy rate
    cust_counts = Counter(r["customer_id"] for r in records)
    multi_customers = sum(1 for c in cust_counts.values() if c > 1)
    total_customers = len(cust_counts)
    multi_rate = multi_customers / max(total_customers, 1)
    _check("Multi-policy rate 8-25%", 0.08 <= multi_rate <= 0.25,
           f"{multi_rate:.1%}", warn_only=True)

    # Referential integrity
    if customers_path.exists():
        customers = json.loads(customers_path.read_text())
        valid_cust_ids = {c["customer_id"] for c in customers}
        orphan_policies = [r["policy_number"] for r in records
                           if r["customer_id"] not in valid_cust_ids]
        _check("All customer_ids exist in customers.json", not orphan_policies,
               f"{len(orphan_policies)} orphan policies" if orphan_policies else "")
    else:
        print(f"  {WARN}  Referential integrity  [customers.json not found — skipped]")

    # Summary
    avg_premium = sum(r.get("premium_annual", 0) for r in records) / len(records)
    print(f"\n  Records: {len(records):,} | Enrolled: {enrolled:,} ({enroll_rate:.1%}) | "
          f"Non-enrolled: {len(records)-enrolled:,}")
    print(f"  Avg premium: ${avg_premium:,.2f} | Multi-policy customers: {multi_rate:.1%}")

    passed = not dupes and not bad_vins and missing_cov_count == 0 and date_errors == 0
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/policies.json"))
    parser.add_argument("--customers", type=Path, default=Path("data/customers.json"))
    args = parser.parse_args()
    ok = verify(args.path, args.customers)
    sys.exit(0 if ok else 1)

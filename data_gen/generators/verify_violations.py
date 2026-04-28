"""
verify_violations.py — data quality checks for violations.json.

Checks:
  1. File exists and is valid JSON
  2. Schema validation against violation.schema.json
  3. Unique violation_id
  4. Violation rate 25–45% of customers
  5. DUI rate within expected range (1–6% of customers with violations)
  6. All violation_date in the past and within lookback window
  7. expiry_date > violation_date
  8. conviction_date >= violation_date when present
  9. conviction_date not in the future
 10. Points match coverage_rules.json point_weights for the violation type
 11. Active violation count > 0 (at least some not yet expired)
 12. Referential integrity — all customer_id exist in customers.json
 13. Referential integrity — all policy_number exist in policies.json
 14. Major violation types have higher average points than minor types
 15. No parking_violation records (excluded by generator design)

Usage:
    uv run python data_gen/generators/verify_violations.py
    uv run python data_gen/generators/verify_violations.py \\
        --path data/violations.json \\
        --customers data/customers.json \\
        --policies data/policies.json \\
        --config data_gen/config/coverage_rules.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


# ── Formatting helpers ─────────────────────────────────────────────────────

PASS  = "  ✓"
FAIL  = "  ✗"
WARN  = "  ⚠"


def _ok(msg: str) -> None:
    print(f"{PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"{FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"{WARN}  {msg}")


# ── Individual checks ──────────────────────────────────────────────────────

def check_file_exists(path: Path) -> list[dict] | None:
    if not path.exists():
        _fail(f"File not found: {path}")
        return None
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON: {e}")
        return None
    if not isinstance(records, list):
        _fail("Root element is not a list")
        return None
    _ok(f"File exists and is valid JSON ({len(records):,} records)")
    return records


def check_schema(records: list[dict], schema_dir: Path) -> bool:
    try:
        import jsonschema
    except ImportError:
        _warn("jsonschema not installed — skipping schema validation")
        return True

    schema_path = schema_dir / "violation.schema.json"
    if not schema_path.exists():
        _warn(f"Schema file not found at {schema_path} — skipping")
        return True

    schema = json.loads(schema_path.read_text())
    errors = []
    for i, rec in enumerate(records):
        try:
            jsonschema.validate(instance=rec, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Record {i} ({rec.get('violation_id','?')}): "
                          f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}")
        if len(errors) >= 5:
            errors.append("... (truncated after 5 errors)")
            break

    if errors:
        _fail(f"Schema validation failed ({len(errors)} error(s)):")
        for err in errors:
            print(f"       {err}")
        return False

    _ok("Schema validation passed")
    return True


def check_unique_ids(records: list[dict]) -> bool:
    ids = [r["violation_id"] for r in records]
    dupes = [vid for vid in set(ids) if ids.count(vid) > 1]
    if dupes:
        _fail(f"Duplicate violation_id values ({len(dupes)}): {dupes[:5]}")
        return False
    _ok(f"All {len(ids):,} violation_id values are unique")
    return True


def check_violation_rate(
    records: list[dict],
    customers: list[dict],
) -> bool:
    total_customers   = len(customers)
    with_violations   = len({r["customer_id"] for r in records})
    rate              = with_violations / total_customers if total_customers else 0

    low, high = 0.25, 0.45
    msg = (f"Violation rate: {with_violations:,} / {total_customers:,} "
           f"customers = {rate:.1%} (expected {low:.0%}–{high:.0%})")
    if low <= rate <= high:
        _ok(msg)
        return True
    _warn(msg)
    return True   # warn only — rate varies with dataset size


def check_dui_rate(records: list[dict]) -> bool:
    customers_with_violations = {r["customer_id"] for r in records}
    dui_customers = {
        r["customer_id"] for r in records if r["violation_type"] == "dui_dwi"
    }
    if not customers_with_violations:
        _warn("No violations found — skipping DUI rate check")
        return True

    rate    = len(dui_customers) / len(customers_with_violations)
    low, high = 0.01, 0.06
    msg = (f"DUI rate: {len(dui_customers):,} / {len(customers_with_violations):,} "
           f"customers with violations = {rate:.1%} (expected {low:.0%}–{high:.0%})")
    if low <= rate <= high:
        _ok(msg)
        return True
    _warn(msg)
    return True   # warn only


def check_dates(records: list[dict], lookback_years: int) -> bool:
    today        = date.today()
    window_start = date(today.year - lookback_years, today.month, today.day)
    errors       = []

    for r in records:
        vid    = r["violation_id"]
        vdate  = date.fromisoformat(r["violation_date"])
        expdt  = date.fromisoformat(r["expiry_date"])
        convdt = (date.fromisoformat(r["conviction_date"])
                  if r.get("conviction_date") else None)

        if vdate > today:
            errors.append(f"{vid}: violation_date {vdate} is in the future")
        if vdate < window_start:
            errors.append(f"{vid}: violation_date {vdate} is outside "
                          f"{lookback_years}-year lookback window")
        if expdt <= vdate:
            errors.append(f"{vid}: expiry_date {expdt} not after violation_date {vdate}")
        if convdt is not None:
            if convdt < vdate:
                errors.append(f"{vid}: conviction_date {convdt} before "
                               f"violation_date {vdate}")
            if convdt > today:
                errors.append(f"{vid}: conviction_date {convdt} is in the future")

        if len(errors) >= 10:
            errors.append("... (truncated after 10 errors)")
            break

    if errors:
        _fail(f"Date validation failed ({len(errors)} issue(s)):")
        for err in errors:
            print(f"       {err}")
        return False

    _ok("All violation, expiry, and conviction dates are valid")
    return True


def check_points(records: list[dict], point_weights: dict[str, int]) -> bool:
    errors = []
    for r in records:
        expected = point_weights.get(r["violation_type"])
        if expected is None:
            errors.append(f"{r['violation_id']}: unknown violation_type "
                          f"'{r['violation_type']}'")
        elif r["points"] != expected:
            errors.append(f"{r['violation_id']}: points={r['points']} "
                          f"expected={expected} for type '{r['violation_type']}'")
        if len(errors) >= 10:
            errors.append("... (truncated after 10 errors)")
            break

    if errors:
        _fail(f"Points mismatch ({len(errors)} issue(s)):")
        for err in errors:
            print(f"       {err}")
        return False

    _ok("All points match coverage_rules.json point_weights")
    return True


def check_active_violations(records: list[dict]) -> bool:
    today  = date.today()
    active = [r for r in records if date.fromisoformat(r["expiry_date"]) >= today]
    if not active:
        _fail("No active violations found (all expired) — data quality issue")
        return False
    pct = len(active) / len(records) if records else 0
    _ok(f"Active violations (not yet expired): {len(active):,} / "
        f"{len(records):,} ({pct:.1%})")
    return True


def check_no_parking_violations(records: list[dict]) -> bool:
    parking = [r for r in records if r["violation_type"] == "parking_violation"]
    if parking:
        _fail(f"Found {len(parking):,} parking_violation records — "
              f"generator should exclude these (0 points, no risk signal)")
        return False
    _ok("No parking_violation records (correctly excluded)")
    return True


def check_referential_integrity(
    records: list[dict],
    customers: list[dict],
    policies: list[dict],
) -> bool:
    customer_ids = {c["customer_id"] for c in customers}
    policy_nums  = {p["policy_number"] for p in policies}

    bad_customers = [
        r["violation_id"] for r in records
        if r["customer_id"] not in customer_ids
    ]
    bad_policies = [
        r["violation_id"] for r in records
        if r.get("policy_number") and r["policy_number"] not in policy_nums
    ]

    passed = True
    if bad_customers:
        _fail(f"Unknown customer_id in {len(bad_customers):,} records: "
              f"{bad_customers[:3]}")
        passed = False
    else:
        _ok("All customer_id values exist in customers.json")

    if bad_policies:
        _fail(f"Unknown policy_number in {len(bad_policies):,} records: "
              f"{bad_policies[:3]}")
        passed = False
    else:
        _ok("All policy_number values exist in policies.json")

    return passed


def check_major_vs_minor_points(
    records: list[dict],
    major_types: list[str],
) -> bool:
    major_pts = [r["points"] for r in records if r["violation_type"] in major_types]
    minor_pts = [r["points"] for r in records if r["violation_type"] not in major_types]

    if not major_pts or not minor_pts:
        _warn("Insufficient major or minor violation records to compare points")
        return True

    avg_major = sum(major_pts) / len(major_pts)
    avg_minor = sum(minor_pts) / len(minor_pts)

    msg = (f"Avg points — major: {avg_major:.2f}, minor: {avg_minor:.2f} "
           f"(major should exceed minor)")
    if avg_major > avg_minor:
        _ok(msg)
        return True
    _fail(msg)
    return False


# ── Main ───────────────────────────────────────────────────────────────────

def main(
    violations_path: Path,
    customers_path: Path,
    policies_path: Path,
    config_path: Path,
) -> bool:
    print("\n" + "=" * 55)
    print("  Violations Verification")
    print("=" * 55)

    # Load config
    config        = json.loads(config_path.read_text())
    rules         = config.get("violation_rules", {})
    point_weights = rules.get("point_weights", {})
    major_types   = rules.get("major_violation_types", [])
    lookback_years = rules.get("lookback_years", 5)

    # Load reference data
    customers: list[dict] = []
    policies:  list[dict] = []

    if customers_path.exists():
        customers = json.loads(customers_path.read_text())
    else:
        _warn(f"customers.json not found at {customers_path} — "
              "skipping referential integrity checks")

    if policies_path.exists():
        policies = json.loads(policies_path.read_text())
    else:
        _warn(f"policies.json not found at {policies_path} — "
              "skipping policy referential integrity check")

    # Run checks
    records = check_file_exists(violations_path)
    if records is None:
        print("\n  Overall: FAIL ✗  (file unreadable — stopping)\n")
        return False

    schema_dir = violations_path.parent.parent / "data_gen" / "schemas"
    if not schema_dir.exists():
        # fallback: look relative to this file
        schema_dir = Path(__file__).parent.parent / "schemas"

    results = [
        check_schema(records, schema_dir),
        check_unique_ids(records),
        check_violation_rate(records, customers) if customers else True,
        check_dui_rate(records),
        check_dates(records, lookback_years),
        check_points(records, point_weights),
        check_active_violations(records),
        check_no_parking_violations(records),
        check_referential_integrity(records, customers, policies)
            if customers and policies else True,
        check_major_vs_minor_points(records, major_types),
    ]

    passed = all(results)
    print("=" * 55)
    print(f"  Overall: {'ALL PASS ✓' if passed else 'FAILURES DETECTED ✗'}")
    print("=" * 55 + "\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify violations.json data quality."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/violations.json"),
        help="Path to violations.json",
    )
    parser.add_argument(
        "--customers",
        type=Path,
        default=Path("data/customers.json"),
        help="Path to customers.json for referential integrity check",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("data/policies.json"),
        help="Path to policies.json for referential integrity check",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data_gen/config/coverage_rules.json"),
        help="Path to coverage_rules.json",
    )
    args = parser.parse_args()

    ok = main(args.path, args.customers, args.policies, args.config)
    sys.exit(0 if ok else 1)
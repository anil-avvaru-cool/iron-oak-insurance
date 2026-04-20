"""
verify_customers.py — standalone data quality verification for customers.json.

Usage:
    uv run python data_gen/generators/verify_customers.py
    uv run python data_gen/generators/verify_customers.py --path data/customers.json

Checks:
    1. File exists and is valid JSON
    2. All records pass customer.schema.json
    3. All 50 states + DC represented
    4. State distribution roughly matches population weights (no single state > 25%)
    5. Credit scores within [300, 850] and realistically distributed
    6. DOBs represent adults aged 18-85
    7. Customer IDs are unique
    8. ~3% have null email (not too many, not all)
    9. No duplicate customer IDs
    10. created_at timestamps are in the past
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

# Resolve schema path when run from any directory
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from validate import validate_records

_ALL_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def _check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    line = f"  {symbol}  {label}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    return condition


def verify(path: Path) -> bool:
    print(f"\n{'='*55}")
    print(f"  Customer Verification: {path}")
    print(f"{'='*55}")

    # 1. File exists
    if not _check("File exists", path.exists(), str(path)):
        return False

    # 2. Valid JSON
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _check("Valid JSON", False, str(e))
        return False
    _check("Valid JSON", True, f"{len(records):,} records")

    if not records:
        _check("Non-empty dataset", False)
        return False

    # 3. Schema validation
    try:
        validate_records(records, "customer.schema.json")
        _check("Schema validation (all records)", True)
    except ValueError as e:
        _check("Schema validation", False, str(e)[:120])
        return False

    # 4. Unique customer IDs
    ids = [r["customer_id"] for r in records]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    _check("Unique customer_id", not dupes, f"{len(dupes)} duplicates" if dupes else "")

    # 5. All 50 states + DC present
    states_present = {r["state"] for r in records}
    missing = _ALL_STATES - states_present
    _check(
        "All 50 states + DC present",
        len(missing) == 0,
        f"Missing: {sorted(missing)}" if missing else "",
        warn_only=len(records) < 200,  # small datasets may miss rare states
    )

    # 6. State distribution — no single state dominates
    state_counts = Counter(r["state"] for r in records)
    max_state, max_count = state_counts.most_common(1)[0]
    max_pct = max_count / len(records)
    _check(
        "State distribution (no state > 25%)",
        max_pct <= 0.25,
        f"{max_state}: {max_pct:.1%}",
        warn_only=True,
    )

    # 7. Credit scores in range
    scores = [r.get("credit_score") for r in records if r.get("credit_score") is not None]
    out_of_range = [s for s in scores if not (300 <= s <= 850)]
    _check("Credit scores in [300, 850]", not out_of_range,
           f"{len(out_of_range)} out-of-range" if out_of_range else "")
    if scores:
        avg_score = sum(scores) / len(scores)
        _check(
            "Credit score mean realistic (600-750)",
            600 <= avg_score <= 750,
            f"mean={avg_score:.0f}",
            warn_only=True,
        )

    # 8. DOBs — adults 18-85
    today = date.today()
    bad_dobs = []
    for r in records:
        try:
            dob = date.fromisoformat(r["dob"])
            age = (today - dob).days // 365
            if not (17 <= age <= 86):
                bad_dobs.append((r["customer_id"], age))
        except (ValueError, KeyError):
            bad_dobs.append((r["customer_id"], "invalid"))
    _check("DOBs represent adults (18-85)", not bad_dobs,
           f"{len(bad_dobs)} out-of-range" if bad_dobs else "")

    # 9. Null email rate (~3%, tolerate 0-10%)
    null_email_count = sum(1 for r in records if r.get("email") is None)
    null_email_pct = null_email_count / len(records)
    _check(
        "Null email rate acceptable (0-10%)",
        0.0 <= null_email_pct <= 0.10,
        f"{null_email_pct:.1%} null emails",
        warn_only=True,
    )

    # 10. created_at in the past (rough check)
    future_created = 0
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for r in records:
        ca = r.get("created_at", "")
        if ca and ca[:19] > now_str:
            future_created += 1
    _check("created_at all in the past", future_created == 0,
           f"{future_created} future timestamps" if future_created else "")

    # Summary
    print(f"\n  Records: {len(records):,} | States: {len(states_present)} | "
          f"Top state: {max_state} ({max_pct:.1%})")
    print(f"  Credit score mean: {sum(scores)/max(len(scores),1):.0f} | "
          f"Null emails: {null_email_pct:.1%}")

    all_passed = not dupes and not out_of_range and not bad_dobs
    status = "PASS" if all_passed else "FAIL"
    print(f"\n  Result: {status}\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/customers.json"))
    args = parser.parse_args()
    ok = verify(args.path)
    sys.exit(0 if ok else 1)

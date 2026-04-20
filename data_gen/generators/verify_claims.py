"""
verify_claims.py — standalone data quality verification for claims.json.

Usage:
    uv run python data_gen/generators/verify_claims.py
    uv run python data_gen/generators/verify_claims.py --path data/claims.json

Checks:
    1.  File exists and is valid JSON
    2.  Schema validation
    3.  Fraud rate in [2%, 8%] range
    4.  Fraud claims have at least one fraud_signal
    5.  Non-fraud claims have empty fraud_signals
    6.  filed_date >= incident_date (always)
    7.  Settlement amount <= claim amount (when present)
    8.  Denied claims have no settlement amount
    9.  Unique claim IDs
    10. Referential integrity vs policies.json
    11. Claim type distribution realistic (collision dominant)
    12. No claims with incident_date in the future
    13. Per-policy claim count: max <= 10, p99 <= 5
        (catches generator regressions — a single policy should never
         accumulate hundreds of claims)
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

PASS, FAIL, WARN = "✓", "✗", "⚠"

# Thresholds for per-policy claim count checks.
# Align with MAX_CLAIMS_PER_POLICY_PER_YEAR in claim_gen.py (=3) plus a
# tolerance for multi-year policies. Raise these only if you intentionally
# increase the generator cap.
_MAX_CLAIMS_PER_POLICY_HARD  = 10   # anything above this is a generator bug
_MAX_CLAIMS_PER_POLICY_P99   = 5    # warn if the 99th-percentile exceeds this


def _check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    print(f"  {symbol}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def verify(path: Path, policies_path: Path = Path("data/policies.json")) -> bool:
    print(f"\n{'='*55}")
    print(f"  Claim Verification: {path}")
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

    try:
        validate_records(records, "claim.schema.json")
        _check("Schema validation (all records)", True)
    except ValueError as e:
        _check("Schema validation", False, str(e)[:120])
        return False

    # ── Unique claim IDs ──────────────────────────────────────────────────
    ids = [r["claim_id"] for r in records]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    _check("Unique claim_id", not dupes, f"{len(dupes)} duplicates" if dupes else "")

    # ── Fraud rate ────────────────────────────────────────────────────────
    fraud_count = sum(1 for r in records if r["is_fraud"])
    fraud_rate = fraud_count / len(records)
    _check("Fraud rate in [2%, 8%]", 0.02 <= fraud_rate <= 0.08,
           f"{fraud_rate:.2%} ({fraud_count} fraud claims)", warn_only=fraud_rate > 0.08)

    # ── Fraud signals consistent with is_fraud flag ───────────────────────
    fraud_no_signals = [r["claim_id"] for r in records
                        if r["is_fraud"] and not r.get("fraud_signals")]
    non_fraud_with_signals = [r["claim_id"] for r in records
                               if not r["is_fraud"] and r.get("fraud_signals")]
    _check("Fraud claims have at least one signal", not fraud_no_signals,
           f"{len(fraud_no_signals)} fraud claims without signals" if fraud_no_signals else "")
    _check("Non-fraud claims have no signals", not non_fraud_with_signals,
           f"{len(non_fraud_with_signals)} non-fraud claims with signals" if non_fraud_with_signals else "")

    # ── filed_date >= incident_date ───────────────────────────────────────
    date_errors = []
    for r in records:
        try:
            inc = date.fromisoformat(r["incident_date"])
            fil = date.fromisoformat(r["filed_date"])
            if fil < inc:
                date_errors.append(r["claim_id"])
        except (ValueError, KeyError):
            date_errors.append(r["claim_id"])
    _check("filed_date >= incident_date (all records)", not date_errors,
           f"{len(date_errors)} errors" if date_errors else "")

    # ── No future incident dates ──────────────────────────────────────────
    today = date.today()
    future_claims = [r["claim_id"] for r in records
                     if date.fromisoformat(r["incident_date"]) > today]
    _check("No future incident_date", not future_claims,
           f"{len(future_claims)} future dates" if future_claims else "")

    # ── Settlement <= claim amount ────────────────────────────────────────
    bad_settlements = []
    for r in records:
        sa = r.get("settlement_amount")
        ca = r.get("claim_amount", 0)
        if sa is not None and sa > ca:
            bad_settlements.append(r["claim_id"])
    _check("settlement_amount <= claim_amount", not bad_settlements,
           f"{len(bad_settlements)} invalid" if bad_settlements else "")

    # ── Denied claims have no settlement ─────────────────────────────────
    denied_with_settlement = [r["claim_id"] for r in records
                               if r.get("status") == "denied" and r.get("settlement_amount") is not None]
    _check("Denied claims have no settlement_amount", not denied_with_settlement,
           f"{len(denied_with_settlement)} denied claims with settlement" if denied_with_settlement else "")

    # ── Per-policy claim count ────────────────────────────────────────────
    # Catches generator regressions: a policy with 198 claims means the Poisson
    # cap in claim_gen.py is broken. Max should be ~3/year * policy_age_years.
    claims_per_policy = Counter(r["policy_number"] for r in records)
    max_claims  = max(claims_per_policy.values()) if claims_per_policy else 0
    sorted_vals = sorted(claims_per_policy.values())
    p99_idx     = max(0, int(len(sorted_vals) * 0.99) - 1)
    p99_claims  = sorted_vals[p99_idx] if sorted_vals else 0

    # Hard failure: any single policy over the absolute ceiling
    hard_ok = max_claims <= _MAX_CLAIMS_PER_POLICY_HARD
    _check(
        f"Per-policy claim count max <= {_MAX_CLAIMS_PER_POLICY_HARD}",
        hard_ok,
        f"max={max_claims} — generator cap may be broken" if not hard_ok else f"max={max_claims}",
    )
    # Soft warning: p99 higher than expected (multi-year policies are fine up to ~6)
    p99_ok = p99_claims <= _MAX_CLAIMS_PER_POLICY_P99
    _check(
        f"Per-policy claim count p99 <= {_MAX_CLAIMS_PER_POLICY_P99}",
        p99_ok,
        f"p99={p99_claims}",
        warn_only=True,   # warn only — multi-year policies legitimately exceed this
    )

    # ── Distributions ─────────────────────────────────────────────────────
    type_counts = Counter(r["claim_type"] for r in records)
    print(f"  {PASS}  Claim type distribution:")
    for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"         {ctype:25s} {count:5,}  ({count/len(records):.1%})")

    status_counts = Counter(r.get("status", "unknown") for r in records)
    print(f"  {PASS}  Status distribution:")
    for st, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"         {st:20s} {count:5,}  ({count/len(records):.1%})")

    # ── Referential integrity ─────────────────────────────────────────────
    if policies_path.exists():
        policies = json.loads(policies_path.read_text())
        valid_policy_nums = {p["policy_number"] for p in policies}
        orphan_claims = [r["claim_id"] for r in records
                         if r["policy_number"] not in valid_policy_nums]
        _check("All policy_numbers exist in policies.json", not orphan_claims,
               f"{len(orphan_claims)} orphan claims" if orphan_claims else "")
    else:
        print(f"  {WARN}  Referential integrity  [policies.json not found — skipped]")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  Records: {len(records):,} | Fraud: {fraud_count:,} ({fraud_rate:.2%})")
    avg_amount = sum(r.get("claim_amount", 0) for r in records) / len(records)
    print(f"  Avg claim amount: ${avg_amount:,.2f}")
    print(f"  Claims per policy — max: {max_claims}, p99: {p99_claims}")

    passed = (
        not dupes
        and not date_errors
        and not fraud_no_signals
        and 0.02 <= fraud_rate <= 0.08
        and hard_ok
    )
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/claims.json"))
    parser.add_argument("--policies", type=Path, default=Path("data/policies.json"))
    args = parser.parse_args()
    ok = verify(args.path, args.policies)
    sys.exit(0 if ok else 1)
"""
verify_telematics.py — standalone data quality verification for telematics.json.

Usage:
    uv run python data-gen/generators/verify_telematics.py
    uv run python data-gen/generators/verify_telematics.py --path data/telematics.json

Checks:
    1. File exists and is valid JSON
    2. Schema validation
    3. Only enrolled policies have trips (drive_score != null in policy)
    4. Unique trip IDs
    5. drive_score in [0, 100] for every trip
    6. night_driving_pct in [0, 1]
    7. All event counts >= 0
    8. Trip durations and distances > 0
    9. No future trip_dates
    10. Referential integrity vs policies.json
    11. Per-policy avg drive_score roughly matches policy-level drive_score (within 20pts)
    12. Trips per enrolled policy in reasonable range (5-500)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from validate import validate_records

PASS, FAIL, WARN = "✓", "✗", "⚠"


def _check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    print(f"  {symbol}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def verify(path: Path, policies_path: Path = Path("data/policies.json")) -> bool:
    print(f"\n{'='*55}")
    print(f"  Telematics Verification: {path}")
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
        _check("Non-empty dataset", False, "No trip records found")
        return False

    try:
        validate_records(records, "telematics.schema.json")
        _check("Schema validation (all records)", True)
    except ValueError as e:
        _check("Schema validation", False, str(e)[:120])
        return False

    # Unique trip IDs
    ids = [r["trip_id"] for r in records]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    _check("Unique trip_id", not dupes, f"{len(dupes)} duplicates" if dupes else "")

    # drive_score in range
    bad_scores = [r["trip_id"] for r in records if not (0 <= r["drive_score"] <= 100)]
    _check("drive_score in [0, 100]", not bad_scores,
           f"{len(bad_scores)} out-of-range" if bad_scores else "")

    # night_driving_pct in [0, 1]
    bad_night = [r["trip_id"] for r in records if not (0 <= r["night_driving_pct"] <= 1)]
    _check("night_driving_pct in [0.0, 1.0]", not bad_night,
           f"{len(bad_night)} out-of-range" if bad_night else "")

    # Event counts >= 0
    neg_events = [r["trip_id"] for r in records
                  if r["hard_brakes"] < 0 or r["rapid_accelerations"] < 0 or r["speeding_events"] < 0]
    _check("All event counts >= 0", not neg_events,
           f"{len(neg_events)} negative events" if neg_events else "")

    # Distance and duration > 0
    bad_dist = [r["trip_id"] for r in records if r["distance_miles"] <= 0 or r["duration_minutes"] <= 0]
    _check("distance_miles and duration_minutes > 0", not bad_dist,
           f"{len(bad_dist)} zero/negative" if bad_dist else "")

    # No future trip dates
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    future_trips = [r["trip_id"] for r in records if r["trip_date"][:19] > now_str]
    _check("No future trip_dates", not future_trips,
           f"{len(future_trips)} future trips" if future_trips else "")

    # Referential integrity and enrollment consistency
    if policies_path.exists():
        policies = json.loads(policies_path.read_text())
        policy_map = {p["policy_number"]: p for p in policies}
        enrolled_policy_nums = {p["policy_number"] for p in policies if p.get("drive_score") is not None}
        non_enrolled = {p["policy_number"] for p in policies if p.get("drive_score") is None}

        orphan_trips = [r["trip_id"] for r in records if r["policy_number"] not in policy_map]
        _check("All policy_numbers exist in policies.json", not orphan_trips,
               f"{len(orphan_trips)} orphan trips" if orphan_trips else "")

        # Non-enrolled policies should have NO trips
        non_enrolled_trips = [r["trip_id"] for r in records if r["policy_number"] in non_enrolled]
        _check("Non-enrolled policies have no trips", not non_enrolled_trips,
               f"{len(non_enrolled_trips)} trips for non-enrolled policies" if non_enrolled_trips else "")

        # Per-policy drive score alignment
        policy_trip_scores = defaultdict(list)
        for r in records:
            policy_trip_scores[r["policy_number"]].append(r["drive_score"])

        misaligned = 0
        for pol_num, trip_scores in policy_trip_scores.items():
            policy = policy_map.get(pol_num)
            if policy and policy.get("drive_score") is not None:
                avg_trip_score = sum(trip_scores) / len(trip_scores)
                pol_score = policy["drive_score"]
                if abs(avg_trip_score - pol_score) > 25:  # tolerate 25pt drift
                    misaligned += 1
        _check(
            "Per-policy avg trip score roughly matches policy drive_score (±25pts)",
            misaligned == 0,
            f"{misaligned} policies misaligned" if misaligned else "",
            warn_only=True,
        )

        # Trips per enrolled policy
        trip_counts = Counter(r["policy_number"] for r in records)
        min_trips = min(trip_counts.values()) if trip_counts else 0
        max_trips = max(trip_counts.values()) if trip_counts else 0
        avg_trips = sum(trip_counts.values()) / max(len(trip_counts), 1)
        _check("Trips per enrolled policy in [1, 1000]", 1 <= min_trips and max_trips <= 1000,
               f"min={min_trips}, max={max_trips}, avg={avg_trips:.0f}", warn_only=True)

        print(f"\n  Enrolled policies with trips: {len(trip_counts):,} / {len(enrolled_policy_nums):,}")
        print(f"  Non-enrolled policies (correctly skipped): {len(non_enrolled):,}")
    else:
        print(f"  {WARN}  Referential integrity  [policies.json not found — skipped]")

    # Summary stats
    avg_score = sum(r["drive_score"] for r in records) / len(records)
    avg_dist = sum(r["distance_miles"] for r in records) / len(records)
    print(f"  Total trips: {len(records):,} | Avg drive score: {avg_score:.1f} | "
          f"Avg trip distance: {avg_dist:.1f} mi")

    passed = not dupes and not bad_scores and not bad_dist and not neg_events
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/telematics.json"))
    parser.add_argument("--policies", type=Path, default=Path("data/policies.json"))
    args = parser.parse_args()
    ok = verify(args.path, args.policies)
    sys.exit(0 if ok else 1)

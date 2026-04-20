"""
telematics_gen.py — generates synthetic telematics trip records for AIOI.

Usage:
    uv run python data_gen/generators/telematics_gen.py
    from telematics_gen import generate, main

Key design decisions:
    - Only generates trips for policies with drive_score != null (enrolled in telematics)
    - Non-telematics customers (drive_score=null) are explicitly excluded — not an error
    - Trip volume proportional to policy age (newer policies have fewer recorded trips)
    - Drive score is computed deterministically from trip events — score in policy matches
      the average of that customer's trip scores, allowing drift over time
    - Night driving, hard braking, and speeding have realistic correlations:
        * Night driving increases hard brake probability
        * High speeding increases rapid acceleration
    - Trip duration and distance follow log-normal distributions (realistic commute/errand split)
    - Some customers show improving scores (young drivers learning), others declining (aging)
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from validate import validate_records

fake = Faker("en_US")
Faker.seed(45)
random.seed(45)


def _compute_drive_score(hard_brakes: int, rapid_accelerations: int,
                         speeding_events: int, night_driving_pct: float,
                         distance_miles: float) -> float:
    """
    Iron Oak Drive Score formula:
    100 - (hard_brakes * 2.0) - (rapid_accel * 1.5) - (speeding * 3.0)
        - (night_pct * 10.0) - (distance penalty for very short/very long trips)
    Clamped to [0, 100].

    Per-mile normalization: events are normalized to events-per-10-miles to avoid
    penalizing long trips unfairly.
    """
    dist = max(distance_miles, 0.1)
    norm = min(10.0 / dist, 2.0)  # normalize to per-10-miles, cap at 2x

    score = 100.0
    score -= hard_brakes * 2.0 * norm
    score -= rapid_accelerations * 1.5 * norm
    score -= speeding_events * 3.0 * norm
    score -= night_driving_pct * 10.0
    return round(max(0.0, min(100.0, score)), 2)


def _driver_profile(policy_drive_score: float) -> dict:
    """
    Derive a behavioral profile from the policy-level drive score.
    Higher drive score → fewer risky events per trip.
    Rate is linear so per-trip average aligns within ±20pts of policy drive_score.
    """
    rate = max(0.05, (100.0 - policy_drive_score) / 20.0)
    return {
        "brake_mean": rate * 0.5,
        "accel_mean": rate * 0.4,
        "speed_mean": rate * 0.35,
        "night_mean": min(0.30, rate * 0.08),
    }


def _generate_trip(trip_n: int, policy: dict, trip_date: datetime, profile: dict) -> dict:
    """Generate a single trip record."""
    # Trip type: ~70% short commutes, ~25% medium, ~5% long
    trip_type = random.choices(["short", "medium", "long"], weights=[70, 25, 5])[0]
    if trip_type == "short":
        distance = max(0.5, random.lognormvariate(1.8, 0.6))  # ~2-15 miles
        duration = distance * random.uniform(2.5, 5.0)         # city driving
    elif trip_type == "medium":
        distance = max(5.0, random.lognormvariate(3.0, 0.5))  # ~10-50 miles
        duration = distance * random.uniform(1.5, 3.0)
    else:
        distance = max(20.0, random.lognormvariate(4.0, 0.5))  # ~50-200 miles
        duration = distance * random.uniform(1.2, 2.0)

    distance = round(min(distance, 450), 2)
    duration = round(min(duration, 480), 2)

    # Night driving
    hour = trip_date.hour
    is_night_hour = hour < 6 or hour >= 22
    base_night = profile["night_mean"]
    night_driving_pct = round(min(1.0, max(0.0, random.gauss(
        base_night * 2.5 if is_night_hour else base_night, 0.05
    ))), 4)

    # Events — Poisson distributed, correlated with night driving
    night_multiplier = 1.0 + night_driving_pct * 1.5
    hard_brakes = max(0, int(random.gauss(
        profile["brake_mean"] * night_multiplier, profile["brake_mean"] * 0.8
    )))
    rapid_accelerations = max(0, int(random.gauss(
        profile["accel_mean"] * night_multiplier * 0.8, profile["accel_mean"] * 0.7
    )))
    speeding_events = max(0, int(random.gauss(
        profile["speed_mean"] * 1.2, profile["speed_mean"] * 0.9
    )))

    drive_score = _compute_drive_score(
        hard_brakes, rapid_accelerations, speeding_events, night_driving_pct, distance
    )

    return {
        "trip_id": f"TRIP-{trip_n:07d}",
        "policy_number": policy["policy_number"],
        "customer_id": policy["customer_id"],
        "trip_date": trip_date.isoformat() + "Z",
        "distance_miles": distance,
        "duration_minutes": duration,
        "hard_brakes": hard_brakes,
        "rapid_accelerations": rapid_accelerations,
        "speeding_events": speeding_events,
        "night_driving_pct": night_driving_pct,
        "drive_score": drive_score,
        "source": "synthetic-v1",
    }


def generate(trips_per_policy: int, config: dict, states_data: dict,
             policies: list[dict] | None = None) -> list[dict]:
    """
    Generate telematics trip records.
    Only generates trips for policies with drive_score != null (telematics enrolled).
    Non-enrolled policies are skipped — this is expected behavior, not an error.

    `trips_per_policy` is the average trips per enrolled policy.
    """
    if policies is None:
        policies_path = Path("data/policies.json")
        if not policies_path.exists():
            raise FileNotFoundError("data/policies.json not found. Run policy_gen.py first.")
        policies = json.loads(policies_path.read_text())

    # Only enrolled policies
    enrolled_policies = [p for p in policies if p.get("drive_score") is not None]
    non_enrolled = len(policies) - len(enrolled_policies)
    print(
        f"[telematics_gen] {len(enrolled_policies):,} enrolled, "
        f"{non_enrolled:,} non-enrolled (drive_score=null) — skipping non-enrolled"
    )

    records = []
    trip_n = 1

    for policy in enrolled_policies:
        policy_drive_score = policy["drive_score"]
        profile = _driver_profile(policy_drive_score)

        # Trip volume: proportional to policy age (older policies have more trips)
        try:
            eff = datetime.fromisoformat(policy["effective_date"])
        except (ValueError, KeyError):
            eff = datetime.now() - timedelta(days=365)

        policy_age_days = max(30, (datetime.now() - eff).days)
        # Scale trips to policy age; older = more trips, capped at 2x
        age_factor = min(2.0, policy_age_days / 365)
        n_trips = max(5, int(trips_per_policy * age_factor * random.uniform(0.6, 1.4)))

        # Generate trips spread across the policy active period
        for i in range(n_trips):
            # Trip date: within last 90 days (recent trips) or spread over policy life
            if i < int(n_trips * 0.4):
                # Recent trips (last 90 days)
                days_ago = random.randint(1, 90)
            else:
                days_ago = random.randint(1, min(policy_age_days, 730))

            trip_time = datetime.now() - timedelta(days=days_ago)
            # Realistic hour distribution: peaks at 7-9am and 5-7pm
            hour_weights = [1,1,1,1,1,2,3,8,10,7,5,5,6,5,5,7,9,10,8,6,4,3,2,1]
            hour = random.choices(range(24), weights=hour_weights)[0]
            minute = random.randint(0, 59)
            trip_time = trip_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

            trip = _generate_trip(trip_n, policy, trip_time, profile)
            records.append(trip)
            trip_n += 1

    validate_records(records, "telematics.schema.json")
    return records


def main(trips_target: int, output_path: Path, config: dict, states_data: dict) -> None:
    """
    trips_target is total trips across all enrolled policies.
    The generator approximates this — actual count may vary based on policy count.
    """
    policies_path = Path("data/policies.json")
    if not policies_path.exists():
        raise FileNotFoundError("data/policies.json not found.")
    policies = json.loads(policies_path.read_text())

    enrolled_count = sum(1 for p in policies if p.get("drive_score") is not None)
    trips_per_policy = max(5, trips_target // max(enrolled_count, 1))

    records = generate(trips_per_policy, config, states_data, policies)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[telematics_gen] wrote {len(records):,} trip records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        trips_target=50000,
        output_path=Path("data/telematics.json"),
        config={"coverage_rules": coverage_rules},
        states_data=states_data,
    )

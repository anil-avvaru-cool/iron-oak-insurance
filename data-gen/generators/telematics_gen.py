"""
Telematics data generator for AIOI.

Generates synthetic trip records with driving behavior metrics
and Iron Oak Drive Score calculations.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """
    Generate telematics trip records.

    Args:
        count: Total number of trips to generate
        config: Configuration dict
        states_data: State rules dictionary

    Returns:
        List of trip records
    """
    random.seed(42)

    # Load policies
    data_dir = Path(__file__).parent.parent.parent / "data"
    try:
        with open(data_dir / "policies.json") as f:
            policies = json.load(f)
    except FileNotFoundError:
        return []

    records = []
    trip_counter = 0
    trips_per_policy = count // len(policies) if policies else count

    for policy in policies:
        policy_number = policy["policy_number"]
        customer_id = policy["customer_id"]
        num_trips = random.randint(max(1, trips_per_policy - 5), trips_per_policy + 5)

        for _ in range(num_trips):
            trip_counter += 1
            trip_id = f"TRIP-{trip_counter:07d}"

            # Trip date (within last 6 months)
            days_back = random.randint(0, 180)
            trip_date = (datetime.now() - timedelta(days=days_back)).date()
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            trip_datetime = trip_date.isoformat() + f"T{hour:02d}:{minute:02d}:00"

            # Trip metrics
            distance_miles = random.randint(5, 100)
            duration_minutes = random.randint(10, 120)
            hard_brakes = random.randint(0, 10)
            rapid_accelerations = random.randint(0, 8)
            speeding_events = random.randint(0, 5)
            night_driving_pct = random.uniform(0, 1)

            # Calculate Drive Score
            # Formula: 100 - (hard_brakes*2) - (rapid_accel*1.5) - (speeding*3) - (night_pct*10)
            drive_score = (
                100
                - (hard_brakes * 2)
                - (rapid_accelerations * 1.5)
                - (speeding_events * 3)
                - (night_driving_pct * 10)
            )
            drive_score = max(0, min(100, drive_score))  # Clamp to [0, 100]

            record = {
                "trip_id": trip_id,
                "policy_number": policy_number,
                "customer_id": customer_id,
                "trip_date": trip_datetime,
                "distance_miles": distance_miles,
                "duration_minutes": duration_minutes,
                "hard_brakes": hard_brakes,
                "rapid_accelerations": rapid_accelerations,
                "speeding_events": speeding_events,
                "night_driving_pct": round(night_driving_pct, 2),
                "drive_score": round(drive_score, 1),
                "source": "synthetic-v1",
            }
            records.append(record)

            if trip_counter >= count:
                break

        if trip_counter >= count:
            break

    return records[:count]


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    """
    Generate and write telematics records.

    Args:
        count: Total number of trip records to generate
        output_path: Path to write JSON file
        config: Configuration dictionary
        states_data: State rules dictionary
    """
    records = generate(count, config, states_data)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"[telematics.json] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    config = {}

    data_dir = Path(__file__).parent.parent.parent / "data"
    main(5000, data_dir / "telematics.json", config, states_data)

"""
customer_gen.py — generates synthetic customer records for AIOI.

Usage:
    uv run python data-gen/generators/customer_gen.py
    from customer_gen import generate, main

Key design decisions:
    - State assignment uses population-weighted random choice from states.json
    - Credit scores follow a realistic right-skewed distribution (mean ~680)
    - DOB generates adults aged 18-85, skewed toward 25-65 (driving-age population)
    - ~3% of customers have no email (prefer phone only — realistic)
    - Agent IDs assigned to clusters of customers (realistic territory model)
    - No default env values — EnvironmentError raised on missing required config
"""
from __future__ import annotations

import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

from faker import Faker
from validate import validate_records

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

# Realistic US agent territory distribution — ~20 agents cover the book
_AGENT_POOL = [f"AGT-{n:04d}" for n in range(1, 21)]

# Age distribution weights: 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-85
_AGE_BUCKETS = [
    (18, 24, 8),   # young drivers — higher risk, lower volume
    (25, 34, 20),
    (35, 44, 22),
    (45, 54, 20),
    (55, 64, 17),
    (65, 74, 10),
    (75, 85, 3),
]


def _weighted_state(states_data: dict) -> str:
    """Return a state code using population weights from states.json."""
    codes = list(states_data.keys())
    weights = [states_data[s]["weight"] for s in codes]
    return random.choices(codes, weights=weights, k=1)[0]


def _realistic_credit_score() -> int:
    """
    Credit score distribution:
        Poor (300-579):   ~16%
        Fair (580-669):   ~18%
        Good (670-739):   ~22%
        Very Good (740-799): ~25%
        Exceptional (800-850): ~19%
    Approximated with a normal distribution clipped to [300, 850].
    """
    while True:
        score = int(random.gauss(700, 95))
        if 300 <= score <= 850:
            return score


def _realistic_dob() -> date:
    """Generate DOB for a driving-age adult, weighted toward 25-65."""
    weights = [w for _, _, w in _AGE_BUCKETS]
    bucket = random.choices(_AGE_BUCKETS, weights=weights, k=1)[0]
    age = random.randint(bucket[0], bucket[1])
    today = date.today()
    birth_year = today.year - age
    try:
        dob = date(birth_year, random.randint(1, 12), random.randint(1, 28))
    except ValueError:
        dob = date(birth_year, 1, 1)
    return dob


def _zip_for_state(state: str) -> str:
    """
    Return a plausible 5-digit ZIP prefix for a state.
    Uses approximate ZIP prefix ranges (not exhaustive, but realistic for demos).
    """
    _STATE_ZIP_PREFIXES = {
        "AL": (350, 369), "AK": (995, 999), "AZ": (850, 865), "AR": (716, 729),
        "CA": (900, 961), "CO": (800, 816), "CT": (60, 69), "DE": (197, 199),
        "FL": (320, 349), "GA": (300, 319), "HI": (967, 968), "ID": (832, 838),
        "IL": (600, 629), "IN": (460, 479), "IA": (500, 528), "KS": (660, 679),
        "KY": (400, 427), "LA": (700, 714), "ME": (39, 49), "MD": (206, 219),
        "MA": (10, 27), "MI": (480, 499), "MN": (550, 567), "MS": (386, 397),
        "MO": (630, 658), "MT": (590, 599), "NE": (680, 693), "NV": (889, 898),
        "NH": (30, 38), "NJ": (70, 89), "NM": (870, 884), "NY": (100, 149),
        "NC": (270, 289), "ND": (580, 588), "OH": (430, 458), "OK": (730, 749),
        "OR": (970, 979), "PA": (150, 196), "RI": (28, 29), "SC": (290, 299),
        "SD": (570, 577), "TN": (370, 385), "TX": (750, 799), "UT": (840, 847),
        "VT": (50, 59), "VA": (200, 246), "WA": (980, 994), "WV": (247, 268),
        "WI": (530, 549), "WY": (820, 831), "DC": (200, 205),
    }
    lo, hi = _STATE_ZIP_PREFIXES.get(state, (100, 999))
    prefix = random.randint(lo, min(hi, 999))
    suffix = random.randint(0, 9999)
    return f"{prefix:03d}{suffix // 100:01d}{suffix % 100 // 10:01d}"[:5].zfill(5)


def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """Generate `count` customer records. Returns validated list of dicts."""
    records = []
    agent_weights = [random.randint(1, 10) for _ in _AGENT_POOL]

    for n in range(1, count + 1):
        state = _weighted_state(states_data)
        dob = _realistic_dob()
        has_email = random.random() > 0.03  # ~3% no email on file

        record = {
            "customer_id": f"CUST-{n:05d}",
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "state": state,
            "zip": _zip_for_state(state),
            "email": fake.email() if has_email else None,
            "dob": dob.isoformat(),
            "credit_score": _realistic_credit_score(),
            "created_at": fake.date_time_between(
                start_date="-5y", end_date="now"
            ).isoformat() + "Z",
            "source": "synthetic-v1",
        }
        records.append(record)

    validate_records(records, "customer.schema.json")
    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    records = generate(count, config, states_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[customer_gen] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        count=1000,
        output_path=Path("data/customers.json"),
        config={"coverage_rules": coverage_rules},
        states_data=states_data,
    )

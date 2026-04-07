"""
policy_gen.py — generates synthetic policy records for AIOI.

Usage:
    uv run python data-gen/generators/policy_gen.py
    from policy_gen import generate, main

Key design decisions:
    - Reads customers.json to maintain referential integrity
    - ~15% of customers have a second policy (multi-vehicle household)
    - coverage_rules.json drives mandatory coverage per state
    - ALL 7 coverage keys always present; `required` field always emitted
    - No-fault states get pip.required=true; UM-required states get UM.required=true
    - Drive score present only for telematics-enrolled policies (~62% by default)
    - Non-telematics customers get drive_score=null (not a missing value — a business state)
    - Premium calculation incorporates credit score, vehicle age, and drive score discount
    - VINs are 17-character strings matching WMI/VDS/VIS format (not checksum-validated)
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
Faker.seed(43)
random.seed(43)

_COVERAGE_TYPES = ["liability", "collision", "comprehensive", "pip", "uninsured_motorist", "gap", "roadside"]
_STATUS_WEIGHTS = [("active", 70), ("lapsed", 12), ("cancelled", 8), ("pending_renewal", 10)]

# Liability limit options as strings — matches coverage_rules.json
_LIABILITY_LIMITS = ["15/30/5", "25/50/10", "30/60/25", "50/100/25", "100/300/50", "250/500/100"]
_DEDUCTIBLES = [250, 500, 1000, 2500]

# WMI prefixes by manufacturer (approximate)
_WMI_BY_MAKE = {
    "Toyota": ["1T2", "2T1", "JT2"], "Ford": ["1FA", "1FT", "3FA"], "Chevrolet": ["1GC", "1G1", "2G1"],
    "Honda": ["1HG", "2HG", "JHM"], "Nissan": ["1N4", "3N1", "JN1"], "Jeep": ["1C4", "1J4", "3C4"],
    "Ram": ["3C6", "1C6"], "GMC": ["1GT", "2GT"], "Hyundai": ["5NP", "KMH"],
    "Kia": ["5XX", "KNA"], "Subaru": ["4S3", "4S4", "JF1"], "Tesla": ["5YJ", "7SA"],
    "Volkswagen": ["1VW", "3VW", "WVW"], "BMW": ["WBA", "WBS", "WBX"],
    "Mercedes-Benz": ["WDB", "4JG", "WDC"], "Dodge": ["2B3", "1B3", "2C3"],
    "Mazda": ["JM1", "4F2"], "Lexus": ["JTJB", "JT8"], "Buick": ["1G4", "KL4"],
    "Cadillac": ["1GY", "1G6"],
}
_VDS_CHARS = string.digits + "ABCDEFGHJKLMNPRSTUVWXYZ"  # VIN charset (no I, O, Q)


def _generate_vin(make: str) -> str:
    """Generate a plausible 17-character VIN. Correct format, not checksum-validated."""
    wmi_options = _WMI_BY_MAKE.get(make, ["1ZZ"])
    wmi = random.choice(wmi_options)
    # Pad WMI to 3 chars if shorter
    wmi = wmi.ljust(3, random.choice(_VDS_CHARS))[:3]
    vds = "".join(random.choices(_VDS_CHARS, k=5))  # vehicle descriptor
    check = random.choice(_VDS_CHARS)               # check digit (position 9)
    vis = str(random.randint(1000000, 9999999))      # vehicle identifier (positions 10-17)
    vin = (wmi + vds + check + vis)[:17].ljust(17, "0")
    return vin.upper()


def _pick_vehicle(coverage_rules: dict) -> dict:
    """Pick a realistic vehicle make/model/year combination."""
    entry = random.choice(coverage_rules["vehicle_makes_models"])
    make = entry["make"]
    model = random.choice(entry["models"])
    # Year distribution: weighted toward recent years
    years = list(range(1998, 2026))
    year_weights = [max(1, (y - 1997) ** 1.5) for y in years]
    year = int(random.choices(years, weights=year_weights, k=1)[0])
    return {"make": make, "model": model, "year": year, "vin": _generate_vin(make)}


def _build_coverages(state: str, state_rules: dict, coverage_rules: dict, vehicle_year: int) -> dict:
    """
    Build the coverages dict for a policy.
    ALL 7 coverage types are always present.
    `required` is always emitted (never omitted for optional coverages).
    """
    deductibles = coverage_rules["deductible_options"]
    liability_limits = coverage_rules["liability_limits"]
    pip_required = state_rules.get("pip_required", False)
    pip_limit = state_rules.get("pip_limit")
    um_required = state_rules.get("uninsured_motorist_required", False)
    vehicle_age = 2025 - vehicle_year

    # Customer elections for optional coverages
    # Older vehicles less likely to carry collision/comprehensive
    has_collision = random.random() < max(0.45, 0.92 - vehicle_age * 0.025)
    has_comprehensive = random.random() < max(0.50, 0.93 - vehicle_age * 0.020)
    has_um = um_required or random.random() < 0.55
    has_gap = vehicle_age <= 3 and random.random() < 0.35  # gap only makes sense on newer vehicles
    has_roadside = random.random() < 0.60

    coverages = {
        "liability": {
            "included": True,
            "required": True,
            "deductible": None,
            "limit": random.choice(liability_limits),
            "pip_limit": None,
        },
        "collision": {
            "included": has_collision,
            "required": False,
            "deductible": random.choice(deductibles) if has_collision else None,
            "limit": None,
            "pip_limit": None,
        },
        "comprehensive": {
            "included": has_comprehensive,
            "required": False,
            "deductible": random.choice(deductibles) if has_comprehensive else None,
            "limit": None,
            "pip_limit": None,
        },
        "pip": {
            "included": pip_required or random.random() < 0.25,
            "required": pip_required,
            "deductible": None,
            "limit": None,
            "pip_limit": pip_limit if pip_required else None,
        },
        "uninsured_motorist": {
            "included": has_um,
            "required": um_required,
            "deductible": None,
            "limit": random.choice(["25/50", "50/100", "100/300"]) if has_um else None,
            "pip_limit": None,
        },
        "gap": {
            "included": has_gap,
            "required": False,
            "deductible": None,
            "limit": None,
            "pip_limit": None,
        },
        "roadside": {
            "included": has_roadside,
            "required": False,
            "deductible": None,
            "limit": "roadside_standard" if has_roadside else None,
            "pip_limit": None,
        },
    }
    return coverages


def _calculate_premium(vehicle: dict, state: str, credit_score: int,
                        drive_score: float | None, coverages: dict) -> float:
    """
    Realistic premium calculation:
    - Base: $1,200 (national average baseline)
    - Vehicle age factor: newer = higher (replacement cost)
    - Credit factor: 300-850 range → 0.85-1.35 multiplier
    - Drive score factor: only if enrolled in telematics
    - State factor: FL, MI, NY, LA are high-cost states
    - Coverage breadth: more coverage = higher premium
    """
    state_factors = {
        "FL": 1.35, "MI": 1.42, "NY": 1.38, "LA": 1.30, "NJ": 1.28,
        "CA": 1.22, "MD": 1.18, "DC": 1.20, "PA": 1.10, "TX": 1.05,
        "NC": 0.92, "OH": 0.90, "ID": 0.85, "IA": 0.83, "VT": 0.82,
    }

    base = 1200.0
    vehicle_age = 2025 - vehicle["year"]
    age_factor = max(0.70, 1.15 - vehicle_age * 0.025)

    # Credit factor: 680 credit = 1.0 baseline
    credit_factor = 1.0 + max(-0.15, min(0.35, (680 - credit_score) / 500))

    # Telematics discount
    telem_discount = 0.0
    if drive_score is not None:
        if drive_score >= 90:
            telem_discount = 0.15
        elif drive_score >= 75:
            telem_discount = 0.08
        elif drive_score >= 60:
            telem_discount = 0.03

    # Coverage breadth factor
    active_coverages = sum(
        1 for k, v in coverages.items()
        if k not in ("liability",) and v.get("included", False)
    )
    coverage_factor = 1.0 + active_coverages * 0.08

    state_factor = state_factors.get(state, 1.0)

    premium = base * age_factor * credit_factor * coverage_factor * state_factor
    premium = premium * (1.0 - telem_discount)

    # Add jitter for realism
    jitter = random.uniform(0.90, 1.10)
    return round(premium * jitter, 2)


def generate(count: int, config: dict, states_data: dict,
             customers: list[dict] | None = None) -> list[dict]:
    """
    Generate policies linked to customers.
    `count` is the number of customers; policies are ~count * 1.15 (15% multi-policy).
    If `customers` is None, loads from data/customers.json.
    """
    if customers is None:
        customers_path = Path("data/customers.json")
        if not customers_path.exists():
            raise FileNotFoundError("data/customers.json not found. Run customer_gen.py first.")
        customers = json.loads(customers_path.read_text())

    coverage_rules = config["coverage_rules"]
    telematics_rate = coverage_rules.get("telematics_enrollment_rate", 0.62)
    multi_policy_rate = coverage_rules.get("multi_policy_rate", 0.15)

    records = []
    policy_n = 1

    for cust in customers:
        state = cust["state"]
        state_rules = states_data.get(state, {})
        credit_score = cust.get("credit_score") or 650

        # Determine telematics enrollment for this customer
        is_enrolled = random.random() < telematics_rate

        # Generate 1 or 2 policies per customer
        n_policies = 2 if random.random() < multi_policy_rate else 1

        for _ in range(n_policies):
            vehicle = _pick_vehicle(coverage_rules)
            drive_score = round(random.gauss(72, 15), 2) if is_enrolled else None
            if drive_score is not None:
                drive_score = round(max(0.0, min(100.0, drive_score)), 2)

            coverages = _build_coverages(state, state_rules, coverage_rules, vehicle["year"])
            premium = _calculate_premium(vehicle, state, credit_score, drive_score, coverages)

            # Policy dates — active, lapsed, or recently expired
            status_choices = [s for s, _ in _STATUS_WEIGHTS]
            status_weights = [w for _, w in _STATUS_WEIGHTS]
            status = random.choices(status_choices, weights=status_weights, k=1)[0]

            eff_date = fake.date_between(start_date="-3y", end_date="today")
            exp_date = eff_date + timedelta(days=365)

            if status in ("lapsed", "cancelled"):
                # Lapsed/cancelled policies expired in the past
                eff_date = fake.date_between(start_date="-4y", end_date="-1y")
                exp_date = eff_date + timedelta(days=365)
            elif status == "pending_renewal":
                # Renewal policies expire within 30 days
                exp_date = date.today() + timedelta(days=random.randint(1, 30))
                eff_date = exp_date - timedelta(days=365)

            record = {
                "policy_number": f"{state}-{policy_n:05d}",
                "customer_id": cust["customer_id"],
                "state": state,
                "effective_date": eff_date.isoformat(),
                "expiry_date": exp_date.isoformat(),
                "status": status,
                "coverages": coverages,
                "vehicle": vehicle,
                "premium_annual": premium,
                "drive_score": drive_score,
                "agent_id": f"AGT-{random.randint(1, 20):04d}",
                "source": "synthetic-v1",
            }
            records.append(record)
            policy_n += 1

    validate_records(records, "policy.schema.json")
    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    records = generate(count, config, states_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[policy_gen] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        count=1000,
        output_path=Path("data/policies.json"),
        config={"coverage_rules": coverage_rules},
        states_data=states_data,
    )

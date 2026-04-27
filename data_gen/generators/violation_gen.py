"""
violation_gen.py — generates driving violation records for AIOI customers.

Generation rules (all rates configurable via coverage_rules.json):
  - ~35% of customers have at least one violation in the lookback window
  - under25 drivers: 2x base violation rate
  - senior65plus drivers: 0.7x base violation rate
  - drive_score < 50: 1.5x violation rate (telematics confirms risky behavior)
  - DUI rate: ~3% of customers WITH violations get at least one DUI
  - Violation dates spread across the lookback window (not clustered)
  - expiry_date: violation_date + 3 years (minor) or + 5 years (major)
  - conviction_date: set probabilistically per violation type
  - parking_violation excluded from generation (0 points, no risk signal)
  - Output validated against violation.schema.json before writing

Usage:
    uv run python -m data_gen.generators.violation_gen
    uv run python data_gen/generators/violation_gen.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta

from data_gen.generators.validate import validate_records


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_config(config_dir: Path) -> tuple[dict, dict]:
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    states_data    = json.loads((config_dir / "states.json").read_text())
    return coverage_rules, states_data


def _driver_age_bucket(dob_str: str, today: date) -> str:
    dob = date.fromisoformat(dob_str)
    age = (today - dob).days // 365
    if age < 25:
        return "under25"
    if age < 65:
        return "standard"
    return "senior65plus"


def _random_date_in_window(today: date, lookback_years: int) -> date:
    """Return a random date within the lookback window, not in the future."""
    window_start = today - relativedelta(years=lookback_years)
    delta_days   = (today - window_start).days
    return window_start + timedelta(days=random.randint(0, delta_days - 1))


def _conviction_date(
    violation_date: date,
    violation_type: str,
    conviction_rates: dict[str, float],
    today: date,
) -> date | None:
    """
    Return a conviction date probabilistically based on violation type.
    Conviction date is 30-180 days after violation date.
    Returns None if no conviction (based on conviction_rate).
    """
    rate = conviction_rates.get(violation_type, 0.30)
    if random.random() > rate:
        return None
    # Conviction lag: 30–180 days after violation
    lag_days = random.randint(30, 180)
    conviction = violation_date + timedelta(days=lag_days)
    # Never in the future
    return min(conviction, today)


def _expiry_date(
    violation_date: date,
    violation_type: str,
    major_types: list[str],
    expiry_years: dict[str, int],
) -> date:
    """Minor violations expire after 3 years; major after 5 years."""
    years = expiry_years["major"] if violation_type in major_types else expiry_years["minor"]
    return violation_date + relativedelta(years=years)


def _select_violation_types(
    n: int,
    type_distribution: dict[str, float],
    force_dui: bool,
) -> list[str]:
    """
    Select n violation types weighted by type_distribution.
    If force_dui is True, the first violation is always dui_dwi.
    parking_violation is excluded (0 weight enforced regardless of config).
    """
    types  = [t for t, w in type_distribution.items() if t != "parking_violation" and w > 0]
    weights = [type_distribution[t] for t in types]

    selected = []
    if force_dui:
        selected.append("dui_dwi")
        n -= 1

    if n > 0:
        selected.extend(random.choices(types, weights=weights, k=n))

    return selected


# ── Core generator ─────────────────────────────────────────────────────────

def generate(
    customers: list[dict],
    policies: list[dict],
    config: dict,
) -> list[dict]:
    """
    Generate violation records for all customers.

    Args:
        customers:  List of customer dicts (from customers.json).
        policies:   List of policy dicts (from policies.json) — used to
                    look up drive_score and policy_number per customer.
        config:     Merged config dict containing 'violation_rules' key.

    Returns:
        List of violation record dicts, validated against violation.schema.json.
    """
    rules             = config["violation_rules"]
    base_rate         = rules["violation_rate_per_customer"]
    max_violations    = rules["max_violations_per_customer"]
    dui_rate          = rules["dui_rate"]
    lookback_years    = rules["lookback_years"]
    age_multipliers   = rules["age_multipliers"]
    low_ds            = rules["low_drive_score_multiplier"]
    expiry_years      = rules["expiry_years"]
    major_types       = rules["major_violation_types"]
    point_weights     = rules["point_weights"]
    conviction_rates  = rules["conviction_rate_by_type"]
    type_distribution = rules["type_distribution"]

    today = date.today()

    # Build customer_id → best policy lookup (prefer active, else latest)
    # drive_score taken from policy; policy_number linked to violation
    policy_by_customer: dict[str, dict] = {}
    for p in policies:
        cid = p["customer_id"]
        existing = policy_by_customer.get(cid)
        if existing is None:
            policy_by_customer[cid] = p
        elif p["status"] == "active" and existing["status"] != "active":
            policy_by_customer[cid] = p

    records: list[dict] = []
    violation_counter   = 1

    for customer in customers:
        cid   = customer["customer_id"]
        state = customer["state"]
        dob   = customer.get("dob", "1980-01-01")

        age_bucket = _driver_age_bucket(dob, today)
        age_mult   = age_multipliers.get(age_bucket, 1.0)

        # Drive score multiplier — use policy drive_score if enrolled
        best_policy  = policy_by_customer.get(cid)
        drive_score  = best_policy.get("drive_score") if best_policy else None
        ds_mult      = (
            low_ds["multiplier"]
            if drive_score is not None and drive_score < low_ds["threshold"]
            else 1.0
        )
        policy_number = best_policy["policy_number"] if best_policy else None

        # Effective violation probability for this customer
        effective_rate = min(base_rate * age_mult * ds_mult, 0.95)

        if random.random() > effective_rate:
            # Customer has no violations
            continue

        # Decide number of violations (1 to max_violations, weighted toward fewer)
        weights_n  = [1 / i for i in range(1, max_violations + 1)]
        n_violations = random.choices(
            range(1, max_violations + 1), weights=weights_n, k=1
        )[0]

        # Decide if this customer gets a DUI (only customers WITH violations)
        force_dui = random.random() < dui_rate

        violation_types = _select_violation_types(n_violations, type_distribution, force_dui)

        for vtype in violation_types:
            vdate   = _random_date_in_window(today, lookback_years)
            exp_dt  = _expiry_date(vdate, vtype, major_types, expiry_years)
            conv_dt = _conviction_date(vdate, vtype, conviction_rates, today)
            points  = point_weights[vtype]

            records.append({
                "violation_id":    f"VIO-{violation_counter:06d}",
                "customer_id":     cid,
                "policy_number":   policy_number,
                "state":           state,
                "violation_date":  vdate.isoformat(),
                "violation_type":  vtype,
                "points":          points,
                "conviction_date": conv_dt.isoformat() if conv_dt else None,
                "expiry_date":     exp_dt.isoformat(),
                "source":          "synthetic-v1",
            })
            violation_counter += 1

    return records


# ── Entry point ────────────────────────────────────────────────────────────

def main(
    output_path: Path,
    config: dict,
    customers: list[dict],
    policies: list[dict],
) -> list[dict]:
    """
    Generate, validate, and write violations.json.

    Args:
        output_path: Destination path (e.g. data/violations.json).
        config:      Merged config dict with violation_rules key.
        customers:   Loaded customers list.
        policies:    Loaded policies list.

    Returns:
        Generated records (also written to output_path).
    """
    records = generate(customers, policies, config)

    validate_records(records, "violation.schema.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))

    total_customers  = len(customers)
    with_violations  = len({r["customer_id"] for r in records})
    dui_count        = sum(1 for r in records if r["violation_type"] == "dui_dwi")
    major_count      = sum(1 for r in records if r["points"] >= 4)
    active_count     = sum(
        1 for r in records
        if date.fromisoformat(r["expiry_date"]) >= date.today()
    )

    print(f"[violation_gen] customers with violations : {with_violations:,} / {total_customers:,} "
          f"({with_violations / total_customers:.1%})")
    print(f"[violation_gen] total violations generated : {len(records):,}")
    print(f"[violation_gen] active (not yet expired)   : {active_count:,}")
    print(f"[violation_gen] major violations (pts >= 4): {major_count:,}")
    print(f"[violation_gen] DUI/DWI violations         : {dui_count:,}")
    print(f"[violation_gen] written → {output_path}")

    return records


if __name__ == "__main__":
    import sys

    config_dir = Path("data_gen/config")
    coverage_rules, _ = _load_config(config_dir)

    data_dir   = Path("data")
    customers  = json.loads((data_dir / "customers.json").read_text())
    policies   = json.loads((data_dir / "policies.json").read_text())

    config = {"violation_rules": coverage_rules["violation_rules"]}

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else data_dir / "violations.json"
    main(output, config, customers, policies)
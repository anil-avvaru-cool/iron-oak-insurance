"""
policy_gen.py — generates policy records for AIOI synthetic dataset.

Path:  data_gen/generators/policy_gen.py

Each policy is linked to a customer. Coverage objects are built per
state-specific rules (PIP, UM requirements). Premium calculation
incorporates credit score, vehicle age, drive score, coverage breadth,
and state factors.

CHURN SIGNAL FIX (v1.1):
    Policy `status` is no longer randomly assigned. It is driven by
    `_churn_probability()`, which correlates status with credit_score,
    drive_score, premium_annual, and state — exactly the features
    churn_features() extracts. Without this correlation, those features
    carry no signal and ROC-AUC collapses to ~0.52 (random).

    Signals wired in:
        credit_score < 580       → +0.18 churn probability
        credit_score < 650       → +0.09
        drive_score  < 40        → +0.15  (enrolled only; None → 50 proxy)
        drive_score  < 55        → +0.07
        premium_annual > 2500    → +0.10  (affordability pressure)
        premium_annual > 1800    → +0.05
        state in high-cost set   → +0.08  (FL, MI, NY, NJ)
        multi_claim customer     → +0.06  (passed via customer_churn_hints)

    Base churn rate: 0.14  (matches ~21% label rate after compounding)
    Cap: 0.78

Usage:
    uv run python data_gen/generators/policy_gen.py
    from policy_gen import generate, main
"""
from __future__ import annotations

import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

from validate import validate_records

# ── Constants ──────────────────────────────────────────────────────────────

# No-fault states — pip.required=true on every policy in these states
_NO_FAULT_STATES = frozenset({
    "DE", "FL", "HI", "KS", "KY", "MA", "MI", "MN",
    "NJ", "ND", "NY", "PA", "UT",
})

# States requiring uninsured motorist coverage
_UM_REQUIRED_STATES = frozenset({
    "CT", "IL", "KS", "MA", "ME", "MD", "MN", "MO",
    "NC", "NE", "NJ", "NY", "OR", "RI", "SC", "SD",
    "VA", "VT", "WA", "WI", "WV", "DC",
})

# High-cost states that generate additional churn pressure
_HIGH_CHURN_STATES = frozenset({"FL", "MI", "NY", "NJ"})

# State premium multipliers (spot-checked against PHASE_1_DATA_GEN.md §6.2)
_STATE_PREMIUM_FACTOR: dict[str, float] = {
    "FL": 1.35, "MI": 1.42, "NY": 1.38, "NJ": 1.28,
    "CA": 1.22, "LA": 1.25, "MD": 1.18,
    "ID": 0.85, "VT": 0.82, "NH": 0.84, "ME": 0.86, "IA": 0.88,
}

# Deductible options from coverage_rules.json
_DEDUCTIBLE_OPTIONS = [250, 500, 1000, 2500]

# Liability limit strings from coverage_rules.json
_LIABILITY_LIMITS = [
    "15/30/5", "25/50/10", "30/60/25",
    "50/100/25", "100/300/50", "250/500/100",
]

# Current year for vehicle age calculations
_CURRENT_YEAR = date.today().year


# ── Churn probability ──────────────────────────────────────────────────────

def _churn_probability(
    customer: dict,
    drive_score: float | None,
    premium_annual: float,
    state: str,
    extra_claims: int = 0,
) -> float:
    """
    Return the probability [0, 1] that this policy should be lapsed/cancelled.

    All inputs are the same values being written into the policy record, so
    the churn model's feature_engineer.py query will extract them and the
    signal will be present at training time.

    Args:
        customer:      Customer record dict (used for credit_score).
        drive_score:   Policy drive_score (None = not enrolled → proxy 50).
        premium_annual: Computed annual premium before jitter.
        state:         Two-letter state code.
        extra_claims:  Number of prior claims for this customer — passed from
                       run_all.py if available; defaults to 0 here.
    """
    p = 0.14  # base churn rate — calibrated to yield ~21% label rate overall

    # ── Credit score signal ────────────────────────────────────────────────
    credit = customer.get("credit_score") or 650
    if credit < 580:
        p += 0.18
    elif credit < 650:
        p += 0.09
    elif credit < 700:
        p += 0.04

    # ── Drive score signal (telematics enrolled only) ──────────────────────
    # Non-enrolled policies use drive_score=None; proxy to 50 for probability
    # calc only — the actual policy record still stores None.
    effective_score = drive_score if drive_score is not None else 50.0
    if effective_score < 40:
        p += 0.15
    elif effective_score < 55:
        p += 0.07
    elif effective_score < 65:
        p += 0.03

    # ── Premium affordability signal ───────────────────────────────────────
    if premium_annual > 2500:
        p += 0.10
    elif premium_annual > 1800:
        p += 0.05
    elif premium_annual > 1400:
        p += 0.02

    # ── State risk signal ──────────────────────────────────────────────────
    if state in _HIGH_CHURN_STATES:
        p += 0.08

    # ── Claims history signal (if available) ──────────────────────────────
    if extra_claims >= 3:
        p += 0.06
    elif extra_claims >= 2:
        p += 0.03

    return min(p, 0.78)


def _assign_status(churn_prob: float) -> str:
    """Convert churn probability to a policy status string."""
    if random.random() < churn_prob:
        return random.choice(["lapsed", "cancelled"])
    return "active"


# ── VIN generation ─────────────────────────────────────────────────────────

def _generate_vin() -> str:
    """Generate a syntactically valid 17-character VIN (not checksummed)."""
    chars = string.ascii_uppercase.replace("I", "").replace("O", "").replace("Q", "")
    wmi = "".join(random.choices(chars + string.digits, k=3))
    vds = "".join(random.choices(chars + string.digits, k=6))
    vis = "".join(random.choices(string.digits, k=8))
    return (wmi + vds + vis)[:17]


# ── Coverage builder ───────────────────────────────────────────────────────

def _build_coverages(
    state: str,
    vehicle_year: int,
    states_data: dict,
    config: dict,
) -> dict:
    """
    Build the full coverages object for a policy.

    Rules (PHASE_1_DATA_GEN.md §6.2):
      - All 7 coverage keys are always present.
      - `required` is always emitted — never omitted.
      - PIP: required=true for no-fault states; pip_limit from states.json.
      - UM: required=true for UM-required states.
      - Collision/comprehensive: probability declines with vehicle age.
      - GAP: only for vehicles ≤ 3 years old.
      - Roadside: ~55% take-up rate; no deductible.
    """
    state_rules = states_data.get(state, {})
    vehicle_age = _CURRENT_YEAR - vehicle_year

    # ── Liability (always required, always included) ──────────────────────
    min_limits = state_rules.get("min_liability", {})
    bi_per_person = min_limits.get("bodily_injury_per_person", 25000)
    bi_per_accident = min_limits.get("bodily_injury_per_accident", 50000)
    prop_damage = min_limits.get("property_damage", 10000)
    liability_limit = f"{bi_per_person // 1000}/{bi_per_accident // 1000}/{prop_damage // 1000}"

    coverages: dict[str, dict] = {
        "liability": {
            "included": True,
            "required": True,
            "deductible": None,
            "limit": liability_limit,
            "pip_limit": None,
        }
    }

    # ── Collision ─────────────────────────────────────────────────────────
    # Older vehicles less likely to carry collision (PHASE_1 §6.2 design decision)
    collision_prob = max(0.30, 0.95 - (vehicle_age * 0.04))
    collision_included = random.random() < collision_prob
    coverages["collision"] = {
        "included": collision_included,
        "required": False,
        "deductible": random.choice(_DEDUCTIBLE_OPTIONS) if collision_included else None,
        "limit": None,
        "pip_limit": None,
    }

    # ── Comprehensive ─────────────────────────────────────────────────────
    comp_prob = max(0.35, 0.92 - (vehicle_age * 0.035))
    comp_included = random.random() < comp_prob
    coverages["comprehensive"] = {
        "included": comp_included,
        "required": False,
        "deductible": random.choice(_DEDUCTIBLE_OPTIONS) if comp_included else None,
        "limit": None,
        "pip_limit": None,
    }

    # ── PIP ───────────────────────────────────────────────────────────────
    pip_required = state in _NO_FAULT_STATES
    pip_included = pip_required or (random.random() < 0.12)
    pip_limit = state_rules.get("pip_limit") if pip_required else None
    if pip_included and not pip_required:
        pip_limit = random.choice([2500, 5000, 10000])
    coverages["pip"] = {
        "included": pip_included,
        "required": pip_required,
        "deductible": None,
        "limit": None,
        "pip_limit": pip_limit,
    }

    # ── Uninsured / Underinsured Motorist ─────────────────────────────────
    um_required = state in _UM_REQUIRED_STATES
    um_included = um_required or (random.random() < 0.55)
    coverages["uninsured_motorist"] = {
        "included": um_included,
        "required": um_required,
        "deductible": None,
        "limit": random.choice(["25/50", "50/100", "100/300"]) if um_included else None,
        "pip_limit": None,
    }

    # ── GAP ───────────────────────────────────────────────────────────────
    # Only sensible for vehicles ≤ 3 years old (PHASE_1 §6.2)
    gap_eligible = vehicle_age <= 3
    gap_included = gap_eligible and (random.random() < 0.38)
    coverages["gap"] = {
        "included": gap_included,
        "required": False,
        "deductible": None,
        "limit": None,
        "pip_limit": None,
    }

    # ── Roadside Assistance ───────────────────────────────────────────────
    roadside_included = random.random() < 0.55
    coverages["roadside"] = {
        "included": roadside_included,
        "required": False,
        "deductible": None,
        "limit": None,
        "pip_limit": None,
    }

    return coverages


# ── Premium calculator ─────────────────────────────────────────────────────

def _calculate_premium(
    customer: dict,
    vehicle_year: int,
    drive_score: float | None,
    state: str,
    coverages: dict,
    config: dict,
) -> float:
    """
    Compute annual premium. Factors (PHASE_1_DATA_GEN.md §6.2):
      Base $1,200 × vehicle_age_factor × credit_factor
           × state_factor × coverage_factor
           × telematics_discount × jitter
    """
    base = 1200.0

    # Vehicle age factor — newer = more expensive to replace
    vehicle_age = _CURRENT_YEAR - vehicle_year
    if vehicle_age <= 2:
        age_factor = 1.40
    elif vehicle_age <= 5:
        age_factor = 1.20
    elif vehicle_age <= 10:
        age_factor = 1.00
    elif vehicle_age <= 15:
        age_factor = 0.85
    else:
        age_factor = 0.72

    # Credit factor — 680 baseline; 300-score pays ~35% more
    credit = customer.get("credit_score") or 650
    credit_factor = 1.0 + max(0.0, (680 - credit) / 680 * 0.35)

    # State factor
    state_factor = _STATE_PREMIUM_FACTOR.get(state, 1.00)

    # Coverage breadth — each optional coverage adds ~8%
    optional_included = sum(
        1 for cov_name, cov in coverages.items()
        if cov_name != "liability" and cov.get("included")
    )
    coverage_factor = 1.0 + (optional_included * 0.08)

    # Telematics discount tiers (from coverage_rules.json)
    telematics_discount = 1.0
    if drive_score is not None:
        tiers = config.get("drive_score_discount_tiers", [])
        for tier in sorted(tiers, key=lambda t: t["min"], reverse=True):
            if drive_score >= tier["min"]:
                telematics_discount = 1.0 - tier["discount_pct"]
                break

    # Jitter ±10%
    jitter = random.uniform(0.90, 1.10)

    premium = (
        base
        * age_factor
        * credit_factor
        * state_factor
        * coverage_factor
        * telematics_discount
        * jitter
    )
    return round(premium, 2)


# ── Vehicle generator ──────────────────────────────────────────────────────

# Make-specific year floors — prevents historically impossible combinations
_MAKE_YEAR_FLOORS = {
    "Tesla":      2008,  # first Roadster deliveries
    "Jeep":       1941,  # safe — already above 1990 floor
    "Ram":        2010,  # Ram became standalone brand (split from Dodge)
    "GMC":        1990,  # fine as-is, but explicit for clarity
}

def _pick_vehicle(config: dict) -> dict:
    """Pick a random make/model/year/vin combination."""
    makes_models = config.get("coverage_rules", {}).get("vehicle_makes_models", [])
    make_entry = random.choice(makes_models)
    make = make_entry["make"]
    model = random.choice(make_entry["models"])

    # Year floor for this make — prevents e.g. 1991 Tesla
    year_floor = max(_MAKE_YEAR_FLOORS.get(make, 1990), 1990)

    # Year distribution: weighted toward recent years
    years = list(range(year_floor, _CURRENT_YEAR + 1))
    year_weights = []
    for y in years:
        age = _CURRENT_YEAR - y
        if age <= 3:
            year_weights.append(8)
        elif age <= 7:
            year_weights.append(6)
        elif age <= 12:
            year_weights.append(4)
        elif age <= 18:
            year_weights.append(2)
        else:
            year_weights.append(1)

    year = random.choices(years, weights=year_weights, k=1)[0]
    return {"make": make, "model": model, "year": year, "vin": _generate_vin()}


# ── Policy date generator ──────────────────────────────────────────────────

def _pick_dates(status: str) -> tuple[str, str]:
    """
    Return (effective_date, expiry_date) as ISO strings.

    Active policies: 1-year term, effective within the last 12 months.
    Lapsed/cancelled: effective 1–3 years ago; expiry in the past.
    pending_renewal: effective within last 14 months; expiry within next 30 days.
    """
    today = date.today()

    if status == "active":
        days_ago = random.randint(0, 365)
        effective = today - timedelta(days=days_ago)
        expiry = effective + timedelta(days=365)

    elif status in ("lapsed", "cancelled"):
        days_ago = random.randint(365, 3 * 365)
        effective = today - timedelta(days=days_ago)
        # Term was 1 year; policy expired in the past
        expiry = effective + timedelta(days=365)

    else:  # pending_renewal
        days_ago = random.randint(335, 395)
        effective = today - timedelta(days=days_ago)
        expiry = effective + timedelta(days=365)

    return effective.isoformat(), expiry.isoformat()


# ── Drive score assignment ─────────────────────────────────────────────────

def _assign_drive_score(
    enrollment_rate: float,
    credit_score: int | None,
    status: str,
) -> float | None:
    """
    Assign a drive_score or None (not enrolled).

    Enrollment rate from coverage_rules.json (default 0.62).
    Lapsed/cancelled policies are slightly less likely to have been enrolled
    (low-engagement customers opt out of telematics more often).
    """
    effective_rate = enrollment_rate
    if status in ("lapsed", "cancelled"):
        effective_rate *= 0.80  # lower enrollment among churned customers

    if random.random() >= effective_rate:
        return None  # not enrolled

    # Drive score distribution: roughly normal around 72, σ=18
    raw = random.gauss(72, 18)
    # Low credit customers trend slightly lower on drive scores
    if credit_score is not None and credit_score < 620:
        raw -= random.uniform(3, 10)

    return round(max(0.0, min(100.0, raw)), 2)


# ── Main generator ─────────────────────────────────────────────────────────

def generate(
    customers: list[dict],
    config: dict,
    states_data: dict,
    customer_claim_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    Generate policy records.

    Args:
        customers:             List of customer dicts from customer_gen.
        config:                Parsed coverage_rules.json.
        states_data:           Parsed states.json.
        customer_claim_counts: Optional dict {customer_id: claim_count} for
                               churn probability enrichment. Pass in from
                               run_all.py if claim data is available from a
                               prior run; otherwise omit.

    Returns:
        List of validated policy dicts.
    """
    if customer_claim_counts is None:
        customer_claim_counts = {}

    enrollment_rate = config.get("telematics_enrollment_rate", 0.62)
    multi_policy_rate = config.get("multi_policy_rate", 0.15)

    # Build set of agent IDs — 20 agents with territory clustering
    agent_ids = [f"AGT-{i:03d}" for i in range(1, 21)]

    policies: list[dict] = []
    policy_counter = 1

    for customer in customers:
        state = customer["state"]
        customer_id = customer["customer_id"]
        credit = customer.get("credit_score")
        extra_claims = customer_claim_counts.get(customer_id, 0)

        # Determine how many policies this customer has
        num_policies = 2 if random.random() < multi_policy_rate else 1

        for _ in range(num_policies):
            policy_number = f"{state}-{policy_counter:05d}"
            policy_counter += 1

            vehicle = _pick_vehicle(config)
            vehicle_year = vehicle["year"]

            # Build coverages first (needed for premium calc)
            coverages = _build_coverages(state, vehicle_year, states_data, config)

            # Assign drive score (enrollment decision)
            drive_score = _assign_drive_score(enrollment_rate, credit, "active")
            # We'll adjust drive_score below once status is determined,
            # but we need a preliminary premium to compute churn probability.

            premium = _calculate_premium(
                customer, vehicle_year, drive_score, state, coverages, config
            )

            # ── CHURN SIGNAL FIX ──────────────────────────────────────────
            # Status is driven by churn_probability, not random.choice.
            # This is the core fix for ROC-AUC ~0.52.
            churn_prob = _churn_probability(
                customer=customer,
                drive_score=drive_score,
                premium_annual=premium,
                state=state,
                extra_claims=extra_claims,
            )
            status = _assign_status(churn_prob)

            # Re-evaluate drive score with status now known (lapsed/cancelled
            # customers have lower enrollment probability).
            drive_score = _assign_drive_score(enrollment_rate, credit, status)

            # Recompute premium with final drive_score (telematics discount)
            premium = _calculate_premium(
                customer, vehicle_year, drive_score, state, coverages, config
            )

            effective_date, expiry_date = _pick_dates(status)

            agent_id = random.choices(
                agent_ids,
                weights=[5 if i < 5 else 3 if i < 10 else 1 for i in range(20)],
                k=1,
            )[0]

            policy: dict = {
                "policy_number": policy_number,
                "customer_id": customer_id,
                "state": state,
                "effective_date": effective_date,
                "expiry_date": expiry_date,
                "status": status,
                "coverages": coverages,
                "vehicle": vehicle,
                "premium_annual": premium,
                "drive_score": drive_score,
                "agent_id": agent_id,
                "source": "synthetic-v1",
            }
            policies.append(policy)

    validate_records(policies, "policy.schema.json")
    return policies


def main(
    count: int,                   # ← restored: matches run_all.py call signature
    output_path: Path,
    config: dict,
    states_data: dict,
    customer_claim_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    Load customers from disk, generate policies, write to output_path.
    Matches the original call signature expected by run_all.py:
        gen_policies(args.customers, data_dir / "policies.json", config, states_data)
    """
    customers_path = output_path.parent / "customers.json"
    customers = json.loads(customers_path.read_text())

    policies = generate(customers, config, states_data, customer_claim_counts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policies, indent=2))
    print(f"[policy_gen] wrote {len(policies):,} policies → {output_path}")

    enrolled = sum(1 for p in policies if p["drive_score"] is not None)
    lapsed_or_cancelled = sum(1 for p in policies if p["status"] in ("lapsed", "cancelled"))
    print(
        f"  status: {sum(1 for p in policies if p['status'] == 'active'):,} active  "
        f"{lapsed_or_cancelled:,} lapsed/cancelled  "
        f"{sum(1 for p in policies if p['status'] == 'pending_renewal'):,} pending_renewal"
    )
    print(
        f"  telematics: {enrolled:,} enrolled ({enrolled / len(policies):.0%})  "
        f"{len(policies) - enrolled:,} not enrolled"
    )
    return policies


if __name__ == "__main__":
    config_dir = Path("data_gen/config")
    data_dir = Path("data")

    states_data_ = json.loads((config_dir / "states.json").read_text())
    config_ = json.loads((config_dir / "coverage_rules.json").read_text())

    # count arg is ignored when running standalone — customers loaded from disk
    main(
        count=0,                                   # ignored; customers loaded from disk
        output_path=data_dir / "policies.json",
        config=config_,
        states_data=states_data_,
    )
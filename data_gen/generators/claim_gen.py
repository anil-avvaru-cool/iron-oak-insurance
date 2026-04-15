"""
claim_gen.py — generates synthetic claim records for AIOI.

Usage:
    uv run python data-gen/generators/claim_gen.py
    from claim_gen import generate, main

Key design decisions:
    - Claim rate defaults to ~28% of policies (configurable via coverage_rules.json)
    - Fraud injected at configurable rate (default 4%); fraud signals are specific and consistent
    - Fraud signals are realistic combinations: claim_delta_high + telematics_anomaly, etc.
    - Adjuster notes and narratives use domain-specific vocabulary
    - Claims match the coverage types on the policy (no PIP claim if PIP not on policy)
    - Filed date is always >= incident date; filing lag is realistic (0-60 days)
    - Settlement amounts are < claim amounts for approved/settled claims
    - No-fault state claims skew toward PIP claim type
    - Per-policy claim count drawn from Poisson distribution and hard-capped at
      MAX_CLAIMS_PER_POLICY_PER_YEAR * policy_age_years to prevent ML training outliers
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker
from validate import validate_records

fake = Faker("en_US")
Faker.seed(44)
random.seed(44)

# ---------------------------------------------------------------------------
# Per-policy claim count limits
# Industry average is ~0.28 claims/policy/year (claim_rate_per_policy).
# Poisson draw models the natural randomness of rare independent events.
# Hard cap prevents synthetic outliers from corrupting ML training targets —
# no real policy accumulates 198 lifetime claims.
# ---------------------------------------------------------------------------
MAX_CLAIMS_PER_POLICY_PER_YEAR = 3   # catastrophic single-policy ceiling
_rng = np.random.default_rng(seed=44)

_CLAIM_TYPES = ["collision", "comprehensive", "liability", "pip", "uninsured_motorist"]

# Adjuster note and narrative templates by claim type
_COLLISION_NARRATIVES = [
    "Insured reports vehicle struck another vehicle at intersection. Airbags deployed. Damage to front bumper, hood, and radiator assessed at repair facility.",
    "Insured was rear-ended at a stoplight. Significant rear damage to trunk and bumper. Other driver admitted fault at scene.",
    "Insured sideswiped a parked vehicle while changing lanes. Door and quarter panel damage on driver side.",
    "Insured vehicle lost control on wet road and struck guardrail. Front-end and suspension damage noted by appraiser.",
    "Multi-vehicle accident on highway. Insured vehicle struck from behind and pushed into vehicle ahead.",
]
_COMPREHENSIVE_NARRATIVES = [
    "Insured reports windshield shattered by road debris on I-35. No other damage noted.",
    "Vehicle vandalized overnight in parking garage. Keyed paint on both sides, side mirror broken.",
    "Hail storm caused extensive denting to hood, roof, and trunk. Windshield cracked.",
    "Tree branch fell on vehicle during storm. Roof dented, sunroof shattered.",
    "Insured reports vehicle was stolen from driveway. Vehicle recovered 3 days later with interior damage.",
    "Flood damage from heavy rainfall. Water entered interior, engine compartment affected.",
]
_LIABILITY_NARRATIVES = [
    "Insured at fault in collision with third party. Third party filing injury and property damage claim.",
    "Insured reversed into pedestrian in parking lot. Minor injury reported. Third-party medical claim filed.",
    "Insured ran red light and struck another vehicle. Liability clear per police report.",
]
_PIP_NARRATIVES = [
    "Insured and passenger sustained injuries in collision. PIP claim filed for medical expenses and lost wages.",
    "Insured sustained whiplash in rear-end collision. Physical therapy prescribed. PIP coverage applies.",
    "Passenger sustained soft tissue injuries. Insured filing PIP for medical treatment received.",
]
_UM_NARRATIVES = [
    "Insured struck by uninsured driver who fled the scene. Police report filed. UM coverage applies.",
    "Insured involved in hit-and-run. Uninsured motorist coverage invoked per policy terms.",
]

_ADJUSTER_NOTES_TEMPLATES = [
    "Initial inspection completed. Estimate obtained from preferred repair facility. Photos documented in claim file.",
    "Third-party appraisal ordered. Waiting on medical records for injury component. Reserves adjusted.",
    "Subrogation potential identified — other party may be at fault. SIU referral not warranted at this time.",
    "Customer cooperative. Documentation complete. Claim straightforward per initial review.",
    "Rental authorized per policy terms. Repair timeline estimated at 7-10 business days.",
    "Independent medical exam (IME) ordered given extended treatment duration.",
    "Police report obtained. Corroborates insured's account. Proceeding to payment.",
    "Customer submitted repair receipts. Within estimated range. Approved for payment.",
]

# Fraud-specific adjuster notes
_FRAUD_ADJUSTER_NOTES = [
    "Claim flagged by automated system for review. Multiple signals identified — refer to SIU.",
    "SIU referral initiated. Claim frequency inconsistent with driving profile. Escalating.",
    "Telematics data inconsistent with reported incident date/location. Further investigation required.",
    "Policy recently reinstated before claim date. Timing anomaly noted. SIU notified.",
    "Claim amount significantly exceeds vehicle ACV. Independent appraisal ordered.",
]

# Fraud signal combinations that are internally consistent
_FRAUD_SIGNAL_COMBOS = [
    ["claim_delta_high", "recent_policy_reinstatement"],
    ["claim_delta_high", "telematics_anomaly", "frequency_spike"],
    ["telematics_anomaly", "incident_location_mismatch"],
    ["frequency_spike", "rapid_refiling"],
    ["claim_delta_high", "staged_accident_pattern", "third_party_attorney_early"],
    ["telematics_anomaly", "claim_filed_after_lapse_reinstatement"],
    ["frequency_spike", "claim_delta_high"],
    ["staged_accident_pattern", "no_police_report", "multiple_claimants"],
]


def _policy_age_years(policy: dict) -> float:
    """
    Returns policy age in years, floored at 0.25 (3 months).
    Used to scale the per-policy claim count cap and Poisson expected value.
    """
    try:
        eff = date.fromisoformat(policy["effective_date"])
        exp = date.fromisoformat(policy["expiry_date"])
        end = min(exp, date.today())
        age_days = max((end - eff).days, 1)
    except (ValueError, KeyError):
        age_days = 365
    return max(age_days / 365.0, 0.25)


def _claims_count_for_policy(claim_rate: float, policy_age_years: float) -> int:
    """
    Draw a realistic claim count for one policy using a Poisson distribution,
    then hard-cap at MAX_CLAIMS_PER_POLICY_PER_YEAR * policy_age_years.

    Poisson is the natural model for rare, independent events (accidents).
    The hard cap prevents the long tail of the Poisson from producing
    unrealistic multi-claim policies that corrupt ML training targets.

    Examples at claim_rate=0.28:
      1-year policy  → expected=0.28, Poisson draw usually 0 or 1, cap=3
      2-year policy  → expected=0.56, draw usually 0-2, cap=6
    """
    expected = claim_rate * policy_age_years
    draw = int(_rng.poisson(expected))
    cap = math.ceil(MAX_CLAIMS_PER_POLICY_PER_YEAR * policy_age_years)
    return min(draw, cap)


def _pick_claim_type(policy: dict, state_rules: dict) -> str:
    """
    Pick a claim type that matches the policy's active coverages.
    No-fault states bias toward PIP.
    """
    available = []
    coverages = policy.get("coverages", {})

    if coverages.get("collision", {}).get("included"):
        available.extend(["collision"] * 3)
    if coverages.get("comprehensive", {}).get("included"):
        available.extend(["comprehensive"] * 2)
    if coverages.get("liability", {}).get("included"):
        available.extend(["liability"] * 2)
    if coverages.get("pip", {}).get("included"):
        weight = 4 if state_rules.get("pip_required") else 1
        available.extend(["pip"] * weight)
    if coverages.get("uninsured_motorist", {}).get("included"):
        available.extend(["uninsured_motorist"] * 1)

    if not available:
        available = ["collision"]  # fallback

    return random.choice(available)


def _narrative_for_type(claim_type: str) -> str:
    templates = {
        "collision": _COLLISION_NARRATIVES,
        "comprehensive": _COMPREHENSIVE_NARRATIVES,
        "liability": _LIABILITY_NARRATIVES,
        "pip": _PIP_NARRATIVES,
        "uninsured_motorist": _UM_NARRATIVES,
    }
    return random.choice(templates.get(claim_type, _COLLISION_NARRATIVES))


def _claim_amount_for_type(claim_type: str, is_fraud: bool) -> float:
    """
    Realistic claim amounts by type. Fraud claims skew higher.
    """
    ranges = {
        "collision": (1500, 18000),
        "comprehensive": (800, 12000),
        "liability": (3000, 35000),
        "pip": (1000, 20000),
        "uninsured_motorist": (5000, 40000),
    }
    lo, hi = ranges.get(claim_type, (1000, 10000))
    if is_fraud:
        # Fraud claims inflate by 30-80%
        lo = int(lo * 1.3)
        hi = int(hi * 1.8)

    # Log-normal distribution within range for realism
    mid = (lo + hi) / 2
    std = (hi - lo) / 6
    amount = random.gauss(mid, std)
    return round(max(lo, min(hi, amount)), 2)


def generate(count: int, config: dict, states_data: dict,
             policies: list[dict] | None = None) -> list[dict]:
    """
    Generate claims. `count` is the number of policies (controls claim volume).
    If `policies` is None, loads from data/policies.json.
    """
    if policies is None:
        policies_path = Path("data/policies.json")
        if not policies_path.exists():
            raise FileNotFoundError("data/policies.json not found. Run policy_gen.py first.")
        policies = json.loads(policies_path.read_text())

    coverage_rules = config["coverage_rules"]
    claim_rate = coverage_rules.get("claim_rate_per_policy", 0.28)
    fraud_rate = config.get("fraud_rate", coverage_rules.get("fraud_rate", 0.04))

    # Build a lookup of policies that are eligible to have claims
    eligible_policies = [
        p for p in policies
        if p["status"] not in ("cancelled",)
    ]

    records = []
    claim_n = 1

    for policy in eligible_policies:
        age_years = _policy_age_years(policy)
        n_claims = _claims_count_for_policy(claim_rate, age_years)

        if n_claims == 0:
            continue

        state = policy["state"]
        state_rules = states_data.get(state, {})

        for _ in range(n_claims):
            is_fraud = random.random() < fraud_rate
            claim_type = _pick_claim_type(policy, state_rules)
            claim_amount = _claim_amount_for_type(claim_type, is_fraud)

            # Incident date within policy effective period, not in future
            try:
                eff = date.fromisoformat(policy["effective_date"])
                exp = date.fromisoformat(policy["expiry_date"])
            except (ValueError, KeyError):
                eff = date.today() - timedelta(days=365)
                exp = date.today()

            end = min(exp, date.today())
            if eff >= end:
                eff = end - timedelta(days=180)

            incident_date = fake.date_between(start_date=eff, end_date=end)

            # Filing lag: 0-60 days, fraud claims sometimes filed faster (urgency signal)
            if is_fraud:
                lag_days = random.choices([0, 1, 2], weights=[40, 35, 25])[0]
            else:
                lag_days = random.choices(
                    range(0, 61), weights=[max(1, 60 - i) for i in range(61)]
                )[0]
            filed_date = incident_date + timedelta(days=lag_days)

            # Claim status
            if policy["status"] == "lapsed":
                status_choices = ["denied"] * 4 + ["under_review"] * 1
            else:
                status_choices = (
                    ["open"] * 15 + ["under_review"] * 20 +
                    ["approved"] * 25 + ["settled"] * 35 + ["denied"] * 5
                )
            status = random.choice(status_choices)

            # Settlement — only for approved/settled claims
            settlement_amount = None
            if status in ("approved", "settled"):
                # Settlement is 70-95% of claim amount (deductible + negotiation)
                ratio = random.uniform(0.70, 0.95)
                settlement_amount = round(claim_amount * ratio, 2)

            # Fraud signals
            fraud_signals: list[str] = []
            if is_fraud:
                fraud_signals = list(random.choice(_FRAUD_SIGNAL_COMBOS))

            adjuster_notes = (
                random.choice(_FRAUD_ADJUSTER_NOTES) if is_fraud
                else random.choice(_ADJUSTER_NOTES_TEMPLATES)
            )

            record = {
                "claim_id": f"CLM-{claim_n:05d}",
                "policy_number": policy["policy_number"],
                "customer_id": policy["customer_id"],
                "state": state,
                "incident_date": incident_date.isoformat(),
                "filed_date": filed_date.isoformat(),
                "claim_type": claim_type,
                "status": status,
                "claim_amount": claim_amount,
                "settlement_amount": settlement_amount,
                "adjuster_notes": adjuster_notes,
                "incident_narrative": _narrative_for_type(claim_type),
                "is_fraud": is_fraud,
                "fraud_signals": fraud_signals,
                "source": "synthetic-v1",
            }
            records.append(record)
            claim_n += 1

    validate_records(records, "claim.schema.json")
    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    records = generate(count, config, states_data)
    fraud_count = sum(1 for r in records if r["is_fraud"])
    fraud_pct = fraud_count / max(len(records), 1) * 100
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(
        f"[claim_gen] wrote {len(records):,} records → {output_path} "
        f"({fraud_count} fraud, {fraud_pct:.1f}%)"
    )


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        count=1000,
        output_path=Path("data/claims.json"),
        config={"coverage_rules": coverage_rules, "fraud_rate": 0.04},
        states_data=states_data,
    )
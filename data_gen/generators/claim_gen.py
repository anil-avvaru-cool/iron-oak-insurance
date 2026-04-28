"""
claim_gen.py — generates synthetic claim records for AIOI.

Usage:
    uv run python data_gen/generators/claim_gen.py
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
    - reported_passengers: fraud claims skew toward more passengers (jump-in signal)
    - num_witnesses: fraud claims skew toward more convenient witnesses

v1.1 fix: _policy_age_years() now uses the full policy term (effective → expiry)
    instead of the elapsed portion (effective → min(expiry, today)).
    Active 1-year policies were being credited only for their elapsed days,
    causing Poisson expected value to collapse to ~0.07 and most policies
    drawing 0 claims. Result: 5000 customers → 855 claims instead of ~4000+.
    The incident_date generator still clamps to (effective, today) so all
    incidents remain in the past.

v1.2 — risk-sensitive claim rates:
    Claim probability is now modulated by a risk multiplier derived from:
      - drive_score (telematics behavioral signal)
      - active violation points and violation type (DUI, major, minor)
      - driver age bucket (under25 = 1.5x, senior65plus = 1.2x)
    This ensures behavioral features (violations, telematics) have genuine
    predictive power in the risk scoring model. Without this, claim_gen
    assigns claims randomly and XGBoost cannot learn a violation→claim signal.
    Multipliers are sourced from violation_rules in coverage_rules.json and
    applied before the Poisson draw so the expected claim count reflects
    the customer's actual risk profile.
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
# ---------------------------------------------------------------------------
MAX_CLAIMS_PER_POLICY_PER_YEAR = 3
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

_FRAUD_ADJUSTER_NOTES = [
    "Claim flagged by automated system for review. Multiple signals identified — refer to SIU.",
    "SIU referral initiated. Claim frequency inconsistent with driving profile. Escalating.",
    "Telematics data inconsistent with reported incident date/location. Further investigation required.",
    "Policy recently reinstated before claim date. Timing anomaly noted. SIU notified.",
    "Claim amount significantly exceeds vehicle ACV. Independent appraisal ordered.",
]

_FRAUD_SIGNAL_COMBOS = [
    ["telematics_anomaly", "incident_location_mismatch"],
    ["telematics_anomaly", "claim_delta_high"],
    ["staged_accident_pattern", "no_police_report", "multiple_claimants"],
    ["staged_accident_pattern", "third_party_attorney_early"],
    ["claim_delta_high", "recent_policy_reinstatement"],
    ["claim_filed_after_lapse_reinstatement", "claim_delta_high"],
    ["frequency_spike", "rapid_refiling"],
]


def _policy_age_years(policy: dict) -> float:
    """
    Returns the full policy term in years for claim count modeling.

    v1.1 fix: use full term (effective → expiry) not elapsed portion.
    Floor at 0.25 (3 months) to prevent zero-exposure edge cases.
    """
    try:
        eff = date.fromisoformat(policy["effective_date"])
        exp = date.fromisoformat(policy["expiry_date"])
        age_days = max((exp - eff).days, 1)
    except (ValueError, KeyError):
        age_days = 365
    return max(age_days / 365.0, 0.25)


def _driver_age_bucket(dob_str: str, today: date) -> str:
    """Return actuarial age tier from date of birth string."""
    try:
        dob = date.fromisoformat(dob_str)
        age = (today - dob).days // 365
    except (ValueError, TypeError):
        return "standard"
    if age < 25:
        return "under25"
    if age < 65:
        return "standard"
    return "senior65plus"


def _claim_risk_multiplier(
    policy: dict,
    customer: dict,
    customer_violations: list[dict],
    violation_rules: dict,
    today: date,
) -> float:
    """
    Compute a risk multiplier for the base claim rate.

    Inputs:
        policy:              policy dict (drive_score, status)
        customer:            customer dict (dob)
        customer_violations: list of violation dicts for this customer
        violation_rules:     violation_rules block from coverage_rules.json
        today:               reference date for expiry checks

    Returns:
        float >= 0.5 — multiplied against base claim_rate before Poisson draw.

    Multiplier components (multiplicative, not additive):
        drive_score   : low score = higher claim probability
        violations    : active points drive severity; DUI = strongest signal
        driver age    : under25 and senior65plus are actuarially higher risk

    Cap at 4.0 to prevent extreme outlier policies from generating
    unrealistically high claim counts that would dominate the training set.
    """
    multiplier = 1.0
    point_weights  = violation_rules.get("point_weights", {})
    age_multipliers = violation_rules.get("age_multipliers", {})

    # ── Drive score component ───────────────────────────────────────────────
    drive_score = policy.get("drive_score")
    if drive_score is not None:
        if drive_score < 40:
            multiplier *= 1.8
        elif drive_score < 60:
            multiplier *= 1.3
        elif drive_score > 80:
            multiplier *= 0.7
    # Non-enrolled policies (drive_score=None) get no adjustment —
    # we have no behavioral signal for them so we leave the base rate.

    # ── Violation component ─────────────────────────────────────────────────
    active_points = sum(
        point_weights.get(v["violation_type"], 0)
        for v in customer_violations
        if v.get("expiry_date", "") >= today.isoformat()
    )
    has_dui = any(
        v["violation_type"] == "dui_dwi"
        and v.get("expiry_date", "") >= today.isoformat()
        for v in customer_violations
    )

    if has_dui:
        multiplier *= 2.5
    elif active_points >= 8:    # reckless driving or multiple majors
        multiplier *= 1.9
    elif active_points >= 4:    # one major violation
        multiplier *= 1.5
    elif active_points >= 1:    # minor violations only
        multiplier *= 1.2

    # ── Driver age component ────────────────────────────────────────────────
    bucket = _driver_age_bucket(customer.get("dob", "1980-01-01"), today)
    age_mult = age_multipliers.get(bucket, 1.0)
    multiplier *= age_mult

    # ── Cap ─────────────────────────────────────────────────────────────────
    return min(multiplier, 4.0)


def _claims_count_for_policy(
    claim_rate: float,
    policy_age_years: float,
    risk_multiplier: float = 1.0,
) -> int:
    """
    Draw a realistic claim count for one policy using a Poisson distribution,
    then hard-cap at MAX_CLAIMS_PER_POLICY_PER_YEAR * policy_age_years.

    risk_multiplier scales the expected value before the Poisson draw so
    high-risk policies naturally generate more claims.
    """
    expected = claim_rate * policy_age_years * risk_multiplier
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
        available = ["collision"]

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
        lo = int(lo * 1.3)
        hi = int(hi * 1.8)

    mid = (lo + hi) / 2
    std = (hi - lo) / 6
    amount = random.gauss(mid, std)
    return round(max(lo, min(hi, amount)), 2)


def _reported_passengers(claim_type: str, is_fraud: bool) -> int:
    if claim_type == "comprehensive":
        return random.choices([0, 1, 2], weights=[70, 25, 5])[0]
    if is_fraud and claim_type in ("pip", "liability", "collision"):
        return random.choices([0, 1, 2, 3, 4, 5], weights=[5, 10, 25, 30, 20, 10])[0]
    return random.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]


def _num_witnesses(is_fraud: bool) -> int:
    if is_fraud:
        return random.choices([0, 1, 2, 3], weights=[20, 35, 30, 15])[0]
    return random.choices([0, 1, 2, 3], weights=[55, 30, 12, 3])[0]


def generate(
    count: int,
    config: dict,
    states_data: dict,
    policies: list[dict] | None = None,
    customers: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> list[dict]:
    """
    Generate claims. `count` is the number of policies (controls claim volume).

    Args:
        count:      Number of customers (used for volume logging only).
        config:     Merged config dict with coverage_rules and fraud_rate keys.
        states_data: States config dict.
        policies:   Policy list. If None, loaded from data/policies.json.
        customers:  Customer list. If None, loaded from data/customers.json.
        violations: Violation list. If None, loaded from data/violations.json
                    if the file exists; silently skipped if absent (Phase 1
                    compatibility — violation_gen runs after claim_gen in the
                    original order but before it in the updated run_all.py).

    v1.2: customers and violations are new optional parameters used to compute
    per-policy risk multipliers. All three default to None for backward
    compatibility with any callers that do not yet pass them.
    """
    data_dir = Path("data")

    if policies is None:
        policies_path = data_dir / "policies.json"
        if not policies_path.exists():
            raise FileNotFoundError("data/policies.json not found. Run policy_gen.py first.")
        policies = json.loads(policies_path.read_text())

    if customers is None:
        customers_path = data_dir / "customers.json"
        if customers_path.exists():
            customers = json.loads(customers_path.read_text())
        else:
            customers = []

    if violations is None:
        violations_path = data_dir / "violations.json"
        if violations_path.exists():
            violations = json.loads(violations_path.read_text())
        else:
            violations = []

    coverage_rules   = config["coverage_rules"]
    claim_rate       = coverage_rules.get("claim_rate_per_policy", 0.28)
    fraud_rate       = config.get("fraud_rate", coverage_rules.get("fraud_rate", 0.04))
    violation_rules  = coverage_rules.get("violation_rules", {})

    # Build lookup maps for O(1) access inside the policy loop
    customer_map: dict[str, dict] = {c["customer_id"]: c for c in customers}
    violations_by_customer: dict[str, list[dict]] = {}
    for v in violations:
        cid = v["customer_id"]
        violations_by_customer.setdefault(cid, []).append(v)

    today = date.today()

    eligible_policies = [
        p for p in policies
        if p["status"] not in ("cancelled",)
    ]

    records  = []
    claim_n  = 1

    for policy in eligible_policies:
        age_years = _policy_age_years(policy)
        cid       = policy["customer_id"]
        customer  = customer_map.get(cid, {})
        cust_violations = violations_by_customer.get(cid, [])

        # v1.2: compute risk multiplier from behavioral signals
        if violation_rules and (customer or cust_violations):
            risk_mult = _claim_risk_multiplier(
                policy, customer, cust_violations, violation_rules, today
            )
        else:
            risk_mult = 1.0

        n_claims = _claims_count_for_policy(claim_rate, age_years, risk_mult)

        if n_claims == 0:
            continue

        state       = policy["state"]
        state_rules = states_data.get(state, {})

        try:
            eff = date.fromisoformat(policy["effective_date"])
            exp = date.fromisoformat(policy["expiry_date"])
        except (ValueError, KeyError):
            eff = today - timedelta(days=365)
            exp = today

        incident_end   = min(exp, today)
        incident_start = eff
        if incident_start >= incident_end:
            incident_start = incident_end - timedelta(days=180)

        for _ in range(n_claims):
            is_fraud   = random.random() < fraud_rate
            claim_type = _pick_claim_type(policy, state_rules)
            claim_amount = _claim_amount_for_type(claim_type, is_fraud)

            incident_date = fake.date_between(
                start_date=incident_start,
                end_date=incident_end,
            )

            if is_fraud:
                if random.random() < 0.50:
                    lag_days = random.choices([0, 1, 2, 3], weights=[30, 30, 25, 15])[0]
                else:
                    lag_days = random.choices(
                        range(0, 61), weights=[max(1, 60 - i) for i in range(61)]
                    )[0]
            else:
                lag_days = random.choices(
                    range(0, 61), weights=[max(1, 60 - i) for i in range(61)]
                )[0]
            filed_date = incident_date + timedelta(days=lag_days)

            if policy["status"] == "lapsed":
                status_choices = ["denied"] * 4 + ["under_review"] * 1
            else:
                status_choices = (
                    ["open"] * 15 + ["under_review"] * 20 +
                    ["approved"] * 25 + ["settled"] * 35 + ["denied"] * 5
                )
            status = random.choice(status_choices)

            settlement_amount = None
            if status in ("approved", "settled"):
                ratio = random.uniform(0.70, 0.95)
                settlement_amount = round(claim_amount * ratio, 2)

            fraud_signals: list[str] = []
            if is_fraud:
                fraud_signals = list(random.choice(_FRAUD_SIGNAL_COMBOS))

            adjuster_notes = (
                random.choice(_FRAUD_ADJUSTER_NOTES) if is_fraud
                else random.choice(_ADJUSTER_NOTES_TEMPLATES)
            )

            record = {
                "claim_id":            f"CLM-{claim_n:05d}",
                "policy_number":       policy["policy_number"],
                "customer_id":         cid,
                "state":               state,
                "incident_date":       incident_date.isoformat(),
                "filed_date":          filed_date.isoformat(),
                "claim_type":          claim_type,
                "status":              status,
                "claim_amount":        claim_amount,
                "settlement_amount":   settlement_amount,
                "adjuster_notes":      adjuster_notes,
                "incident_narrative":  _narrative_for_type(claim_type),
                "is_fraud":            is_fraud,
                "fraud_signals":       fraud_signals,
                "reported_passengers": _reported_passengers(claim_type, is_fraud),
                "num_witnesses":       _num_witnesses(is_fraud),
                "source":              "synthetic-v1",
            }
            records.append(record)
            claim_n += 1

    validate_records(records, "claim.schema.json")
    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    records = generate(count, config, states_data)
    fraud_count = sum(1 for r in records if r["is_fraud"])
    fraud_pct   = fraud_count / max(len(records), 1) * 100
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(
        f"[claim_gen] wrote {len(records):,} records → {output_path} "
        f"({fraud_count} fraud, {fraud_pct:.1f}%)"
    )


if __name__ == "__main__":
    config_dir  = Path(__file__).parent.parent / "config"
    states_data    = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        count=1000,
        output_path=Path("data/claims.json"),
        config={"coverage_rules": coverage_rules, "fraud_rate": 0.04},
        states_data=states_data,
    )
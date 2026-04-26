"""
iso_gen.py — generates synthetic cross-carrier claim history (ISO ClaimSearch simulation).

Usage:
    uv run python data_gen/generators/iso_gen.py
    from iso_gen import generate, main

Design decisions:
    - Fraud-tagged customers get 2-4 prior cross-carrier claims; clean get 0-1
    - fraud_indicator=True on 30-40% of fraud customer records, rarely on clean
    - VIN re-use: staged-accident fraud customers occasionally share a VIN with
      a different customer_id — simulates salvage/re-title fraud
    - role distribution: claimant is most common; witness/third_party are minority
    - prior_claim_date is always before the customer's earliest policy effective_date
      so ISO history predates AIOI relationship (realistic)
    - Carriers never include AIOI itself — this is cross-carrier history only
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from validate import validate_records

random.seed(45)

_CARRIERS = ["StateFarm", "Allstate", "Progressive", "Geico", "Farmers", "USAA", "Travelers", "Liberty"]

_CLAIM_TYPES = ["collision", "comprehensive", "liability", "pip", "uninsured_motorist"]

_AMOUNT_RANGES = {
    "collision":           (1500, 18000),
    "comprehensive":       (800,  12000),
    "liability":           (3000, 35000),
    "pip":                 (1000, 20000),
    "uninsured_motorist":  (5000, 40000),
}


def _prior_claim_date(earliest_policy_date: date) -> date:
    """
    Return a date before the customer's earliest AIOI policy.
    Spread across 1-5 years prior — realistic cross-carrier history window.
    """
    days_back = random.randint(30, 5 * 365)
    candidate = earliest_policy_date - timedelta(days=days_back)
    # Floor at 2015-01-01 — no need for ancient history
    floor = date(2015, 1, 1)
    return max(candidate, floor)


def _claim_amount(claim_type: str, fraud_indicator: bool) -> float:
    lo, hi = _AMOUNT_RANGES.get(claim_type, (1000, 10000))
    if fraud_indicator:
        lo = int(lo * 1.2)
        hi = int(hi * 1.6)
    mid = (lo + hi) / 2
    std = (hi - lo) / 6
    amount = random.gauss(mid, std)
    return round(max(lo, min(hi, amount)), 2)


def _role(is_fraud_customer: bool) -> str:
    """
    Fraud customers appear more often as claimants.
    Witnesses and third_party roles add network link signal.
    """
    if is_fraud_customer:
        return random.choices(
            ["claimant", "witness", "third_party"],
            weights=[70, 20, 10]
        )[0]
    return random.choices(
        ["claimant", "witness", "third_party"],
        weights=[60, 25, 15]
    )[0]


def generate(
    customers: list[dict],
    policies: list[dict],
    claims: list[dict],
) -> list[dict]:
    # Index fraud customers — still needed for fraud_indicator rate and VIN re-use
    fraud_customer_ids: set[str] = {
        c["customer_id"] for c in claims if c["is_fraud"]
    }

    # Count total claims per customer — drives ISO volume, not fraud label
    customer_claim_count: dict[str, int] = {}
    for c in claims:
        cid = c["customer_id"]
        customer_claim_count[cid] = customer_claim_count.get(cid, 0) + 1

    # Index earliest policy effective_date per customer
    earliest_policy: dict[str, date] = {}
    for p in policies:
        cid = p["customer_id"]
        eff = date.fromisoformat(p["effective_date"])
        if cid not in earliest_policy or eff < earliest_policy[cid]:
            earliest_policy[cid] = eff

    # Index VINs per customer for VIN re-use injection
    customer_vins: dict[str, str] = {
        p["customer_id"]: p["vehicle"]["vin"]
        for p in policies
    }

    # Build a small pool of VINs owned by fraud customers for cross-customer re-use
    fraud_vins: list[tuple[str, str]] = [
        (cid, vin)
        for cid, vin in customer_vins.items()
        if cid in fraud_customer_ids
    ]

    records: list[dict] = []
    iso_n = 1

    for customer in customers:
        cid = customer["customer_id"]
        is_fraud = cid in fraud_customer_ids
        eff_date = earliest_policy.get(cid, date.today() - timedelta(days=365))
        vin = customer_vins.get(cid, "00000000000000000")
        n_claims = customer_claim_count.get(cid, 0)

        # ISO volume is driven by claim activity, not fraud label.
        # Customers with no AIOI claims still have a 20% chance of cross-carrier
        # history (they existed before AIOI). Each AIOI claim adds a ~40% chance
        # of a corresponding cross-carrier entry — realistic cross-carrier coverage.
        if n_claims == 0:
            n_records = random.choices([0, 1], weights=[80, 20])[0]
        elif n_claims == 1:
            n_records = random.choices([0, 1, 2], weights=[45, 40, 15])[0]
        elif n_claims == 2:
            n_records = random.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]
        else:
            # 3+ AIOI claims — more likely to have cross-carrier history
            n_records = random.choices([1, 2, 3, 4], weights=[25, 40, 25, 10])[0]

        for _ in range(n_records):
            # fraud_indicator rate: elevated for fraud customers, rare for clean.
            # This is the legitimate signal — prior fraud flags, not prior claim count.
            fraud_indicator = (
                random.random() < 0.35 if is_fraud else random.random() < 0.02
            )
            claim_type = random.choice(_CLAIM_TYPES)

            records.append({
                "iso_id":             f"ISO-{iso_n:06d}",
                "customer_id":        cid,
                "vin":                vin,
                "prior_carrier":      random.choice(_CARRIERS),
                "prior_claim_date":   _prior_claim_date(eff_date).isoformat(),
                "prior_claim_type":   claim_type,
                "prior_claim_amount": _claim_amount(claim_type, fraud_indicator),
                "role":               _role(is_fraud),
                "fraud_indicator":    fraud_indicator,
                "source":             "synthetic-iso-v1",
            })
            iso_n += 1

        # VIN re-use: unchanged — this is a legitimate fraud-specific signal
        if is_fraud and fraud_vins and random.random() < 0.25:
            other_cid, other_vin = random.choice(fraud_vins)
            if other_cid != cid:
                claim_type = random.choice(["collision", "liability"])
                records.append({
                    "iso_id":             f"ISO-{iso_n:06d}",
                    "customer_id":        cid,
                    "vin":                other_vin,
                    "prior_carrier":      random.choice(_CARRIERS),
                    "prior_claim_date":   _prior_claim_date(eff_date).isoformat(),
                    "prior_claim_type":   claim_type,
                    "prior_claim_amount": _claim_amount(claim_type, True),
                    "role":               "claimant",
                    "fraud_indicator":    True,
                    "source":             "synthetic-iso-v1",
                })
                iso_n += 1

    validate_records(records, "iso_claim_history.schema.json")
    return records


def main(output_path: Path, customers: list[dict], policies: list[dict], claims: list[dict]) -> None:
    records = generate(customers, policies, claims)
    fraud_flagged = sum(1 for r in records if r["fraud_indicator"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(
        f"[iso_gen] wrote {len(records):,} records → {output_path} "
        f"({fraud_flagged} fraud_indicator=True)"
    )


if __name__ == "__main__":
    data_dir = Path("data")
    customers = json.loads((data_dir / "customers.json").read_text())
    policies  = json.loads((data_dir / "policies.json").read_text())
    claims    = json.loads((data_dir / "claims.json").read_text())
    main(
        output_path=Path("data/iso_claim_history.json"),
        customers=customers,
        policies=policies,
        claims=claims,
    )
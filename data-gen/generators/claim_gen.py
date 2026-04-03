"""
Claims data generator for AIOI.

Generates synthetic insurance claims with fraud injection and
realistic adjuster narratives.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker


def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """
    Generate claim records.

    Args:
        count: Number of customers
        config: Configuration dict (fraud_rate, coverage_rules)
        states_data: State rules dictionary

    Returns:
        List of claim records
    """
    fake = Faker("en_US")
    Faker.seed(42)
    random.seed(42)

    fraud_rate = config.get("fraud_rate", 0.04)

    # Load policies
    data_dir = Path(__file__).parent.parent.parent / "data"
    try:
        with open(data_dir / "policies.json") as f:
            policies = json.load(f)
    except FileNotFoundError:
        return []

    claim_types = [
        "collision",
        "comprehensive",
        "liability",
        "pip",
        "uninsured_motorist",
    ]

    # Fraud signal types
    fraud_signals_list = [
        ["claim_delta_high"],
        ["frequency_spike"],
        ["telematics_anomaly"],
        ["rapid_refiling"],
        ["claim_delta_high", "frequency_spike"],
        ["claim_delta_high", "telematics_anomaly"],
    ]

    records = []
    claim_counter = 0

    # 30% of policies get at least 1 claim
    policies_with_claims = random.sample(policies, int(len(policies) * 0.30))

    for policy in policies_with_claims:
        # 1-3 claims per policy
        num_claims = random.randint(1, 3)

        for _ in range(num_claims):
            claim_counter += 1
            claim_id = f"CLM-{claim_counter:05d}"
            policy_number = policy["policy_number"]
            customer_id = policy["customer_id"]
            state = policy["state"]

            # Claim type
            claim_type = random.choice(claim_types)

            # Is fraud?
            is_fraud = random.random() < fraud_rate
            fraud_signals = random.choice(fraud_signals_list) if is_fraud else []

            # Dates
            days_back = random.randint(10, 180)
            incident_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
            filed_date = (
                (datetime.fromisoformat(incident_date) + timedelta(days=random.randint(0, 5)))
                .date()
                .isoformat()
            )

            # Claim amount (fraud claims tend to be higher)
            if is_fraud:
                claim_amount = random.randint(15000, 50000)
            else:
                claim_amount = random.randint(1000, 15000)

            # Settlement amount (80% of claims are approved/settled)
            if random.random() < 0.80:
                settlement_amount = claim_amount * random.uniform(0.7, 1.0)
                status = random.choice(["approved", "settled"])
            else:
                settlement_amount = None
                status = random.choice(["denied", "under_review"])

            # Narratives
            adjuster_notes = fake.sentence(nb_words=10)
            incident_narrative = fake.paragraph(nb_sentences=3)

            record = {
                "claim_id": claim_id,
                "policy_number": policy_number,
                "customer_id": customer_id,
                "state": state,
                "incident_date": incident_date,
                "filed_date": filed_date,
                "claim_type": claim_type,
                "status": status,
                "claim_amount": claim_amount,
                "settlement_amount": round(settlement_amount, 2) if settlement_amount else None,
                "adjuster_notes": adjuster_notes,
                "incident_narrative": incident_narrative,
                "is_fraud": is_fraud,
                "fraud_signals": fraud_signals,
                "source": "synthetic-v1",
            }
            records.append(record)

    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    """
    Generate and write claim records.

    Args:
        count: Number of customers
        output_path: Path to write JSON file
        config: Configuration dictionary
        states_data: State rules dictionary
    """
    records = generate(count, config, states_data)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"[claim.json] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    config = {"fraud_rate": 0.04}

    data_dir = Path(__file__).parent.parent.parent / "data"
    main(100, data_dir / "claims.json", config, states_data)

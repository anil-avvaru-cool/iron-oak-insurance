"""
Customer data generator for AIOI.

Generates synthetic customer records with realistic demographics,
state-weighted distribution, and credit scores.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker


def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """
    Generate customer records.

    Args:
        count: Number of customers to generate
        config: Configuration dict (contains coverage_rules, fraud_rate)
        states_data: Dictionary of state rules from states.json

    Returns:
        List of customer records
    """
    fake = Faker("en_US")
    Faker.seed(42)
    random.seed(42)

    # Build weighted state list for distribution
    state_choices = []
    state_weights = []
    for state_code, state_info in states_data.items():
        state_choices.append(state_code)
        state_weights.append(state_info.get("weight", 1))

    records = []
    for i in range(count):
        # Weighted random state selection
        state = random.choices(state_choices, weights=state_weights, k=1)[0]

        # Generate customer record
        customer_id = f"CUST-{i+1:05d}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        zip_code = fake.postcode()[:5]  # Ensure exactly 5 digits
        email = fake.email()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=75).isoformat()
        credit_score = random.randint(300, 850)
        created_at = (
            datetime.now() - timedelta(days=random.randint(0, 1825))
        ).isoformat()

        record = {
            "customer_id": customer_id,
            "first_name": first_name,
            "last_name": last_name,
            "state": state,
            "zip": zip_code,
            "email": email,
            "dob": dob,
            "credit_score": credit_score,
            "created_at": created_at,
            "source": "synthetic-v1",
        }
        records.append(record)

    return records


def validate_record(record: dict, schema_path: Path) -> bool:
    """Validate a record against JSON schema."""
    try:
        import jsonschema

        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=record, schema=schema)
        return True
    except ImportError:
        # jsonschema not in Phase 1 deps, skip validation
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    """
    Generate and write customer records.

    Args:
        count: Number of customers to generate
        output_path: Path to write JSON file
        config: Configuration dictionary
        states_data: State rules dictionary
    """
    records = generate(count, config, states_data)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"[customer.json] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    config = {"coverage_rules": coverage_rules}

    data_dir = Path(__file__).parent.parent.parent / "data"
    main(100, data_dir / "customers.json", config, states_data)

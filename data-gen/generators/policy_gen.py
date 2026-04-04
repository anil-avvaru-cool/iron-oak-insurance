"""
Policy data generator for AIOI.

Generates synthetic auto insurance policies linked to customers,
respecting state-specific coverage requirements.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """
    Generate policy records.

    Args:
        count: Number of customers (policies will be >= count due to multi-policy holders)
        config: Configuration dict (coverage_rules)
        states_data: Dictionary of state rules

    Returns:
        List of policy records
    """
    random.seed(42)
    coverage_rules = config.get("coverage_rules", {})

    # Load customer data to link policies
    customers_path = (
        Path(__file__).parent.parent / "generators" / ".." / ".." / "data" / "customers.json"
    )
    try:
        with open(customers_path) as f:
            customers = json.load(f)
    except FileNotFoundError:
        # Fallback: create minimal customer references
        customers = [{"customer_id": f"CUST-{i + 1:05d}", "state": "CA"} for i in range(count)]

    coverage_types = coverage_rules.get("coverage_types", [])
    liability_limits = coverage_rules.get("liability_limits", ["15/30/5"])
    deductible_options = coverage_rules.get("deductible_options", [250, 500, 1000])
    drive_score_tiers = coverage_rules.get("drive_score_discount_tiers", [])

    records = []
    policy_counter = 0

    for customer in customers[:count]:
        customer_id = customer["customer_id"]
        state = customer.get("state", "CA")
        state_rules = states_data.get(state, {})

        # 1-3 policies per customer (15% get 2 policies)
        num_policies = 2 if random.random() < 0.15 else 1

        for _ in range(num_policies):
            policy_counter += 1
            policy_number = f"{state[:2]}-{policy_counter:05d}"

            # Build coverages object respecting state rules and schema contract
            coverages = {}
            for cov_type in coverage_types:
                if cov_type == "pip" and state_rules.get("pip_required"):
                    coverages[cov_type] = {
                        "included": True,
                        "required": True,
                        "deductible": None,
                        "limit": state_rules.get("pip_limit", 5000),
                        "pip_limit": state_rules.get("pip_limit", 5000),
                    }
                elif cov_type == "uninsured_motorist" and state_rules.get(
                    "uninsured_motorist_required"
                ):
                    coverages[cov_type] = {
                        "included": True,
                        "required": True,
                        "deductible": None,
                        "limit": "25/50/10",
                        "pip_limit": None,
                    }
                else:
                    is_core = cov_type in ["liability", "collision", "comprehensive"]
                    included = True if is_core else random.choice([True, False])
                    coverages[cov_type] = {
                        "included": included,
                        "required": False,
                        "deductible": random.choice(deductible_options)
                        if cov_type in ["collision", "comprehensive", "gap", "roadside"]
                        and included
                        else None,
                        "limit": None,
                        "pip_limit": None,
                    }

            # Liability limits and required flags
            liability_limit = random.choice(liability_limits)
            coverages["liability"] = {
                "included": True,
                "required": False,
                "deductible": None,
                "limit": liability_limit,
                "pip_limit": None,
            }

            # Drive score
            drive_score = random.randint(0, 100)

            # Premium calculation (simplified)
            base_premium = 1200
            drive_discount = 0
            for tier in drive_score_tiers:
                if drive_score >= tier["min"]:
                    drive_discount = tier["discount_pct"]
                    break
            premium_annual = base_premium * (1 - drive_discount)

            # Status distribution (5% lapsed, 3% cancelled, rest active or pending)
            status_choice = random.random()
            if status_choice < 0.05:
                status = "lapsed"
            elif status_choice < 0.08:
                status = "cancelled"
            elif status_choice < 0.15:
                status = "pending_renewal"
            else:
                status = "active"

            # Policy effective date
            days_back = random.randint(0, 730)
            effective_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
            expiry_date = (
                (datetime.fromisoformat(effective_date) + timedelta(days=365)).date().isoformat()
            )

            # Vehicle info
            vehicle = {
                "year": random.randint(2015, 2024),
                "make": random.choice(["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Tesla"]),
                "model": random.choice(
                    ["Camry", "Civic", "F-150", "Malibu", "3 Series", "Model 3"]
                ),
                "vin": "".join([str(random.randint(0, 9)) for _ in range(17)]),
            }

            record = {
                "policy_number": policy_number,
                "customer_id": customer_id,
                "state": state,
                "effective_date": effective_date,
                "expiry_date": expiry_date,
                "status": status,
                "coverages": coverages,
                "vehicle": vehicle,
                "premium_annual": round(premium_annual, 2),
                "drive_score": drive_score,
                "agent_id": f"AGENT-{random.randint(1000, 9999)}",
                "source": "synthetic-v1",
            }
            records.append(record)

    return records


def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    """
    Generate and write policy records.

    Args:
        count: Number of customers to generate policies for
        output_path: Path to write JSON file
        config: Configuration dictionary
        states_data: State rules dictionary
    """
    records = generate(count, config, states_data)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"[policy.json] wrote {len(records):,} records → {output_path}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    config = {"coverage_rules": coverage_rules}

    data_dir = Path(__file__).parent.parent.parent / "data"
    main(100, data_dir / "policies.json", config, states_data)

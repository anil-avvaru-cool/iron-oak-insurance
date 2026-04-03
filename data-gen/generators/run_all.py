"""
run_all.py — Single entry point for all AIOI data generation.

Usage:
  uv run python data-gen/generators/run_all.py
  uv run python data-gen/generators/run_all.py --customers 500 --fraud-rate 0.05
  uv run python data-gen/generators/run_all.py --pdf-docs 2000
"""

import argparse
import json
from pathlib import Path


def main():
    """Orchestrate all data generators."""
    parser = argparse.ArgumentParser(
        description="AIOI synthetic data generation for all entities"
    )
    parser.add_argument(
        "--customers", type=int, default=1000, help="Number of customers to generate"
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.04,
        help="Fraud rate for claims (0.0-1.0)",
    )
    parser.add_argument(
        "--trips-per-policy",
        type=int,
        default=50,
        help="Average trips per policy",
    )
    parser.add_argument(
        "--pdf-docs",
        type=int,
        default=1000,
        help="Total PDFs to generate (declaration + claim + renewal combined)",
    )
    parser.add_argument(
        "--state-focus",
        type=str,
        default=None,
        help="Comma-separated state codes to oversample, e.g., TX,PA",
    )

    args = parser.parse_args()

    # Load config files
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    config = {"fraud_rate": args.fraud_rate, "coverage_rules": coverage_rules}

    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("AIOI SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print(
        f"Customers: {args.customers:,} | Fraud Rate: {args.fraud_rate:.1%} | "
        f"PDFs: {args.pdf_docs:,}"
    )
    print("=" * 70 + "\n")

    # Import generators in dependency order
    from customer_gen import main as gen_customers
    from policy_gen import main as gen_policies
    from claim_gen import main as gen_claims
    from telematics_gen import main as gen_telematics
    from document_gen import main as gen_documents

    # 1. Generate customers (independent)
    print("[1/5] Generating customers...")
    gen_customers(args.customers, data_dir / "customers.json", config, states_data)

    # 2. Generate policies (depends on customers)
    print("[2/5] Generating policies...")
    gen_policies(args.customers, data_dir / "policies.json", config, states_data)

    # 3. Generate claims (depends on policies)
    print("[3/5] Generating claims...")
    gen_claims(args.customers, data_dir / "claims.json", config, states_data)

    # 4. Generate telematics (depends on policies)
    print("[4/5] Generating telematics...")
    total_trips = args.customers * args.trips_per_policy
    gen_telematics(total_trips, data_dir / "telematics.json", config, states_data)

    # 5. Generate documents (depends on policies + claims)
    print("[5/5] Generating documents...")
    docs_dir = Path(__file__).parent.parent.parent / "documents"
    gen_documents(args.pdf_docs, docs_dir, config, states_data)

    # Summary
    print("\n" + "=" * 70)
    print("✓ ALL DATA GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  customers:  {data_dir / 'customers.json'}")
    print(f"  policies:   {data_dir / 'policies.json'}")
    print(f"  claims:     {data_dir / 'claims.json'}")
    print(f"  telematics: {data_dir / 'telematics.json'}")
    print(f"  documents:  {docs_dir} (PDFs)")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  - Verify data in data/ directory")
    print("  - Run Phase 2: Load into PostgreSQL (see PHASE_2_DATABASE.md)")
    print()


if __name__ == "__main__":
    main()

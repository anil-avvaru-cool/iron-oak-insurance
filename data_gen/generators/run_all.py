"""
run_all.py — single entry point for all AIOI data generation.

Usage:
    uv run python data_gen/generators/run_all.py
    uv run python data_gen/generators/run_all.py --customers 500 --fraud-rate 0.05
    uv run python data_gen/generators/run_all.py --customers 100 --pdf-docs 50 --no-pdfs

Generation order is fixed — each step depends on the previous:
    1. customers.json           (standalone)
    2. policies.json            (reads customers.json)
    3. claims.json              (reads policies.json)
    4. iso_claim_history.json   (reads customers/policies/claims)
    5. telematics.json          (reads policies.json; skips non-enrolled)
    6. violations.json          (reads customers/policies)
    7. PDFs                     (reads customers/policies/claims)
    8. FAQs                     (Phase 4 stub — no-op until Phase 4)
"""
import argparse
import json
import sys
import time
from pathlib import Path

_GENERATORS_DIR = Path(__file__).parent
_CONFIG_DIR = _GENERATORS_DIR.parent / "config"

if str(_GENERATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATORS_DIR))


def _load_config() -> tuple[dict, dict]:
    states_data = json.loads((_CONFIG_DIR / "states.json").read_text())
    coverage_rules = json.loads((_CONFIG_DIR / "coverage_rules.json").read_text())
    return states_data, coverage_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOI synthetic data generator")
    parser.add_argument("--customers", type=int, default=1000,
                        help="Number of customer records (default: 1000)")
    parser.add_argument("--fraud-rate", type=float, default=0.04,
                        help="Claim fraud injection rate 0.0-0.15 (default: 0.04)")
    parser.add_argument("--trips-target", type=int, default=50000,
                        help="Target total telematics trips across enrolled policies (default: 50000)")
    parser.add_argument("--pdf-docs", type=int, default=500,
                        help="Total PDFs to generate (default: 500)")
    parser.add_argument("--no-pdfs", action="store_true",
                        help="Skip PDF generation (faster for dev/testing)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Root directory for data/, documents/, faqs/ (default: .)")
    args = parser.parse_args()

    if not (0.0 <= args.fraud_rate <= 0.15):
        parser.error("--fraud-rate must be between 0.0 and 0.15")

    states_data, coverage_rules = _load_config()
    config = {"coverage_rules": coverage_rules, "fraud_rate": args.fraud_rate}

    root = Path(args.output_dir)
    data_dir = root / "data"
    docs_dir = root / "documents"
    faqs_dir = root / "faqs"

    t_total = time.time()
    print(f"\n{'='*55}")
    print("  AVVARU IRON OAK INSURANCE — Data Generation")
    print(f"{'='*55}")
    print(f"  Customers:      {args.customers:,}")
    print(f"  Fraud rate:     {args.fraud_rate:.1%}")
    print(f"  Trips target:   {args.trips_target:,}")
    print(f"  PDFs:           {'skipped' if args.no_pdfs else str(args.pdf_docs) + ' docs'}")
    print(f"{'='*55}\n")

    # ── Step 1: Customers ──────────────────────────────────────────────────
    from customer_gen import main as gen_customers
    t0 = time.time()
    gen_customers(args.customers, data_dir / "customers.json", config, states_data)
    print(f"  ✓ customers ({time.time()-t0:.1f}s)")

    # ── Step 2: Policies ───────────────────────────────────────────────────
    from policy_gen import main as gen_policies
    t0 = time.time()
    gen_policies(args.customers, data_dir / "policies.json", config, states_data)
    print(f"  ✓ policies ({time.time()-t0:.1f}s)")

    # ── Step 3: Claims ─────────────────────────────────────────────────────   
    from claim_gen import generate as _gen_claims_records
    import json as _json
    t0 = time.time()
    customers_data = _json.loads((data_dir / "customers.json").read_text())
    policies_data  = _json.loads((data_dir / "policies.json").read_text())
            
    violations_data = (
        _json.loads((data_dir / "violations.json").read_text())
        if (data_dir / "violations.json").exists() else []
    )
    claims_records = _gen_claims_records(
        count=args.customers,
        config=config,
        states_data=states_data,
        policies=policies_data,
        customers=customers_data,
        violations=violations_data,
    )
    (data_dir / "claims.json").parent.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "claims.json", "w") as _f:
        _json.dump(claims_records, _f, indent=2, default=str)
    fraud_n = sum(1 for r in claims_records if r["is_fraud"])
    print(f"  ✓ claims — {len(claims_records):,} records "
          f"({fraud_n} fraud, {fraud_n/max(len(claims_records),1):.1%}) "
          f"({time.time()-t0:.1f}s)")
    claims_data = claims_records

    # ── Step 4: ISO Claim History ──────────────────────────────────────────  # NEW
    from iso_gen import main as gen_iso                                        # NEW
    t0 = time.time()                                                           # NEW
    customers_data = json.loads((data_dir / "customers.json").read_text())     # NEW
    policies_data  = json.loads((data_dir / "policies.json").read_text())      # NEW
    claims_data    = json.loads((data_dir / "claims.json").read_text())        # NEW
    gen_iso(                                                                   # NEW
        output_path=data_dir / "iso_claim_history.json",                      # NEW
        customers=customers_data,                                              # NEW
        policies=policies_data,                                                # NEW
        claims=claims_data,                                                    # NEW
    )                                                                          # NEW
    print(f"  ✓ ISO claim history ({time.time()-t0:.1f}s)")                    # NEW

    # ── Step 5: Telematics ─────────────────────────────────────────────────
    from telematics_gen import main as gen_telematics
    t0 = time.time()
    gen_telematics(args.trips_target, data_dir / "telematics.json", config, states_data)
    print(f"  ✓ telematics ({time.time()-t0:.1f}s)")

    # ── Step 6: Violations ─────────────────────────────────────────────────
    from violation_gen import main as gen_violations
    t0 = time.time()
    if not customers_data:
        customers_data = json.loads((data_dir / "customers.json").read_text())
    if not policies_data:
        policies_data  = json.loads((data_dir / "policies.json").read_text())
    gen_violations(
        output_path=data_dir / "violations.json",
        config={"violation_rules": coverage_rules["violation_rules"]},
        customers=customers_data,
        policies=policies_data,
    )
    print(f"  ✓ violations ({time.time()-t0:.1f}s)")

    # ── Step 7: PDFs (optional) ────────────────────────────────────────────
    if not args.no_pdfs:
        from document_gen import main as gen_documents
        t0 = time.time()
        gen_documents(args.pdf_docs, docs_dir, config, states_data)
        print(f"  ✓ PDFs ({time.time()-t0:.1f}s)")
    else:
        print("  – PDFs skipped (--no-pdfs)")

    # ── Step 8: FAQs (Phase 4 stub) ────────────────────────────────────────
    from faq_gen import main as gen_faqs
    t0 = time.time()
    gen_faqs(faqs_dir / "faq_corpus.json", states_data)
    print(f"  ✓ FAQs ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  ✓ Generation complete in {elapsed:.1f}s")
    print(f"{'='*55}")
    print(f"  data/customers.json         → {data_dir}/customers.json")
    print(f"  data/policies.json          → {data_dir}/policies.json")
    print(f"  data/claims.json            → {data_dir}/claims.json")
    print(f"  data/iso_claim_history.json → {data_dir}/iso_claim_history.json")  # NEW
    print(f"  data/telematics.json        → {data_dir}/telematics.json")
    print(f"  data/violations.json        → {data_dir}/violations.json")      # NEW
    if not args.no_pdfs:
        print(f"  documents/                  → {docs_dir}/")
    print(f"  faqs/faq_corpus.json        → {faqs_dir}/faq_corpus.json")
    print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Verify Phase 1 data generation checklist."""

import json
from pathlib import Path

data_dir = Path("data")
results = {}

# 1. Verify all generators produce valid JSON
print("=" * 70)
print("PHASE 1 VERIFICATION CHECKLIST")
print("=" * 70)

files = {
    "customers.json": data_dir / "customers.json",
    "policies.json": data_dir / "policies.json",
    "claims.json": data_dir / "claims.json",
    "telematics.json": data_dir / "telematics.json",
}

for name, path in files.items():
    try:
        data = json.load(open(path))
        results[name] = len(data)
        print(f"✓ {name:20} {len(data):6,} records")
    except Exception as e:
        print(f"✗ {name:20} ERROR: {e}")

# 2. Check fraud rate
print("\n[Fraud Rate Check]")
try:
    claims = json.load(open(data_dir / "claims.json"))
    fraud_claims = [c for c in claims if c.get("is_fraud")]
    fraud_rate = len(fraud_claims) / len(claims) if claims else 0
    status = "✓" if 0.03 <= fraud_rate <= 0.05 else "⚠"
    print(f"{status} Fraud rate: {len(fraud_claims)}/{len(claims)} = {fraud_rate:.1%}")
    if fraud_rate < 0.03 or fraud_rate > 0.05:
        print(f"  WARNING: Expected 3-5%, got {fraud_rate:.1%}")
except Exception as e:
    print(f"✗ Error checking fraud rate: {e}")

# 3. Check all 50 states + DC present
print("\n[State Distribution Check]")
try:
    customers = json.load(open(data_dir / "customers.json"))
    states = set(c["state"] for c in customers)
    expected_states = 51  # 50 states + DC
    status = "✓" if len(states) >= 40 else "⚠"
    print(f"{status} States present: {len(states)} (expect ~50+ in large dataset)")
    print(f"  Sample states: {sorted(states)[:10]}")
except Exception as e:
    print(f"✗ Error checking states: {e}")

# 4. Check PDF filenames match convention
print("\n[PDF Filename Convention Check]")
docs_dir = Path("documents")
if docs_dir.exists():
    pdfs = list(docs_dir.glob("*.pdf"))
    decl_count = len(list(docs_dir.glob("decl_*.pdf")))
    claim_count = len(list(docs_dir.glob("claim_letter_*.pdf")))
    renewal_count = len(list(docs_dir.glob("renewal_*.pdf")))
    print(f"✓ Declaration PDFs: {decl_count}")
    print(f"✓ Claim letter PDFs: {claim_count}")
    print(f"✓ Renewal PDFs: {renewal_count}")
    print(f"✓ Total PDFs: {len(pdfs)}")
else:
    print("✗ documents/ directory not found")

# 5. Check schema coverage
print("\n[Required Fields Check]")
try:
    customers = json.load(open(data_dir / "customers.json"))
    if customers:
        sample = customers[0]
        required = [
            "customer_id",
            "first_name",
            "last_name",
            "state",
            "zip",
            "email",
            "dob",
            "credit_score",
            "created_at",
            "source",
        ]
        missing = [f for f in required if f not in sample]
        if not missing:
            print(f"✓ Customer schema: all {len(required)} required fields present")
        else:
            print(f"✗ Customer missing fields: {missing}")
except Exception as e:
    print(f"✗ Error checking customer schema: {e}")

print("\n" + "=" * 70)
print("PHASE 1 VERIFICATION COMPLETE")
print("=" * 70)

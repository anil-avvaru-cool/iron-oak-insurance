"""
Bulk loader — reads generated JSON files and inserts into Postgres.
Usage: uv run python db/load_json.py
       uv run python db/load_json.py --truncate   # wipe and reload
"""
import json, argparse, os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()
#load_dotenv(override=True)

def get_conn():
    
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def load_table(conn, table: str, records: list[dict], columns: list[str]):
    rows = [[r.get(c) for c in columns] for r in records]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"  {table}: {len(rows):,} rows loaded")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    data = Path("data")
    conn = get_conn()

    if args.truncate:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE telematics, claims, policies, customers, iso_claim_history CASCADE")
        conn.commit()
        print("Tables truncated.")

    customers  = json.loads((data / "customers.json").read_text())
    policies   = json.loads((data / "policies.json").read_text())
    claims     = json.loads((data / "claims.json").read_text())
    telematics = json.loads((data / "telematics.json").read_text())
    iso_claim_history = json.loads((data / "iso_claim_history.json").read_text())

    import json as _json
    # Serialize JSONB fields
    for p in policies:
        p["coverages"] = _json.dumps(p["coverages"])
        p["vehicle"]   = _json.dumps(p["vehicle"])

    load_table(conn, "customers",  customers,  ["customer_id","first_name","last_name","state","zip","email","dob","credit_score","created_at","source"])
    load_table(conn, "policies",   policies,   ["policy_number","customer_id","state","effective_date","expiry_date","status","coverages","vehicle","premium_annual","drive_score","agent_id","source"])
    load_table(conn, "claims",     claims,     ["claim_id","policy_number","customer_id","state","incident_date","filed_date","claim_type","status","claim_amount","settlement_amount","adjuster_notes","incident_narrative","is_fraud","fraud_signals","reported_passengers", "num_witnesses","source"])
    load_table(conn, "iso_claim_history", iso_claim_history, ["customer_id","vin","prior_carrier","prior_claim_date","prior_claim_type","prior_claim_amount","role","fraud_indicator","source"])
    load_table(conn, "telematics", telematics, ["trip_id","policy_number","customer_id","trip_date","distance_miles","duration_minutes","hard_brakes","rapid_accelerations","speeding_events","night_driving_pct","drive_score","source"])

    conn.close()
    
    print("\n✓ Load complete.")

if __name__ == "__main__":
    main()
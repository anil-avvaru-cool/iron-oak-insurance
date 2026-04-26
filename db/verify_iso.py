# Quick check — run after loading iso_claim_history from DB
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return val


def get_engine():
    url = (
        f"postgresql+psycopg2://{_require_env('DB_USER')}:{_require_env('DB_PASSWORD')}"
        f"@{_require_env('DB_HOST')}:{_require_env('DB_PORT')}/{_require_env('DB_NAME')}"
    )
    return create_engine(url)

engine = get_engine()
df = pd.read_sql(text("""
    SELECT 
        c.is_fraud,
        COUNT(h.id) / COUNT(DISTINCT c.claim_id)::float AS avg_iso_entries_per_claim
    FROM claims c
    JOIN customers cu ON cu.customer_id = c.customer_id
    LEFT JOIN iso_claim_history h ON h.customer_id = c.customer_id
    GROUP BY c.is_fraud
"""), engine)
print(df)
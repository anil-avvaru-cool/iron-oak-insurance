"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def fraud_features(engine=None) -> pd.DataFrame:
    """Returns one row per claim with fraud label and signal features."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            c.claim_id,
            c.is_fraud                                              AS label,
            c.claim_amount,
            c.claim_amount / NULLIF(p.premium_annual, 0)           AS claim_to_premium_ratio,
            (c.filed_date::date - c.incident_date::date) AS days_to_file,
            COUNT(c2.claim_id) OVER (PARTITION BY c.customer_id)   AS customer_claim_count,
            COALESCE(t.avg_drive_score, 50)                        AS avg_drive_score,
            COALESCE(t.hard_brakes_90d, 0)                         AS hard_brakes_90d,
            p.state,
            c.claim_type,
            ARRAY_LENGTH(c.fraud_signals, 1)                       AS fraud_signal_count
        FROM claims c
        JOIN policies p ON p.policy_number = c.policy_number
        LEFT JOIN (
            SELECT policy_number,
                   AVG(drive_score)   AS avg_drive_score,
                   SUM(hard_brakes)   AS hard_brakes_90d
            FROM telematics
            WHERE trip_date >= NOW() - INTERVAL '90 days'
            GROUP BY policy_number
        ) t ON t.policy_number = c.policy_number
        LEFT JOIN claims c2 ON c2.customer_id = c.customer_id
    """)
    return pd.read_sql(sql, engine)

def risk_features(engine=None) -> pd.DataFrame:
    """Returns one row per policy for risk scoring."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            p.policy_number,
            p.state,
            p.premium_annual,
            p.drive_score,
            cust.credit_score,
            (p.vehicle->>'year')::int                              AS vehicle_year,
            p.vehicle->>'make'                                     AS vehicle_make,
            COUNT(c.claim_id)                                      AS total_claims,
            COALESCE(SUM(c.claim_amount), 0)                       AS total_claim_amount,
            COALESCE(AVG(t.drive_score), 50)                       AS avg_drive_score_12m
        FROM policies p
        JOIN customers cust ON cust.customer_id = p.customer_id
        LEFT JOIN claims c ON c.policy_number = p.policy_number
        LEFT JOIN telematics t ON t.policy_number = p.policy_number
            AND t.trip_date >= NOW() - INTERVAL '365 days'
        GROUP BY p.policy_number, p.state, p.premium_annual,
                 p.drive_score, cust.credit_score,
                 p.vehicle->>'year', p.vehicle->>'make'
    """)
    return pd.read_sql(sql, engine)

def churn_features(engine=None) -> pd.DataFrame:
    """Returns one row per customer with churn label (lapsed/cancelled = 1)."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            cust.customer_id,
            (MAX(p.status) IN ('lapsed','cancelled'))::int         AS label,
            cust.credit_score,
            COUNT(p.policy_number)                                 AS policy_count,
            AVG(p.premium_annual)                                  AS avg_premium,
            COALESCE(AVG(p.drive_score), 50)                       AS avg_drive_score,
            COUNT(c.claim_id)                                      AS total_claims,
            MAX(CASE WHEN p.status = 'active' THEN 1 ELSE 0 END)  AS has_active_policy
        FROM customers cust
        LEFT JOIN policies p ON p.customer_id = cust.customer_id
        LEFT JOIN claims c ON c.customer_id = cust.customer_id
        GROUP BY cust.customer_id, cust.credit_score
    """)
    return pd.read_sql(sql, engine)
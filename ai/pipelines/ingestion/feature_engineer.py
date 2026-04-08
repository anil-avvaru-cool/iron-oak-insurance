"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
"""
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


def fraud_features(engine=None) -> pd.DataFrame:
    """Returns one row per claim with fraud label and signal features."""
    engine = engine or get_engine()
    sql = text("""
        SELECT
            c.claim_id,
            c.is_fraud                                              AS label,
            c.claim_amount,
            c.claim_amount / NULLIF(p.premium_annual, 0)           AS claim_to_premium_ratio,
            EXTRACT(DAY FROM c.filed_date::date - c.incident_date::date) AS days_to_file,
            COUNT(c2.claim_id) OVER (PARTITION BY c.customer_id)   AS customer_claim_count,
            COALESCE(t.avg_drive_score, 50)                        AS avg_drive_score,
            COALESCE(t.hard_brakes_90d, 0)                         AS hard_brakes_90d,
            p.state,
            p.vehicle->>'make'                                     AS vehicle_make,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            c.claim_type,
            COALESCE(ARRAY_LENGTH(c.fraud_signals, 1), 0)          AS fraud_signal_count
        FROM claims c
        JOIN policies p ON p.policy_number = c.policy_number
        JOIN customers cust ON cust.customer_id = c.customer_id
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
    """Returns one row per policy for risk scoring.

    Extra columns included for fairness audit (state, vehicle_make, zip_prefix)
    but excluded from model features via RISK_EXCLUDE in risk model.
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            p.policy_number,
            p.state,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            p.premium_annual,
            COALESCE(p.drive_score, 50)                            AS drive_score,
            COALESCE(cust.credit_score, 650)                       AS credit_score,
            COALESCE((p.vehicle->>'year')::int, 2015)              AS vehicle_year,
            p.vehicle->>'make'                                     AS vehicle_make,
            COUNT(c.claim_id)                                      AS total_claims,
            COALESCE(SUM(c.claim_amount), 0)                       AS total_claim_amount,
            COALESCE(AVG(t12.drive_score), 50)                     AS avg_drive_score_12m,
            COALESCE(AVG(t3.drive_score), 50)                      AS avg_drive_score_3m
        FROM policies p
        JOIN customers cust ON cust.customer_id = p.customer_id
        LEFT JOIN claims c ON c.policy_number = p.policy_number
        LEFT JOIN telematics t12 ON t12.policy_number = p.policy_number
            AND t12.trip_date >= NOW() - INTERVAL '365 days'
        LEFT JOIN telematics t3 ON t3.policy_number = p.policy_number
            AND t3.trip_date >= NOW() - INTERVAL '90 days'
        GROUP BY p.policy_number, p.state, cust.zip,
                 p.premium_annual, p.drive_score, cust.credit_score,
                 p.vehicle->>'year', p.vehicle->>'make'
    """)
    return pd.read_sql(sql, engine)


def churn_features(engine=None) -> pd.DataFrame:
    """Returns one row per customer with churn label (lapsed/cancelled = 1).

    Extra columns included for fairness audit (state, zip_prefix).
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            cust.customer_id,
            cust.state,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            (MAX(p.status) IN ('lapsed','cancelled'))::int         AS label,
            COALESCE(cust.credit_score, 650)                       AS credit_score,
            COUNT(p.policy_number)                                 AS policy_count,
            COALESCE(AVG(p.premium_annual), 0)                     AS avg_premium,
            COALESCE(AVG(p.drive_score), 50)                       AS avg_drive_score,
            COALESCE(AVG(t12.drive_score), 50)                     AS avg_drive_score_12m,
            COALESCE(AVG(t3.drive_score), 50)                      AS avg_drive_score_3m,
            -- Drive score delta: negative = deteriorating driving, churn signal
            COALESCE(AVG(t3.drive_score), 50)
              - COALESCE(AVG(t12.drive_score), 50)                 AS drive_score_delta,
            COUNT(c.claim_id)                                      AS total_claims,
            MAX(CASE WHEN p.status = 'active' THEN 1 ELSE 0 END)  AS has_active_policy
        FROM customers cust
        LEFT JOIN policies p ON p.customer_id = cust.customer_id
        LEFT JOIN claims c ON c.customer_id = cust.customer_id
        LEFT JOIN telematics t12 ON t12.policy_number = p.policy_number
            AND t12.trip_date >= NOW() - INTERVAL '365 days'
        LEFT JOIN telematics t3 ON t3.policy_number = p.policy_number
            AND t3.trip_date >= NOW() - INTERVAL '90 days'
        GROUP BY cust.customer_id, cust.state, cust.zip, cust.credit_score
    """)
    return pd.read_sql(sql, engine)
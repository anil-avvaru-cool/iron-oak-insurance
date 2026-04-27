"""
waterfall_params.py — extract build_waterfall() parameters from DB and model.

Called from model.py main() after train() completes so detection_rate is available.

Usage:
    from ai.models.fraud_detection.waterfall_params import extract_params
    params = extract_params(detection_rate=roc)
    build_waterfall(**params)
"""
from __future__ import annotations
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── Investigation cost: flat estimate (no DB column exists).
# Source: avg adjuster hours per flagged claim × hourly rate × expected flagged volume.
# Override via env var AIOI_INVESTIGATION_COST_ANNUAL if tracked in finance system.
DEFAULT_INVESTIGATION_COST = 85_000

# ── Model infrastructure cost: from strategy Section 7.2 Use Case 4.
# RAG Q&A ($59) + Fraud Agent ($162) + base ($62) per month × 12.
# Override via env var AIOI_MODEL_COST_ANNUAL.
DEFAULT_MODEL_COST = 45_000


def _get_conn():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def extract_params(detection_rate: float) -> dict:
    """
    Pull premium, legit_claims, total_fraud_loss from DB.
    Combine with detection_rate from the just-trained model.

    Args:
        detection_rate: returned directly by train() — use it as detection_rate proxy.
                 For a more conservative estimate pass detection_rate * 0.85.

    Returns:
        Dict ready to unpack into build_waterfall(**params).
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(p.premium_annual), 0)          AS premium,

                    COALESCE(SUM(
                        CASE
                            WHEN c.is_fraud = false
                             AND c.status   = 'settled'
                            THEN COALESCE(c.settlement_amount, 0)
                            ELSE 0
                        END
                    ), 0)                                       AS legit_claims,

                    COALESCE(SUM(
                        CASE
                            WHEN c.is_fraud = true
                             AND c.status   = 'settled'
                            THEN COALESCE(c.settlement_amount, 0)
                            ELSE 0
                        END
                    ), 0)                                       AS total_fraud_loss

                FROM policies p
                LEFT JOIN claims c ON c.policy_number = p.policy_number
                WHERE p.status = 'active'
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    premium, legit_claims, total_fraud_loss = (float(v) for v in row)

    return {
        "premium":            premium,
        "legit_claims":       legit_claims,
        "total_fraud_loss":   total_fraud_loss,
        "detection_rate":     detection_rate,
        "investigation_cost": float(
            os.environ.get("AIOI_INVESTIGATION_COST_ANNUAL", DEFAULT_INVESTIGATION_COST)
        ),
        "model_cost": float(
            os.environ.get("AIOI_MODEL_COST_ANNUAL", DEFAULT_MODEL_COST)
        ),
    }
"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

Fan-out fix (all three functions):
  All joins that could produce multiple rows per anchor entity (claim / policy /
  customer) are pre-aggregated into subqueries before joining. Raw multi-row
  joins against the same table in a single FROM clause cause cartesian
  multiplication before GROUP BY, inflating counts and averages silently.
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
    """Returns one row per claim with fraud label and signal features.

    Fan-out fix: replaced raw 'LEFT JOIN claims c2 ON c2.customer_id = ...'
    with a pre-aggregated subquery. The original join multiplied each claim row
    by the number of other claims for that customer before the window function
    ran, producing inflated customer_claim_count values.

    Telematics join was already pre-aggregated (safe — no change needed).
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            c.claim_id,
            c.is_fraud                                              AS label,
            c.claim_amount,
            c.claim_amount / NULLIF(p.premium_annual, 0)           AS claim_to_premium_ratio,
            c.filed_date::date - c.incident_date::date             AS days_to_file,
            COALESCE(cust_claims.claim_count, 0)                   AS customer_claim_count,
            COALESCE(t.avg_drive_score, 50)                        AS avg_drive_score,
            COALESCE(t.hard_brakes_90d, 0)                         AS hard_brakes_90d,
            p.state,
            p.vehicle->>'make'                                     AS vehicle_make,
            LEFT(cust.zip, 3)                                      AS zip_prefix,
            c.claim_type,
            COALESCE(ARRAY_LENGTH(c.fraud_signals, 1), 0)          AS fraud_signal_count
        FROM claims c
        JOIN policies p   ON p.policy_number  = c.policy_number
        JOIN customers cust ON cust.customer_id = c.customer_id

        -- Pre-aggregated telematics: safe, was already correct
        LEFT JOIN (
            SELECT
                policy_number,
                AVG(drive_score)  AS avg_drive_score,
                SUM(hard_brakes)  AS hard_brakes_90d
            FROM telematics
            WHERE trip_date >= NOW() - INTERVAL '90 days'
            GROUP BY policy_number
        ) t ON t.policy_number = c.policy_number

        -- Pre-aggregated claim count per customer.
        -- Previously: raw LEFT JOIN claims c2 ON c2.customer_id = c.customer_id
        -- caused each claim row to be multiplied by the number of sibling claims
        -- before the window COUNT ran. Fixed: aggregate first, then join.
        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(claim_id) AS claim_count
            FROM claims
            GROUP BY customer_id
        ) cust_claims ON cust_claims.customer_id = c.customer_id
    """)
    return pd.read_sql(sql, engine)



# Max claims and amount to aggregate per policy — prevents synthetic outliers
# from polluting the risk target. Tune these for production data volume.
_RISK_MAX_CLAIMS_PER_POLICY = 5
_RISK_CLAIM_WINDOW_DAYS     = 365   # only look at claims in the past 12 months

def risk_features(engine=None) -> pd.DataFrame:
    """Returns one row per policy for risk scoring.

    Target: premium_annual
      The generator builds premiums from drive score, credit score, vehicle age,
      state factors, and coverage breadth — a reasonable actuarial proxy for risk.
      The model learns to replicate that pricing function from pre-loss features.

    Why NOT annualized_loss_cost:
      Claims are outcomes of risk, not predictors. Using total_claims or
      total_claim_amount as inputs (even indirectly via the target) causes
      the model to learn a tautology: high-claim policies are high-risk.
      This is circular and useless for scoring new or renewal policies.

    Features:
      - drive_score / avg_drive_score_12m / avg_drive_score_3m  (telematics)
      - is_telematics_enrolled  (flag distinguishes real 50 from imputed 50)
      - drive_score_delta       (trend signal: improving vs. deteriorating)
      - credit_score
      - vehicle_year            (age proxy for replacement cost)
      - state                   (regulatory + loss environment)
      - coverage_count          (breadth of coverage elected)
      - has_collision, has_comprehensive, has_pip  (specific high-cost coverages)

    Audit-only columns (excluded from model features via RISK_EXCLUDE):
      policy_number, premium_annual, zip_prefix, vehicle_make
    """
    engine = engine or get_engine()
    sql = text("""
        SELECT
            p.policy_number,
            p.state,
            LEFT(cust.zip, 3)                                        AS zip_prefix,
            p.premium_annual,

            -- Telematics enrollment flag: distinguishes real score from imputed
            CASE WHEN p.drive_score IS NOT NULL THEN 1 ELSE 0 END    AS is_telematics_enrolled,

            -- Drive score: use actual if enrolled, neutral 50 if not
            COALESCE(p.drive_score, 50)                              AS drive_score,

            COALESCE(cust.credit_score, 650)                         AS credit_score,
            COALESCE((p.vehicle->>'year')::int, 2015)                AS vehicle_year,
            p.vehicle->>'make'                                       AS vehicle_make,

            -- Telematics trend signals (12m vs 3m averages)
            -- Non-enrolled policies get 50 (neutral) for both
            COALESCE(AVG(t12.drive_score), 50)                       AS avg_drive_score_12m,
            COALESCE(AVG(t3.drive_score), 50)                        AS avg_drive_score_3m,

            -- Coverage elections: high-cost coverages signal higher insured value
            (
                CASE WHEN (p.coverages->'collision'->>'included')::boolean     THEN 1 ELSE 0 END +
                CASE WHEN (p.coverages->'comprehensive'->>'included')::boolean THEN 1 ELSE 0 END +
                CASE WHEN (p.coverages->'pip'->>'included')::boolean           THEN 1 ELSE 0 END +
                CASE WHEN (p.coverages->'uninsured_motorist'->>'included')::boolean THEN 1 ELSE 0 END +
                CASE WHEN (p.coverages->'gap'->>'included')::boolean           THEN 1 ELSE 0 END +
                CASE WHEN (p.coverages->'roadside'->>'included')::boolean      THEN 1 ELSE 0 END
            )                                                        AS coverage_count,

            CASE WHEN (p.coverages->'collision'->>'included')::boolean     THEN 1 ELSE 0 END AS has_collision,
            CASE WHEN (p.coverages->'comprehensive'->>'included')::boolean THEN 1 ELSE 0 END AS has_comprehensive,
            CASE WHEN (p.coverages->'pip'->>'included')::boolean           THEN 1 ELSE 0 END AS has_pip

        FROM policies p
        JOIN customers cust ON cust.customer_id = p.customer_id

        LEFT JOIN telematics t12
            ON  t12.policy_number = p.policy_number
            AND t12.trip_date >= NOW() - INTERVAL '365 days'

        LEFT JOIN telematics t3
            ON  t3.policy_number = p.policy_number
            AND t3.trip_date >= NOW() - INTERVAL '90 days'

        GROUP BY p.policy_number, p.state, cust.zip,
                 p.premium_annual, p.drive_score, cust.credit_score,
                 p.vehicle->>'year', p.vehicle->>'make', p.coverages
    """)
    df = pd.read_sql(sql, engine)

    # Drive score delta: 3m minus 12m average.
    # Positive = improving, negative = deteriorating.
    # For non-enrolled policies both averages are 50, so delta is 0 — correct behavior.
    df["drive_score_delta"] = df["avg_drive_score_3m"] - df["avg_drive_score_12m"]

    return df

def churn_features(engine=None) -> pd.DataFrame:
    """Returns one row per customer with churn label and enriched features.

    Changes from original:
    - Label uses CASE WHEN MAX(status) to avoid alphabetic-sort bug
    - is_enrolled flag added as explicit model feature (not imputed away)
    - drive_score_delta uses policy-level drive_score vs. telematics avg
      so non-enrolled customers get a real signal instead of 0 vs. 0
    - avg_drive_score uses policy drive_score (set at enrollment),
      not telematics avg which is NULL for non-enrolled
    - telematics joins scoped to p.policy_number via a subquery to avoid
      fan-out row multiplication when a customer has multiple policies
      each with many telematics rows
    - days_since_last_claim added: recency of claims is a churn signal
    - tenure_days added: longer-tenured customers churn less
    - Extra columns for fairness audit: state, zip_prefix (excluded from
      model features via EXCLUDE_COLS in churn model)
    """
    engine = engine or get_engine()
    sql = text("""
        WITH policy_summary AS (
            -- One row per policy with telematics aggregates pre-joined.
            -- Aggregating here prevents fan-out when joining telematics
            -- to a customer who holds multiple policies.
            SELECT
                p.policy_number,
                p.customer_id,
                p.status,
                p.premium_annual,
                p.effective_date,
                p.drive_score                                          AS policy_drive_score,
                CASE WHEN p.drive_score IS NOT NULL THEN 1 ELSE 0 END  AS is_enrolled,

                -- Telematics 12-month average (NULL for non-enrolled)
                t12.avg_drive_score_12m,

                -- Telematics 3-month average (NULL for non-enrolled)
                t3.avg_drive_score_3m,

                -- Delta: policy baseline minus recent telematics avg.
                -- Non-enrolled: NULL (handled in Python imputation, not here).
                -- Negative = driving deteriorating since enrollment.
                CASE
                    WHEN p.drive_score IS NOT NULL
                         AND t12.avg_drive_score_12m IS NOT NULL
                    THEN t12.avg_drive_score_12m - p.drive_score
                    ELSE NULL
                END                                                    AS drive_score_delta

            FROM policies p

            LEFT JOIN (
                SELECT
                    policy_number,
                    AVG(drive_score) AS avg_drive_score_12m
                FROM telematics
                WHERE trip_date >= NOW() - INTERVAL '365 days'
                GROUP BY policy_number
            ) t12 ON t12.policy_number = p.policy_number

            LEFT JOIN (
                SELECT
                    policy_number,
                    AVG(drive_score) AS avg_drive_score_3m
                FROM telematics
                WHERE trip_date >= NOW() - INTERVAL '90 days'
                GROUP BY policy_number
            ) t3 ON t3.policy_number = p.policy_number
        ),

        claim_summary AS (
            -- One row per customer: claim count and days since last claim.
            -- Scoped to customer level so it joins cleanly without fan-out.
            SELECT
                customer_id,
                COUNT(claim_id)                                        AS total_claims,
                EXTRACT(
                    DAY FROM NOW() - MAX(filed_date::timestamptz)
                )::int                                                 AS days_since_last_claim
            FROM claims
            GROUP BY customer_id
        )

        SELECT
            cust.customer_id,

            -- Fairness audit columns (excluded from model features in EXCLUDE_COLS)
            cust.state,
            LEFT(cust.zip, 3)                                          AS zip_prefix,

            -- Label: 1 if any policy is lapsed or cancelled.
            -- Uses CASE WHEN MAX() to avoid alphabetic-sort bug with MAX(status).
            MAX(CASE WHEN ps.status IN ('lapsed', 'cancelled') THEN 1 ELSE 0 END)
                                                                       AS label,

            -- Customer-level features
            COALESCE(cust.credit_score, 650)                           AS credit_score,
            COUNT(DISTINCT ps.policy_number)                           AS policy_count,
            COALESCE(AVG(ps.premium_annual), 0)                        AS avg_premium,

            -- Tenure: days from earliest policy effective date to today.
            -- Longer-tenured customers churn less.
            COALESCE(
                EXTRACT(DAY FROM NOW() - MIN(ps.effective_date::timestamptz))::int,
                0
            )                                                          AS tenure_days,

            -- Enrollment: 1 if any policy is telematics-enrolled.
            MAX(ps.is_enrolled)                                        AS is_enrolled,

            -- Drive score: policy-level baseline for enrolled customers.
            -- NULL for non-enrolled; imputed in Python with sentinel -1.
            AVG(CASE WHEN ps.is_enrolled = 1 THEN ps.policy_drive_score ELSE NULL END)
                                                                       AS avg_drive_score,

            -- Telematics averages: NULL for non-enrolled; imputed in Python.
            AVG(ps.avg_drive_score_12m)                                AS avg_drive_score_12m,
            AVG(ps.avg_drive_score_3m)                                 AS avg_drive_score_3m,

            -- Drive score delta: avg across enrolled policies.
            -- NULL for fully non-enrolled customers; imputed in Python.
            AVG(ps.drive_score_delta)                                  AS drive_score_delta,

            -- Claim features
            COALESCE(cs.total_claims, 0)                               AS total_claims,
            cs.days_since_last_claim,

            -- Active policy flag
            MAX(CASE WHEN ps.status = 'active' THEN 1 ELSE 0 END)     AS has_active_policy

        FROM customers cust
        LEFT JOIN policy_summary ps  ON ps.customer_id = cust.customer_id
        LEFT JOIN claim_summary  cs  ON cs.customer_id = cust.customer_id
        GROUP BY
            cust.customer_id,
            cust.state,
            cust.zip,
            cust.credit_score,
            cs.total_claims,
            cs.days_since_last_claim
    """)
    return pd.read_sql(sql, engine)

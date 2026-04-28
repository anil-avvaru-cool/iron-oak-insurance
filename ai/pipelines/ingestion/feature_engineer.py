"""
feature_engineer.py — extract model-ready features from Postgres.

Returns DataFrames with a consistent column contract so models can be
retrained or swapped without touching the API layer.

Design changes (v2):
  - fraud_features(): replaced fraud_signal_count with 11 binary sig_* columns
    via LATERAL unnest pivot. The model now sees each signal individually.
  - days_to_file: kept but transformed to log1p to reduce leakage dominance.
    Raw value was 0.82 importance due to synthetic filing-lag artifact.
  - claim_to_premium_ratio: fixed — now uses monthly premium equivalent
    so ratios are meaningful across policies of different premium levels.
  - risk_features() and churn_features(): unchanged.

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
    """
    Returns one row per claim with fraud label and feature columns.

    Key design decisions:
      - fraud signals are unpivoted from TEXT[] into 11 binary columns (sig_*)
        via LATERAL unnest + BOOL_OR. This makes each signal an independent
        feature the model can learn from, rather than a useless count.
      - days_to_file is included but the model.py applies log1p transform
        to reduce its dominance from the synthetic filing-lag artifact.
      - claim_to_monthly_premium_ratio uses monthly premium (premium_annual/12)
        as denominator so a $10K claim on a $1200/yr policy reads as 10.0x
        monthly, not 0.83x annual — more intuitive and consistent scale.
      - customer_claim_count uses prior claims only (excludes self and future)
        so repeat filers score higher regardless of which claim is evaluated.
      - days_since_inception / near_inception / near_expiry: claims filed
        within 30 days of policy start or 10 days before expiry are a strong
        fraud timing signal.
      - late_reporting: binary flag for filing lag > 3 days. Kept separate
        from days_to_file_log so the model has both continuous and threshold
        representations of the same signal.
      - premium_to_credit_ratio: expensive policy relative to credit score
        proxies for vehicle-value-vs-income mismatch.
      - ISO features: cross-carrier claim history aggregated per customer.
        iso_prior_fraud_flag_count and iso_prior_carrier_count are expected
        to rank in top-5 SHAP importances.
      - iso_vin_claimant_count: number of distinct customers who have filed
        claims on the same VIN across carriers — VIN re-use / salvage signal.
    """
    engine = engine or get_engine()
    sql = text("""
    WITH base AS (
    SELECT
        c.claim_id,
        c.is_fraud,
        c.claim_amount,
        c.claim_amount / NULLIF(p.premium_annual / 12.0, 0) AS claim_to_monthly_premium_ratio,
        c.filed_date::date - c.incident_date::date           AS days_to_file,
        CASE WHEN (c.filed_date::date - c.incident_date::date) > 3
             THEN 1 ELSE 0 END                               AS late_reporting,
        EXTRACT(DOW FROM c.incident_date::date)              AS incident_day_of_week,

        -- Policy timing signals
        c.incident_date::date - p.effective_date             AS days_since_inception,
        p.expiry_date - c.incident_date::date                AS days_until_expiry,
        CASE WHEN (c.incident_date::date - p.effective_date) <= 30
             THEN 1 ELSE 0 END                               AS near_inception,
        CASE WHEN (p.expiry_date - c.incident_date::date) <= 10
             THEN 1 ELSE 0 END                               AS near_expiry,

        -- Vehicle value vs income proxy
        p.premium_annual / NULLIF(cust.credit_score, 0)      AS premium_to_credit_ratio,

        -- Prior claims (excludes self and future)
        (
            SELECT COUNT(*)
            FROM claims c2
            WHERE c2.customer_id = c.customer_id
              AND c2.claim_id   != c.claim_id
              AND c2.filed_date  < c.filed_date
        ) AS customer_claim_count,

        (
            SELECT COUNT(*)
            FROM claims c2
            WHERE c2.customer_id = c.customer_id
              AND c2.claim_id   != c.claim_id
              AND c2.filed_date  < c.filed_date
              AND c2.filed_date >= c.filed_date - INTERVAL '90 days'
        ) AS claims_last_90d,

        COALESCE(t.avg_drive_score, 50)       AS avg_drive_score,
        COALESCE(t.hard_brakes_90d, 0)        AS hard_brakes_90d,
        COALESCE(t.speeding_events_90d, 0)    AS speeding_events_90d,
        p.state,
        p.vehicle->>'make'                    AS vehicle_make,
        LEFT(cust.zip, 3)                     AS zip_prefix,
        c.claim_type,
        c.reported_passengers,
        c.num_witnesses,
        s.signal,

        -- ISO aggregate features
        COALESCE(iso.iso_prior_claim_count, 0)      AS iso_prior_claim_count,
        COALESCE(iso.iso_prior_carrier_count, 0)    AS iso_prior_carrier_count,
        COALESCE(iso.iso_prior_fraud_flag_count, 0) AS iso_prior_fraud_flag_count,
        COALESCE(
            c.incident_date::date - iso.iso_last_claim_date, 999
        )                                           AS iso_days_since_last_claim,

        -- VIN re-use signal
        COALESCE(vin_iso.iso_vin_claimant_count, 1) AS iso_vin_claimant_count

    FROM claims c
    JOIN policies p     ON p.policy_number  = c.policy_number
    JOIN customers cust ON cust.customer_id = c.customer_id
    LEFT JOIN (
        SELECT policy_number,
               AVG(drive_score)       AS avg_drive_score,
               SUM(hard_brakes)       AS hard_brakes_90d,
               SUM(speeding_events)   AS speeding_events_90d
        FROM telematics
        WHERE trip_date >= NOW() - INTERVAL '90 days'
        GROUP BY policy_number
    ) t ON t.policy_number = c.policy_number
    LEFT JOIN LATERAL (
        SELECT
            LEAST(COUNT(*), 5)                    AS iso_prior_claim_count,
            COUNT(DISTINCT prior_carrier)         AS iso_prior_carrier_count,
            LEAST(SUM(fraud_indicator::int), 3)   AS iso_prior_fraud_flag_count,
            MAX(prior_claim_date)                 AS iso_last_claim_date
        FROM iso_claim_history h
        WHERE h.customer_id = c.customer_id
          AND h.prior_claim_date < c.filed_date
    ) iso ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            LEAST(COUNT(DISTINCT customer_id), 5) AS iso_vin_claimant_count
        FROM iso_claim_history h
        WHERE h.vin = p.vehicle->>'vin'
          AND h.prior_claim_date < c.filed_date
    ) vin_iso ON TRUE
    LEFT JOIN LATERAL unnest(c.fraud_signals) AS s(signal) ON TRUE
    )
    SELECT
        claim_id,
        is_fraud AS label,
        claim_amount,
        claim_to_monthly_premium_ratio,
        days_to_file,
        late_reporting,
        incident_day_of_week,
        days_since_inception,
        days_until_expiry,
        near_inception,
        near_expiry,
        premium_to_credit_ratio,
        customer_claim_count,
        claims_last_90d,
        avg_drive_score,
        hard_brakes_90d,
        speeding_events_90d,
        state,
        vehicle_make,
        zip_prefix,
        claim_type,
        reported_passengers,
        num_witnesses,
        iso_prior_claim_count,
        iso_prior_carrier_count,
        iso_prior_fraud_flag_count,
        iso_days_since_last_claim,
        iso_vin_claimant_count,
        BOOL_OR(signal = 'claim_delta_high')                      AS sig_claim_delta_high,
        BOOL_OR(signal = 'telematics_anomaly')                    AS sig_telematics_anomaly,
        BOOL_OR(signal = 'staged_accident_pattern')               AS sig_staged_accident,
        BOOL_OR(signal = 'frequency_spike')                       AS sig_frequency_spike,
        BOOL_OR(signal = 'incident_location_mismatch')            AS sig_location_mismatch,
        BOOL_OR(signal = 'multiple_claimants')                    AS sig_multiple_claimants,
        BOOL_OR(signal = 'no_police_report')                      AS sig_no_police_report,
        BOOL_OR(signal = 'third_party_attorney_early')            AS sig_attorney_early,
        BOOL_OR(signal = 'claim_filed_after_lapse_reinstatement') AS sig_lapse_reinstatement,
        BOOL_OR(signal = 'rapid_refiling')                        AS sig_rapid_refiling,
        BOOL_OR(signal = 'recent_policy_reinstatement')           AS sig_recent_reinstatement
    FROM base
    GROUP BY
        claim_id, is_fraud,
        claim_amount, claim_to_monthly_premium_ratio,
        days_to_file, late_reporting, incident_day_of_week,
        days_since_inception, days_until_expiry, near_inception, near_expiry,
        premium_to_credit_ratio,
        customer_claim_count, claims_last_90d,
        avg_drive_score, hard_brakes_90d, speeding_events_90d,
        state, vehicle_make, zip_prefix, claim_type,
        reported_passengers, num_witnesses,
        iso_prior_claim_count, iso_prior_carrier_count,
        iso_prior_fraud_flag_count, iso_days_since_last_claim,
        iso_vin_claimant_count;
    """)
    df = pd.read_sql(sql, engine)

    # log1p transform on days_to_file — see original comment above
    import numpy as np
    df["days_to_file_log"] = np.log1p(df["days_to_file"].clip(lower=0))

    # Cast boolean signal columns to int
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    df[sig_cols] = df[sig_cols].astype("Int64")

    return df

def risk_features(engine=None) -> pd.DataFrame:
    """
    Returns one row per policy for pure behavioral risk scoring.

    Target: loss_score (built in model.py from claims components) — NOT premium_annual.
    premium_annual is fully removed to prevent target leakage via the pricing formula.

    Feature groups:
      Vehicle:     vehicle_age, vehicle_make (severity predictors — repair cost)
      Driver:      driver_age_bucket, credit_score (frequency predictors)
      Geographic:  state, zip_prefix (environmental risk — static exposure)
      Stability:   has_lapse (adverse selection signal)
      Telematics:  drive_score, avg_drive_score_12m, avg_drive_score_3m,
                   drive_score_delta, annual_trips, annual_miles,
                   avg_night_driving_pct, hard_brakes_per_mile,
                   speeding_per_mile, avg_trip_distance
      Claims:      total_claims, total_claim_amount, claims_last_12m,
                   claims_last_90d, amount_last_12m
                   (claims columns are loss_score components AND lagged predictors)
      Violations:  active_violation_points, active_violation_count,
                   has_dui, has_major_violation, violations_last_3y

    Excluded from model features (identity + loss_score target components):
      policy_number, total_claim_amount, claims_last_12m,
      claims_last_90d, amount_last_12m
    """
    engine = engine or get_engine()
    sql = text("""
        WITH claim_stats AS (
            SELECT
                c.policy_number,
                COUNT(*)                                                AS total_claims,
                COALESCE(SUM(c.claim_amount), 0)                        AS total_claim_amount,
                COUNT(*) FILTER (
                    WHERE c.filed_date >= NOW() - INTERVAL '365 days'
                )                                                       AS claims_last_12m,
                COUNT(*) FILTER (
                    WHERE c.filed_date >= NOW() - INTERVAL '90 days'
                )                                                       AS claims_last_90d,
                COALESCE(SUM(c.claim_amount) FILTER (
                    WHERE c.filed_date >= NOW() - INTERVAL '365 days'
                ), 0)                                                   AS amount_last_12m
            FROM claims c
            GROUP BY c.policy_number
        ),
        telem_stats AS (
            SELECT
                t.policy_number,
                COUNT(*)                                                AS annual_trips,
                COALESCE(SUM(t.distance_miles), 0)                      AS annual_miles,
                COALESCE(AVG(t.drive_score), 50)                        AS avg_drive_score_12m,
                COALESCE(AVG(t.night_driving_pct), 0)                   AS avg_night_driving_pct,
                COALESCE(
                    SUM(t.hard_brakes)
                    / NULLIF(SUM(t.distance_miles), 0), 0
                )                                                       AS hard_brakes_per_mile,
                COALESCE(
                    SUM(t.speeding_events)
                    / NULLIF(SUM(t.distance_miles), 0), 0
                )                                                       AS speeding_per_mile,
                COALESCE(
                    SUM(t.distance_miles)
                    / NULLIF(COUNT(*), 0), 0
                )                                                       AS avg_trip_distance
            FROM telematics t
            WHERE t.trip_date >= NOW() - INTERVAL '365 days'
            GROUP BY t.policy_number
        ),
        telem_recent AS (
            SELECT
                t.policy_number,
                COALESCE(AVG(t.drive_score), 50)                        AS avg_drive_score_3m
            FROM telematics t
            WHERE t.trip_date >= NOW() - INTERVAL '90 days'
            GROUP BY t.policy_number
        ),
        violation_stats AS (
            SELECT
                v.policy_number,
                COALESCE(SUM(v.points)  FILTER (
                    WHERE v.expiry_date >= CURRENT_DATE
                ), 0)                                                   AS active_violation_points,
                COUNT(*) FILTER (
                    WHERE v.expiry_date >= CURRENT_DATE
                )                                                       AS active_violation_count,
                MAX(CASE
                    WHEN v.violation_type = 'dui_dwi'
                     AND v.expiry_date >= CURRENT_DATE
                    THEN 1 ELSE 0
                END)                                                    AS has_dui,
                MAX(CASE
                    WHEN v.violation_type IN (
                        'dui_dwi','reckless_driving',
                        'speeding_major','at_fault_accident'
                    )
                     AND v.expiry_date >= CURRENT_DATE
                    THEN 1 ELSE 0
                END)                                                    AS has_major_violation,
                COUNT(*) FILTER (
                    WHERE v.violation_date >= CURRENT_DATE - INTERVAL '3 years'
                )                                                       AS violations_last_3y
            FROM violations v
            GROUP BY v.policy_number
        )
        SELECT
            p.policy_number,

            -- Audit / identity (state and zip_prefix are ALSO model features)
            p.state,
            LEFT(cust.zip, 3)                                           AS zip_prefix,

            -- Vehicle features (severity predictors)
            (EXTRACT(YEAR FROM NOW())::int
                - COALESCE((p.vehicle->>'year')::int, 2015))            AS vehicle_age,
            p.vehicle->>'make'                                          AS vehicle_make,

            -- Driver features (frequency predictors)
            CASE
                WHEN DATE_PART('year', AGE(cust.dob)) < 25  THEN 'under25'
                WHEN DATE_PART('year', AGE(cust.dob)) < 65  THEN 'standard'
                ELSE 'senior65plus'
            END                                                         AS driver_age_bucket,
            COALESCE(cust.credit_score, 650)                            AS credit_score,

            -- Policy stability
            CASE WHEN p.status IN ('lapsed','cancelled') THEN 1 ELSE 0 END
                                                                        AS has_lapse,

            -- Telematics behavioral features
            COALESCE(p.drive_score, 50)                                 AS drive_score,
            COALESCE(ts.avg_drive_score_12m, 50)                        AS avg_drive_score_12m,
            COALESCE(tr.avg_drive_score_3m, 50)                         AS avg_drive_score_3m,
            -- Negative delta = deteriorating behavior = rising risk
            COALESCE(tr.avg_drive_score_3m, 50)
                - COALESCE(ts.avg_drive_score_12m, 50)                  AS drive_score_delta,
            COALESCE(ts.annual_trips, 0)                                AS annual_trips,
            COALESCE(ts.annual_miles, 0)                                AS annual_miles,
            COALESCE(ts.avg_night_driving_pct, 0)                       AS avg_night_driving_pct,
            COALESCE(ts.hard_brakes_per_mile, 0)                        AS hard_brakes_per_mile,
            COALESCE(ts.speeding_per_mile, 0)                           AS speeding_per_mile,
            COALESCE(ts.avg_trip_distance, 0)                           AS avg_trip_distance,

            -- Claims history (loss_score components + lagged predictors)
            COALESCE(cs.total_claims, 0)                                AS total_claims,
            COALESCE(cs.total_claim_amount, 0)                          AS total_claim_amount,
            COALESCE(cs.claims_last_12m, 0)                             AS claims_last_12m,
            COALESCE(cs.claims_last_90d, 0)                             AS claims_last_90d,
            COALESCE(cs.amount_last_12m, 0)                             AS amount_last_12m,

            -- Violation features (driving record — highest actuarial impact)
            COALESCE(vs.active_violation_points, 0)                     AS active_violation_points,
            COALESCE(vs.active_violation_count, 0)                      AS active_violation_count,
            COALESCE(vs.has_dui, 0)                                     AS has_dui,
            COALESCE(vs.has_major_violation, 0)                         AS has_major_violation,
            COALESCE(vs.violations_last_3y, 0)                          AS violations_last_3y

        FROM policies p
        JOIN customers cust
            ON  cust.customer_id = p.customer_id
        LEFT JOIN claim_stats cs
            ON  cs.policy_number = p.policy_number
        LEFT JOIN telem_stats ts
            ON  ts.policy_number = p.policy_number
        LEFT JOIN telem_recent tr
            ON  tr.policy_number = p.policy_number
        LEFT JOIN violation_stats vs
            ON  vs.policy_number = p.policy_number
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
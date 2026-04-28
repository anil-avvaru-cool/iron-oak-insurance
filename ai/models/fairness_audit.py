"""
Fairness Audit — post-training disparate impact analysis.

Module run:
  uv run python -m ai.models.fairness_audit --model fraud
  uv run python -m ai.models.fairness_audit --model risk
  uv run python -m ai.models.fairness_audit --model churn

Called programmatically after each model trains:
  from ai.models.fairness_audit import run_audit
  run_audit(df, model_name="fraud", score_col="predicted_score", label_col="label")

Slice dimensions:
  state         — 50 states + DC
  zip_prefix    — first 3 digits of ZIP (geographic proxy)
  vehicle_make  — vehicle manufacturer (proxy for owner demographics in fraud/risk)

Threshold: flag if a slice's predicted positive rate deviates > 2× from the
overall rate without a corresponding deviation in the labeled positive rate.
Minimum slice size: MIN_SLICE_SIZE rows (slices smaller than this are skipped
to avoid high-variance flags on tiny populations).

Output: prints a report to stdout; logs summary via structured logger.
Writes a JSON report to ai/models/fairness_reports/<model>_<timestamp>.json
for record-keeping.

AK/HI flags: small population states (n<65). Label rate estimates 
are high-variance at this sample size (4-5 events). 
Flags are statistically expected with n<65 and should be 
re-evaluated when dataset >= 5,000 customers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from ai.utils.log import get_logger

log = get_logger(__name__)

MIN_SLICE_SIZE = 50       # skip slices smaller than this
DEVIATION_THRESHOLD = 2.0  # flag if predicted rate / overall rate > threshold
REPORT_DIR = Path("ai/models/fairness_reports")

# Slice columns available per model
SLICE_COLS: dict[str, list[str]] = {
    "fraud":  ["state", "zip_prefix", "vehicle_make"],
    "risk":   ["state", "zip_prefix", "vehicle_make", "violation_tier"],
    "churn":  ["state", "zip_prefix"],
}


def _positive_rate(series: pd.Series) -> float:
    """Fraction of values >= 0.5 (predicted positive rate)."""
    return float((series >= 0.5).mean())


def _analyze_slice(
    df: pd.DataFrame,
    slice_col: str,
    score_col: str,
    label_col: str,
    overall_pred_rate: float,
    overall_label_rate: float,
) -> list[dict[str, Any]]:
    """Return flagged slice records for one dimension."""
    flags = []
    for val, grp in df.groupby(slice_col):
        if len(grp) < MIN_SLICE_SIZE:
            continue
        pred_rate = _positive_rate(grp[score_col])
        label_rate = float(grp[label_col].mean())

        ratio = pred_rate / overall_pred_rate if overall_pred_rate > 0 else 0.0
        label_ratio = label_rate / overall_label_rate if overall_label_rate > 0 else 0.0

        # Flag: predicted rate deviates > threshold from overall,
        # but label rate does NOT — suggests model bias, not ground truth
        predicted_deviation = abs(ratio - 1.0) > (DEVIATION_THRESHOLD - 1.0)
        label_explains_it = abs(label_ratio - 1.0) > (DEVIATION_THRESHOLD - 1.0)
        flagged = predicted_deviation and not label_explains_it

        if flagged:
            flags.append({
                "slice_col": slice_col,
                "slice_val": str(val),
                "n": len(grp),
                "pred_rate": round(pred_rate, 4),
                "label_rate": round(label_rate, 4),
                "overall_pred_rate": round(overall_pred_rate, 4),
                "overall_label_rate": round(overall_label_rate, 4),
                "pred_ratio": round(ratio, 3),
                "label_ratio": round(label_ratio, 3),
                "flagged": True,
            })
    return flags


def run_audit(
    df: pd.DataFrame,
    model_name: str,
    score_col: str,
    label_col: str,
) -> dict[str, Any]:
    """
    Run disparate impact analysis. Prints report, logs summary, writes JSON.

    Args:
        df:          DataFrame containing score_col, label_col, and slice columns.
        model_name:  "fraud" | "risk" | "churn"
        score_col:   Column with model predicted probability / score (0–1 or 0–100).
        label_col:   Column with ground truth binary label (0/1).

    Returns:
        Report dict (same structure written to JSON).
    """
    t0 = time.time()
    slice_cols = SLICE_COLS.get(model_name, ["state"])
    available_slices = [c for c in slice_cols if c in df.columns]

    if score_col not in df.columns or label_col not in df.columns:
        log.warning(
            "fairness_audit_skipped",
            model=model_name,
            reason="score_col or label_col missing from dataframe",
        )
        return {}

    # Normalize score to 0-1 if it looks like 0-100 range (risk model)
    if df[score_col].max() > 1.5:
        df = df.copy()
        df[score_col] = df[score_col] / 100.0

    overall_pred_rate = _positive_rate(df[score_col])
    overall_label_rate = float(df[label_col].mean())
    total_rows = len(df)

    all_flags: list[dict[str, Any]] = []
    slice_summaries: list[dict[str, Any]] = []

    for col in available_slices:
        flags = _analyze_slice(
            df, col, score_col, label_col, overall_pred_rate, overall_label_rate
        )
        all_flags.extend(flags)
        slice_summaries.append({"col": col, "flags": len(flags)})

    # Print report
    print(f"\n{'='*60}")
    print(f"FAIRNESS AUDIT — {model_name.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Rows: {total_rows:,}  |  Overall predicted positive rate: {overall_pred_rate:.2%}")
    print(f"Overall labeled positive rate: {overall_label_rate:.2%}")
    print(f"Deviation threshold: {DEVIATION_THRESHOLD}×  |  Min slice size: {MIN_SLICE_SIZE}")
    print()

    if not all_flags:
        print("✓ No disparate impact flags found.")
    else:
        print(f"⚠  {len(all_flags)} slice(s) flagged for review:\n")
        for f in all_flags:
            print(
                f"  [{f['slice_col']}={f['slice_val']}] n={f['n']}  "
                f"pred_rate={f['pred_rate']:.2%} ({f['pred_ratio']:.2f}× overall)  "
                f"label_rate={f['label_rate']:.2%} ({f['label_ratio']:.2f}× overall)"
            )
        print(
            "\n  Action: review flagged slices before production deployment.",
        )
    print(f"{'='*60}\n")

    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_rows": total_rows,
        "overall_pred_rate": round(overall_pred_rate, 4),
        "overall_label_rate": round(overall_label_rate, 4),
        "threshold": DEVIATION_THRESHOLD,
        "min_slice_size": MIN_SLICE_SIZE,
        "slices_analyzed": available_slices,
        "total_flags": len(all_flags),
        "flags": all_flags,
        "elapsed_s": round(time.time() - t0, 3),
    }

    # Write JSON report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    report_path = REPORT_DIR / f"{model_name}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))

    log.info(
        "fairness_audit_complete",
        model=model_name,
        total_rows=total_rows,
        total_flags=len(all_flags),
        report_path=str(report_path),
        elapsed_s=report["elapsed_s"],
    )

    return report


def _load_and_score(model_name: str) -> tuple[pd.DataFrame, str, str]:
    """Load features and a saved model, return scored DataFrame."""
    if model_name == "fraud":
        from ai.pipelines.ingestion.feature_engineer import fraud_features
        from ai.models.fraud_detection.model import preprocess, _feature_cols, MODEL_PATH
        import xgboost as xgb

        df = fraud_features()
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        df_proc, _ = preprocess(df.copy())
        df["predicted_score"] = model.predict_proba(df_proc[_feature_cols(df_proc)])[:, 1]
        return df, "predicted_score", "label"

    elif model_name == "risk":
        from ai.pipelines.ingestion.feature_engineer import risk_features
        from ai.models.risk_scoring.model import preprocess, _feature_cols, MODEL_PATH, TIER_THRESHOLDS
        import xgboost as xgb
        import json as _json
        import numpy as np

        df = risk_features()
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        bounds = _json.loads(MODEL_PATH.with_suffix(".bounds.json").read_text())
        df_proc, _ = preprocess(df.copy())
        raw = model.predict(df_proc[_feature_cols(df_proc)])
        raw = np.clip(raw, bounds["min"], bounds["max"])
        df["predicted_score"] = (raw - bounds["min"]) / max(bounds["max"] - bounds["min"], 1) * 100
        df["label"] = (df["predicted_score"] >= TIER_THRESHOLDS["medium"]).astype(int)
        return df, "predicted_score", "label"

    elif model_name == "churn":
        from ai.pipelines.ingestion.feature_engineer import churn_features
        from ai.models.churn_prediction.model import preprocess, _feature_cols, MODEL_PATH
        import xgboost as xgb

        df = churn_features()
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        df_proc, _ = preprocess(df.copy())
        df["predicted_score"] = model.predict_proba(df_proc[_feature_cols(df_proc)])[:, 1]
        return df, "predicted_score", "label"

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: fraud | risk | churn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fairness audit on a trained model.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["fraud", "risk", "churn"],
        help="Model to audit.",
    )
    args = parser.parse_args()
    df, score_col, label_col = _load_and_score(args.model)
    run_audit(df, model_name=args.model, score_col=score_col, label_col=label_col)


if __name__ == "__main__":
    main()

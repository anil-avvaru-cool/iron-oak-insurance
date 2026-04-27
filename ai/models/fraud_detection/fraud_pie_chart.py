"""
fraud_pie_chart.py — Executive-ready fraud detection results pie chart.

Generates a polished .png showing:
  - Fraud vs. Clean claim counts and percentages
  - Model-predicted breakdown (is_fraud_predicted)
  - Saves to: ai/models/fraud_detection/fraud_detection_summary.png

Usage (from repo root):
    uv run python ai/models/fraud_detection/fraud_pie_chart.py

Standalone demo (no DB needed — uses synthetic counts):
    uv run python ai/models/fraud_detection/fraud_pie_chart.py --demo

Integration: called at the end of model.py main() after predict() runs on
the full dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Brand palette ────────────────────────────────────────────────────────────
COLORS = {
    "clean":         "#1A6B3C",   # Iron Oak deep green — clean / safe
    "fraud":         "#C0392B",   # Alert red — fraud
    "clean_light":   "#A8D5B5",   # Light green accent
    "fraud_light":   "#F1948A",   # Light red accent
    "background":    "#FAFAFA",
    "text_dark":     "#1C1C1C",
    "text_muted":    "#666666",
    "border":        "#E0E0E0",
}

FONT_FAMILY = "DejaVu Sans"


def build_chart(
    total_claims: int,
    fraud_predicted: int,
    clean_predicted: int,
    model_version: str = "xgboost-fraud-v1",
    output_path: Path | None = None,
    roc_auc: float | None = None,
) -> Path:
    """
    Build and save the executive pie chart.

    Args:
        total_claims:     Total claims scored.
        fraud_predicted:  Count predicted as fraud (score >= 0.5).
        clean_predicted:  Count predicted as clean.
        model_version:    Label shown in subtitle.
        output_path:      Where to save. Defaults to script directory.
        roc_auc:          Optional ROC-AUC to show in footer.

    Returns:
        Path to saved .png file.
    """
    if output_path is None:
        output_path = Path(__file__).parent / "fraud_detection_summary.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraud_pct = fraud_predicted / total_claims * 100
    clean_pct = clean_predicted / total_claims * 100

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor=COLORS["background"])

    # Title block
    fig.text(
        0.5, 0.94,
        "Fraud Detection Model — Claim Risk Summary",
        ha="center", va="top",
        fontsize=20, fontweight="bold",
        color=COLORS["text_dark"], fontfamily=FONT_FAMILY,
    )
    fig.text(
        0.5, 0.88,
        f"Model: {model_version}  ·  Total Claims Scored: {total_claims:,}",
        ha="center", va="top",
        fontsize=12, color=COLORS["text_muted"], fontfamily=FONT_FAMILY,
    )

    # ── Pie chart (left panel) ─────────────────────────────────────────────
    ax_pie = fig.add_axes([0.05, 0.12, 0.50, 0.70])
    ax_pie.set_facecolor(COLORS["background"])

    sizes  = [clean_predicted, fraud_predicted]
    colors = [COLORS["clean"], COLORS["fraud"]]
    explode = (0, 0.06)   # slight pop on fraud slice for emphasis

    wedges, texts, autotexts = ax_pie.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        shadow=False,
    )

    for at in autotexts:
        at.set_fontsize(15)
        at.set_fontweight("bold")
        at.set_color("white")
        at.set_fontfamily(FONT_FAMILY)

    # Center donut label
    ax_pie.text(
        0, 0, f"{total_claims:,}\nClaims",
        ha="center", va="center",
        fontsize=13, fontweight="bold",
        color=COLORS["text_dark"], fontfamily=FONT_FAMILY,
        multialignment="center",
    )

    # Make it a donut
    centre_circle = plt.Circle((0, 0), 0.48, fc=COLORS["background"])
    ax_pie.add_patch(centre_circle)

    # ── Stats panel (right) ────────────────────────────────────────────────
    ax_stats = fig.add_axes([0.58, 0.12, 0.38, 0.70])
    ax_stats.set_facecolor(COLORS["background"])
    ax_stats.axis("off")

    def stat_block(y: float, label: str, count: int, pct: float, color: str):
        """Draw a colored stat card."""
        # Card background
        rect = mpatches.FancyBboxPatch(
            (0.0, y - 0.09), 1.0, 0.22,
            boxstyle="round,pad=0.02",
            facecolor=color + "18",   # 18 = ~10% opacity hex
            edgecolor=color,
            linewidth=1.5,
            transform=ax_stats.transAxes,
            clip_on=False,
        )
        ax_stats.add_patch(rect)

        ax_stats.text(
            0.08, y + 0.08, label.upper(),
            transform=ax_stats.transAxes,
            fontsize=10, color=color, fontweight="bold",
            fontfamily=FONT_FAMILY, va="top",
        )
        ax_stats.text(
            0.08, y - 0.01, f"{count:,}",
            transform=ax_stats.transAxes,
            fontsize=28, fontweight="bold", color=COLORS["text_dark"],
            fontfamily=FONT_FAMILY, va="top",
        )
        ax_stats.text(
            0.72, y - 0.01, f"{pct:.1f}%",
            transform=ax_stats.transAxes,
            fontsize=22, fontweight="bold", color=color,
            fontfamily=FONT_FAMILY, va="top",
        )

    stat_block(0.72, "✓  Clean Claims",   clean_predicted, clean_pct, COLORS["clean"])
    stat_block(0.35, "⚠  Flagged Fraud",  fraud_predicted, fraud_pct, COLORS["fraud"])

    # ROC-AUC badge (optional)
    if roc_auc is not None:
        ax_stats.text(
            0.5, 0.07,
            f"Model ROC-AUC:  {roc_auc:.4f}",
            transform=ax_stats.transAxes,
            ha="center", fontsize=11,
            color=COLORS["text_muted"], fontfamily=FONT_FAMILY,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=COLORS["border"],
                edgecolor=COLORS["border"],
            ),
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")
    fig.text(
        0.5, 0.04,
        f"Avvaru Iron Oak Insurance  ·  Fraud Detection Dashboard  ·  {today}",
        ha="center", va="bottom",
        fontsize=9, color=COLORS["text_muted"], fontfamily=FONT_FAMILY,
        style="italic",
    )

    # Thin top border line under title
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [0.86, 0.86],
                   transform=fig.transFigure,
                   color=COLORS["border"], linewidth=1)
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=COLORS["background"],
        edgecolor="none",
    )
    plt.close(fig)
    print(f"[fraud_pie_chart] saved → {output_path}")
    return output_path


def from_scored_records(
    records: list[dict],
    output_path: Path | None = None,
    roc_auc: float | None = None,
) -> Path:
    """
    Build chart directly from the list of scored claim dicts returned by predict().

    Each record must have 'is_fraud_predicted' (bool) added by predict().
    """
    total   = len(records)
    flagged = sum(1 for r in records if r.get("is_fraud_predicted", False))
    clean   = total - flagged
    return build_chart(
        total_claims=total,
        fraud_predicted=flagged,
        clean_predicted=clean,
        output_path=output_path,
        roc_auc=roc_auc,
    )


def from_json_results(
    results_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Build chart from a saved JSON file of scored records
    (e.g. the output of the /models/fraud/score endpoint).
    """
    data = json.loads(results_path.read_text())
    records = data if isinstance(data, list) else data.get("results", [])
    return from_scored_records(records, output_path=output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate executive fraud detection pie chart."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic demo data (no DB or model file needed).",
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to a JSON file of scored claim records.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .png path. Default: ai/models/fraud_detection/fraud_detection_summary.png",
    )
    parser.add_argument(
        "--roc-auc", type=float, default=None,
        help="Optional ROC-AUC score to display in the chart.",
    )
    args = parser.parse_args()

    if args.demo:
        # Realistic synthetic numbers for demo/testing
        build_chart(
            total_claims=3_200,
            fraud_predicted=128,
            clean_predicted=3_072,
            model_version="xgboost-fraud-v1 (demo)",
            output_path=args.output,
            roc_auc=args.roc_auc or 0.9312,
        )
        return

    if args.results:
        from_json_results(args.results, output_path=args.output)
        return

    # Default: load from DB and run model scoring
    try:
        from ai.pipelines.ingestion.feature_engineer import fraud_features
        from ai.models.fraud_detection.model import predict, preprocess, _feature_cols, MODEL_PATH
        import xgboost as xgb
        import pandas as pd

        df = fraud_features()
        records = df.to_dict("records")
        scored  = predict(records)

        # Try to extract ROC-AUC from most recent fairness report
        roc_auc = args.roc_auc
        if roc_auc is None:
            reports = sorted(
                Path("ai/models/fairness_reports").glob("fraud_*.json"),
                reverse=True,
            )
            if reports:
                report = json.loads(reports[0].read_text())
                # ROC-AUC isn't in fairness report — use arg only
                pass

        from_scored_records(scored, output_path=args.output, roc_auc=roc_auc)

    except ImportError as e:
        print(f"DB/model not available ({e}). Run with --demo for a standalone chart.")
        raise


if __name__ == "__main__":
    main()
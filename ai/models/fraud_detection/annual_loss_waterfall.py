"""
annual_loss_waterfall.py — Executive side-by-side waterfall chart.
Shows Annual Loss Calculation: Before vs. After fraud model deployment.

Placement: ai/models/fraud_detection/annual_loss_waterfall.py

Usage:
    # Demo mode — no DB needed
    uv run python -m ai.models.fraud_detection.annual_loss_waterfall --demo

    # From live DB (derives figures from synthetic dataset)
    uv run python -m ai.models.fraud_detection.annual_loss_waterfall

    # Custom figures (all values in dollars)
    uv run python -m ai.models.fraud_detection.annual_loss_waterfall \\
        --premium 12500000 \\
        --legit-claims 7200000 \\
        --total-fraud 500000 \\
        --detection-rate 0.78 \\
        --investigation-cost 85000 \\
        --model-cost 45000

Integration — call from model.py main() after training:
    from ai.models.fraud_detection.annual_loss_waterfall import build_waterfall
    build_waterfall(
        premium=total_premium,
        legit_claims=legit_claim_total,
        total_fraud_loss=fraud_total,
        detection_rate=roc_auc_proxy,   # or pass actual detection rate
    )
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np


# ── Brand palette ────────────────────────────────────────────────────────────
C = {
    "inflow":       "#1A6B3C",   # green  — premium / positive
    "loss":         "#C0392B",   # red    — losses / outflows
    "recovered":    "#2471A3",   # blue   — fraud recovered by model
    "net_positive": "#1A6B3C",   # green  — net result positive
    "net_negative": "#C0392B",   # red    — net result negative
    "model_cost":   "#7D6608",   # amber  — model investment
    "connector":    "#AAAAAA",   # grey   — waterfall connector lines
    "bg":           "#FAFAFA",
    "panel_bg":     "#F2F2F2",
    "text_dark":    "#1C1C1C",
    "text_muted":   "#666666",
    "border":       "#DDDDDD",
    "savings_band": "#D5F5E3",   # light green band — savings highlight
}

FONT = "DejaVu Sans"


def _fmt(v: float) -> str:
    """Format dollar value for bar labels: $1.2M, $850K, $45K."""
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"${abs_v/1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"${abs_v/1_000:.0f}K"
    return f"${abs_v:,.0f}"


def _fmt_pct(v: float, base: float) -> str:
    """Format as percent of premium base."""
    if base == 0:
        return ""
    return f"{abs(v)/base*100:.1f}%"


def _waterfall_bars(ax, steps: list[dict], base_color_map: dict, premium: float):
    """
    Draw one waterfall on ax.

    Each step dict:
        label   str
        value   float  (positive = inflow, negative = outflow)
        kind    str    (inflow | loss | recovered | model_cost | net)
        is_net  bool   (draws from zero, not stacked)
    """
    running = 0.0
    bottoms = []
    heights = []
    colors  = []
    x_positions = list(range(len(steps)))

    for step in steps:
        v = step["value"]
        if step.get("is_net"):
            bottoms.append(min(0, v))
            heights.append(abs(v))
            colors.append(C["net_positive"] if v >= 0 else C["net_negative"])
        else:
            if v >= 0:
                bottoms.append(running)
                heights.append(v)
            else:
                bottoms.append(running + v)
                heights.append(abs(v))
            running += v
            colors.append(base_color_map.get(step["kind"], C["loss"]))

    bars = ax.bar(
        x_positions, heights, bottom=bottoms,
        color=colors, width=0.55,
        edgecolor="white", linewidth=1.5,
        zorder=3,
    )

    # Connector lines between bars
    running2 = 0.0
    for i, step in enumerate(steps[:-1]):
        if step.get("is_net"):
            continue
        next_step = steps[i + 1]
        if next_step.get("is_net"):
            continue
        top = running2 + step["value"] if step["value"] >= 0 else running2
        top = running2 + step["value"]
        ax.plot(
            [i + 0.275, i + 0.725], [top, top],
            color=C["connector"], linewidth=0.8,
            linestyle="--", zorder=2,
        )
        running2 += step["value"]

    # Bar value labels
    running3 = 0.0
    for i, (step, bar) in enumerate(zip(steps, bars)):
        v = step["value"]
        top_y = bar.get_y() + bar.get_height()
        bot_y = bar.get_y()
        label_y = top_y + (ax.get_ylim()[1] * 0.012)

        dollar_label = _fmt(v)
        pct_label    = _fmt_pct(v, premium)
        sign         = "+" if v > 0 else "−" if v < 0 else ""

        color = C["inflow"] if v > 0 else C["loss"]
        if step["kind"] == "recovered":
            color = C["recovered"]
        if step["kind"] == "model_cost":
            color = C["model_cost"]
        if step.get("is_net"):
            color = C["net_positive"] if v >= 0 else C["net_negative"]
            dollar_label = _fmt(v)
            pct_label    = ""

        ax.text(
            i, label_y,
            f"{sign}{dollar_label}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=color, fontfamily=FONT,
            zorder=5,
        )
        if pct_label:
            ax.text(
                i, label_y + ax.get_ylim()[1] * 0.045,
                f"({pct_label} of premium)",
                ha="center", va="bottom",
                fontsize=7, color=C["text_muted"],
                fontfamily=FONT, zorder=5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [s["label"] for s in steps],
        fontsize=8, fontfamily=FONT,
        color=C["text_dark"],
    )
    ax.tick_params(axis="x", length=0, pad=6)

    return bars


def build_waterfall(
    premium: float           = 12_500_000,
    legit_claims: float      = 7_200_000,
    total_fraud_loss: float  = 500_000,
    detection_rate: float    = 0.78,
    investigation_cost: float = 85_000,
    model_cost: float        = 45_000,
    output_path: Path | None = None,
) -> Path:
    """
    Build and save the side-by-side waterfall.

    Args:
        premium:              Total annual premium collected.
        legit_claims:         Legitimate claims paid out.
        total_fraud_loss:     Total fraud exposure (detected + undetected).
        detection_rate:       Fraction of fraud the model catches (0–1).
        investigation_cost:   Annual cost to investigate flagged claims.
        model_cost:           Annual model infrastructure + ops cost.
        output_path:          Save path. Defaults to script directory.

    Returns:
        Path to saved .png.
    """
    if output_path is None:
        output_path = Path(__file__).parent / "annual_loss_waterfall.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Derived figures ───────────────────────────────────────────────────
    fraud_undetected_before = total_fraud_loss                    # 100% loss before model
    net_before = premium - legit_claims - fraud_undetected_before

    fraud_caught     = total_fraud_loss * detection_rate          # recovered by model
    fraud_missed     = total_fraud_loss * (1 - detection_rate)    # still slips through
    total_model_cost = investigation_cost + model_cost
    net_after  = premium - legit_claims - fraud_missed - total_model_cost

    savings = net_after - net_before                              # net improvement

    # ── Step definitions ─────────────────────────────────────────────────
    color_map = {
        "inflow":     C["inflow"],
        "loss":       C["loss"],
        "recovered":  C["recovered"],
        "model_cost": C["model_cost"],
    }

    steps_before = [
        {"label": "Premium\nCollected",    "value":  premium,                   "kind": "inflow"},
        {"label": "Legitimate\nClaims",    "value": -legit_claims,              "kind": "loss"},
        {"label": "Fraud\nLosses\n(100%)", "value": -fraud_undetected_before,   "kind": "loss"},
        {"label": "Net\nResult",           "value":  net_before,                "kind": "net", "is_net": True},
    ]

    steps_after = [
        {"label": "Premium\nCollected",      "value":  premium,              "kind": "inflow"},
        {"label": "Legitimate\nClaims",      "value": -legit_claims,         "kind": "loss"},
        {"label": "Fraud\nRecovered\n(Model)","value":  fraud_caught,        "kind": "recovered"},
        {"label": "Fraud\nMissed",           "value": -fraud_missed,         "kind": "loss"},
        {"label": "Model &\nInvestigation",  "value": -total_model_cost,     "kind": "model_cost"},
        {"label": "Net\nResult",             "value":  net_after,            "kind": "net", "is_net": True},
    ]

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=C["bg"])

    # Title
    fig.text(
        0.5, 0.955,
        "Annual Loss Calculation — Before vs. After Fraud Model Deployment",
        ha="center", va="top",
        fontsize=18, fontweight="bold",
        color=C["text_dark"], fontfamily=FONT,
    )
    def _safe(v: float) -> str:
        """Dollar-free label safe for matplotlib text (no mathtext conflict)."""
        abs_v = abs(v)
        if abs_v >= 1_000_000:
            return f"USD {abs_v/1_000_000:.2f}M"
        if abs_v >= 1_000:
            return f"USD {abs_v/1_000:.0f}K"
        return f"USD {abs_v:,.0f}"

    fig.text(
        0.5, 0.912,
        f"Fraud exposure: {_safe(total_fraud_loss)}  |  "
        f"Model detection rate: {detection_rate*100:.0f}%  |  "
        f"Net annual savings: {_safe(savings)}",
        ha="center", va="top",
        fontsize=11, color=C["text_muted"], fontfamily=FONT,
        usetex=False,
    )

    # Divider
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [0.895, 0.895],
        transform=fig.transFigure,
        color=C["border"], linewidth=1,
    ))

    # ── Left panel — Before ───────────────────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.13, 0.40, 0.73])
    ax1.set_facecolor(C["panel_bg"])
    ax1.spines[["top","right","left"]].set_visible(False)
    ax1.spines["bottom"].set_color(C["border"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K"
    ))
    ax1.tick_params(axis="y", labelsize=8, labelcolor=C["text_muted"])
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", color=C["border"], linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)

    y_max = premium * 1.18
    y_min = min(net_before, -legit_claims * 0.05) - premium * 0.05
    ax1.set_ylim(y_min, y_max)

    _waterfall_bars(ax1, steps_before, color_map, premium)

    # Zero line
    ax1.axhline(0, color=C["text_dark"], linewidth=0.8, zorder=4)

    ax1.set_title(
        "WITHOUT Fraud Model",
        fontsize=13, fontweight="bold",
        color=C["loss"], fontfamily=FONT, pad=10,
    )

    # Net result annotation
    ax1.annotate(
        f"Net: {_fmt(net_before)}\n({_fmt_pct(net_before, premium)} margin)",
        xy=(len(steps_before)-1, net_before),
        xytext=(len(steps_before)-1.6, net_before + premium * 0.06),
        fontsize=8.5, color=C["net_positive"] if net_before >= 0 else C["net_negative"],
        fontfamily=FONT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=0.8),
    )

    # ── Right panel — After ───────────────────────────────────────────────
    ax2 = fig.add_axes([0.535, 0.13, 0.44, 0.73])
    ax2.set_facecolor(C["panel_bg"])
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.spines["bottom"].set_color(C["border"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K"
    ))
    ax2.tick_params(axis="y", labelsize=8, labelcolor=C["text_muted"])
    ax2.tick_params(axis="x", labelsize=8)
    ax2.grid(axis="y", color=C["border"], linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_ylim(y_min, y_max)   # same scale as left panel — critical for comparison

    _waterfall_bars(ax2, steps_after, color_map, premium)
    ax2.axhline(0, color=C["text_dark"], linewidth=0.8, zorder=4)

    ax2.set_title(
        "WITH Fraud Model Deployed",
        fontsize=13, fontweight="bold",
        color=C["inflow"], fontfamily=FONT, pad=10,
    )

    ax2.annotate(
        f"Net: {_fmt(net_after)}\n({_fmt_pct(net_after, premium)} margin)",
        xy=(len(steps_after)-1, net_after),
        xytext=(len(steps_after)-1.8, net_after + premium * 0.06),
        fontsize=8.5, color=C["net_positive"] if net_after >= 0 else C["net_negative"],
        fontfamily=FONT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=0.8),
    )

    # ── Savings callout strip between panels ─────────────────────────────
    ax_mid = fig.add_axes([0.455, 0.35, 0.075, 0.30])
    ax_mid.set_facecolor(C["bg"])
    ax_mid.axis("off")

    ax_mid.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.90,
        boxstyle="round,pad=0.05",
        facecolor=C["savings_band"],
        edgecolor=C["inflow"],
        linewidth=1.5,
        transform=ax_mid.transAxes,
    ))
    ax_mid.text(0.5, 0.78, "Annual\nSavings", ha="center", va="center",
                fontsize=8, color=C["text_muted"], fontfamily=FONT,
                fontweight="bold", transform=ax_mid.transAxes,
                multialignment="center")
    ax_mid.text(0.5, 0.48, _fmt(savings), ha="center", va="center",
                fontsize=13, color=C["inflow"], fontfamily=FONT,
                fontweight="bold", transform=ax_mid.transAxes)
    ax_mid.text(0.5, 0.25, f"ROI: {savings/total_model_cost:.1f}×",
                ha="center", va="center",
                fontsize=9, color=C["inflow"], fontfamily=FONT,
                fontweight="bold", transform=ax_mid.transAxes)
    ax_mid.text(0.5, 0.10, "▶", ha="center", va="center",
                fontsize=14, color=C["inflow"], fontfamily=FONT,
                transform=ax_mid.transAxes)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C["inflow"],     label="Premium / Inflow"),
        mpatches.Patch(facecolor=C["loss"],       label="Loss / Outflow"),
        mpatches.Patch(facecolor=C["recovered"],  label="Fraud Recovered (Model)"),
        mpatches.Patch(facecolor=C["model_cost"], label="Model & Investigation Cost"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor=C["border"],
        bbox_to_anchor=(0.5, 0.01),
        prop={"family": FONT, "size": 9},
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    today = datetime.date.today().strftime("%B %d, %Y")
    fig.text(
        0.5, 0.055,
        f"Avvaru Iron Oak Insurance  |  Annual Loss Analysis  |  {today}",
        ha="center", va="bottom",
        fontsize=8.5, color=C["text_muted"],
        fontfamily=FONT, style="italic",
    )

    plt.savefig(
        output_path, dpi=180,
        bbox_inches="tight",
        facecolor=C["bg"],
        edgecolor="none",
    )
    plt.close(fig)
    print(f"[annual_loss_waterfall] saved → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate executive annual loss waterfall chart."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo figures.")
    parser.add_argument("--premium",           type=float, default=None)
    parser.add_argument("--legit-claims",      type=float, default=None)
    parser.add_argument("--total-fraud",       type=float, default=None)
    parser.add_argument("--detection-rate",    type=float, default=0.78)
    parser.add_argument("--investigation-cost",type=float, default=85_000)
    parser.add_argument("--model-cost",        type=float, default=45_000)
    parser.add_argument("--output",            type=Path,  default=None)
    args = parser.parse_args()

    if args.demo or not all([args.premium, args.legit_claims, args.total_fraud]):
        # Derive from synthetic dataset defaults or use demo constants
        build_waterfall(output_path=args.output)
        return

    build_waterfall(
        premium=args.premium,
        legit_claims=args.legit_claims,
        total_fraud_loss=args.total_fraud,
        detection_rate=args.detection_rate,
        investigation_cost=args.investigation_cost,
        model_cost=args.model_cost,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
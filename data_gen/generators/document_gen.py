"""
document_gen.py — generates synthetic insurance PDFs for AIOI using ReportLab.

Produces three document types:
    decl_{POLICY_NUMBER}.pdf    — Declaration pages (coverage tables, vehicle details)
    claim_letter_{CLAIM_ID}.pdf — Claim outcome letters (narrative prose)
    renewal_{POLICY_NUMBER}.pdf — Renewal notices (premium change table + prose)

Usage:
    uv run python data_gen/generators/document_gen.py
    from document_gen import generate, main

Key design decisions:
    - Filename convention is load-bearing: chunk_router.py uses prefix matching in Phase 4
    - Declaration pages render the coverages JSONB as a table — one row per coverage type
      with Limit and Deductible kept together in the same chunk (critical for RAG accuracy)
    - All currency values use locale formatting for realism
    - AIOI branding: Iron & Oak color scheme (dark forest green #2D5016, copper #B8763A)
    - PDFs are searchable (text-layer, not scanned images)
    - No real PII — all names/addresses from Faker
"""
from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

from faker import Faker

fake = Faker("en_US")
Faker.seed(46)
random.seed(46)

# ── ReportLab imports ──────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# AIOI brand colors
_FOREST_GREEN = colors.HexColor("#2D5016")
_COPPER = colors.HexColor("#B8763A")
_LIGHT_GRAY = colors.HexColor("#F5F5F5")
_MID_GRAY = colors.HexColor("#CCCCCC")

_COVERAGE_LABELS = {
    "liability": "Bodily Injury / Property Damage Liability",
    "collision": "Collision",
    "comprehensive": "Comprehensive",
    "pip": "Personal Injury Protection (PIP)",
    "uninsured_motorist": "Uninsured / Underinsured Motorist",
    "gap": "GAP Coverage",
    "roadside": "Roadside Assistance",
}


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("DocTitle", parent=base["Normal"],
                                fontSize=16, fontName="Helvetica-Bold",
                                textColor=_FOREST_GREEN, spaceAfter=4),
        "subtitle": ParagraphStyle("DocSub", parent=base["Normal"],
                                   fontSize=10, fontName="Helvetica",
                                   textColor=colors.gray, spaceAfter=8),
        "heading": ParagraphStyle("SecHeading", parent=base["Normal"],
                                  fontSize=11, fontName="Helvetica-Bold",
                                  textColor=_FOREST_GREEN, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("Body", parent=base["Normal"],
                               fontSize=9, fontName="Helvetica",
                               leading=13, spaceAfter=4),
        "small": ParagraphStyle("Small", parent=base["Normal"],
                                fontSize=7.5, fontName="Helvetica",
                                textColor=colors.gray, leading=10),
        "footer": ParagraphStyle("Footer", parent=base["Normal"],
                                 fontSize=7, fontName="Helvetica",
                                 textColor=colors.gray, alignment=TA_CENTER),
    }


def _header_block(styles: dict, doc_type: str, policy_number: str) -> list:
    elements = []
    elements.append(Paragraph("AVVARU IRON OAK INSURANCE", styles["title"]))
    elements.append(Paragraph(
        f"1000 Market Street, Harrisburg, PA 17101  |  1-800-555-AIOI  |  www.ironoakinsurance.com",
        styles["subtitle"]
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=_COPPER, spaceAfter=8))
    elements.append(Paragraph(doc_type, styles["heading"]))
    elements.append(Paragraph(f"Policy Number: {policy_number}", styles["body"]))
    return elements


def _coverage_table(coverages: dict, styles: dict) -> Table:
    header = [
        Paragraph("<b>Coverage</b>", styles["body"]),
        Paragraph("<b>Status</b>", styles["body"]),
        Paragraph("<b>Limit</b>", styles["body"]),
        Paragraph("<b>Deductible</b>", styles["body"]),
    ]
    rows = [header]
    for key in ["liability", "collision", "comprehensive", "pip",
                "uninsured_motorist", "gap", "roadside"]:
        cov = coverages.get(key, {})
        label = _COVERAGE_LABELS.get(key, key.replace("_", " ").title())
        included = cov.get("included", False)
        status = "Included" if included else "Not Included"
        limit = cov.get("limit") or ("N/A" if not included else "See Policy")
        deductible = cov.get("deductible")
        ded_str = f"${deductible:,}" if deductible else ("N/A" if not included else "None")
        pip_limit = cov.get("pip_limit")
        if pip_limit and included:
            limit = f"${pip_limit:,}"

        rows.append([
            Paragraph(label, styles["body"]),
            Paragraph(status, styles["body"]),
            Paragraph(str(limit), styles["body"]),
            Paragraph(ded_str, styles["body"]),
        ])

    tbl = Table(rows, colWidths=[3.2 * inch, 1.0 * inch, 1.4 * inch, 1.0 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _FOREST_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, _MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _generate_declaration_page(policy: dict, customer: dict,
                                output_path: Path, styles: dict) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    elements = _header_block(styles, "DECLARATIONS PAGE — PERSONAL AUTO POLICY",
                             policy["policy_number"])

    # Named insured block
    elements.append(Paragraph("Named Insured & Policy Details", styles["heading"]))
    insured_data = [
        ["Named Insured:", f"{customer['first_name']} {customer['last_name']}"],
        ["Mailing Address:", f"{fake.street_address()}, {fake.city()}, {policy['state']} {customer.get('zip', '00000')}"],
        ["Policy Number:", policy["policy_number"]],
        ["Effective Date:", policy["effective_date"]],
        ["Expiration Date:", policy["expiry_date"]],
        ["Annual Premium:", f"${policy.get('premium_annual', 0):,.2f}"],
        ["Policy Status:", policy.get("status", "active").title()],
    ]
    drive_score = policy.get("drive_score")
    if drive_score is not None:
        insured_data.append(["Iron Oak Drive Score:", f"{drive_score:.1f} / 100.0"])
    else:
        insured_data.append(["Iron Oak Drive Score:", "Not enrolled in telematics program"])

    insured_table = Table(insured_data, colWidths=[2.0*inch, 4.6*inch])
    insured_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(insured_table)
    elements.append(Spacer(1, 8))

    # Vehicle block
    vehicle = policy.get("vehicle", {})
    elements.append(Paragraph("Covered Vehicle", styles["heading"]))
    veh_data = [
        ["Year:", str(vehicle.get("year", "N/A"))],
        ["Make:", vehicle.get("make", "N/A")],
        ["Model:", vehicle.get("model", "N/A")],
        ["VIN:", vehicle.get("vin", "N/A")],
    ]
    veh_table = Table(veh_data, colWidths=[2.0*inch, 4.6*inch])
    veh_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(veh_table)
    elements.append(Spacer(1, 8))

    # Coverage table
    elements.append(Paragraph("Coverages", styles["heading"]))
    elements.append(_coverage_table(policy.get("coverages", {}), styles))
    elements.append(Spacer(1, 12))

    # State-specific notice
    state_notice = (
        f"This policy is governed by the laws of the state of {policy['state']}. "
        f"Please review your policy document for complete terms and conditions."
    )
    elements.append(Paragraph(state_notice, styles["small"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Avvaru Iron Oak Insurance | Confidential — For Policyholder Use Only | "
        "Source: synthetic-v1 (demo data)",
        styles["footer"]
    ))

    doc.build(elements)


def _generate_claim_letter(claim: dict, policy: dict, customer: dict,
                            output_path: Path, styles: dict) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    elements = _header_block(styles, "CLAIM DETERMINATION LETTER", policy["policy_number"])

    elements.append(Paragraph(f"Date: {date.today().isoformat()}", styles["body"]))
    elements.append(Paragraph(
        f"{customer['first_name']} {customer['last_name']}<br/>"
        f"{fake.street_address()}<br/>{fake.city()}, {policy['state']} "
        f"{customer.get('zip', '00000')}",
        styles["body"]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"RE: Claim Number {claim['claim_id']} — "
                               f"{claim['claim_type'].replace('_', ' ').title()} Claim",
                               styles["heading"]))

    status_text = {
        "approved": "We are pleased to inform you that your claim has been approved.",
        "settled": "We are writing to confirm that your claim has been settled.",
        "denied": "After careful review, we must inform you that your claim has been denied.",
        "under_review": "Your claim is currently under review by our claims team.",
        "open": "We have received your claim and it is being processed.",
    }.get(claim.get("status", "open"), "Your claim is being processed.")

    body_paragraphs = [
        f"Dear {customer['first_name']} {customer['last_name']},",
        "",
        status_text,
        "",
        f"Claim Details:",
        f"  Claim Number: {claim['claim_id']}",
        f"  Policy Number: {policy['policy_number']}",
        f"  Incident Date: {claim.get('incident_date', 'N/A')}",
        f"  Claim Type: {claim.get('claim_type', 'N/A').replace('_', ' ').title()}",
        f"  Claim Amount: ${claim.get('claim_amount', 0):,.2f}",
    ]
    if claim.get("settlement_amount"):
        body_paragraphs.append(
            f"  Settlement Amount: ${claim['settlement_amount']:,.2f}"
        )

    body_paragraphs += [
        "",
        claim.get("incident_narrative", ""),
        "",
        claim.get("adjuster_notes", ""),
        "",
        "If you have questions regarding this determination, please contact our claims "
        "department at 1-800-555-AIOI or visit your online account portal.",
        "",
        "Sincerely,",
        "Claims Department",
        "Avvaru Iron Oak Insurance",
    ]

    for para in body_paragraphs:
        if para == "":
            elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph(para, styles["body"]))

    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=_MID_GRAY))
    elements.append(Paragraph(
        "Avvaru Iron Oak Insurance | Source: synthetic-v1 (demo data) | "
        f"Claim {claim['claim_id']} | Policy {policy['policy_number']}",
        styles["footer"]
    ))
    doc.build(elements)


def _generate_renewal_notice(policy: dict, customer: dict,
                              output_path: Path, styles: dict) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    elements = _header_block(styles, "POLICY RENEWAL NOTICE", policy["policy_number"])

    current_premium = policy.get("premium_annual", 1200.0)
    change_pct = random.uniform(-0.08, 0.18)  # -8% to +18% change
    new_premium = round(current_premium * (1 + change_pct), 2)
    change_direction = "increased" if change_pct > 0 else "decreased"
    change_abs = abs(round(current_premium * change_pct, 2))

    elements.append(Paragraph(
        f"Dear {customer['first_name']} {customer['last_name']},",
        styles["body"]
    ))
    elements.append(Paragraph(
        f"Your policy <b>{policy['policy_number']}</b> is scheduled for renewal on "
        f"<b>{policy.get('expiry_date', 'N/A')}</b>. Please review the changes to "
        f"your premium below.",
        styles["body"]
    ))
    elements.append(Spacer(1, 8))

    # Premium change table
    elements.append(Paragraph("Premium Summary", styles["heading"]))
    premium_data = [
        [Paragraph("<b>Item</b>", styles["body"]),
         Paragraph("<b>Current</b>", styles["body"]),
         Paragraph("<b>Renewal</b>", styles["body"])],
        ["Annual Premium",
         f"${current_premium:,.2f}",
         f"${new_premium:,.2f}"],
        ["Monthly Installment",
         f"${current_premium/12:,.2f}",
         f"${new_premium/12:,.2f}"],
        ["Change",
         "",
         f"{'+' if change_pct > 0 else ''}{change_pct*100:.1f}% (${'+' if change_pct > 0 else '-'}{change_abs:,.2f})"],
    ]
    drive_score = policy.get("drive_score")
    if drive_score is not None:
        discount = 0.0
        if drive_score >= 90: discount = 0.15
        elif drive_score >= 75: discount = 0.08
        elif drive_score >= 60: discount = 0.03
        if discount > 0:
            premium_data.append([
                "Iron Oak Drive Score Discount",
                f"Drive Score: {drive_score:.1f}",
                f"-{discount*100:.0f}% applied"
            ])

    prem_tbl = Table(premium_data, colWidths=[3.2*inch, 1.7*inch, 1.7*inch])
    prem_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _FOREST_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, _MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(prem_tbl)
    elements.append(Spacer(1, 8))

    # Coverage table
    elements.append(Paragraph("Coverage Summary", styles["heading"]))
    elements.append(_coverage_table(policy.get("coverages", {}), styles))
    elements.append(Spacer(1, 10))

    # Prose explanation
    reason_texts = [
        f"Your premium has {change_direction} by ${change_abs:,.2f} this renewal period.",
        f"Factors affecting your rate include your claims history, vehicle age, "
        f"{'telematics driving data, ' if drive_score is not None else ''}"
        f"and regional market conditions in {policy['state']}.",
        f"Your current coverages are shown in the table above. "
        f"To make changes to your coverage or payment plan, please contact us at "
        f"1-800-555-AIOI or log into your account portal.",
        f"To accept this renewal, no action is required. Your policy will automatically "
        f"renew on {policy.get('expiry_date', 'the expiration date')} at the new premium.",
    ]
    for text in reason_texts:
        elements.append(Paragraph(text, styles["body"]))

    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=_MID_GRAY))
    elements.append(Paragraph(
        "Avvaru Iron Oak Insurance | Source: synthetic-v1 (demo data) | "
        f"Renewal Notice | Policy {policy['policy_number']}",
        styles["footer"]
    ))
    doc.build(elements)


def generate(pdf_count: int, config: dict, states_data: dict,
             policies: list[dict] | None = None,
             customers: list[dict] | None = None,
             claims: list[dict] | None = None,
             output_dir: Path = Path("documents")) -> list[Path]:
    """
    Generate PDFs. Returns list of generated file paths.
    Balances doc types: ~45% declarations, ~35% claim letters, ~20% renewals.
    """
    if policies is None:
        policies = json.loads(Path("data/policies.json").read_text())
    if customers is None:
        customers = json.loads(Path("data/customers.json").read_text())

    # Build lookup maps
    cust_map = {c["customer_id"]: c for c in customers}
    claims_list = claims or []
    if not claims_list:
        claims_path = Path("data/claims.json")
        if claims_path.exists():
            claims_list = json.loads(claims_path.read_text())

    claim_map = {c["claim_id"]: (c, cust_map.get(c["customer_id"], {}))
                 for c in claims_list}

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    generated: list[Path] = []

    # Determine document type split
    n_decl = int(pdf_count * 0.45)
    n_claim = int(pdf_count * 0.35)
    n_renewal = pdf_count - n_decl - n_claim

    # Shuffle for variety
    shuffled_policies = policies.copy()
    random.shuffle(shuffled_policies)

    # Declaration pages
    for policy in shuffled_policies[:n_decl]:
        cust = cust_map.get(policy["customer_id"], {})
        if not cust:
            continue
        out = output_dir / f"decl_{policy['policy_number']}.pdf"
        try:
            _generate_declaration_page(policy, cust, out, styles)
            generated.append(out)
        except Exception as e:
            print(f"  [WARN] skipped decl_{policy['policy_number']}: {e}")

    # Claim letters
    claim_items = list(claim_map.items())
    random.shuffle(claim_items)
    for claim_id, (claim, cust) in claim_items[:n_claim]:
        if not cust:
            continue
        policy = next((p for p in policies if p["policy_number"] == claim["policy_number"]), None)
        if not policy:
            continue
        out = output_dir / f"claim_letter_{claim_id}.pdf"
        try:
            _generate_claim_letter(claim, policy, cust, out, styles)
            generated.append(out)
        except Exception as e:
            print(f"  [WARN] skipped claim_letter_{claim_id}: {e}")

    # Renewal notices
    renewal_policies = shuffled_policies[n_decl:n_decl + n_renewal]
    for policy in renewal_policies:
        cust = cust_map.get(policy["customer_id"], {})
        if not cust:
            continue
        out = output_dir / f"renewal_{policy['policy_number']}.pdf"
        try:
            _generate_renewal_notice(policy, cust, out, styles)
            generated.append(out)
        except Exception as e:
            print(f"  [WARN] skipped renewal_{policy['policy_number']}: {e}")

    return generated


def main(pdf_count: int, output_dir: Path, config: dict, states_data: dict) -> None:
    generated = generate(pdf_count, config, states_data, output_dir=output_dir)
    print(f"[document_gen] generated {len(generated):,} PDFs → {output_dir}/")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        pdf_count=200,
        output_dir=Path("documents"),
        config={"coverage_rules": coverage_rules},
        states_data=states_data,
    )

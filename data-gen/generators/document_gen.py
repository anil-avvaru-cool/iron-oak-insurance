"""
Document generator for AIOI.

Generates synthetic PDF documents (declarations, claim letters, renewals)
using reportlab.
"""

import json
import random
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_declaration_pdf(
    doc_path: Path, policy_number: str, customer_name: str, state: str
) -> None:
    """Generate a declaration page PDF."""
    doc = SimpleDocTemplate(
        str(doc_path),
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#003366"),
        spaceAfter=12,
        alignment=1,
    )
    elements.append(Paragraph("POLICY DECLARATION PAGE", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Info table
    info_data = [
        ["Policy Number:", policy_number],
        ["Insured Name:", customer_name],
        ["State:", state],
        ["Effective Date:", datetime.now().strftime("%Y-%m-%d")],
        ["Expiration Date:", (datetime.now() + __import__("datetime").timedelta(days=365)).strftime("%Y-%m-%d")],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(info_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Coverage info
    coverage_title = ParagraphStyle(
        "CoverageTitle", parent=styles["Heading2"], fontSize=12, spaceAfter=8
    )
    elements.append(Paragraph("COVERAGES", coverage_title))

    coverage_text = "This policy provides comprehensive coverage including liability, collision, comprehensive, and optional coverages as selected."
    elements.append(Paragraph(coverage_text, styles["Normal"]))

    doc.build(elements)


def generate_claim_letter_pdf(
    doc_path: Path, claim_id: str, policy_number: str, claim_amount: float
) -> None:
    """Generate a claim letter PDF."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Header
    elements.append(Paragraph("CLAIM LETTER", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Claim details
    claim_data = [
        ["Claim ID:", claim_id],
        ["Policy Number:", policy_number],
        ["Date:", datetime.now().strftime("%Y-%m-%d")],
        ["Claim Amount:", f"${claim_amount:,.2f}"],
    ]
    claim_table = Table(claim_data, colWidths=[2 * inch, 4 * inch])
    claim_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(claim_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Letter body
    body_text = "Your claim has been received and is being processed. An adjuster will contact you within 2-3 business days with an update."
    elements.append(Paragraph(body_text, styles["Normal"]))

    doc.build(elements)


def generate_renewal_pdf(
    doc_path: Path, policy_number: str, customer_name: str, premium: float
) -> None:
    """Generate a renewal notice PDF."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("POLICY RENEWAL NOTICE", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Renewal details
    renewal_data = [
        ["Policy Number:", policy_number],
        ["Insured Name:", customer_name],
        ["Current Premium:", f"${premium:,.2f}"],
        ["Renewal Effective:", (
            datetime.now() + __import__("datetime").timedelta(days=365)
        ).strftime("%Y-%m-%d")],
    ]
    renewal_table = Table(renewal_data, colWidths=[2 * inch, 4 * inch])
    renewal_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(renewal_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Renewal text
    renewal_text = "Your policy is eligible for renewal. Please review the enclosed documents and confirm your renewal preference."
    elements.append(Paragraph(renewal_text, styles["Normal"]))

    doc.build(elements)


def generate(count: int, config: dict, states_data: dict) -> int:
    """
    Generate PDF documents.

    Args:
        count: Number of documents to generate
        config: Configuration dict
        states_data: State rules dictionary

    Returns:
        Number of documents generated
    """
    # Load policies and claims
    data_dir = Path(__file__).parent.parent.parent / "data"
    try:
        with open(data_dir / "policies.json") as f:
            policies = json.load(f)
    except FileNotFoundError:
        policies = []

    try:
        with open(data_dir / "claims.json") as f:
            claims = json.load(f)
    except FileNotFoundError:
        claims = []

    random.seed(42)
    doc_counter = 0
    docs_dir = Path(__file__).parent.parent.parent / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Generate PDFs
    while doc_counter < count and (policies or claims):
        doc_type = random.choice(["declaration", "claim_letter", "renewal"])

        try:
            if doc_type == "declaration" and policies:
                policy = random.choice(policies)
                policy_num = policy["policy_number"]
                customer_id = policy["customer_id"]
                state = policy["state"]
                pdf_path = docs_dir / f"decl_{policy_num}.pdf"

                if not pdf_path.exists():
                    generate_declaration_pdf(
                        pdf_path,
                        policy_num,
                        f"Customer {customer_id}",
                        state,
                    )
                    doc_counter += 1

            elif doc_type == "claim_letter" and claims:
                claim = random.choice(claims)
                claim_id = claim["claim_id"]
                policy_num = claim["policy_number"]
                claim_amt = claim["claim_amount"]
                pdf_path = docs_dir / f"claim_letter_{claim_id}.pdf"

                if not pdf_path.exists():
                    generate_claim_letter_pdf(
                        pdf_path,
                        claim_id,
                        policy_num,
                        claim_amt,
                    )
                    doc_counter += 1

            elif doc_type == "renewal" and policies:
                policy = random.choice(policies)
                policy_num = policy["policy_number"]
                customer_id = policy["customer_id"]
                premium = policy["premium_annual"]
                pdf_path = docs_dir / f"renewal_{policy_num}.pdf"

                if not pdf_path.exists():
                    generate_renewal_pdf(
                        pdf_path,
                        policy_num,
                        f"Customer {customer_id}",
                        premium,
                    )
                    doc_counter += 1

        except Exception as e:
            print(f"Error generating {doc_type}: {e}")

    return doc_counter


def main(count: int, output_dir: Path, config: dict, states_data: dict) -> None:
    """
    Generate and write PDF documents.

    Args:
        count: Number of documents to generate
        output_dir: Directory to write PDFs
        config: Configuration dictionary
        states_data: State rules dictionary
    """
    num_generated = generate(count, config, states_data)
    print(f"[documents/] wrote {num_generated} PDFs → {output_dir}")


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    config = {"coverage_rules": coverage_rules}

    docs_dir = Path(__file__).parent.parent.parent / "documents"
    main(500, docs_dir, config, states_data)

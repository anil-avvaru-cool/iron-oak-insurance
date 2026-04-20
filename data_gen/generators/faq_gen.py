"""
faq_gen.py — generates the AIOI FAQ corpus from states.json and coverage_rules.json.

All five categories are fully implemented:
  coverage_concepts  — liability, collision, comprehensive, pip, UM, gap, roadside
  state_rules        — no-fault, min limits, total loss, PIP req, UM req (per state)
  claims_process     — filing through settlement
  costs_discounts    — premiums, drive score, telematics, discounts
  policy_management  — add vehicle/driver, coverage changes, renewal, SR-22

Module run:  uv run python -m data_gen.generators.faq_gen
Direct run:  uv run python data_gen/generators/faq_gen.py
"""
from __future__ import annotations

import json
from pathlib import Path

# ── Coverage concept templates (applicable_states: ["ALL"]) ──────────────────

_COVERAGE_TEMPLATES = [
    {
        "subcategory": "liability",
        "question": "What does liability insurance cover?",
        "answer": (
            "Liability insurance pays for injuries and property damage you cause to others in an "
            "accident where you are at fault. It is split into bodily injury (BI) and property "
            "damage (PD) components, expressed as split limits such as 30/60/25 — meaning $30,000 "
            "per person, $60,000 per accident, and $25,000 for property damage. It does not cover "
            "your own injuries or vehicle damage."
        ),
        "tags": ["liability","bodily-injury","property-damage","split-limits"],
    },
    {
        "subcategory": "liability",
        "question": "What are split limits on a liability policy?",
        "answer": (
            "Split limits express three separate maximums: per-person bodily injury, per-accident "
            "bodily injury, and property damage. For example, 25/50/10 means $25,000 maximum per "
            "injured person, $50,000 maximum for all injuries in one accident, and $10,000 for "
            "property damage. If one person's injuries exceed the per-person limit, the excess is "
            "your personal responsibility."
        ),
        "tags": ["liability","split-limits","bodily-injury"],
    },
    {
        "subcategory": "collision",
        "question": "When does collision coverage apply?",
        "answer": (
            "Collision coverage pays to repair or replace your vehicle when it collides with "
            "another vehicle or object — a guardrail, tree, or parked car — regardless of fault. "
            "You pay the deductible first; the insurer covers the rest up to the actual cash value "
            "(ACV) of the vehicle. It does not apply to theft, weather, or hitting an animal."
        ),
        "tags": ["collision","deductible","acv","fault"],
    },
    {
        "subcategory": "collision",
        "question": "What is the difference between collision and comprehensive?",
        "answer": (
            "Collision covers damage from hitting something — another vehicle, a pole, or rolling "
            "over. Comprehensive covers damage from everything else: theft, fire, hail, flood, "
            "fallen trees, vandalism, and hitting an animal. Both require a deductible. Together "
            "they are called 'full coverage,' though that is not an official insurance term."
        ),
        "tags": ["collision","comprehensive","full-coverage","deductible"],
    },
    {
        "subcategory": "comprehensive",
        "question": "What perils does comprehensive coverage cover?",
        "answer": (
            "Comprehensive covers non-collision losses including theft, fire, explosion, windstorm, "
            "hail, flood, falling objects, vandalism, and collision with an animal such as a deer. "
            "It pays actual cash value minus your deductible. It does not cover normal wear and "
            "tear, mechanical breakdown, or damage from collision with another vehicle."
        ),
        "tags": ["comprehensive","theft","hail","flood","acv"],
    },
    {
        "subcategory": "pip",
        "question": "What is Personal Injury Protection (PIP) coverage?",
        "answer": (
            "PIP covers medical expenses, lost wages, and related costs for you and your passengers "
            "after an accident, regardless of who was at fault. It is required in no-fault states "
            "such as MI, FL, NY, NJ, and PA. Benefit limits vary by state — Michigan offers the "
            "highest at up to $500,000, while Utah requires only $3,000 minimum."
        ),
        "tags": ["pip","no-fault","medical","lost-wages","coverage"],
    },
    {
        "subcategory": "pip",
        "question": "Does PIP cover passengers in my vehicle?",
        "answer": (
            "Yes. In states where PIP is required, it covers you, resident family members, and "
            "passengers in your vehicle regardless of fault. In some states it also covers you as "
            "a pedestrian struck by a vehicle. PIP does not cover vehicle damage — only medical "
            "and wage-related costs."
        ),
        "tags": ["pip","passengers","no-fault","medical"],
    },
    {
        "subcategory": "uninsured_motorist",
        "question": "What happens if the other driver has no insurance?",
        "answer": (
            "Uninsured Motorist (UM) coverage pays for your injuries and, in some states, your "
            "vehicle damage when the at-fault driver carries no insurance. Underinsured Motorist "
            "(UIM) coverage steps in when the at-fault driver's limits are too low to cover your "
            "losses. Without UM/UIM you would need to sue the at-fault driver directly, which is "
            "often impractical if they have no assets."
        ),
        "tags": ["uninsured-motorist","um","uim","at-fault","coverage"],
    },
    {
        "subcategory": "gap",
        "question": "When does gap insurance matter?",
        "answer": (
            "Gap insurance covers the difference between what your insurer pays (actual cash value) "
            "and what you still owe on your auto loan or lease if your vehicle is totaled or stolen. "
            "New vehicles depreciate quickly — in the first year a car can lose 15–25% of its value "
            "while the loan balance drops much more slowly. Gap is most valuable in the first "
            "2–3 years of ownership and is typically not needed once the loan balance falls below "
            "the vehicle's ACV."
        ),
        "tags": ["gap","total-loss","loan","lease","acv","depreciation"],
    },
    {
        "subcategory": "roadside",
        "question": "What is included in roadside assistance coverage?",
        "answer": (
            "Roadside assistance typically covers towing to the nearest qualified repair facility, "
            "battery jump-starts, flat tire changes (using your spare), lockout service, and fuel "
            "delivery for an empty tank. Some policies include winching if the vehicle is stuck "
            "off-road. Roadside is low-cost and worth adding if you do not already have it through "
            "a membership program."
        ),
        "tags": ["roadside","towing","battery","lockout","flat-tire"],
    },
]

# ── Claims process templates (applicable_states: ["ALL"]) ───────────────────

_CLAIMS_TEMPLATES = [
    {
        "subcategory": "filing",
        "question": "How do I report an accident to AIOI?",
        "answer": (
            "You can file a claim through Oak Assist (our AI intake agent), the AIOI mobile app, "
            "or by calling our claims line. You will need your policy number, the date and location "
            "of the incident, a description of what happened, and contact information for any other "
            "parties involved. Filing within 24–48 hours is strongly recommended — delays can "
            "complicate investigation."
        ),
        "tags": ["filing","fnol","claim","oak-assist","report"],
    },
    {
        "subcategory": "filing",
        "question": "What is FNOL?",
        "answer": (
            "FNOL stands for First Notice of Loss — the initial report you file when a covered "
            "incident occurs. It starts the claims process and assigns a claim number. Oak Assist "
            "handles FNOL intake for straightforward claims and escalates complex situations to a "
            "human adjuster. Providing accurate information at FNOL speeds up your settlement."
        ),
        "tags": ["fnol","filing","claims-process","oak-assist"],
    },
    {
        "subcategory": "after_filing",
        "question": "What happens after I file a claim?",
        "answer": (
            "After filing, AIOI acknowledges receipt within the state-mandated window (typically "
            "7–15 days). An adjuster is assigned to investigate, which may include inspecting "
            "your vehicle, reviewing photos, and contacting other parties. You receive a coverage "
            "determination and, if approved, a settlement offer. Complex or disputed claims take "
            "longer; straightforward claims are often settled within 30 days."
        ),
        "tags": ["after-filing","adjuster","investigation","settlement","timeline"],
    },
    {
        "subcategory": "documentation",
        "question": "What documents do I need to support my claim?",
        "answer": (
            "Useful documentation includes: photos of all vehicle damage and the accident scene, "
            "a police report if one was filed, contact and insurance information from all parties, "
            "witness names and contact info, medical records and bills for injury claims, and "
            "repair estimates. The more documentation you provide upfront, the faster the "
            "adjuster can process your claim."
        ),
        "tags": ["documentation","police-report","photos","records","adjuster"],
    },
    {
        "subcategory": "adjuster",
        "question": "What does a claims adjuster do?",
        "answer": (
            "A claims adjuster investigates the incident, determines coverage, assesses the value "
            "of the loss, and negotiates a settlement. They may inspect your vehicle in person or "
            "use photos and repair estimates. The adjuster verifies that the loss is covered under "
            "your policy, checks for fraud indicators, and calculates the payout after applying "
            "your deductible."
        ),
        "tags": ["adjuster","investigation","coverage","settlement","deductible"],
    },
    {
        "subcategory": "rental",
        "question": "Does my policy cover a rental car while mine is being repaired?",
        "answer": (
            "Rental reimbursement coverage pays for a rental car while your vehicle is being "
            "repaired due to a covered claim. It is an optional add-on with a daily and total "
            "cap — for example $40/day up to $1,200. Check your declarations page under "
            "'rental reimbursement' to see if it is included and what your limits are."
        ),
        "tags": ["rental","reimbursement","coverage","repair","declarations"],
    },
    {
        "subcategory": "total_loss_process",
        "question": "What happens if my car is totaled?",
        "answer": (
            "If repair costs exceed your state's total loss threshold (as a percentage of actual "
            "cash value), AIOI declares the vehicle a total loss. We pay you the ACV of your "
            "vehicle at the time of loss, minus your deductible. You sign over the title to AIOI. "
            "If you have gap coverage and owe more than the ACV on a loan, gap pays the difference. "
            "State thresholds range from 60% (Oklahoma) to 100% (several states including TX and AZ)."
        ),
        "tags": ["total-loss","acv","gap","threshold","title"],
    },
    {
        "subcategory": "settlement",
        "question": "How is a claim settlement calculated?",
        "answer": (
            "Settlement is based on the actual cash value (ACV) of the loss, minus your deductible. "
            "ACV is the replacement cost of a like-kind vehicle or repair, adjusted for depreciation "
            "and condition. For vehicle damage, we use market data and repair shop estimates. "
            "For total losses, we use vehicle valuation guides. You can negotiate if you disagree "
            "with the ACV — provide comparable vehicles or independent appraisals as evidence."
        ),
        "tags": ["settlement","acv","deductible","depreciation","negotiation"],
    },
]

# ── Costs and discounts templates (applicable_states: ["ALL"]) ───────────────

_COSTS_TEMPLATES = [
    {
        "subcategory": "premium_calc",
        "question": "What factors affect my insurance premium?",
        "answer": (
            "Key rating factors include: your driving record (accidents and violations), age and "
            "years of experience, the vehicle's make, model, year, and safety ratings, your annual "
            "mileage, where the vehicle is garaged (state and ZIP code), your credit score (in most "
            "states), and the coverages and deductibles you select. Telematics enrollment through "
            "the Iron Oak Drive Score program can provide additional discounts based on actual "
            "driving behavior."
        ),
        "tags": ["premium","rating-factors","driving-record","credit","telematics"],
    },
    {
        "subcategory": "drive_score",
        "question": "What is the Iron Oak Drive Score and how does it affect my rate?",
        "answer": (
            "The Iron Oak Drive Score (0–100) is calculated from telematics data including hard "
            "braking, rapid acceleration, speeding events, and night driving percentage — normalized "
            "per 10 miles driven. Scores of 90+ earn up to a 15% discount. Scores 75–89 earn 8%, "
            "scores 60–74 earn 3%, and scores below 60 receive no telematics discount. Your score "
            "is recalculated each renewal period based on the most recent 12 months of trip data."
        ),
        "tags": ["drive-score","telematics","discount","ubi","hard-braking"],
    },
    {
        "subcategory": "telematics",
        "question": "What driving data does the telematics program collect?",
        "answer": (
            "The Iron Oak telematics program collects trip-level data including distance driven, "
            "trip duration, hard braking events, rapid acceleration events, speeding events, and "
            "the percentage of miles driven at night (10 PM–5 AM). Location data is used only to "
            "detect trip boundaries and is not stored long-term. Data collection requires the "
            "Iron Oak mobile app or an OBD-II device."
        ),
        "tags": ["telematics","data","privacy","ubi","collection"],
    },
    {
        "subcategory": "discounts",
        "question": "What discounts does AIOI offer?",
        "answer": (
            "AIOI offers discounts for: safe driver (clean record for 3+ years), Iron Oak Drive "
            "Score (telematics enrollment), multi-policy (bundling auto with another AIOI product), "
            "paid-in-full (paying annual premium upfront), anti-theft devices, defensive driving "
            "course completion, good student (GPA 3.0+), and paperless billing. Not all discounts "
            "are available in every state."
        ),
        "tags": ["discounts","safe-driver","multi-policy","telematics","good-student"],
    },
    {
        "subcategory": "multi_policy",
        "question": "Does bundling policies with AIOI save money?",
        "answer": (
            "Yes. Customers with two or more AIOI policies (for example, two vehicles, or auto "
            "plus renters) typically receive a 5–12% multi-policy discount on each policy. The "
            "discount is applied automatically when the policies share the same named insured. "
            "Contact your agent or log in to verify the discount appears on your declarations page."
        ),
        "tags": ["multi-policy","bundle","discount","savings"],
    },
    {
        "subcategory": "good_driver",
        "question": "How is a good driver discount earned and maintained?",
        "answer": (
            "The good driver discount applies when the primary driver has had no at-fault "
            "accidents, major violations (DUI, reckless driving), or more than one minor violation "
            "in the past three years. The discount is re-evaluated at each renewal. A single "
            "at-fault accident typically removes the discount for three years from the accident date."
        ),
        "tags": ["good-driver","discount","violations","at-fault","renewal"],
    },
    {
        "subcategory": "credit",
        "question": "Does my credit score affect my insurance rate?",
        "answer": (
            "In most states, insurers use a credit-based insurance score — distinct from your "
            "lending credit score — as a rating factor. Research shows it correlates with claim "
            "frequency. States that prohibit its use include California, Hawaii, Massachusetts, "
            "and Michigan. If your credit improves significantly, you can request a re-rate at "
            "renewal. AIOI uses credit as one factor among many, not as the sole determinant."
        ),
        "tags": ["credit","insurance-score","rating","state-rule"],
    },
]

# ── Policy management templates (applicable_states: ["ALL"]) ────────────────

_POLICY_MGMT_TEMPLATES = [
    {
        "subcategory": "add_vehicle",
        "question": "How do I add a vehicle to my policy?",
        "answer": (
            "Contact AIOI or log in to your account to add a vehicle. You will need the VIN, "
            "year, make, model, and current odometer reading. Coverage for the new vehicle begins "
            "as of the add date. Your premium is prorated for the remaining policy term. If you "
            "replace an existing vehicle, the old vehicle is removed from the date of sale."
        ),
        "tags": ["add-vehicle","policy-change","vin","premium","mid-term"],
    },
    {
        "subcategory": "add_driver",
        "question": "What happens when I add a teenage driver to my policy?",
        "answer": (
            "Adding a teenage driver typically increases your premium, as young drivers have "
            "statistically higher claim rates. The increase varies by age (16–19), gender, and "
            "state. A good student discount (GPA 3.0+) can offset some of the increase. Teens "
            "can also enroll in the Drive Score program — good scores earn discounts and "
            "reinforce safe habits. The teen must be added; driving on the policy without being "
            "listed can result in a coverage denial."
        ),
        "tags": ["add-driver","teen","premium","good-student","drive-score"],
    },
    {
        "subcategory": "coverage_changes",
        "question": "Can I change my coverage mid-term?",
        "answer": (
            "Yes. Most coverage changes can be made mid-term — adding or removing optional "
            "coverages, changing deductibles, or updating limits. Changes take effect on the "
            "requested date and your premium is adjusted pro-rata. Some changes (such as removing "
            "collision on a financed vehicle) may be restricted by your lender. Contact AIOI or "
            "make changes through the online portal."
        ),
        "tags": ["coverage-change","mid-term","deductible","pro-rata","lender"],
    },
    {
        "subcategory": "renewal_lapse",
        "question": "What happens if I miss a premium payment?",
        "answer": (
            "AIOI provides a grace period (typically 10–30 days depending on your state) after "
            "the due date before the policy lapses for non-payment. During the grace period you "
            "remain covered but should pay immediately to avoid cancellation. A lapsed policy "
            "means no coverage — any claims filed after the lapse date will be denied. "
            "Reinstating after a lapse may require a new application and premium recalculation."
        ),
        "tags": ["payment","grace-period","lapse","cancellation","reinstatement"],
    },
    {
        "subcategory": "cancellation",
        "question": "How do I cancel my AIOI policy?",
        "answer": (
            "To cancel, contact AIOI by phone, mail, or through your agent. You will need your "
            "policy number and the desired cancellation date. If you have paid ahead, you will "
            "receive a prorated refund for the unused premium (minus any short-rate fee if "
            "cancelling mid-term at your request). Cancellation without replacement coverage "
            "creates a coverage gap that can raise future premiums."
        ),
        "tags": ["cancellation","refund","pro-rata","coverage-gap","policy"],
    },
    {
        "subcategory": "sr22",
        "question": "What is an SR-22 and when is it required?",
        "answer": (
            "An SR-22 is a certificate of financial responsibility filed by your insurer with "
            "your state's DMV confirming you carry the minimum required coverage. It is typically "
            "required after a DUI/DWI conviction, serious traffic violation, license suspension, "
            "or being caught driving uninsured. The filing requirement usually lasts 3 years. "
            "Not all insurers file SR-22s — AIOI does file in all states where we operate. "
            "An SR-22 itself is not insurance; it is proof of insurance."
        ),
        "tags": ["sr22","dui","suspension","financial-responsibility","dmv"],
    },
]


def _make_record(
    idx: int,
    subcategory: str,
    category: str,
    template: dict,
    applicable_states: list[str],
) -> dict:
    slug = subcategory.replace("_", "-")
    return {
        "faq_id": f"faq-{slug}-{idx:03d}",
        "category": category,
        "subcategory": subcategory,
        "question": template["question"],
        "answer": template["answer"],
        "applicable_states": applicable_states,
        "tags": template["tags"],
        "source": "synthetic-faq-v1",
        "version": "1.0",
    }


def generate_coverage_faqs() -> list[dict]:
    """Generic coverage concept FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "coverage_concepts", t, ["ALL"])
        for i, t in enumerate(_COVERAGE_TEMPLATES)
    ]


def generate_claims_faqs() -> list[dict]:
    """Claims process FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "claims_process", t, ["ALL"])
        for i, t in enumerate(_CLAIMS_TEMPLATES)
    ]


def generate_costs_faqs() -> list[dict]:
    """Cost and discount FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "costs_discounts", t, ["ALL"])
        for i, t in enumerate(_COSTS_TEMPLATES)
    ]


def generate_policy_mgmt_faqs() -> list[dict]:
    """Policy management FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "policy_management", t, ["ALL"])
        for i, t in enumerate(_POLICY_MGMT_TEMPLATES)
    ]


def generate_state_faqs(states_data: dict) -> list[dict]:
    """
    State-specific FAQs generated directly from states.json.
    Produces: no-fault explainer, total loss threshold, min liability,
    PIP requirement, and UM requirement per applicable state.
    """
    records: list[dict] = []
    idx = 1

    for state, rules in states_data.items():
        # No-fault explainer
        if rules.get("no_fault"):
            pip_limit = rules.get("pip_limit") or 0
            records.append({
                "faq_id": f"faq-nofault-{idx:03d}",
                "category": "state_rules",
                "subcategory": "no_fault",
                "question": f"Is {state} a no-fault state?",
                "answer": (
                    f"Yes, {state} is a no-fault state. After an accident, your own PIP "
                    f"coverage pays your medical expenses up to the state limit "
                    f"(${pip_limit:,}) regardless of who caused the accident. Your ability "
                    f"to sue the at-fault driver for pain and suffering is restricted unless "
                    f"your injuries meet a defined threshold."
                ),
                "applicable_states": [state],
                "tags": ["no-fault", "pip", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # Total loss threshold
        tlt = rules.get("total_loss_threshold")
        if tlt:
            records.append({
                "faq_id": f"faq-totalloss-{idx:03d}",
                "category": "state_rules",
                "subcategory": "total_loss",
                "question": f"At what damage percentage is a car declared a total loss in {state}?",
                "answer": (
                    f"In {state}, a vehicle is declared a total loss when repair costs reach "
                    f"{int(tlt * 100)}% or more of the vehicle's actual cash value (ACV)."
                ),
                "applicable_states": [state],
                "tags": ["total-loss", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # Minimum liability limits
        ml = rules.get("min_liability", {})
        if ml:
            bi_pp = ml.get("bodily_injury_per_person", 0)
            bi_pa = ml.get("bodily_injury_per_accident", 0)
            pd = ml.get("property_damage", 0)
            records.append({
                "faq_id": f"faq-minliability-{idx:03d}",
                "category": "state_rules",
                "subcategory": "minimum_liability",
                "question": f"What are the minimum liability insurance limits required in {state}?",
                "answer": (
                    f"{state} requires minimum liability limits of ${bi_pp:,} per person / "
                    f"${bi_pa:,} per accident for bodily injury and ${pd:,} for property damage "
                    f"(expressed as {bi_pp // 1000}/{bi_pa // 1000}/{pd // 1000}). These are "
                    f"minimums only — most drivers benefit from higher limits to protect their assets."
                ),
                "applicable_states": [state],
                "tags": ["minimum-liability", "state-rule", "split-limits", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # PIP requirement
        if rules.get("pip_required"):
            pip_limit = rules.get("pip_limit") or 0
            records.append({
                "faq_id": f"faq-pipreq-{idx:03d}",
                "category": "state_rules",
                "subcategory": "pip_requirements",
                "question": f"Is PIP coverage required in {state}?",
                "answer": (
                    f"Yes, Personal Injury Protection (PIP) is mandatory in {state} with a minimum "
                    f"benefit limit of ${pip_limit:,}. Every policy issued in {state} must include PIP."
                ),
                "applicable_states": [state],
                "tags": ["pip", "required", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # UM requirement
        if rules.get("uninsured_motorist_required"):
            records.append({
                "faq_id": f"faq-umreq-{idx:03d}",
                "category": "state_rules",
                "subcategory": "um_coverage",
                "question": f"Is uninsured motorist coverage required in {state}?",
                "answer": (
                    f"Yes, uninsured motorist (UM) coverage is mandatory in {state}. "
                    f"It protects you when you are injured by a driver who has no insurance. "
                    f"Your UM limits must match your liability limits unless you sign a waiver "
                    f"to select lower limits."
                ),
                "applicable_states": [state],
                "tags": ["uninsured-motorist", "required", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

    return records


def generate(states_data: dict) -> list[dict]:
    """Generate the full FAQ corpus. Returns all records combined."""
    return (
        generate_coverage_faqs()
        + generate_claims_faqs()
        + generate_costs_faqs()
        + generate_policy_mgmt_faqs()
        + generate_state_faqs(states_data)
    )


def main(
    output_path: Path | None = None,
    states_data: dict | None = None,
) -> list[dict]:
    config_dir = Path("data_gen/config")
    if states_data is None:
        states_data = json.loads((config_dir / "states.json").read_text())
    if output_path is None:
        output_path = Path("faqs/faq_corpus.json")

    records = generate(states_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"[faq_gen] wrote {len(records):,} FAQ records → {output_path}")
    return records


if __name__ == "__main__":
    main()
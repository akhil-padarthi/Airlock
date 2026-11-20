"""Synthetic 1,000+ page contract with planted facts, for demos and evaluation.

Generates a realistic, boilerplate-heavy Master Services Agreement where 15
specific facts are buried on known pages, plus a question set with gold page
numbers and 8 questions the document cannot answer. Requires `reportlab`.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

DEFAULT_PAGES = 1040


@dataclass
class Fact:
    page: int
    heading: str
    paragraph: str
    questions: list[tuple[str, list[str]]]   # (question, any-of answer strings)


FACTS = [
    Fact(37, "Initial Term and Renewal",
         "The relationship between the parties is intended to be stable and long-lived. The initial term of "
         "this Agreement is thirty-six (36) months from the Effective Date, after which it renews automatically "
         "for successive twelve (12) month renewal terms unless either party gives notice of non-renewal. Each "
         "renewal term remains subject to the fee provisions of this Agreement.",
         [("How long is the initial term of the agreement?", ["36", "thirty-six", "three years", "3 years"])]),
    Fact(112, "Payment Terms",
         "Customer shall pay each undisputed invoice within forty-five (45) days of receipt of a valid invoice. "
         "Amounts not paid when due accrue a late charge of one and one-half percent (1.5%) per month until paid. "
         "Disputed amounts shall be identified in writing with reasonable supporting detail.",
         [("Within how many days must invoices be paid?", ["45", "forty-five"]),
          ("What late charge applies to overdue amounts?", ["1.5"])]),
    Fact(205, "Service Level Commitment",
         "Provider shall make the Hosted Services available at least 99.95% of the time in each calendar month, "
         "excluding scheduled maintenance windows. For each 0.1% by which monthly availability falls below this "
         "commitment, Customer is entitled to a service credit equal to ten percent (10%) of the monthly fees for "
         "the affected service.",
         [("What monthly uptime does the provider commit to?", ["99.95"]),
          ("What service credit does the customer get when availability falls short?", ["10%", "ten percent", "10 percent"])]),
    Fact(288, "Security Incident Notification",
         "Provider shall notify Customer in writing of any confirmed Security Incident affecting Customer Data "
         "within forty-eight (48) hours of discovery. The notice shall describe the nature of the incident, the "
         "categories of data involved and the remediation steps taken or planned.",
         [("How quickly must the provider notify the customer of a security incident?", ["48", "forty-eight"])]),
    Fact(351, "Limitation of Liability",
         "Except for the excluded claims, each party's aggregate liability arising out of this Agreement shall "
         "not exceed the fees paid in the twelve (12) months preceding the claim, and in no event shall it exceed "
         "$4,250,000. Neither party shall be liable for indirect, consequential or punitive damages.",
         [("What is the maximum aggregate liability under the agreement?", ["4,250,000", "4.25 million"])]),
    Fact(433, "Insurance Requirements",
         "Throughout the term, Provider shall maintain cyber liability insurance with limits of not less than "
         "$10,000,000 per claim, together with commercial general liability coverage customary in the industry. "
         "Certificates of insurance shall be furnished to Customer upon request.",
         [("How much cyber liability insurance must the provider carry?", ["10,000,000", "10 million"])]),
    Fact(517, "Governing Law and Venue",
         "This Agreement is governed by the laws of the State of Delaware, without regard to its conflict of laws "
         "principles. The state and federal courts located in Wilmington, Delaware have exclusive jurisdiction "
         "over any dispute arising under this Agreement.",
         [("Which state's law governs the agreement?", ["Delaware"])]),
    Fact(602, "Non-Solicitation of Personnel",
         "During the term and for eighteen (18) months thereafter, neither party shall knowingly solicit for "
         "employment any employee of the other party who was directly involved in performing the Services. "
         "General advertisements not targeted at such employees do not breach this restriction.",
         [("How long after the agreement ends does the non-solicitation restriction last?", ["18", "eighteen"])]),
    Fact(689, "Audit Rights",
         "Customer may audit Provider's compliance with this Agreement no more than once per calendar year, upon "
         "at least thirty (30) days' prior written notice and during normal business hours. Each party bears its "
         "own costs of any audit unless a material breach is discovered.",
         [("How often may the customer audit the provider?", ["once", "annual", "one time"])]),
    Fact(744, "Force Majeure",
         "A party affected by a Force Majeure Event is excused from performance for the duration of the event. "
         "If a Force Majeure Event continues for more than one hundred twenty (120) consecutive days, the "
         "unaffected party may terminate the affected Services upon written notice.",
         [("After how many days of a force majeure event can the other party terminate?", ["120", "one hundred twenty"])]),
    Fact(811, "Fee Adjustments",
         "Provider may increase the fees at the start of each renewal term, provided that any such increase shall "
         "not exceed four percent (4%) over the fees in effect for the preceding term. Fee changes require sixty "
         "(60) days' advance written notice.",
         [("By how much can the provider raise fees at renewal?", ["4%", "four percent", "4 percent"])]),
    Fact(873, "Termination for Convenience",
         "This Section governs how the relationship may be wound down without cause. Either party may terminate "
         "this Agreement for convenience upon ninety (90) days' prior written notice to the other party. Upon "
         "such termination, Customer shall pay all undisputed fees accrued through the effective date.",
         [("How much notice is required to terminate the agreement for convenience?", ["90", "ninety"])]),
    Fact(921, "Subprocessors",
         "Provider shall give Customer at least fifteen (15) business days' notice before engaging any new "
         "subprocessor that will process Customer Data, during which Customer may object on reasonable data "
         "protection grounds.",
         [("How much notice must be given before a new subprocessor is engaged?", ["15", "fifteen"])]),
    Fact(978, "Return and Deletion of Customer Data",
         "Within thirty (30) days after termination or expiration, Provider shall return Customer Data in a "
         "commonly used format and thereafter securely delete all remaining copies, certifying the deletion in "
         "writing upon request.",
         [("How soon after termination must customer data be returned and deleted?", ["30", "thirty"])]),
    Fact(1012, "Survival of Confidentiality",
         "The confidentiality obligations in this Agreement survive for five (5) years after termination or "
         "expiration, except that obligations regarding trade secrets continue for as long as the information "
         "remains a trade secret under applicable law.",
         [("How long do confidentiality obligations survive after termination?", ["5", "five"])]),
]

UNANSWERABLE = [
    "What is the annual salary of the Provider's chief executive officer?",
    "Where are the Provider's corporate headquarters located?",
    "How many employees does the Customer have?",
    "On which stock exchange is the Provider listed?",
    "In what year was the Provider founded?",
    "What parental leave policy applies to Provider personnel?",
    "In which cloud region is customer data hosted?",
    "What penalty applies to late delivery of hardware?",
]

PARTIES = ["Provider", "Customer", "each party", "the Project Manager", "the Steering Committee",
           "the Account Executive", "the receiving party"]
DOCS = ["the Statement of Work", "the applicable Order Form", "Exhibit B", "the Project Plan", "Schedule 4",
        "the Service Description", "the Governance Charter", "Annex C"]
ADJ = ["reasonable", "commercially reasonable", "good faith", "diligent", "timely", "prompt", "proper"]
WHEN = ["promptly", "without undue delay", "as soon as practicable", "in accordance with the Project Plan",
        "at the next governance meeting", "within a reasonable period"]

FAMILIES: dict[str, list[str]] = {
    "Deliverables": [
        "{P} shall prepare each Deliverable described in {D} using {A} efforts and qualified resources.",
        "Each Deliverable shall conform in all material respects to the specifications set out in {D}.",
        "Draft Deliverables may be circulated for comment, and {P} shall consider comments {W}.",
        "Where a Deliverable depends on inputs from {P}, the delivery schedule adjusts to reflect any delay in those inputs.",
        "Deliverables shall be provided in the electronic formats reasonably requested by the receiving team.",
        "Interim work product is provided for convenience only and is not a Deliverable unless {D} says otherwise.",
    ],
    "Acceptance": [
        "Upon delivery, {P} shall review the Deliverable against the acceptance criteria in {D}.",
        "If a Deliverable fails to meet the acceptance criteria, the reviewer shall describe each nonconformity in writing.",
        "Provider shall correct identified nonconformities {W} and resubmit the Deliverable for review.",
        "Acceptance shall not be unreasonably withheld, conditioned or delayed by {P}.",
        "Use of a Deliverable in a production environment constitutes acceptance unless otherwise agreed in {D}.",
        "Acceptance testing shall be carried out in a test environment that reasonably represents live conditions.",
    ],
    "Change Control": [
        "Either party may request a change to the scope of the Services by submitting a change request to {P}.",
        "Each change request shall describe the proposed change, its rationale and its expected impact on {D}.",
        "Provider shall assess each change request {W} and provide an impact analysis covering schedule and resources.",
        "No change is binding until it is documented in a change order signed by authorised representatives.",
        "Pending agreement on a change, the parties shall continue to perform in accordance with {D}.",
        "Minor operational adjustments that do not affect scope or schedule may be agreed by {P} by email.",
    ],
    "Governance": [
        "The parties shall establish a governance structure consisting of operational and executive tiers.",
        "{P} shall chair regular review meetings and circulate minutes {W}.",
        "Governance meetings shall review progress against {D}, open risks and pending decisions.",
        "Each party shall appoint representatives with sufficient authority to make decisions at governance meetings.",
        "The Steering Committee shall resolve matters referred to it by the operational tier using {A} judgement.",
        "Changes to governance membership shall be communicated to the other party in writing.",
    ],
    "Personnel": [
        "Provider shall assign personnel with the skills and experience appropriate to the Services.",
        "{P} may request the replacement of any individual whose performance is unsatisfactory, acting in a {A} manner.",
        "Provider remains responsible for the acts and omissions of its personnel performing the Services.",
        "Key personnel identified in {D} shall not be reassigned without prior consultation with Customer.",
        "Provider personnel working at Customer sites shall comply with site rules communicated to them in advance.",
        "Each party is solely responsible for the compensation and benefits of its own personnel.",
    ],
    "Documentation": [
        "Provider shall maintain documentation describing the configuration and operation of the Services.",
        "Documentation shall be updated {W} following any material change to the Services.",
        "{P} may reproduce the documentation for internal use in connection with the Services.",
        "Documentation shall be written in clear language suitable for trained operational staff.",
        "Where documentation conflicts with {D}, the provisions of {D} prevail.",
        "Runbooks shall be stored in a repository accessible to authorised personnel of both parties.",
    ],
    "Training": [
        "Provider shall deliver training sessions to Customer's designated users as described in {D}.",
        "Training materials shall reflect the version of the Services deployed in the production environment.",
        "{P} shall coordinate training logistics, including scheduling and attendance, {W}.",
        "Additional training beyond the agreed scope may be requested through the change control procedure.",
        "Training shall be delivered remotely unless the parties agree that on-site delivery is more effective.",
        "Customer may record training sessions for internal reference by its own personnel.",
    ],
    "Escalation": [
        "Operational issues shall first be raised with the responsible service manager of the other party.",
        "If an issue is not resolved at the operational level, {P} may escalate it to the Steering Committee.",
        "Escalated matters shall be addressed {W} and in a spirit of cooperation.",
        "The parties shall use {A} efforts to resolve disputes informally before pursuing formal remedies.",
        "Escalation does not relieve either party of its obligation to continue performing under {D}.",
        "A written summary of each escalated matter and its resolution shall be kept by {P}.",
    ],
    "Reporting": [
        "Provider shall deliver periodic status reports summarising activities, risks and upcoming milestones.",
        "Reports shall be prepared in the format set out in {D} or as otherwise agreed by {P}.",
        "{P} shall review each report and raise any questions {W}.",
        "Operational dashboards shall be made available to authorised Customer personnel through the portal.",
        "Reports shall identify any dependency on Customer that may affect delivery of the Services.",
        "Ad hoc reports may be requested where reasonably necessary for Customer's internal governance.",
    ],
    "Policy Compliance": [
        "Each party shall comply with applicable laws in performing its obligations under this Agreement.",
        "Provider shall comply with Customer policies provided to it in writing, as described in {D}.",
        "{P} shall notify the other party {W} if it becomes aware of any conflict between a policy and applicable law.",
        "Policy updates take effect for Provider only after reasonable notice and an opportunity to assess their impact.",
        "Compliance obligations shall be interpreted in a {A} manner consistent with industry practice.",
        "Each party shall maintain internal controls designed to support its compliance obligations.",
    ],
    "Records": [
        "Provider shall keep accurate records relating to the performance of the Services.",
        "Records shall be retained for the period required by applicable law or by {D}, whichever is longer.",
        "{P} shall make relevant records available for inspection {W} upon reasonable request.",
        "Records may be maintained in electronic form provided that their integrity is preserved.",
        "Each party shall take {A} steps to protect records against unauthorised alteration.",
        "Archived records shall be indexed so that they can be located efficiently when required.",
    ],
    "Cooperation": [
        "The parties shall cooperate in good faith to achieve the objectives of this Agreement.",
        "Customer shall provide timely access to information, systems and facilities reasonably required by Provider.",
        "{P} shall inform the other party {W} of any circumstance likely to affect performance.",
        "Each party shall act in a {A} manner when exercising discretion under {D}.",
        "Third parties engaged by Customer shall be instructed to cooperate reasonably with Provider.",
        "Neither party shall unreasonably delay a decision that the other party requires to proceed.",
    ],
    "Intellectual Property": [
        "Each party retains ownership of the intellectual property it owned before the Effective Date.",
        "Provider grants Customer a licence to use the Deliverables for its internal business purposes.",
        "{P} shall not remove any proprietary notices from materials provided under {D}.",
        "Improvements to Provider's pre-existing tools remain the property of Provider.",
        "Customer materials remain the property of Customer and may be used only to perform the Services.",
        "Open source components shall be identified in {D} together with their applicable licences.",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    s = template.format(P=rng.choice(PARTIES), D=rng.choice(DOCS), A=rng.choice(ADJ), W=rng.choice(WHEN))
    return s[0].upper() + s[1:]


def _paragraph(family: str, rng: random.Random, n: int) -> str:
    templates = FAMILIES[family]
    return " ".join(_fill(t, rng) for t in rng.sample(templates, k=min(n, len(templates))))


def fact_pages(total_pages: int) -> list[int]:
    """Spread the planted facts over the document, scaling for shorter test PDFs."""
    pages, used = [], set()
    for f in FACTS:
        p = max(2, min(total_pages, round(f.page * total_pages / DEFAULT_PAGES)))
        while p in used:
            p += 1
        used.add(p)
        pages.append(p)
    return pages


def build_sample_pdf(pages: int = DEFAULT_PAGES, seed: int = 7) -> tuple[bytes, list[dict]]:
    """Return (pdf_bytes, questions). Questions carry gold pages for evaluation."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    rng = random.Random(seed)
    placed = dict(zip(fact_pages(pages), FACTS))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin, leading, font, size = 72, 13.5, "Helvetica", 10.5
    family_names = list(FAMILIES)

    for page_no in range(1, pages + 1):
        c.setFont("Helvetica", 8)
        c.drawString(margin, height - 40, "Helios Dynamics / Northwind Logistics - Master Services Agreement - Confidential")
        c.drawRightString(width - margin, 36, f"Page {page_no} of {pages}")

        fact = placed.get(page_no)
        family = family_names[(page_no * 7 + rng.randint(0, 3)) % len(family_names)]
        section = f"Section {page_no // 20 + 1}.{page_no % 20 + 1}"
        heading = f"{section} {fact.heading if fact else family}"
        paragraphs = [_paragraph(family, rng, rng.randint(4, 6))]
        if fact:
            paragraphs.append(fact.paragraph)
            paragraphs.append(_paragraph(family_names[(page_no + 5) % len(family_names)], rng, 4))
        else:
            other = family_names[(page_no * 3 + 1) % len(family_names)]
            paragraphs.append(_paragraph(family, rng, rng.randint(4, 6)))
            paragraphs.append(_paragraph(other, rng, rng.randint(4, 5)))

        y = height - 80
        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(margin, y, heading)
        y -= leading * 1.8
        c.setFont(font, size)
        for para in paragraphs:
            for line in simpleSplit(para, font, size, width - 2 * margin):
                if y < 60:
                    break
                c.drawString(margin, y, line)
                y -= leading
            y -= leading * 0.8
        c.showPage()
    c.save()

    questions: list[dict] = []
    for page, fact in placed.items():
        for q, contains in fact.questions:
            questions.append({"question": q, "answerable": True, "pages": [page], "answer_contains": contains})
    questions.extend({"question": q, "answerable": False, "pages": [], "answer_contains": []}
                     for q in UNANSWERABLE)
    return buf.getvalue(), questions

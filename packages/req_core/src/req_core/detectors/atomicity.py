"""atomicity_split: one clause number carrying several separable obligations.

"The Contractor shall supply and install the cameras, and shall commission each within
thirty days" is three obligations wearing one number. It matters because everything
downstream is keyed on the number: the compliance matrix has one row, the bidder writes one
YES, and two of the three obligations are never separately priced, scheduled or checked.

**This detector makes no model call.** The split has already happened — `req_core.extraction`
asked the model to separate a compound clause and Python built the children with lineage.
Asking again would pay twice for the same answer and risk getting a different one. The
detector reads `Requirement.parent_id` and reports what the extraction already concluded.

That is worth being explicit about, because it is the shape the rest of this engine wants:
the model was asked one question about the text, once, and every consumer of that answer is
Python.

Confidence, and why: the number rises with how many obligations came out, because a clause
that separates into four is less likely to be an over-eager split than one that separates
into two. 0.7 for two, 0.8 for three, 0.9 for four or more. The split itself is not in doubt
— it is recorded lineage — so the confidence is about whether separating was *right*, which
is a judgement the number leaves room to disagree with.

Deliberately does not: split anything (that is `extraction`), fire on a clause that produced
one child or none, or judge whether the children are individually well-formed — each child
is a requirement in its own right and every other detector sees it.

Belongs in `req_core`: a compound requirement is a compound requirement in any document that
numbers its clauses.
"""

from req_core.detectors.base import DetectorContext, FindingDraft, quoted
from req_core.findings import Evidence

__all__ = ["detect"]

DETECTOR = "atomicity_split"

# How sure the split was worth making, by how many obligations came out of it.
_CONFIDENCE_BY_COUNT = {2: 0.7, 3: 0.8}
_CONFIDENCE_MANY = 0.9


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Report the clause as compound when extraction separated it into two or more."""
    requirement = context.requirement
    if requirement.parent_id is not None:
        # A child is not itself compound. Reporting one here would count the same defect
        # once per obligation it was split into.
        return []

    children = list(context.siblings)
    if len(children) < 2:
        return []

    confidence = _CONFIDENCE_BY_COUNT.get(len(children), _CONFIDENCE_MANY)
    return [
        FindingDraft(
            detector=DETECTOR,
            statement=(
                f"The clause carries {len(children)} separable obligations under one number. "
                f"Downstream each gets one matrix row, one response and one price between "
                f"them, so {len(children) - 1} of them are never separately answered."
            ),
            evidence=Evidence(
                summary=f"Extraction separated this clause into {len(children)} obligations.",
                observations=[
                    f"children: {quoted(children, limit=6)}",
                    f"parent clause is atomic: {requirement.is_atomic}",
                ],
            ),
            confidence=confidence,
            children=children,
        )
    ]

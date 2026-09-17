"""ordinal_trap: an ordered-category rating compared as though it were a number.

`IP66` and `IP54` both read as numbers and neither is "12 better" than the other. The two
digits index two independent ordered scales — solids ingress and water ingress — so "not
less than IP54" is satisfied by IP66 on one axis and says nothing coherent about the other.
The same trap sits in insulation classes, fire ratings, security levels and every other
scheme where somebody numbered the categories.

It is worth catching because it survives every kind of review. The clause contains a number,
a comparison and a unit-looking token; it reads like a specification. A bidder offering the
"higher" rating is compliant by arithmetic and possibly non-compliant in fact.

**The catalogue is configuration.** Which scales exist is a fact about an industry, and
`req_core` has no business asserting one. `OrdinalScales` arrives from the caller. The
catalogue records only that a scale is ordinal — never which end is better, because no
detector here needs to know that, and encoding it would be encoding an engineering opinion.

**The model names the scale; Python decides what that means.** `Constraint.scale` is a name
the model read off the text. Whether arithmetic applies to it is a lookup, in code.

Confidence, and why:

- **Named scale in the catalogue, compared with an ordering operator — 0.85.** Both halves
  of the trap are present and both were read from the text.
- **Same, but the clause also states a test method — 0.7.** A stated test may resolve which
  axis is meant, so the defect is likelier to be a wording problem than a real ambiguity.

Deliberately does not: rank values on a scale, decide which of two ratings is better,
propose the correct comparison, or fire on a scale the catalogue does not name — an unknown
scale is unknown, and guessing that it is ordinal would produce findings about ordinary
numbers.

Belongs in `req_core`: the mechanism is "ordered categories are not arithmetic", which is
not a fact about tenders. Only the catalogue is domain knowledge, and it is an argument.
"""

from req_core.claims import NUMERIC_COMPARISONS
from req_core.detectors.base import DetectorContext, FindingDraft
from req_core.findings import Evidence

__all__ = ["detect"]

DETECTOR = "ordinal_trap"


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Every constraint comparing an ordinal rating as if the ordering were arithmetic."""
    claims = context.claims
    if claims is None:
        return []

    states_test = claims.states_test_method
    drafts: list[FindingDraft] = []
    for constraint in claims.constraints:
        if constraint.operator not in NUMERIC_COMPARISONS:
            continue
        if not context.ordinal_scales.is_ordinal(constraint.scale):
            continue
        drafts.append(
            FindingDraft(
                detector=DETECTOR,
                statement=(
                    f"{constraint.attribute} of {constraint.subject} is compared as a number "
                    f"({constraint.operator} {constraint.value}) on the {constraint.scale} "
                    f"scale, whose values are ordered categories rather than quantities. A "
                    f"bidder can satisfy the arithmetic without satisfying the requirement."
                ),
                evidence=Evidence(
                    summary=(
                        f"An ordering comparison applied to a value on the "
                        f"{constraint.scale} scale, which the catalogue "
                        f"`{context.ordinal_scales.name}` records as ordinal."
                    ),
                    observations=[
                        f"scale as reported by the reading: {constraint.scale!r}",
                        f"operator: {constraint.operator}, value: {constraint.value!r}",
                        f"catalogue: {context.ordinal_scales.name}",
                        f"clause states a test method: {states_test}",
                    ],
                ),
                confidence=0.7 if states_test else 0.85,
            )
        )
    return drafts

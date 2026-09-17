"""unverifiable: an obligation with no criterion by which it could be shown to have been met.

"The system shall be user-friendly." "Workmanship shall be of the highest standard." Nobody
can demonstrate compliance and nobody can refuse to accept. The clause survives review
because it reads like a requirement; it fails at handover, when the only remedy is argument.

The test applied here is deliberately narrow, because the wide version is a matter of taste.
A clause is unverifiable when **all** of these hold:

1. it places at least one obligation
2. it states no test method — no test, calculation, certificate, inspection or named
   standard to test against
3. it states no bounded constraint either, so there is nothing to measure instead

Any one of the three failing means the clause can be demonstrated somehow. A clause that
bounds a quantity is checkable by measuring it even if it names no test; a clause that names
a standard is checkable against the standard. Requiring all three keeps this away from the
clauses `missing_tolerance` already owns, so a reader does not get the same defect twice
under two names.

Confidence, and why:

- **Mandatory modality, no test, no bounded constraint, no constraints at all — 0.8.**
  Nothing to measure and nothing to test against.
- **Mandatory modality, constraints exist but all unbounded — 0.7.** The clause is about a
  quantity, which makes the gap slightly likelier to be an omission than a style.
- **Non-mandatory modality — 0.45.** Deliberately below the default threshold. An advisory
  clause with no test is usually intended as guidance, and asserting it as a defect would
  fill a report with things nobody has to satisfy.

Deliberately does not: judge whether a stated test method is *adequate* — "tested to the
Engineer's satisfaction" names a test and this detector accepts it, because the difference
between a weak test and no test is a judgement and this one is a reading. It also does not
propose the criterion the clause should have had.

Belongs in `req_core`: an unverifiable obligation is unverifiable in a construction
contract, a service agreement or a safety case. Nothing about the test needs a commercial
frame.
"""

from req_core.detectors.base import DetectorContext, FindingDraft, quoted
from req_core.findings import Evidence
from req_core.policy import Modality

__all__ = ["detect"]

DETECTOR = "unverifiable"

_BOUNDED_OPERATORS = frozenset({"lt", "lte", "eq", "gte", "gt", "between", "one_of"})


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Whether this clause requires something nobody could demonstrate having done."""
    claims = context.claims
    if claims is None:
        return []

    obligations = [item for item in claims.obligations if item.text.strip()]
    if not obligations:
        return []
    if claims.states_test_method:
        return []

    bounded = [item for item in claims.constraints if item.operator in _BOUNDED_OPERATORS]
    if bounded:
        # Measurable, so demonstrable. Whether the bound is usable is missing_tolerance's
        # question, and answering it here too would report one defect under two names.
        return []

    modality = context.modality.classify(context.requirement.text).modality
    if modality is not Modality.MANDATORY:
        confidence = 0.45
    else:
        confidence = 0.7 if claims.constraints else 0.8

    return [
        FindingDraft(
            detector=DETECTOR,
            statement=(
                f"The clause places {len(obligations)} obligation(s) and states no way to "
                f"show they were met: no test, no criterion, nothing bounded to measure. "
                f"It can neither be demonstrated nor refused at handover."
            ),
            evidence=Evidence(
                summary="An obligation with no stated test method and no bounded constraint.",
                observations=[
                    f"obligations: {quoted([item.text for item in obligations])}",
                    "states_test_method: False",
                    f"constraints reported: {len(claims.constraints)}, of which bounded: 0",
                    f"modality under {context.modality.name}: {modality}",
                ],
            ),
            confidence=confidence,
        )
    ]

"""missing_tolerance: a requirement that constrains a property without bounding it.

Two shapes, and they are different defects wearing one name:

- **No bound at all.** The clause requires a property and never says how much — "adequate
  illumination", "sufficient storage". The model reports `operator="unbounded"`, which is a
  statement about the text; that this makes the clause unpriceable is decided here.
- **A bound with no measurement condition.** "Not less than 80 px/m" is a number until you
  ask where it is measured. Two bidders reading it differently both comply and quote
  different prices, and the one who read it the cheap way wins.

The second is the one worth building for. The first is visible to anyone reading carefully;
the second looks complete.

Confidence, and why:

- **unbounded, no test method stated — 0.85.** Two independent signs of the same gap.
- **unbounded, but a test method is stated — 0.65.** The test may carry the bound the
  clause omits.
- **A bound, no measurement condition, unit present — 0.7.** A physical quantity almost
  always has one.
- **A bound, no measurement condition, no unit — 0.5.** Dimensionless counts often need
  none. Below the default threshold, so this shape abstains rather than asserts.

Deliberately does not: fire on a clause with no constraints at all (that is a clause not
being about a quantity, not a clause missing one), invent the bound it says is absent, or
decide what the bound should have been.

Belongs in `req_core`: nothing above needs a commercial or regulatory frame. A specification
for an engine, a drug trial protocol or a building code all have quantities, and all have
this defect.
"""

from req_core.claims import Constraint
from req_core.detectors.base import DetectorContext, FindingDraft, quoted
from req_core.findings import Evidence

__all__ = ["detect"]

DETECTOR = "missing_tolerance"


def _unbounded_draft(context: DetectorContext, constraint: Constraint) -> FindingDraft:
    """A property required without any bound on it."""
    claims = context.claims
    states_test = bool(claims and claims.states_test_method)
    confidence = 0.65 if states_test else 0.85
    observations = [
        f"constraint on {constraint.attribute!r} of {constraint.subject!r} reported as "
        f"operator=unbounded",
        f"clause states a test method: {states_test}",
    ]
    return FindingDraft(
        detector=DETECTOR,
        statement=(
            f"{constraint.attribute} of {constraint.subject} is required but never bounded. "
            f"Two bidders may read it differently and both comply."
        ),
        evidence=Evidence(
            summary=(
                "The clause constrains a property and states no bound for it"
                + (", though it does state a test method." if states_test else ".")
            ),
            observations=observations,
        ),
        confidence=confidence,
    )


def _unconditioned_draft(context: DetectorContext, constraint: Constraint) -> FindingDraft:
    """A bound with nothing saying under what conditions it is to be met."""
    has_unit = bool(constraint.unit.strip())
    confidence = 0.7 if has_unit else 0.5
    return FindingDraft(
        detector=DETECTOR,
        statement=(
            f"{constraint.attribute} of {constraint.subject} is bounded at "
            f"{constraint.value}{' ' + constraint.unit if has_unit else ''} with no stated "
            f"measurement condition. Where and how it is measured decides who complies."
        ),
        evidence=Evidence(
            summary="A numeric bound is stated and the conditions for measuring it are not.",
            observations=[
                f"operator={constraint.operator}, value={constraint.value!r}, "
                f"unit={constraint.unit!r}",
                "measurement_conditions reported: (none)",
                f"unit present: {has_unit}",
            ],
        ),
        confidence=confidence,
    )


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Every unbounded or unconditioned constraint in one clause."""
    claims = context.claims
    if claims is None or not claims.constraints:
        return []

    conditions = [item for item in claims.measurement_conditions if item.strip()]
    drafts: list[FindingDraft] = []
    for constraint in claims.constraints:
        if constraint.operator == "unbounded":
            drafts.append(_unbounded_draft(context, constraint))
        elif not conditions and constraint.operator != "one_of":
            # `one_of` is a membership test — "one of the types listed at Annex B" — and
            # has nothing to measure, so the absence of a condition says nothing about it.
            drafts.append(_unconditioned_draft(context, constraint))
    if drafts and conditions:
        # Recorded once rather than on each draft: the conditions that DO exist are what a
        # reviewer checks first when deciding whether this fired fairly.
        drafts = [
            draft.model_copy(
                update={
                    "evidence": draft.evidence.model_copy(
                        update={
                            "observations": [
                                *draft.evidence.observations,
                                f"conditions stated elsewhere in the clause: {quoted(conditions)}",
                            ]
                        }
                    )
                }
            )
            for draft in drafts
        ]
    return drafts

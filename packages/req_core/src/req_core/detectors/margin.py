"""zero_margin: a constraint satisfiable only exactly, with no engineering margin.

"The cycle shall complete in exactly ten seconds." "Coverage shall be 100%." "The range
shall be 50 to 50 dB." Nothing physical holds a value exactly, so a clause written this way
is either satisfied by nothing at all or satisfied by a tolerance the clause never granted
— and which tolerance applies is decided after the contract is signed, by whoever is
holding the meter.

Three shapes, all read from the operator rather than from the prose:

- **Equality on a physical quantity.** `operator="eq"` with a unit. The clause admits no
  deviation and physics admits no exactness.
- **A range of zero width.** `operator="between"` whose two ends read as the same number.
  Usually a copy-and-paste that nobody noticed, and it hard-fails every supplier.
- **Equality at a scale endpoint.** `operator="eq"` on a value of 0 or 100 with a
  percentage unit — no headroom exists on either side, so any measurement error is a breach.

Confidence, and why:

- **Zero-width range — 0.85.** Not a judgement: the two ends are the same number, read from
  the text.
- **Equality with a unit — 0.75.** A physical quantity pinned to a point.
- **Equality at 0 or 100 percent — 0.75.** Same reasoning, and the endpoint makes the
  absence of headroom one-sided rather than merely tight.
- **Equality with no unit — 0.4.** Deliberately below the default threshold. Counts are
  legitimately exact — "exactly two redundant controllers" is a design, not a defect — so
  this shape abstains and a human decides.

Deliberately does not: propose the tolerance that should have been given, decide how much
margin is enough (that is engineering, and it varies by quantity), or fire on an inequality.
`>= 80` has margin on one side by construction and is `missing_tolerance`'s business if it
has any problem at all.

Belongs in `req_core`: a constraint with no margin is the same defect in a specification for
a bridge, a pacemaker or a data centre. Nothing here needs a commercial frame.
"""

from req_core.detectors.base import DetectorContext, FindingDraft, read_number, read_range
from req_core.findings import Evidence

__all__ = ["detect"]

DETECTOR = "zero_margin"

# Units on which an exact value has no headroom on at least one side.
_PERCENT_UNITS = frozenset({"%", "percent", "pct"})
_ENDPOINTS = (0.0, 100.0)


def _zero_width_range(value: str) -> tuple[float, float] | None:
    """The two ends of a range, when they are written and identical."""
    ends = read_range(value)
    return ends if ends is not None and ends[0] == ends[1] else None


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Every constraint in this clause that admits exactly one value."""
    claims = context.claims
    if claims is None:
        return []

    drafts: list[FindingDraft] = []
    for constraint in claims.constraints:
        unit = constraint.unit.strip()
        has_unit = bool(unit)

        if constraint.operator == "between":
            ends = _zero_width_range(constraint.value)
            if ends is None:
                continue
            drafts.append(
                FindingDraft(
                    detector=DETECTOR,
                    statement=(
                        f"{constraint.attribute} of {constraint.subject} is given as a range "
                        f"from {ends[0]:g} to {ends[1]:g}{' ' + unit if has_unit else ''} — a "
                        f"range of zero width, which no supplier can hold and none can price."
                    ),
                    evidence=Evidence(
                        summary="A stated range whose two ends are the same number.",
                        observations=[
                            f"value as written: {constraint.value!r}",
                            f"ends read: {ends[0]:g} and {ends[1]:g}",
                            f"unit: {unit!r}" if has_unit else "unit: (none)",
                        ],
                    ),
                    confidence=0.85,
                )
            )
            continue

        if constraint.operator != "eq":
            continue

        number = read_number(constraint.value)
        at_endpoint = (
            number in _ENDPOINTS and unit.lower() in _PERCENT_UNITS if number is not None else False
        )
        read_as = "(unreadable)" if number is None else f"{number:g}"
        if at_endpoint:
            confidence = 0.75
            why = "an exact value at the end of its own scale, so no headroom exists"
        elif has_unit:
            confidence = 0.75
            why = "an exact value on a physical quantity, which nothing holds exactly"
        else:
            confidence = 0.4
            why = "an exact value with no unit, which is often a deliberate count"

        drafts.append(
            FindingDraft(
                detector=DETECTOR,
                statement=(
                    f"{constraint.attribute} of {constraint.subject} is required to equal "
                    f"{constraint.value}{' ' + unit if has_unit else ''} exactly, with no "
                    f"margin stated. Whatever tolerance actually applies is decided after "
                    f"signature."
                ),
                evidence=Evidence(
                    summary=f"Equality constraint: {why}.",
                    observations=[
                        f"operator=eq, value={constraint.value!r}",
                        f"unit: {unit!r}" if has_unit else "unit: (none)",
                        f"value read as a number: {read_as}",
                        f"at a percentage endpoint: {at_endpoint}",
                    ],
                ),
                confidence=confidence,
            )
        )
    return drafts

"""What a model is allowed to say about a clause: typed claims, never a verdict.

This is the whole of the model's job in the findings engine. Given a clause, it reports
what the clause *says*:

- the constraints it expresses, as (subject, attribute, operator, value, unit)
- whether it states a test by which compliance could be demonstrated, and what that test is
- the conditions any measurement is to be made under
- the separable obligations it places on the supplier

Nothing here is a judgement. There is no `is_defective`, no `severity`, no `confidence`, no
`finding_type` — and that is a property of the schema rather than of the prompt, so it
cannot be eroded by someone rewording the instructions later. A model cannot return a
verdict through this type because the type has nowhere to put one.

**Why the split is drawn exactly here.** "Is this requirement unverifiable?" is a question
whose answer moves with the asker; "does this clause state a test?" is a question about the
text. The second is what a model is reliable at and what a reader can check by looking at
the page. Whether the absence of a test makes a clause defective, how bad that is, and
whether to report it at all are decisions taken in `detectors` by Python reading these
records — see `req_core.detectors` for each one.

`operator` is a closed set including `unbounded`, which is the load-bearing value: a clause
that constrains an attribute without bounding it is exactly what `missing_tolerance` exists
to catch, and it is reported here as a shape rather than as a problem.

`scale` is reported as a **name**, never as a ranking. A model saying "this value is on the
IP scale" is a statement about the text. A model saying "IP66 is better than IP54" is an
engineering judgement, and `detectors.ordinal` makes it from a configured catalogue instead.

Deliberately does not: decide anything, rank anything, or carry a confidence. It also does
not parse `value` — the string is what the clause wrote, and turning it into a number is
Python's job in `detectors.numbers`, where a failure to parse is visible rather than
guessed at.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CLAIMS_PURPOSE",
    "NUMERIC_COMPARISONS",
    "ClauseClaims",
    "Constraint",
    "Obligation",
    "Operator",
    "PageClaims",
]

CLAIMS_PURPOSE = "req_core.read_claims"

# How a clause relates an attribute to a value. `unbounded` is not a failure to answer: it
# is the answer for "the camera shall have adequate resolution", where an attribute is
# constrained in words and bounded by nothing.
Operator = Literal[
    "lt",
    "lte",
    "eq",
    "gte",
    "gt",
    "between",
    "one_of",
    "unbounded",
]

# The operators that assert an ordering. Comparing on a scale where ordering is not
# arithmetic is what `detectors.ordinal` looks for, and this is the set it looks at.
NUMERIC_COMPARISONS: frozenset[str] = frozenset({"lt", "lte", "gte", "gt", "between"})


class Constraint(BaseModel):
    """One (subject, attribute, operator, value, unit) the clause asserts."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description='What the constraint is about, e.g. "perimeter camera".')
    attribute: str = Field(description='The property constrained, e.g. "pixel density".')
    operator: Operator = Field(
        description="How value bounds attribute. Use `unbounded` when the clause requires a "
        "property without bounding it at all — that is a real answer, not a gap in yours."
    )
    value: str = Field(
        default="",
        description="The bound exactly as the clause writes it, digits and words alike. Do "
        "not convert units, round, or compute. Empty when the operator is `unbounded`.",
    )
    unit: str = Field(default="", description='As written, e.g. "px/m", "°C", "s". May be empty.')
    scale: str = Field(
        default="",
        description="The name of a named rating scale this value sits on, if any — for "
        'example "IP", "NEMA", "IEC 62443 SL". The NAME only. Do not say which end is '
        "better: that is not a question about the text.",
    )


class Obligation(BaseModel):
    """One separable thing the clause requires somebody to do."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The obligation as one self-contained sentence.")
    actor: str = Field(default="", description='Who must do it, as named, e.g. "the Contractor".')


class ClauseClaims(BaseModel):
    """Everything the model was asked to report about one clause."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(description="The clause identifier, copied exactly from the input.")
    constraints: list[Constraint] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    states_test_method: bool = Field(
        default=False,
        description="Whether the clause states a means by which compliance could be "
        "demonstrated — a test, a calculation, a certificate, a named standard to test "
        "against, an inspection. Whether that means is adequate is not being asked.",
    )
    test_method: str = Field(
        default="",
        description="The stated means, quoted or closely paraphrased. Empty when there is none.",
    )
    measurement_conditions: list[str] = Field(
        default_factory=list,
        description='The conditions a measurement is to be taken under, as written — "at '
        '25 °C", "at the far edge of the detection zone", "averaged over 24 hours". Empty '
        "when the clause states none.",
    )


class PageClaims(BaseModel):
    """The model's reading of every clause on one page."""

    model_config = ConfigDict(extra="forbid")

    clauses: list[ClauseClaims] = Field(default_factory=list)

    def by_identifier(self) -> dict[str, ClauseClaims]:
        """The claims keyed by identifier, first occurrence winning."""
        found: dict[str, ClauseClaims] = {}
        for claims in self.clauses:
            found.setdefault(claims.identifier.strip(), claims)
        return found

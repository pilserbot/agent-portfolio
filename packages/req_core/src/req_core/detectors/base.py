"""What every detector is given, what it may return, and the number-reading they share.

A detector is a pure function from a `DetectorContext` to a list of `FindingDraft`. It gets
one requirement, the typed claims for it, and the configuration it needs; it returns what it
concluded and why. It cannot reach a document, make a model call, or see another clause —
this step is single-clause by construction, and the type is what makes that true rather than
a rule in a comment.

**A detector cannot set severity, and cannot mint an id.** `FindingDraft` has no field for
either. The engine stamps both: severity from the caller's `SeverityPolicy`, the id from the
clause and the detector name. A detector that wanted to call its own finding critical would
have to change this type to do it, which is a conversation rather than a commit.

**Confidence is structural.** Each detector states, in its own module, a small table of what
it saw and what that is worth. The numbers are arguable and are meant to be argued with —
what they are not is a model's opinion of its own output. `Evidence` travels beside the
number so a reader can check the reasoning against the page instead of taking the figure.

`read_number` is here because four detectors need it and none of them should differ on what
counts as a number. It returns None rather than guessing, and a detector that cannot read a
value declines to fire on it: a defect concluded from a misparsed number is worse than a
missed one, because it is confidently wrong and takes a reviewer's time to disprove.

Deliberately does not: call a model, read a file, compare clauses, or hold state. Anything
needing two clauses belongs to the second half of this engine, not here.
"""

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from req_core.claims import ClauseClaims
from req_core.contracts import Requirement
from req_core.findings import DetectorName, Evidence
from req_core.policy import ModalityPolicy

__all__ = [
    "Detector",
    "DetectorContext",
    "FindingDraft",
    "OrdinalScales",
    "read_number",
    "read_range",
]

# A number as specifications write one: an optional sign, digits, optional decimal, optional
# thousands separators. Trailing text (a unit, a parenthetical) is left to the caller.
_NUMBER = re.compile(r"[-+−]?\d[\d,]*\.?\d*")

# Specifications write the same number twice — "eighty (80)" — and the digits are the part
# that can be compared. Words alone are deliberately not parsed: a detector that guessed at
# "several" would be inventing a bound the clause did not state.
_RANGE_SEPARATOR = re.compile(r"\s*(?:to|and|–|—|-|\.\.)\s*", re.IGNORECASE)


class OrdinalScales(BaseModel):
    """Named rating scales whose values are ordered but not arithmetic.

    Configuration, and necessarily so: which scales exist is a fact about an industry, not
    about requirements in general. `IP66` and `IP54` are both readable as numbers and
    neither is "12 more" than the other — the digits index two independent ordered
    categories. A detector that compared them arithmetically would be wrong in a way that
    looks right.

    Matching is on the scale NAME the model reported, case-insensitively. The catalogue
    never says which end is better, because no detector here needs to know: it needs to know
    that arithmetic does not apply.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description='Whose catalogue this is, e.g. "default".')
    scales: frozenset[str] = Field(
        default_factory=frozenset,
        description="Scale names, lowercase. A value on one of these is ordinal.",
    )

    def is_ordinal(self, scale: str) -> bool:
        """Whether a reported scale name is one this catalogue calls ordinal."""
        return bool(scale.strip()) and scale.strip().lower() in self.scales


class DetectorContext(BaseModel):
    """One clause, everything known about it, and the configuration to read it by."""

    model_config = ConfigDict(frozen=True)

    requirement: Requirement
    claims: ClauseClaims | None = Field(
        default=None,
        description="What the model reported about this clause, or None when nothing was "
        "read for it. A detector that needs claims declines to fire rather than assuming.",
    )
    siblings: tuple[str, ...] = Field(
        default=(),
        description="Requirement ids split out of this clause, for lineage. Ids only: a "
        "detector cannot reach another clause's text, which is what keeps this step "
        "single-clause.",
    )
    modality: ModalityPolicy
    ordinal_scales: OrdinalScales


class FindingDraft(BaseModel):
    """What a detector concluded. The engine turns it into a `Finding`.

    No severity and no id, on purpose — see the module docstring.
    """

    model_config = ConfigDict(frozen=True)

    detector: DetectorName
    statement: str = Field(min_length=1)
    evidence: Evidence
    confidence: float = Field(ge=0.0, le=1.0)
    children: list[str] = Field(default_factory=list)


class Detector(Protocol):
    """A pure function from one clause's context to what it concluded about it."""

    def __call__(self, context: DetectorContext) -> list[FindingDraft]:
        """Return every draft this detector concluded, or an empty list."""
        ...


def read_number(text: str) -> float | None:
    """The first number in a piece of text, or None when there is not one to read.

    None is a real answer and the common one: "adequate", "as required", "to the
    satisfaction of the Engineer". A detector that turned any of those into a figure would
    be manufacturing the bound whose absence it is supposed to report.
    """
    match = _NUMBER.search(text.replace("−", "-"))
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - the pattern cannot produce an unparsable match
        return None


def read_range(text: str) -> tuple[float, float] | None:
    """The two ends of a written range, low first, or None when two cannot be read.

    Used only to tell a range of zero width from a real one. A range that reads as a single
    number is not a range, and returns None rather than a pair with itself.
    """
    cleaned = text.replace("−", "-").strip()
    numbers = _NUMBER.findall(cleaned)
    if len(numbers) < 2:
        return None
    # Two numbers separated by a range word, not two unrelated figures in a sentence.
    if not _RANGE_SEPARATOR.search(cleaned):
        return None
    try:
        low, high = float(numbers[0].replace(",", "")), float(numbers[1].replace(",", ""))
    except ValueError:  # pragma: no cover - as above
        return None
    return (low, high) if low <= high else (high, low)


def quoted(values: Sequence[str], *, limit: int = 3) -> str:
    """A short, stable rendering of some strings for an evidence line."""
    shown = [repr(value) for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"and {len(values) - limit} more")
    return ", ".join(shown) if shown else "(none)"

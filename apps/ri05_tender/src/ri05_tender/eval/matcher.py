"""Deciding which findings answer which gold items, deterministically and one-to-one.

A finding matches a gold item when **both** hold:

1. **anchor** — their normalised clause references intersect. Normalising uppercases,
   removes whitespace anywhere in the reference (so ``ITB - 9.2`` reaches ``ITB-9.2``),
   folds en and em dashes to a plain hyphen, and collapses a run of hyphens to one. It does
   *not* remove hyphens: ``TS-B.2`` and ``TSB.2`` stay different references, because
   collapsing that distinction would manufacture matches between neighbouring clauses.
2. **type** — the finding's type is in the set derived from **all** of the item's classes.
   All of them: an item classed ``["IMPLICIT", "COMMERCIAL"]`` is answerable either way, and
   reading only the first would mark a correct finding wrong.

Five outcomes, and only the first counts as found:

- ``MATCH`` — both hold, and every output the item requires is present.
- ``OUTPUT_MISS`` — both hold, but a required ``expects_*`` output is absent. The defect was
  seen; the work it demanded was not done.
- ``PARTIAL`` — anchor only. The finding is looking at the right clause and saying the wrong
  thing about it, which is a different failure from not looking.
- ``UNMATCHED`` — the finding anchors nothing. Not the same as wrong: see `adjudication`.
- ``MISS`` — a gold item no finding was credited with.

**Assignment is one-to-one.** One finding cannot satisfy two gold items and one gold item
cannot be credited to two findings, so a single finding citing two planted defects earns
credit for one. Competition is resolved by highest anchor overlap, then highest confidence,
then finding id — the last purely so that identical input always produces an identical
assignment, which a scoring harness has to.

One consequence of that ordering is worth knowing about: a candidate that would be
``OUTPUT_MISS`` outranks one that would be ``MATCH`` if it has the higher overlap or
confidence, and the item then reads as not found while a finding that fully answered it is
left over. The rule is the one specified, and it is applied as specified rather than
quietly improved; `test_the_tiebreak_can_prefer_an_output_miss_over_a_match` pins the
behaviour so a change to it is visible.

Deliberately does not: call a model. Nothing in this module asks anything to judge
similarity — matching is set intersection and dictionary lookup, and that is what makes a
score reproducible. It also does not decide whether an unmatched finding is wrong; that is
a human's call, recorded in `adjudication`.
"""

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ri05_tender.eval.models import Finding, GoldItem

# Each defect class the gold set uses, and the finding type that answers it. A class with no
# entry here stops the run: findings for it could never match, so every item carrying it
# would read as missed and the recall would be wrong in the direction that looks like a
# failing system rather than a broken harness.
CLASS_TO_FINDING_TYPE: dict[str, str] = {
    "COMPOUND": "atomicity_split",
    "IMPLICIT": "implicit_requirement",
    "MODALITY": "modality_inconsistency",
    "CONFLICT": "internal_conflict",
    "ENG-CONFLICT": "engineering_conflict",
    "UNVERIFIABLE": "unverifiable_requirement",
    "TOLERANCE": "missing_tolerance",
    "ZERO-MARGIN": "zero_margin",
    "ORDINAL": "ordinal_trap",
    "COMPUTED": "computed_compliance",
    "REGULATORY": "regulatory_misapplication",
    "IMPOSSIBLE": "undeliverable_requirement",
    "OPERATIONAL": "operational_risk",
    "OVERSPEC": "alternative_candidate",
    "SUPPLY": "supply_chain_screen",
    "COMMERCIAL": "contract_red_flag",
    "PROCESS": "submission_constraint",
    "ELIGIBILITY": "eligibility_gap",
    "BUDGET": "budget_variance",
}

IMPLICIT_CLASS = "IMPLICIT"

Outcome = Literal["match", "output_miss", "partial", "unmatched", "miss"]

_WHITESPACE = re.compile(r"\s+")
_DASHES = re.compile(r"[‐-―−]")
_HYPHEN_RUN = re.compile(r"-{2,}")

__all__ = [
    "CLASS_TO_FINDING_TYPE",
    "IMPLICIT_CLASS",
    "Assignment",
    "MatchReport",
    "MatcherError",
    "Outcome",
    "Partial",
    "allowed_finding_types",
    "anchor_overlap",
    "match_findings",
    "normalise_ref",
    "normalise_refs",
]


class MatcherError(Exception):
    """The gold set names something this matcher cannot interpret."""


def normalise_ref(ref: str) -> str:
    """Put one clause reference into the form two documents can be compared in.

    Uppercase, whitespace removed anywhere in the string, unicode dashes folded to a plain
    hyphen, runs of hyphens collapsed to one. Hyphens are kept: they separate a section from
    a clause, and dropping them would let `TS-B.2` answer for `TSB.2`.
    """
    folded = _DASHES.sub("-", ref)
    squeezed = _WHITESPACE.sub("", folded)
    return _HYPHEN_RUN.sub("-", squeezed).upper()


def normalise_refs(refs: Sequence[str]) -> set[str]:
    """A set of normalised references, with blanks dropped."""
    return {normalised for ref in refs if (normalised := normalise_ref(ref))}


def allowed_finding_types(item: GoldItem) -> set[str]:
    """Every finding type that may answer this gold item, from all of its classes.

    Raises `MatcherError` on a class the map does not cover, rather than returning a
    smaller set: an unmapped class silently makes its items unmatchable forever.
    """
    unknown = sorted(name for name in item.classes if name not in CLASS_TO_FINDING_TYPE)
    if unknown:
        raise MatcherError(
            f"gold item {item.id!r} carries class(es) {', '.join(unknown)} that "
            f"CLASS_TO_FINDING_TYPE does not map. Add them, or every item carrying one can "
            f"never be matched and the recall will read as a failing system."
        )
    return {CLASS_TO_FINDING_TYPE[name] for name in item.classes}


def anchor_overlap(finding: Finding, item: GoldItem) -> int:
    """How many normalised references the two have in common."""
    return len(normalise_refs(finding.refs) & normalise_refs(item.refs))


def missing_outputs(finding: Finding, item: GoldItem) -> list[str]:
    """The outputs this item required that the finding did not carry, in a stable order."""
    carried = finding.outputs.present()
    return [name for name in item.expected_outputs if name not in carried]


class Assignment(BaseModel):
    """One finding credited against one gold item, and how completely."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    gold_id: str
    outcome: Literal["match", "output_miss"]
    anchor_overlap: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_outputs: list[str] = Field(default_factory=list)


class Partial(BaseModel):
    """A finding that cited the right clause and said the wrong kind of thing about it."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    gold_ids: list[str] = Field(min_length=1, description="Items it anchored but cannot answer.")
    finding_type: str


class MatchReport(BaseModel):
    """Every finding and every scored gold item, each placed in exactly one outcome."""

    model_config = ConfigDict(frozen=True)

    matches: list[Assignment] = Field(default_factory=list)
    output_misses: list[Assignment] = Field(default_factory=list)
    partials: list[Partial] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list, description="Finding ids.")
    misses: list[str] = Field(default_factory=list, description="Gold ids nothing answered.")

    @property
    def matched_gold_ids(self) -> set[str]:
        """The gold items a finding fully answered. The only ones that count as found."""
        return {assignment.gold_id for assignment in self.matches}

    @property
    def matched_finding_ids(self) -> set[str]:
        """The findings credited with a full match."""
        return {assignment.finding_id for assignment in self.matches}


def match_findings(findings: Sequence[Finding], gold: Sequence[GoldItem]) -> MatchReport:
    """Assign findings to gold items one-to-one and place everything left over.

    Only scored items should be passed in: this module counts what it is given, and the
    decision to exclude an item belongs to the caller that loaded it.
    """
    allowed = {item.id: allowed_finding_types(item) for item in gold}
    by_id = {item.id: item for item in gold}

    candidates: list[Assignment] = []
    anchored: dict[str, list[str]] = {}
    for finding in findings:
        for item in gold:
            overlap = anchor_overlap(finding, item)
            if not overlap:
                continue
            anchored.setdefault(finding.finding_id, []).append(item.id)
            if finding.finding_type not in allowed[item.id]:
                continue
            absent = missing_outputs(finding, item)
            candidates.append(
                Assignment(
                    finding_id=finding.finding_id,
                    gold_id=item.id,
                    outcome="output_miss" if absent else "match",
                    anchor_overlap=overlap,
                    confidence=finding.confidence,
                    missing_outputs=absent,
                )
            )

    assignments = _assign_one_to_one(candidates)

    taken_findings = {assignment.finding_id for assignment in assignments}
    taken_gold = {assignment.gold_id for assignment in assignments}

    partials = [
        Partial(
            finding_id=finding.finding_id,
            gold_ids=sorted(anchored[finding.finding_id]),
            finding_type=finding.finding_type,
        )
        for finding in findings
        if finding.finding_id not in taken_findings and anchored.get(finding.finding_id)
    ]
    unmatched = [
        finding.finding_id
        for finding in findings
        if finding.finding_id not in taken_findings and not anchored.get(finding.finding_id)
    ]

    return MatchReport(
        matches=[a for a in assignments if a.outcome == "match"],
        output_misses=[a for a in assignments if a.outcome == "output_miss"],
        partials=partials,
        unmatched=unmatched,
        misses=sorted(gold_id for gold_id in by_id if gold_id not in taken_gold),
    )


def _assign_one_to_one(candidates: Sequence[Assignment]) -> list[Assignment]:
    """Greedily take the strongest candidate pairs, never reusing either side.

    Ordered by anchor overlap descending, then confidence descending, then finding id and
    gold id ascending. The last two are not tiebreaks anybody cares about on their own —
    they are there so that the same input always produces the same assignment, which a
    score has to if it is to be compared with yesterday's.
    """
    ordered = sorted(
        candidates,
        key=lambda a: (-a.anchor_overlap, -a.confidence, a.finding_id, a.gold_id),
    )
    used_findings: set[str] = set()
    used_gold: set[str] = set()
    taken: list[Assignment] = []
    for candidate in ordered:
        if candidate.finding_id in used_findings or candidate.gold_id in used_gold:
            continue
        used_findings.add(candidate.finding_id)
        used_gold.add(candidate.gold_id)
        taken.append(candidate)
    return taken

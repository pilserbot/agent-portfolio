"""What a gold item is, and what a finding will be when the pipeline exists.

`GoldItem` mirrors one line of `data/tenders/<name>/gold/gold_set.jsonl` field for field.
It is `extra="forbid"` on purpose: a key added to the gold set that this model does not know
about would otherwise be dropped in silence, and a scoring harness quietly ignoring part of
its answer key is worse than one that will not start.

`Finding` is the shape the pipeline will emit. Nothing produces one yet — that is the point
of this package. The measurement is defined before the thing it measures, so the pipeline is
built against a target rather than the target being drawn around whatever the pipeline
happened to do.

Deliberately does not: read a file, decide whether a finding matches anything, or compute a
score. Loading is `loader`'s, matching is `matcher`'s, arithmetic is `metrics`'. It also
holds no threshold and no weight: what counts as good is a decision, made where it can be
read.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import EvidenceRef

# The severities a gold item may carry, and the tiers it may belong to. Closed sets: an
# unknown value fails the load naming the item, because a severity nobody recognises would
# silently drop out of the weighted recall and the no-bid gate.
Severity = Literal["no_bid", "critical", "major", "minor"]
Tier = Literal["cold_start", "configured"]

__all__ = [
    "Finding",
    "FindingOutputs",
    "GoldItem",
    "GoldReview",
    "Severity",
    "Tier",
]


class GoldReview(BaseModel):
    """Who looked at a gold item, and what they concluded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    reviewer: str | None = None
    date: str | None = None
    severity_override: str | None = None
    note: str = ""


class GoldItem(BaseModel):
    """One planted defect in a labelled tender, as the answer key records it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    document: int = Field(ge=1, description="Which numbered tender document it sits in.")
    document_name: str = ""
    refs: list[str] = Field(
        min_length=1, description="The clause references a finding must cite to anchor here."
    )
    ref_raw: str = ""
    classes: list[str] = Field(
        min_length=1,
        description="Every defect class this item belongs to. All of them widen the set of "
        "finding types that may match, not just the first.",
    )
    finding_type: str = Field(min_length=1)
    severity: Severity
    tier: Tier
    requires: list[str] | None = Field(
        default=None, description="Inputs the pipeline needs before it could find this."
    )
    evidence_basis: str = ""
    statement: str = ""
    expected_action: str = ""
    demo_set: bool = False
    demo_note: str | None = None

    expects_clarification_question: bool = False
    expects_price_impact: bool = False
    expects_alternative: bool = False
    expects_split: bool = False
    expects_checklist_entry: bool = False
    expects_no_bid: bool = False

    review_priority: str = "normal"
    review: GoldReview
    severity_machine: str = ""
    scored: bool = Field(
        description="False excludes the item from every metric. It still loads, and the "
        "report says how many were excluded, because a shrinking denominator nobody "
        "mentioned is how a score improves without the system improving."
    )

    # Present on only a few items. Optional rather than absent from the model, so
    # extra="forbid" can stay on and catch a genuinely unknown key.
    dispute: str | None = None
    reviewer_note: str | None = None

    @property
    def expected_outputs(self) -> list[str]:
        """The output names a finding must carry to fully satisfy this item."""
        wanted = {
            "clarification_question": self.expects_clarification_question,
            "price_impact_usd": self.expects_price_impact,
            "alternative": self.expects_alternative,
            "split_children": self.expects_split,
            "checklist_entry": self.expects_checklist_entry,
            "no_bid": self.expects_no_bid,
        }
        return [name for name, required in wanted.items() if required]


class FindingOutputs(BaseModel):
    """The artefacts a finding may carry beyond the observation itself.

    Which of these a finding must produce is decided by the gold item it matches, through
    that item's `expects_*` flags — never by this model, which only records what is there.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clarification_question: str | None = None
    price_impact_usd: float | None = None
    alternative: str | None = None
    split_children: list[str] = Field(default_factory=list)
    checklist_entry: str | None = None
    no_bid: bool = False

    def present(self) -> set[str]:
        """The output names actually carried, treating blank text as absent.

        A clarification question of `""` is not a clarification question, and crediting it
        would let an empty field satisfy a requirement to ask something.
        """
        carried: set[str] = set()
        for name in ("clarification_question", "alternative", "checklist_entry"):
            value = getattr(self, name)
            if value is not None and value.strip():
                carried.add(name)
        if self.price_impact_usd is not None:
            carried.add("price_impact_usd")
        if self.split_children:
            carried.add("split_children")
        if self.no_bid:
            carried.add("no_bid")
        return carried


class Finding(BaseModel):
    """One defect the pipeline reports, and what it produced about it.

    Frozen, like everything else that records something that happened. `evidence` is
    `spine.contracts.EvidenceRef` rather than a type of this package's own, so a citation
    made here resolves the same way as a citation made anywhere else in the portfolio.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(min_length=1)
    finding_type: str = Field(min_length=1)
    refs: list[str] = Field(default_factory=list)
    severity: Severity
    statement: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    outputs: FindingOutputs = Field(default_factory=FindingOutputs)

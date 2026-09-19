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

# How an item that used to carry the single IMPLICIT class was split at rev 3, and nothing
# else. Closed, because the value is a record of a decision that was made once: a free
# string here would let a later edit write a third story about what happened.
Reclassification = Literal["IMPLICIT -> UNSTATED", "IMPLICIT -> DISPLACED"]

__all__ = [
    "Finding",
    "FindingOutputs",
    "GoldItem",
    "GoldReview",
    "Reclassification",
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
    expects_recovered_statement: bool = Field(
        default=False,
        description="The obligation must be stated back in the finding's own words. Set on "
        "every item of the UNSTATED and DISPLACED classes, because for those the whole "
        "task is to say the thing the package does not say plainly. `matcher` enforces it "
        "against the tender text: a finding that only quotes the source line fails.",
    )

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

    # Set on the 72 items that carried IMPLICIT before rev 3. Kept on the record rather than
    # only in the correction note, so anyone reading one item can see it was moved and which
    # way, without having to know a document exists.
    reclassified_rev3: Reclassification | None = None

    # Set on the 23 items whose `refs` were re-derived from `ref_raw`. The builder's regex
    # could not match the `XX-Y.NN` clause shape and fell back to storing the raw text as a
    # single reference, so `['TS-B.36 + TS-B.38']` was one token equal to no clause id and
    # the item could not anchor. The old value is kept here rather than discarded: a
    # reference that changed is a change to what the answer key claims, and the previous
    # claim should be visible on the record that makes the new one.
    refs_prior: list[str] | None = None

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
            # Last because it is the one output `FindingOutputs` cannot report on its own:
            # whether it was produced depends on the tender text, so `matcher` decides it.
            "recovered_statement": self.expects_recovered_statement,
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
    recovered_statement: str | None = Field(
        default=None,
        description="The obligation stated in the finding's own words, for an item whose "
        "`expects_recovered_statement` is true. Not on `FindingOutputs` with the rest, "
        "because whether it counts cannot be read off this record alone — `matcher` tests "
        "it against the tender text, and a verbatim quote of the source does not count.",
    )
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    outputs: FindingOutputs = Field(default_factory=FindingOutputs)

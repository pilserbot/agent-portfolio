"""Turning a match report into the numbers that say whether the pipeline is any good.

Every ratio here is computed by `spine.eval.metrics`, reached through a thin adapter rather
than reimplemented: `_gold_verdicts` and `_finding_verdicts` turn this domain's records into
the `Verdict` shape the spine already counts, and the spine's own conventions then apply —
a zero denominator is 0.0 over a support of 0, never an error and never a NaN. Forking the
arithmetic would give this package a second definition of recall to keep in step with the
first.

Three things this module refuses to let a caller do:

- **Report one precision without the other.** `Precisions` carries both as required fields,
  so there is no way to construct half of it. Strict precision counts every unmatched
  finding against the system; adjudicated precision counts only the ones a human called
  wrong. Quoting either alone is a choice about how flattering the number is, and the type
  removes the choice.
- **Pass the no-bid gate on average.** The gate is not a rate. It is True only when every
  scored `no_bid` item was fully matched, because a bid that goes out on a tender the
  system should have refused is not offset by finding ninety other things.
- **Quote a recall without its ceiling.** `Addressability` states how many of the scored
  items the registered detectors could answer at all. The first full scored run read
  0 / 186, and 154 of those 186 were never reachable by the six detector families that
  existed: no detector emits a finding type their classes map to. A denominator nobody
  can reach is not a measurement of the system, it is a measurement of the gap between
  the gold set and what has been built so far, and the two say different things.

- **Add UNSTATED recovery to DISPLACED recovery.** There is no combined figure and there is
  no field to put one in. The two used to be one class, IMPLICIT, and summing them is what
  hid the difference: finding an obligation that is written out somewhere nobody looks is a
  reading problem, and finding one that is written nowhere is an inference problem. A single
  ratio over both is an average of two different capabilities, and it moves when the mix
  changes rather than when the system does.

Deliberately does not: decide what a good score is, weight anything the caller did not ask
for, or call a model. The severity weights are the ones specified and they are constants
here where they can be read, not buried in a formula.
"""

from collections.abc import Collection, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ri05_tender.eval.matcher import (
    DISPLACED_CLASS,
    UNSTATED_CLASS,
    MatchReport,
    allowed_finding_types,
)
from ri05_tender.eval.models import GoldItem, Severity
from spine.contracts import Verdict
from spine.eval import metrics as spine_metrics

# What a miss costs, relative to the others. A no_bid item is eight minors: missing one
# means the bid went out on a tender the system should have refused.
SEVERITY_WEIGHTS: dict[Severity, int] = {"no_bid": 8, "critical": 4, "major": 2, "minor": 1}

FOUND = "found"
NOT_FOUND = "not_found"
RELEVANT = "relevant"
SPURIOUS = "spurious"

__all__ = [
    "SEVERITY_WEIGHTS",
    "Breakdown",
    "Precisions",
    "ScoreCard",
    "score",
]


def _gold_verdicts(gold: Sequence[GoldItem], report: MatchReport) -> list[Verdict]:
    """One Verdict per scored gold item, in the shape `spine.eval.metrics` counts.

    `expected` is FOUND for every item — the answer key says each one is there to be found —
    and `actual` is FOUND only where a finding fully matched. Recall over that is exactly
    matches over gold items, computed by the spine rather than here.
    """
    matched = report.matched_gold_ids
    return [
        Verdict(
            item_id=item.id,
            expected=FOUND,
            actual=FOUND if item.id in matched else NOT_FOUND,
            passed=item.id in matched,
            score=1.0 if item.id in matched else 0.0,
            rationale=f"{item.severity}/{item.tier}: "
            f"{'matched' if item.id in matched else 'missed'}",
            judge_model=None,
        )
        for item in gold
    ]


def _finding_verdicts(finding_ids: Sequence[str], relevant: set[str]) -> list[Verdict]:
    """One Verdict per finding: the pipeline called it relevant, and the key says whether it is."""
    return [
        Verdict(
            item_id=finding_id,
            expected=RELEVANT if finding_id in relevant else SPURIOUS,
            actual=RELEVANT,
            passed=finding_id in relevant,
            score=1.0 if finding_id in relevant else 0.0,
            rationale="reported by the pipeline",
            judge_model=None,
        )
        for finding_id in finding_ids
    ]


def _recall_of(items: Sequence[GoldItem], report: MatchReport) -> spine_metrics.MetricValue:
    """Recall over a subset of the gold items, through the spine."""
    return spine_metrics.recall(_gold_verdicts(items, report), positive_label=FOUND)


class Breakdown(BaseModel):
    """One recall figure and the number of items behind it."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: float = Field(ge=0.0, le=1.0)
    found: int = Field(ge=0)
    total: int = Field(ge=0, description="Scored items in this group. Zero means no evidence.")


def _breakdowns(
    gold: Sequence[GoldItem], report: MatchReport, keys: dict[str, list[GoldItem]]
) -> list[Breakdown]:
    """Recall for each named group, in key order."""
    matched = report.matched_gold_ids
    return [
        Breakdown(
            key=key,
            value=_recall_of(items, report).value,
            found=sum(1 for item in items if item.id in matched),
            total=len(items),
        )
        for key, items in sorted(keys.items())
    ]


# What a per-item row says happened. "partial" is not one of the matcher's own gold-item
# outcomes: a partially answered item IS a miss for recall, and this vocabulary keeps the
# reason visible — a finding cited the clause and said the wrong kind of thing about it,
# which is a different problem from nothing citing the clause at all.
ItemOutcome = Literal["match", "output_miss", "partial", "miss"]


class AddressableOutcome(BaseModel):
    """One gold item the detectors could in principle answer, and what actually happened."""

    model_config = ConfigDict(frozen=True)

    gold_id: str = Field(min_length=1)
    classes: list[str]
    severity: Severity
    outcome: ItemOutcome
    demands: list[str] = Field(
        default_factory=list, description="Outputs this item requires beyond the finding."
    )
    unemitted_demands: list[str] = Field(
        default_factory=list,
        description="Of `demands`, the ones no registered detector emits at all. An item "
        "with any of these cannot reach MATCH however well the detectors read the clause.",
    )
    blocked_by: str = Field(
        default="", description="Why it is not a MATCH, in words. Empty when it is one."
    )

    @computed_field
    @property
    def reachable(self) -> bool:
        """Whether a MATCH is possible today, or is waiting on an output nobody emits."""
        return not self.unemitted_demands


class Addressability(BaseModel):
    """How many scored items the detectors that exist could answer, and what stops them.

    Recall's denominator is every scored item, which is the right denominator for the
    product and the wrong one for reading a single step's result. This states the other
    one beside it: of the scored items, how many are addressable at all, and of those, how
    many are held back by a missing output rather than by a detector that did not fire.
    """

    model_config = ConfigDict(frozen=True)

    scored_total: int = Field(ge=0, description="Every scored gold item — recall's denominator.")
    emitted_finding_types: list[str]
    emitted_outputs: list[str] = Field(
        description="Output names some registered detector actually produces. Empty is a "
        "real answer and the one that held in the first scored run."
    )
    items: list[AddressableOutcome] = Field(default_factory=list)

    @computed_field
    @property
    def total(self) -> int:
        """Items whose classes map to a finding type some registered detector emits."""
        return len(self.items)

    @computed_field
    @property
    def found(self) -> int:
        """Addressable items fully answered."""
        return sum(1 for item in self.items if item.outcome == "match")

    @computed_field
    @property
    def reachable_total(self) -> int:
        """Addressable items demanding no output that nothing emits — today's MATCH ceiling."""
        return sum(1 for item in self.items if item.reachable)

    @computed_field
    @property
    def blocked_on_outputs(self) -> int:
        """Addressable items that cannot MATCH until some output is produced at all."""
        return self.total - self.reachable_total

    @computed_field
    @property
    def recall(self) -> float:
        """Found over addressable. 0.0 over nothing addressable, never an error."""
        return self.found / self.total if self.total else 0.0

    @computed_field
    @property
    def recall_reachable(self) -> float:
        """Found over the items a MATCH is actually possible for."""
        return self.found / self.reachable_total if self.reachable_total else 0.0


def addressability(
    gold: Sequence[GoldItem],
    report: MatchReport,
    *,
    emitted_finding_types: Collection[str],
    emitted_outputs: Collection[str] = (),
) -> Addressability:
    """Which scored items the registered detectors could answer, and what blocked each one.

    `emitted_finding_types` and `emitted_outputs` are configuration, supplied by whichever
    application registered the detectors. This module does not know what is wired up and
    must not guess: a hardcoded list here would go stale the moment a detector is added and
    would then overstate the ceiling, which is the one direction it must not be wrong in.
    """
    emitted = set(emitted_finding_types)
    produced = set(emitted_outputs)

    matched = {assignment.gold_id: assignment for assignment in report.matches}
    missed_outputs = {assignment.gold_id: assignment for assignment in report.output_misses}
    anchored_wrongly: dict[str, list[str]] = {}
    for partial in report.partials:
        for gold_id in partial.gold_ids:
            anchored_wrongly.setdefault(gold_id, []).append(partial.finding_type)

    rows: list[AddressableOutcome] = []
    for item in sorted(gold, key=lambda entry: entry.id):
        if not (allowed_finding_types(item) & emitted):
            continue
        demands = list(item.expected_outputs)
        unemitted = [name for name in demands if name not in produced]

        if item.id in matched:
            outcome: ItemOutcome = "match"
            blocked = ""
        elif item.id in missed_outputs:
            outcome = "output_miss"
            absent = ", ".join(missed_outputs[item.id].missing_outputs)
            blocked = f"output(s) absent: {absent}"
        elif item.id in anchored_wrongly:
            outcome = "partial"
            types = ", ".join(sorted(set(anchored_wrongly[item.id])))
            blocked = f"wrong finding_type: {types}"
        else:
            outcome = "miss"
            blocked = "no finding cited this clause"

        rows.append(
            AddressableOutcome(
                gold_id=item.id,
                classes=list(item.classes),
                severity=item.severity,
                outcome=outcome,
                demands=demands,
                unemitted_demands=unemitted,
                blocked_by=blocked,
            )
        )

    return Addressability(
        scored_total=len(gold),
        emitted_finding_types=sorted(emitted),
        emitted_outputs=sorted(produced),
        items=rows,
    )


class Precisions(BaseModel):
    """Both precisions, because neither may be quoted alone.

    `strict` treats every unmatched finding as wrong. `adjudicated` treats only the ones a
    human called wrong as wrong, and counts the ones they called genuinely new as right.
    The gold set records what was planted, not every defect in the package, so the truth is
    between them — which is why both are always shown.
    """

    model_config = ConfigDict(frozen=True)

    strict: float = Field(ge=0.0, le=1.0)
    strict_support: int = Field(
        ge=0, description="Matches, plus unmatched findings, plus duplicates."
    )
    adjudicated: float = Field(ge=0.0, le=1.0)
    adjudicated_support: int = Field(ge=0, description="All findings.")
    true_new: int = Field(ge=0, description="Unmatched findings a human confirmed as real.")
    pending: int = Field(ge=0, description="Unmatched findings nobody has ruled on yet.")


class ScoreCard(BaseModel):
    """Everything one scoring run concluded."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    scored_items: int = Field(ge=0)
    excluded_items: int = Field(ge=0)
    findings: int = Field(ge=0)

    recall_overall: float = Field(ge=0.0, le=1.0)
    recall_weighted: float = Field(ge=0.0, le=1.0)

    # Reported side by side and never summed — see the module docstring. Both carry their
    # denominator, because "68%" of a group whose size nobody stated is not a figure.
    unstated_recovery: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall over items no sentence in the package states. Nothing to read; "
        "the obligation has to be inferred from a quantity, a row or a scope word.",
    )
    unstated_total: int = Field(ge=0)
    displaced_recovery: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall over items written out plainly, but somewhere a requirements "
        "review never goes. Readable — which is the point: it is the half a keyword rule "
        "has any claim on, so it is where an uplift figure has to be earned.",
    )
    displaced_total: int = Field(ge=0)

    by_class: list[Breakdown] = Field(default_factory=list)
    by_tier: list[Breakdown] = Field(default_factory=list)
    by_severity: list[Breakdown] = Field(default_factory=list)

    precisions: Precisions
    no_bid_gate: bool = Field(
        description="True only when every scored no_bid item was fully matched."
    )
    no_bid_total: int = Field(ge=0)
    no_bid_found: int = Field(ge=0)

    matches: int = Field(ge=0)
    partials: int = Field(ge=0)
    output_misses: int = Field(ge=0)
    duplicates: int = Field(
        ge=0, description="Findings restating a defect another finding was credited with."
    )
    unmatched: int = Field(ge=0)
    misses: int = Field(ge=0)

    addressable: Addressability | None = Field(
        default=None,
        description="The ceiling behind `recall_overall`, when the caller said which "
        "finding types and outputs are actually wired up. None when nobody said, and the "
        "report then prints no ceiling rather than inventing one.",
    )

    @property
    def passed_gate(self) -> str:
        """The gate as the word a report prints."""
        return "PASS" if self.no_bid_gate else "FAIL"


def _weighted_recall(gold: Sequence[GoldItem], report: MatchReport) -> float:
    """Recall with each item counted by what missing it would cost."""
    matched = report.matched_gold_ids
    total = sum(SEVERITY_WEIGHTS[item.severity] for item in gold)
    if not total:
        return 0.0
    found = sum(SEVERITY_WEIGHTS[item.severity] for item in gold if item.id in matched)
    return found / total


def score(
    *,
    tender_name: str,
    gold: Sequence[GoldItem],
    excluded: Sequence[GoldItem] = (),
    finding_ids: Sequence[str],
    report: MatchReport,
    true_new: Sequence[str] = (),
    pending: Sequence[str] = (),
    emitted_finding_types: Collection[str] | None = None,
    emitted_outputs: Collection[str] = (),
) -> ScoreCard:
    """Compute every figure from a match report and the gold items behind it.

    `gold` must already be the scored items — excluding them is the loader's decision, and
    `excluded` is passed only so the report can say how many were left out. `true_new` and
    `pending` come from `adjudication`; with neither, adjudicated precision equals what a
    run with no human review can honestly claim.

    `emitted_finding_types` says which finding types the detectors actually registered by
    the caller can produce, and `emitted_outputs` which of a gold item's `expects_*`
    artefacts any of them carries. Give them and the card states recall's ceiling beside
    recall. Leave them out and it states none — better than a ceiling this module guessed.
    """
    matched = report.matched_gold_ids
    confirmed_new = set(true_new)

    by_class: dict[str, list[GoldItem]] = {}
    for item in gold:
        for name in item.classes:
            by_class.setdefault(name, []).append(item)
    by_tier: dict[str, list[GoldItem]] = {}
    by_severity: dict[str, list[GoldItem]] = {}
    for item in gold:
        by_tier.setdefault(item.tier, []).append(item)
        by_severity.setdefault(item.severity, []).append(item)

    unstated = [item for item in gold if UNSTATED_CLASS in item.classes]
    displaced = [item for item in gold if DISPLACED_CLASS in item.classes]
    no_bid_items = [item for item in gold if item.severity == "no_bid"]

    # Which findings strict precision is measured over is the matcher's definition, not a
    # list rebuilt here: matches, plus the unmatched, plus the duplicates. The arithmetic
    # below is the spine's `precision` either way.
    strict = spine_metrics.precision(
        _finding_verdicts(report.strict_denominator, report.matched_finding_ids),
        positive_label=RELEVANT,
    )
    adjudicated = spine_metrics.precision(
        _finding_verdicts(list(finding_ids), report.matched_finding_ids | confirmed_new),
        positive_label=RELEVANT,
    )

    return ScoreCard(
        tender_name=tender_name,
        scored_items=len(gold),
        excluded_items=len(excluded),
        findings=len(finding_ids),
        recall_overall=_recall_of(gold, report).value,
        recall_weighted=_weighted_recall(gold, report),
        unstated_recovery=_recall_of(unstated, report).value,
        unstated_total=len(unstated),
        displaced_recovery=_recall_of(displaced, report).value,
        displaced_total=len(displaced),
        by_class=_breakdowns(gold, report, by_class),
        by_tier=_breakdowns(gold, report, by_tier),
        by_severity=_breakdowns(gold, report, by_severity),
        precisions=Precisions(
            strict=strict.value,
            strict_support=strict.support,
            adjudicated=adjudicated.value,
            adjudicated_support=adjudicated.support,
            true_new=len(confirmed_new),
            pending=len(pending),
        ),
        no_bid_gate=all(item.id in matched for item in no_bid_items),
        no_bid_total=len(no_bid_items),
        no_bid_found=sum(1 for item in no_bid_items if item.id in matched),
        matches=len(report.matches),
        partials=len(report.partials),
        output_misses=len(report.output_misses),
        duplicates=len(report.duplicates),
        unmatched=len(report.unmatched),
        misses=len(report.misses),
        addressable=(
            None
            if emitted_finding_types is None
            else addressability(
                gold,
                report,
                emitted_finding_types=emitted_finding_types,
                emitted_outputs=emitted_outputs,
            )
        ),
    )

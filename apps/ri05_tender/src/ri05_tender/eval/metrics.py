"""Turning a match report into the numbers that say whether the pipeline is any good.

Every ratio here is computed by `spine.eval.metrics`, reached through a thin adapter rather
than reimplemented: `_gold_verdicts` and `_finding_verdicts` turn this domain's records into
the `Verdict` shape the spine already counts, and the spine's own conventions then apply —
a zero denominator is 0.0 over a support of 0, never an error and never a NaN. Forking the
arithmetic would give this package a second definition of recall to keep in step with the
first.

Two things this module refuses to let a caller do:

- **Report one precision without the other.** `Precisions` carries both as required fields,
  so there is no way to construct half of it. Strict precision counts every unmatched
  finding against the system; adjudicated precision counts only the ones a human called
  wrong. Quoting either alone is a choice about how flattering the number is, and the type
  removes the choice.
- **Pass the no-bid gate on average.** The gate is not a rate. It is True only when every
  scored `no_bid` item was fully matched, because a bid that goes out on a tender the
  system should have refused is not offset by finding ninety other things.

Deliberately does not: decide what a good score is, weight anything the caller did not ask
for, or call a model. The severity weights are the ones specified and they are constants
here where they can be read, not buried in a formula.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ri05_tender.eval.matcher import IMPLICIT_CLASS, MatchReport
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
    implicit_recovery: float = Field(ge=0.0, le=1.0)
    implicit_total: int = Field(ge=0)

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
) -> ScoreCard:
    """Compute every figure from a match report and the gold items behind it.

    `gold` must already be the scored items — excluding them is the loader's decision, and
    `excluded` is passed only so the report can say how many were left out. `true_new` and
    `pending` come from `adjudication`; with neither, adjudicated precision equals what a
    run with no human review can honestly claim.
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

    implicit = [item for item in gold if IMPLICIT_CLASS in item.classes]
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
        implicit_recovery=_recall_of(implicit, report).value,
        implicit_total=len(implicit),
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
    )

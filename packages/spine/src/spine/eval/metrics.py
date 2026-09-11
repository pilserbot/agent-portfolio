"""Pure functions that aggregate Verdicts into the numbers a report shows.

Every function here is deterministic arithmetic over a list of Verdicts: no I/O, no state,
no model. Given the same Verdicts they return the same figures, which is what makes an
evaluation comparable between runs.

Two different notions of "right" are in play, and keeping them apart matters:

- `accuracy` counts `verdict.passed`, the checker's own ruling. A numeric answer inside a
  tolerance passed, even though its text differs from the expected text.
- `precision`, `recall` and `f1` compare `verdict.expected` against `verdict.actual` as
  labels, against a named positive label. They are classification metrics and need to know
  which label is the positive one.

Conventions for the degenerate cases, chosen once and applied everywhere: a metric whose
denominator is zero is 0.0 with a support of 0, rather than an error or a NaN. An empty
input is not a failure — it is an evaluation of nothing, and reports as zero.

Deliberately does not: decide what counts as correct. That happened already, in
`spine.eval.checkers`; this module only counts. It also does not weight, threshold or
otherwise tune anything — a metric that quietly flattered a run would be worse than none.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import Verdict

DEFAULT_CALIBRATION_BINS = 10

__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "CalibrationBin",
    "CalibrationResult",
    "ConfusionCounts",
    "MapeResult",
    "MetricValue",
    "TagMacroF1",
    "accuracy",
    "confusion",
    "expected_calibration_error",
    "f1",
    "macro_f1_by_tag",
    "mean_absolute_percentage_error",
    "precision",
    "recall",
]


class MetricValue(BaseModel):
    """One computed metric, with the number of items behind it."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float = Field(ge=0.0)
    support: int = Field(ge=0, description="How many items contributed to the value.")


class ConfusionCounts(BaseModel):
    """The four counts every classification metric is built from."""

    model_config = ConfigDict(frozen=True)

    positive_label: str
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_negative: int = Field(ge=0)

    @property
    def total(self) -> int:
        """Every item counted."""
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def predicted_positive(self) -> int:
        """Items the system labelled positive."""
        return self.true_positive + self.false_positive

    @property
    def actual_positive(self) -> int:
        """Items that are positive in the gold set."""
        return self.true_positive + self.false_negative


class TagMacroF1(BaseModel):
    """F1 per tag, and the unweighted mean across tags."""

    model_config = ConfigDict(frozen=True)

    positive_label: str
    per_tag: dict[str, MetricValue] = Field(default_factory=dict)
    macro_f1: float = Field(default=0.0, ge=0.0)
    tags: list[str] = Field(default_factory=list)


class MapeResult(BaseModel):
    """Mean absolute percentage error, and what had to be left out of it."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, description="Mean absolute percentage error, as a fraction.")
    support: int = Field(ge=0)
    skipped_zero_expected: int = Field(
        default=0, ge=0, description="Items excluded because the expected value was zero."
    )
    skipped_unparsable: int = Field(
        default=0, ge=0, description="Items excluded because a value was not a number."
    )


class CalibrationBin(BaseModel):
    """One confidence bin: how many landed in it, how sure they were, how often they were right."""

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)

    @property
    def gap(self) -> float:
        """How far this bin's confidence sits from its accuracy."""
        return abs(self.mean_confidence - self.accuracy)


class CalibrationResult(BaseModel):
    """Expected calibration error, and the bins it was computed from."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    bin_count: int = Field(gt=0)
    support: int = Field(ge=0)
    bins: list[CalibrationBin] = Field(default_factory=list)


def confusion(verdicts: Sequence[Verdict], *, positive_label: str) -> ConfusionCounts:
    """Count true and false positives and negatives against a named positive label.

    A verdict is a predicted positive when its `actual` equals the label, and an actual
    positive when its `expected` does.
    """
    true_positive = false_positive = false_negative = true_negative = 0
    for verdict in verdicts:
        predicted = verdict.actual == positive_label
        truth = verdict.expected == positive_label
        if predicted and truth:
            true_positive += 1
        elif predicted and not truth:
            false_positive += 1
        elif not predicted and truth:
            false_negative += 1
        else:
            true_negative += 1
    return ConfusionCounts(
        positive_label=positive_label,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """A ratio, with a zero denominator reported as zero rather than raising."""
    return numerator / denominator if denominator else 0.0


def precision(verdicts: Sequence[Verdict], *, positive_label: str) -> MetricValue:
    """Of the items called positive, the fraction that are positive.

    Support is the number of predicted positives: with none, precision is 0.0 over 0.
    """
    counts = confusion(verdicts, positive_label=positive_label)
    return MetricValue(
        name="precision",
        value=_ratio(counts.true_positive, counts.predicted_positive),
        support=counts.predicted_positive,
    )


def recall(verdicts: Sequence[Verdict], *, positive_label: str) -> MetricValue:
    """Of the items that are positive, the fraction that were called positive.

    Support is the number of actual positives: with none, recall is 0.0 over 0.
    """
    counts = confusion(verdicts, positive_label=positive_label)
    return MetricValue(
        name="recall",
        value=_ratio(counts.true_positive, counts.actual_positive),
        support=counts.actual_positive,
    )


def f1(verdicts: Sequence[Verdict], *, positive_label: str) -> MetricValue:
    """The harmonic mean of precision and recall, zero when both are zero.

    Support is the number of actual positives, matching recall, since that is the
    population the score speaks about.
    """
    computed_precision = precision(verdicts, positive_label=positive_label).value
    computed_recall = recall(verdicts, positive_label=positive_label).value
    total = computed_precision + computed_recall
    value = 2 * computed_precision * computed_recall / total if total else 0.0
    counts = confusion(verdicts, positive_label=positive_label)
    return MetricValue(name="f1", value=value, support=counts.actual_positive)


def accuracy(verdicts: Sequence[Verdict]) -> MetricValue:
    """The fraction of items the checker passed.

    This reads `passed`, not the labels: a numeric answer inside its tolerance passed even
    though its text differs from the expected text.
    """
    total = len(verdicts)
    correct = sum(1 for verdict in verdicts if verdict.passed)
    return MetricValue(name="accuracy", value=_ratio(correct, total), support=total)


def macro_f1_by_tag(
    verdicts: Sequence[Verdict],
    tags_by_item: Mapping[str, Sequence[str]],
    *,
    positive_label: str,
) -> TagMacroF1:
    """F1 within each tag, and the unweighted mean across tags.

    Unweighted on purpose: the macro average exists so a small tag cannot be drowned out by
    a large one. An item may carry several tags and counts once in each. Items whose id is
    absent from the mapping, or which carry no tags, contribute to no tag.
    """
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        for tag in tags_by_item.get(verdict.item_id, ()):
            grouped.setdefault(tag, []).append(verdict)

    per_tag = {
        tag: f1(group, positive_label=positive_label) for tag, group in sorted(grouped.items())
    }
    macro = sum(metric.value for metric in per_tag.values()) / len(per_tag) if per_tag else 0.0
    return TagMacroF1(
        positive_label=positive_label,
        per_tag=per_tag,
        macro_f1=macro,
        tags=sorted(per_tag),
    )


def mean_absolute_percentage_error(verdicts: Sequence[Verdict]) -> MapeResult:
    """Mean of |expected - actual| / |expected|, over items where that is defined.

    Returned as a fraction, not a percentage: 0.05 is five per cent. Items with an expected
    value of zero have no defined percentage error and are excluded rather than skewing the
    mean toward infinity; the count of those, and of items whose values are not numbers, is
    reported so an excluded majority cannot hide behind a flattering figure.
    """
    errors: list[float] = []
    skipped_zero = 0
    skipped_unparsable = 0

    for verdict in verdicts:
        try:
            expected_value = float(verdict.expected)
            actual_value = float(verdict.actual)
        except (TypeError, ValueError):
            skipped_unparsable += 1
            continue
        if expected_value == 0:
            skipped_zero += 1
            continue
        errors.append(abs(expected_value - actual_value) / abs(expected_value))

    value = sum(errors) / len(errors) if errors else 0.0
    return MapeResult(
        value=value,
        support=len(errors),
        skipped_zero_expected=skipped_zero,
        skipped_unparsable=skipped_unparsable,
    )


def expected_calibration_error(
    verdicts: Sequence[Verdict], *, bins: int = DEFAULT_CALIBRATION_BINS
) -> CalibrationResult:
    """How far the scores' confidence sits from their actual accuracy.

    Verdict scores are treated as confidence and bucketed into `bins` equal-width bins over
    0..1; each bin's gap between mean confidence and observed accuracy is averaged, weighted
    by how many items landed in it. A score of exactly 1.0 belongs to the last bin.

    Empty bins contribute nothing and are still reported, so a result shows where the items
    actually were rather than only the headline number.
    """
    if bins <= 0:
        raise ValueError(f"bin count must be positive; got {bins}")

    buckets: list[list[Verdict]] = [[] for _ in range(bins)]
    for verdict in verdicts:
        index = min(int(verdict.score * bins), bins - 1)
        buckets[index].append(verdict)

    total = len(verdicts)
    reported: list[CalibrationBin] = []
    weighted_gap = 0.0

    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        count = len(bucket)
        mean_confidence = sum(v.score for v in bucket) / count if count else 0.0
        bin_accuracy = sum(1 for v in bucket if v.passed) / count if count else 0.0
        reported.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=bin_accuracy,
            )
        )
        if count:
            weighted_gap += (count / total) * abs(mean_confidence - bin_accuracy)

    return CalibrationResult(value=weighted_gap, bin_count=bins, support=total, bins=reported)

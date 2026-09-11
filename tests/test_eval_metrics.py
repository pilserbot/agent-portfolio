"""Unit tests for the evaluation metrics.

Every expected figure below is worked out by hand in a comment beside the assertion. A
metric that silently changed definition would otherwise still pass its own tests.

Deliberately does not compare against another library's implementation: the point is that
these numbers are what this project means by them.
"""

import pytest

from spine.contracts import Verdict
from spine.eval.metrics import (
    accuracy,
    confusion,
    expected_calibration_error,
    f1,
    macro_f1_by_tag,
    mean_absolute_percentage_error,
    precision,
    recall,
)

YES = "yes"


def a_verdict(
    item_id: str,
    expected: str,
    actual: str,
    *,
    passed: bool | None = None,
    score: float = 1.0,
) -> Verdict:
    return Verdict(
        item_id=item_id,
        expected=expected,
        actual=actual,
        passed=expected == actual if passed is None else passed,
        score=score,
        rationale="test fixture",
    )


def a_population() -> list[Verdict]:
    """4 true positives, 2 false positives, 3 false negatives, 1 true negative."""
    verdicts = []
    verdicts += [a_verdict(f"tp-{n}", "yes", "yes") for n in range(4)]
    verdicts += [a_verdict(f"fp-{n}", "no", "yes") for n in range(2)]
    verdicts += [a_verdict(f"fn-{n}", "yes", "no") for n in range(3)]
    verdicts += [a_verdict("tn-0", "no", "no")]
    return verdicts


# --- confusion counts --------------------------------------------------------------------


def test_the_four_counts() -> None:
    counts = confusion(a_population(), positive_label=YES)

    assert (counts.true_positive, counts.false_positive) == (4, 2)
    assert (counts.false_negative, counts.true_negative) == (3, 1)
    assert counts.total == 10
    assert counts.predicted_positive == 6  # 4 TP + 2 FP
    assert counts.actual_positive == 7  # 4 TP + 3 FN


def test_the_positive_label_decides_the_counts() -> None:
    # Flipping which label is positive swaps positives for negatives.
    counts = confusion(a_population(), positive_label="no")

    assert counts.true_positive == 1  # the single expected-no, actual-no
    assert counts.false_positive == 3  # expected yes, called no
    assert counts.false_negative == 2  # expected no, called yes
    assert counts.true_negative == 4


# --- precision, recall, f1 -----------------------------------------------------------------


def test_precision_is_tp_over_predicted_positive() -> None:
    # 4 / (4 + 2) = 4/6 = 0.6666...
    metric = precision(a_population(), positive_label=YES)

    assert metric.value == pytest.approx(4 / 6)
    assert metric.support == 6
    assert metric.name == "precision"


def test_recall_is_tp_over_actual_positive() -> None:
    # 4 / (4 + 3) = 4/7 = 0.5714...
    metric = recall(a_population(), positive_label=YES)

    assert metric.value == pytest.approx(4 / 7)
    assert metric.support == 7


def test_f1_is_the_harmonic_mean() -> None:
    # P = 2/3, R = 4/7. P+R = 26/21, P*R = 8/21.
    # F1 = 2 * (8/21) / (26/21) = 16/26 = 8/13 = 0.615384...
    metric = f1(a_population(), positive_label=YES)

    assert metric.value == pytest.approx(8 / 13)
    assert metric.value == pytest.approx(0.6153846153846154)
    assert metric.support == 7


def test_a_perfect_classifier_scores_one() -> None:
    verdicts = [a_verdict("a", "yes", "yes"), a_verdict("b", "no", "no")]

    assert precision(verdicts, positive_label=YES).value == 1.0
    assert recall(verdicts, positive_label=YES).value == 1.0
    assert f1(verdicts, positive_label=YES).value == 1.0
    assert accuracy(verdicts).value == 1.0


def test_no_predicted_positives_gives_zero_precision_over_zero_support() -> None:
    # Nothing was called positive: 0/0 is reported as 0.0 rather than raising.
    verdicts = [a_verdict("a", "yes", "no"), a_verdict("b", "no", "no")]

    metric = precision(verdicts, positive_label=YES)

    assert metric.value == 0.0
    assert metric.support == 0


def test_no_actual_positives_gives_zero_recall_over_zero_support() -> None:
    verdicts = [a_verdict("a", "no", "no"), a_verdict("b", "no", "yes")]

    metric = recall(verdicts, positive_label=YES)

    assert metric.value == 0.0
    assert metric.support == 0


def test_f1_is_zero_when_precision_and_recall_are_both_zero() -> None:
    # Every guess wrong in both directions.
    verdicts = [a_verdict("a", "yes", "no"), a_verdict("b", "no", "yes")]

    assert f1(verdicts, positive_label=YES).value == 0.0


def test_every_metric_is_zero_on_empty_input() -> None:
    assert precision([], positive_label=YES).value == 0.0
    assert recall([], positive_label=YES).value == 0.0
    assert f1([], positive_label=YES).value == 0.0
    assert accuracy([]).value == 0.0
    assert accuracy([]).support == 0


# --- accuracy ------------------------------------------------------------------------------


def test_accuracy_counts_passed_not_label_equality() -> None:
    # A numeric answer inside its tolerance passed even though the texts differ; accuracy
    # must follow the checker's ruling rather than re-deciding it.
    verdicts = [
        a_verdict("a", "100", "102", passed=True),
        a_verdict("b", "100", "999", passed=False),
        a_verdict("c", "yes", "yes", passed=True),
        a_verdict("d", "yes", "no", passed=False),
    ]

    metric = accuracy(verdicts)

    assert metric.value == 0.5  # 2 of 4 passed
    assert metric.support == 4


def test_accuracy_when_nothing_passed() -> None:
    verdicts = [a_verdict("a", "yes", "no"), a_verdict("b", "yes", "no")]

    assert accuracy(verdicts).value == 0.0


# --- macro F1 by tag -------------------------------------------------------------------------


def test_macro_f1_averages_tags_without_weighting() -> None:
    # tag "x": a is TP, b is FN  -> P=1/1=1, R=1/2=0.5, F1 = 2*1*0.5/1.5 = 2/3
    # tag "y": c is TP, d is TN  -> P=1/1=1, R=1/1=1,   F1 = 1
    # macro = (2/3 + 1) / 2 = 5/6 = 0.8333...
    verdicts = [
        a_verdict("a", "yes", "yes"),
        a_verdict("b", "yes", "no"),
        a_verdict("c", "yes", "yes"),
        a_verdict("d", "no", "no"),
    ]
    tags = {"a": ["x"], "b": ["x"], "c": ["y"], "d": ["y"]}

    result = macro_f1_by_tag(verdicts, tags, positive_label=YES)

    assert result.per_tag["x"].value == pytest.approx(2 / 3)
    assert result.per_tag["y"].value == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(5 / 6)
    assert result.tags == ["x", "y"]


def test_a_small_tag_is_not_drowned_out_by_a_large_one() -> None:
    # tag "big": 6 perfect items -> F1 = 1. tag "small": 1 item, wrong -> F1 = 0.
    # Macro = 0.5 even though 6 of 7 items are right; that is the point of a macro average.
    verdicts = [a_verdict(f"b-{n}", "yes", "yes") for n in range(6)]
    verdicts.append(a_verdict("s-0", "yes", "no"))
    tags = {f"b-{n}": ["big"] for n in range(6)} | {"s-0": ["small"]}

    result = macro_f1_by_tag(verdicts, tags, positive_label=YES)

    assert result.macro_f1 == pytest.approx(0.5)


def test_an_item_with_several_tags_counts_in_each() -> None:
    verdicts = [a_verdict("a", "yes", "yes"), a_verdict("b", "yes", "no")]
    tags = {"a": ["x", "y"], "b": ["y"]}

    result = macro_f1_by_tag(verdicts, tags, positive_label=YES)

    assert result.per_tag["x"].value == 1.0  # only item a
    # tag y: a is TP, b is FN -> P=1, R=0.5, F1=2/3
    assert result.per_tag["y"].value == pytest.approx(2 / 3)


def test_an_untagged_item_contributes_to_no_tag() -> None:
    verdicts = [a_verdict("a", "yes", "yes"), a_verdict("orphan", "yes", "no")]

    result = macro_f1_by_tag(verdicts, {"a": ["x"]}, positive_label=YES)

    assert result.tags == ["x"]
    assert result.per_tag["x"].value == 1.0


def test_macro_f1_with_no_tags_at_all_is_zero() -> None:
    result = macro_f1_by_tag([a_verdict("a", "yes", "yes")], {}, positive_label=YES)

    assert result.macro_f1 == 0.0
    assert result.tags == []


def test_macro_f1_on_empty_input_is_zero() -> None:
    assert macro_f1_by_tag([], {}, positive_label=YES).macro_f1 == 0.0


# --- mean absolute percentage error -------------------------------------------------------------


def test_mape_on_hand_computed_values() -> None:
    # |100-110|/100 = 0.10; |200-180|/200 = 0.10; |50-50|/50 = 0.
    # mean = (0.10 + 0.10 + 0) / 3 = 0.0666...
    verdicts = [
        a_verdict("a", "100", "110"),
        a_verdict("b", "200", "180"),
        a_verdict("c", "50", "50"),
    ]

    result = mean_absolute_percentage_error(verdicts)

    assert result.value == pytest.approx(0.2 / 3)
    assert result.support == 3
    assert result.skipped_zero_expected == 0


def test_mape_is_a_fraction_not_a_percentage() -> None:
    # |100-105|/100 = 0.05, reported as 0.05 rather than 5.
    assert mean_absolute_percentage_error([a_verdict("a", "100", "105")]).value == pytest.approx(
        0.05
    )


def test_mape_is_zero_when_every_answer_is_exact() -> None:
    verdicts = [a_verdict("a", "10", "10"), a_verdict("b", "20", "20")]

    assert mean_absolute_percentage_error(verdicts).value == 0.0


def test_mape_excludes_a_zero_expected_value_and_says_so() -> None:
    # A percentage of zero is undefined; excluding it beats a mean of infinity.
    verdicts = [a_verdict("a", "100", "110"), a_verdict("zero", "0", "5")]

    result = mean_absolute_percentage_error(verdicts)

    assert result.value == pytest.approx(0.10), "only the defined item counts"
    assert result.support == 1
    assert result.skipped_zero_expected == 1


def test_mape_excludes_values_that_are_not_numbers_and_says_so() -> None:
    verdicts = [a_verdict("a", "100", "110"), a_verdict("text", "yes", "no")]

    result = mean_absolute_percentage_error(verdicts)

    assert result.support == 1
    assert result.skipped_unparsable == 1


def test_mape_on_empty_input_is_zero() -> None:
    result = mean_absolute_percentage_error([])

    assert result.value == 0.0 and result.support == 0


def test_mape_when_everything_was_excluded() -> None:
    # No defined item survives; the figure is 0.0 over a support of 0, and the counts say why.
    result = mean_absolute_percentage_error([a_verdict("z", "0", "1")])

    assert result.value == 0.0
    assert result.support == 0
    assert result.skipped_zero_expected == 1


def test_mape_handles_an_over_estimate_and_an_under_estimate_alike() -> None:
    # |100-150|/100 = 0.5 and |100-50|/100 = 0.5; mean 0.5.
    verdicts = [a_verdict("a", "100", "150"), a_verdict("b", "100", "50")]

    assert mean_absolute_percentage_error(verdicts).value == pytest.approx(0.5)


# --- expected calibration error -------------------------------------------------------------------


def test_ece_on_hand_computed_bins() -> None:
    # Two bins over [0,1). Bin index = min(int(score*2), 1).
    # bin 1 (0.5..1.0): scores 0.9 T, 0.8 T, 0.7 F
    #   mean confidence = (0.9+0.8+0.7)/3 = 0.8, accuracy = 2/3, gap = |0.8 - 2/3| = 0.13333
    # bin 0 (0.0..0.5): score 0.1 F
    #   mean confidence = 0.1, accuracy = 0.0, gap = 0.1
    # ECE = (3/4)*0.133333 + (1/4)*0.1 = 0.1 + 0.025 = 0.125
    verdicts = [
        a_verdict("a", "yes", "yes", passed=True, score=0.9),
        a_verdict("b", "yes", "yes", passed=True, score=0.8),
        a_verdict("c", "yes", "no", passed=False, score=0.7),
        a_verdict("d", "yes", "no", passed=False, score=0.1),
    ]

    result = expected_calibration_error(verdicts, bins=2)

    assert result.value == pytest.approx(0.125)
    assert result.bin_count == 2
    assert result.support == 4
    assert [b.count for b in result.bins] == [1, 3]


def test_a_perfectly_calibrated_set_scores_zero() -> None:
    # Full confidence and right; no confidence and wrong.
    verdicts = [
        a_verdict("a", "yes", "yes", passed=True, score=1.0),
        a_verdict("b", "yes", "no", passed=False, score=0.0),
    ]

    assert expected_calibration_error(verdicts).value == pytest.approx(0.0)


def test_total_overconfidence_scores_one() -> None:
    # Certain every time and wrong every time: the gap is the whole range.
    verdicts = [a_verdict(f"a-{n}", "yes", "no", passed=False, score=1.0) for n in range(4)]

    assert expected_calibration_error(verdicts).value == pytest.approx(1.0)


def test_a_score_of_exactly_one_lands_in_the_last_bin() -> None:
    result = expected_calibration_error(
        [a_verdict("a", "yes", "yes", passed=True, score=1.0)], bins=10
    )

    assert result.bins[-1].count == 1
    assert result.bins[-1].lower == pytest.approx(0.9)


def test_the_bin_count_is_configurable_and_bins_are_all_reported() -> None:
    verdicts = [a_verdict("a", "yes", "yes", passed=True, score=0.5)]

    for bins in (1, 2, 5, 20):
        result = expected_calibration_error(verdicts, bins=bins)
        assert len(result.bins) == bins
        assert sum(b.count for b in result.bins) == 1


def test_empty_bins_are_reported_without_contributing() -> None:
    # One item at 0.95 with ten bins: nine bins are empty and must not drag the figure.
    result = expected_calibration_error(
        [a_verdict("a", "yes", "yes", passed=True, score=0.95)], bins=10
    )

    assert result.value == pytest.approx(0.05)  # |0.95 - 1.0|
    assert sum(1 for b in result.bins if b.count == 0) == 9


def test_ece_on_empty_input_is_zero() -> None:
    result = expected_calibration_error([])

    assert result.value == 0.0 and result.support == 0
    assert len(result.bins) == 10


def test_a_non_positive_bin_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="bin count"):
        expected_calibration_error([], bins=0)


def test_the_bin_gap_is_reported_per_bin() -> None:
    verdicts = [a_verdict("a", "yes", "no", passed=False, score=0.85)]

    result = expected_calibration_error(verdicts, bins=10)
    occupied = next(b for b in result.bins if b.count)

    assert occupied.mean_confidence == pytest.approx(0.85)
    assert occupied.accuracy == 0.0
    assert occupied.gap == pytest.approx(0.85)


# --- the three modules compose ----------------------------------------------------------------


def test_a_gold_set_scored_by_a_checker_aggregates_into_metrics() -> None:
    # datasets -> checkers -> metrics, over the committed example set, with no model.
    from pathlib import Path

    from spine.eval.checkers import exact_match
    from spine.eval.datasets import DEFAULT_GOLD_ROOT, load_gold_set

    gold = load_gold_set("example", root=Path(__file__).resolve().parents[1] / DEFAULT_GOLD_ROOT)

    # A stand-in system that gets every "yes" right and calls one "no" a "yes".
    def answer(item_id: str, expected: str) -> str:
        return "yes" if item_id == "ex-002" else expected

    verdicts = [
        exact_match(
            str(item.expected["is_requirement"]),
            answer(item.item_id, str(item.expected["is_requirement"])),
            item_id=item.item_id,
        )
        for item in gold.items
    ]

    # The example set holds 7 "yes" and 3 "no"; ex-002 is a "no" called "yes".
    counts = confusion(verdicts, positive_label=YES)
    assert (counts.true_positive, counts.false_positive) == (7, 1)
    assert (counts.false_negative, counts.true_negative) == (0, 2)

    assert precision(verdicts, positive_label=YES).value == pytest.approx(7 / 8)
    assert recall(verdicts, positive_label=YES).value == pytest.approx(1.0)
    # F1 = 2 * (7/8) * 1 / (7/8 + 1) = (7/4) / (15/8) = 14/15
    assert f1(verdicts, positive_label=YES).value == pytest.approx(14 / 15)
    assert accuracy(verdicts).value == pytest.approx(0.9)  # 9 of 10 exact

    by_tag = macro_f1_by_tag(verdicts, gold.tags_by_item(), positive_label=YES)
    assert "requirement" in by_tag.tags
    assert by_tag.per_tag["requirement"].value == pytest.approx(1.0), "no requirement missed"

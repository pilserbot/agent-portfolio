"""Unit tests for the regression comparison and the standalone gate command.

The runner gates itself on the same code these tests exercise, so a gate that disagreed
with the runner would fail here rather than in CI.

Deliberately does not touch the network, the filesystem outside pytest's tmp_path, or any
environment variable.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from spine.contracts import KPISnapshot
from spine.eval.baseline import (
    Baseline,
    MetricReading,
    MetricReadings,
    baseline_from_snapshots,
    check_regressions,
    readings_from_snapshots,
)
from spine.eval.gate import evaluate_gate
from spine.eval.gate import main as gate_main
from spine.eval.run import EvalReport

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def some_readings(**values: float) -> MetricReadings:
    return MetricReadings(
        readings=[MetricReading(metric=name, value=value) for name, value in values.items()]
    )


def a_snapshot(name: str, value: float, **overrides: object) -> KPISnapshot:
    settings: dict[str, object] = {
        "project": "example",
        "metric_name": name,
        "value": value,
        "unit": "ratio",
        "target": 0.8,
        "direction": "higher_is_better",
        "measured_at": AT,
        "sample_size": 10,
        "method": "test",
    }
    settings.update(overrides)
    return KPISnapshot.model_validate(settings)


def a_report(*snapshots: KPISnapshot) -> EvalReport:
    return EvalReport(
        project="example",
        scope="full",
        sample_size=None,
        items_evaluated=10,
        gold_set="example",
        run_id="run-1",
        measured_at=AT,
        snapshots=list(snapshots),
    )


# --- the comparison ---------------------------------------------------------------------


def test_an_empty_baseline_compares_nothing_and_passes() -> None:
    verdict = check_regressions(Baseline.model_validate_json("{}"), some_readings(accuracy=0.9))

    assert verdict.passed
    assert verdict.compared == 0


def test_a_drop_beyond_tolerance_is_a_regression() -> None:
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9, "tolerance": 0.02}}})

    verdict = check_regressions(baseline, some_readings(accuracy=0.85))

    assert not verdict.passed
    assert [r.metric for r in verdict.regressions] == ["accuracy"]
    # 0.9 - 0.85 = 0.05, against a tolerance of 0.02.
    assert verdict.regressions[0].delta == pytest.approx(0.05)


def test_a_drift_within_tolerance_passes() -> None:
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9, "tolerance": 0.05}}})

    assert check_regressions(baseline, some_readings(accuracy=0.86)).passed


def test_a_drift_exactly_at_the_tolerance_passes() -> None:
    # The boundary is inclusive: a tolerance of 0.05 permits a drop of 0.05.
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9, "tolerance": 0.05}}})

    assert check_regressions(baseline, some_readings(accuracy=0.85)).passed


def test_an_improvement_is_never_a_regression() -> None:
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9}}})

    assert check_regressions(baseline, some_readings(accuracy=0.99)).passed


def test_a_lower_is_better_metric_regresses_when_it_rises() -> None:
    # The direction is why one rule catches both a falling accuracy and a rising cost.
    baseline = Baseline.model_validate(
        {"metrics": {"cost": {"value": 0.01, "tolerance": 0.0, "direction": "lower_is_better"}}}
    )

    rose = check_regressions(baseline, some_readings(cost=0.02))
    fell = check_regressions(baseline, some_readings(cost=0.005))

    assert not rose.passed
    assert fell.passed


def test_a_baseline_metric_absent_from_the_report_is_reported_not_failed() -> None:
    baseline = Baseline.model_validate({"metrics": {"nonexistent": {"value": 1.0}}})

    verdict = check_regressions(baseline, some_readings(accuracy=0.9))

    assert verdict.passed
    assert verdict.missing == ["nonexistent"]
    assert verdict.compared == 0


def test_a_metric_the_baseline_does_not_record_is_simply_not_compared() -> None:
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9}}})

    verdict = check_regressions(baseline, some_readings(accuracy=0.9, brand_new=0.1))

    assert verdict.passed
    assert verdict.compared == 1


# --- turning a run into the next baseline -----------------------------------------------


def test_a_new_baseline_takes_its_values_from_the_snapshots() -> None:
    baseline = baseline_from_snapshots([a_snapshot("accuracy", 0.9), a_snapshot("f1", 0.8)])

    assert {name: metric.value for name, metric in baseline.metrics.items()} == {
        "accuracy": 0.9,
        "f1": 0.8,
    }


def test_a_new_baseline_keeps_each_metrics_direction() -> None:
    # Accepting a new baseline must not quietly flip which way a metric should move.
    baseline = baseline_from_snapshots(
        [a_snapshot("cost", 0.01, direction="lower_is_better", unit="usd")]
    )

    assert baseline.metrics["cost"].direction == "lower_is_better"


def test_tolerances_come_from_the_project_not_from_the_measurement() -> None:
    # How much drift is acceptable is a decision, not an observation.
    baseline = baseline_from_snapshots(
        [a_snapshot("accuracy", 0.9), a_snapshot("f1", 0.8)], tolerances={"accuracy": 0.02}
    )

    assert baseline.metrics["accuracy"].tolerance == 0.02
    assert baseline.metrics["f1"].tolerance == 0.0


def test_a_run_compared_against_its_own_results_always_passes() -> None:
    snapshots = [a_snapshot("accuracy", 0.9), a_snapshot("f1", 0.8)]

    verdict = check_regressions(
        baseline_from_snapshots(snapshots), readings_from_snapshots(snapshots)
    )

    assert verdict.passed
    assert verdict.compared == 2


# --- the standalone command -------------------------------------------------------------


def test_the_gate_reads_the_same_numbers_the_runner_wrote(tmp_path: Path) -> None:
    report = a_report(a_snapshot("accuracy", 0.9))
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"metrics": {"accuracy": {"value": 0.95, "tolerance": 0.01}}}), encoding="utf-8"
    )

    assert gate_main(["--report", str(path), "--baseline", str(baseline_path)]) == 1


def test_the_gate_exits_zero_when_nothing_regressed(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(a_report(a_snapshot("accuracy", 0.9)).model_dump_json(), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text('{"metrics": {"accuracy": {"value": 0.9}}}', encoding="utf-8")

    assert gate_main(["--report", str(path), "--baseline", str(baseline_path)]) == 0


def test_the_gate_and_the_comparison_agree_on_a_report() -> None:
    report = a_report(a_snapshot("accuracy", 0.7))
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9}}})

    assert evaluate_gate(baseline, report) == check_regressions(
        baseline, readings_from_snapshots(report.snapshots)
    )


def test_a_drift_a_whisker_beyond_the_tolerance_is_still_a_regression() -> None:
    # The boundary is forgiving of float noise, not of a real drop. 0.05 + 1e-6 is not noise.
    baseline = Baseline.model_validate({"metrics": {"accuracy": {"value": 0.9, "tolerance": 0.05}}})

    assert not check_regressions(baseline, some_readings(accuracy=0.849999)).passed

"""Unit tests for KPI computation, the ROI model, rendering and the replay-exclusion rule.

Every expected value here is worked out by hand and the arithmetic is written beside the
assertion, so a change to a formula shows up as a failing number rather than as a passing
rewrite.

Deliberately does not: touch a network, read a config file, or construct a Router. The
runs and Verdicts are built inline.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from spine.contracts import AgentRun, CallMode, KPISnapshot, ModelCall, StepTrace, Verdict
from spine.kpi import (
    KPIError,
    KPISpec,
    KPISpecSet,
    NoPaybackError,
    ReplayedCostError,
    ROIInputs,
    RoleHours,
    compute_kpis,
    compute_roi,
    cost_basis,
    meets_target,
    render_markdown,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def a_call(cost: str, *, mode: CallMode = "live", latency_ms: int = 100) -> ModelCall:
    return ModelCall(
        provider="anthropic",
        model="claude-sonnet-5",
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd=Decimal(cost),
        latency_ms=latency_ms,
        timestamp=AT,
        purpose="extract",
        mode=mode,
    )


def a_run(*calls: ModelCall, status: str = "completed", failed: int = 0) -> AgentRun:
    steps = [
        StepTrace(
            step_name="extract",
            started_at=AT,
            finished_at=AT,
            status="ok",
            model_calls=list(calls),
        )
    ]
    steps += [
        StepTrace(step_name=f"broken-{n}", started_at=AT, finished_at=AT, status="failed")
        for n in range(failed)
    ]
    return AgentRun(
        run_id="run-1",
        project="example",
        started_at=AT,
        finished_at=AT,
        steps=steps,
        status="completed_with_errors" if failed else status,
    )


def a_verdict(item_id: str, *, expected: str, actual: str, score: float | None = None) -> Verdict:
    passed = expected == actual
    return Verdict(
        item_id=item_id,
        expected=expected,
        actual=actual,
        passed=passed,
        score=(1.0 if passed else 0.0) if score is None else score,
        rationale="hand-built for a test",
    )


def a_spec(**overrides: object) -> KPISpec:
    settings: dict[str, object] = {
        "name": "accuracy",
        "unit": "ratio",
        "target": 0.8,
        "direction": "higher_is_better",
        "method": "test",
        "computation": "accuracy",
    }
    settings.update(overrides)
    return KPISpec.model_validate(settings)


def a_spec_set(*specs: KPISpec, **overrides: object) -> KPISpecSet:
    settings: dict[str, object] = {"project": "example", "kpis": list(specs)}
    settings.update(overrides)
    return KPISpecSet.model_validate(settings)


def some_roi(**overrides: object) -> ROIInputs:
    settings: dict[str, object] = {
        "unit": "tender",
        "period_label": "month",
        "units_per_period": 10,
        "implementation_cost_usd": "20000",
        "automated_model_cost_per_unit_usd": "0",
        "assumptions_source": "hand-built for a test",
        "baseline_roles": [
            {"role": "manager", "hours_per_unit": 6.0, "hourly_cost_usd": "100"},
            {"role": "reviewer", "hours_per_unit": 4.0, "hourly_cost_usd": "50"},
        ],
        "residual_roles": [
            {"role": "manager", "hours_per_unit": 1.0, "hourly_cost_usd": "100"},
        ],
    }
    settings.update(overrides)
    return ROIInputs.model_validate(settings)


# --- the replay-exclusion rule ----------------------------------------------------------
#
# The point of the whole `mode` field. A replayed call carries the cost the real call had,
# so a demo that counted it would report a per-unit cost that looks measured and is not,
# and a demo that dropped it would report a cost of nothing. Both are plausible and wrong,
# so the module refuses and makes the caller choose in the open.


def test_a_live_run_totals_what_it_spent() -> None:
    basis = cost_basis(a_run(a_call("0.02"), a_call("0.03")))

    assert basis.total_usd == Decimal("0.05")
    assert (basis.counted_calls, basis.billed_calls, basis.replayed_calls) == (2, 2, 0)
    assert basis.is_measured and not basis.includes_replayed


def test_a_replayed_run_refuses_to_state_a_cost() -> None:
    run = a_run(a_call("0.02", mode="replay"))

    with pytest.raises(ReplayedCostError) as error:
        cost_basis(run)

    assert "cassette" in str(error.value)
    assert "include_replayed_cost" in str(error.value)


def test_one_replayed_call_among_live_ones_is_enough_to_refuse() -> None:
    # A partly replayed run's spend is not the cost of doing the work either.
    run = a_run(a_call("0.02"), a_call("0.03", mode="replay"))

    with pytest.raises(ReplayedCostError):
        cost_basis(run)


def test_replayed_costs_are_counted_only_when_explicitly_included() -> None:
    run = a_run(a_call("0.02"), a_call("0.03", mode="replay"))

    basis = cost_basis(run, include_replayed=True)

    assert basis.total_usd == Decimal("0.05")
    assert basis.replayed_calls == 1
    assert basis.includes_replayed
    assert not basis.is_measured


def test_a_recorded_call_counts_as_spent_because_it_was() -> None:
    basis = cost_basis(a_run(a_call("0.04", mode="record")))

    assert basis.total_usd == Decimal("0.04")
    assert basis.is_measured


def test_a_replayed_run_cannot_silently_produce_a_cost_per_unit_number() -> None:
    # The headline rule. Asking for cost_per_item_usd over a replayed run raises rather
    # than answering 0.0 — which is what dropping the replayed calls would have reported.
    spec = a_spec_set(
        a_spec(
            name="cost_per_item_usd",
            unit="usd",
            target=0.05,
            computation="cost_per_item_usd",
            direction="lower_is_better",
            method="analysis",
        )
    )
    run = a_run(a_call("0.02", mode="replay"), a_call("0.02", mode="replay"))
    verdicts = [a_verdict("i-1", expected="yes", actual="yes")]

    with pytest.raises(ReplayedCostError):
        compute_kpis(verdicts, run, spec)


def test_the_same_replayed_run_answers_once_the_config_opts_in() -> None:
    spec = a_spec_set(
        a_spec(
            name="cost_per_item_usd",
            unit="usd",
            target=0.05,
            computation="cost_per_item_usd",
            direction="lower_is_better",
            method="analysis",
        ),
        include_replayed_cost=True,
    )
    run = a_run(a_call("0.02", mode="replay"), a_call("0.02", mode="replay"))
    verdicts = [a_verdict("i-1", expected="yes", actual="yes")]

    # 0.04 usd over one item.
    assert compute_kpis(verdicts, run, spec)[0].value == pytest.approx(0.04)


def test_a_replayed_run_still_reports_its_quality_metrics() -> None:
    # Only the money is in doubt; whether the answers were right is not.
    spec = a_spec_set(a_spec())
    run = a_run(a_call("0.02", mode="replay"))
    verdicts = [a_verdict("i-1", expected="yes", actual="yes")]

    assert compute_kpis(verdicts, run, spec)[0].value == 1.0


def test_nothing_is_measured_when_a_cost_metric_will_refuse() -> None:
    # Fails up front rather than part way, so a caller never gets a half-written report
    # whose quality numbers are real and whose money numbers are missing.
    spec = a_spec_set(
        a_spec(),
        a_spec(
            name="total_cost_usd",
            unit="usd",
            target=1.0,
            computation="total_cost_usd",
            direction="lower_is_better",
            method="analysis",
        ),
    )

    with pytest.raises(ReplayedCostError):
        compute_kpis(
            [a_verdict("i-1", expected="yes", actual="yes")],
            a_run(a_call("0.02", mode="replay")),
            spec,
        )


# --- KPI computation --------------------------------------------------------------------


def four_verdicts() -> list[Verdict]:
    # 3 of 4 correct. Labels: expected yes,yes,yes,no / actual yes,yes,no,no.
    # Against positive_label "yes": TP=2, FP=0, FN=1, TN=1.
    return [
        a_verdict("i-1", expected="yes", actual="yes"),
        a_verdict("i-2", expected="yes", actual="yes"),
        a_verdict("i-3", expected="yes", actual="no"),
        a_verdict("i-4", expected="no", actual="no"),
    ]


def test_accuracy_counts_the_checkers_own_ruling() -> None:
    spec = a_spec_set(a_spec())

    assert compute_kpis(four_verdicts(), a_run(), spec)[0].value == pytest.approx(0.75)


def test_precision_recall_and_f1_come_from_the_labels() -> None:
    spec = a_spec_set(
        a_spec(name="p", computation="precision", positive_label="yes"),
        a_spec(name="r", computation="recall", positive_label="yes"),
        a_spec(name="f", computation="f1", positive_label="yes"),
    )

    values = [snapshot.value for snapshot in compute_kpis(four_verdicts(), a_run(), spec)]

    # P = 2/2 = 1.0; R = 2/3 = 0.6667; F1 = 2 * (1 * 2/3) / (1 + 2/3) = (4/3)/(5/3) = 0.8.
    assert values == pytest.approx([1.0, 2 / 3, 0.8])


def test_macro_f1_averages_the_tags_unweighted() -> None:
    spec = a_spec_set(a_spec(name="m", computation="macro_f1_by_tag", positive_label="yes"))
    tags = {"i-1": ["a"], "i-2": ["a"], "i-3": ["b"], "i-4": ["b"]}

    value = compute_kpis(four_verdicts(), a_run(), spec, tags_by_item=tags)[0].value

    # Tag a: TP=2, FP=0, FN=0 -> F1 1.0. Tag b: TP=0, FP=0, FN=1 -> F1 0.0. Mean 0.5.
    assert value == pytest.approx(0.5)


def test_items_evaluated_counts_the_verdicts() -> None:
    spec = a_spec_set(a_spec(name="n", unit="count", target=1, computation="items_evaluated"))

    assert compute_kpis(four_verdicts(), a_run(), spec)[0].value == 4.0


def test_mean_score_averages_the_verdict_scores() -> None:
    spec = a_spec_set(a_spec(name="s", computation="mean_score"))
    verdicts = [
        a_verdict("i-1", expected="yes", actual="yes", score=1.0),
        a_verdict("i-2", expected="yes", actual="no", score=0.4),
        a_verdict("i-3", expected="yes", actual="no", score=0.1),
    ]

    # (1.0 + 0.4 + 0.1) / 3 = 0.5.
    assert compute_kpis(verdicts, a_run(), spec)[0].value == pytest.approx(0.5)


def test_run_level_metrics_read_the_run_not_the_verdicts() -> None:
    spec = a_spec_set(
        a_spec(
            name="failed",
            unit="count",
            target=0,
            direction="lower_is_better",
            computation="failed_steps",
        ),
        a_spec(
            name="tokens",
            unit="count",
            target=0,
            direction="lower_is_better",
            computation="total_tokens",
        ),
        a_spec(
            name="latency",
            unit="ms",
            target=0,
            direction="lower_is_better",
            computation="mean_latency_ms",
        ),
    )
    run = a_run(a_call("0.01", latency_ms=100), a_call("0.01", latency_ms=300), failed=2)

    values = [snapshot.value for snapshot in compute_kpis([], run, spec)]

    # Two failed steps; 2 calls x (100 prompt + 20 completion) = 240 tokens; (100+300)/2 = 200ms.
    assert values == [2.0, 240.0, 200.0]


def test_cost_per_item_divides_the_measured_spend_by_the_items() -> None:
    spec = a_spec_set(
        a_spec(
            name="cost_per_item_usd",
            unit="usd",
            target=0.05,
            direction="lower_is_better",
            method="analysis",
            computation="cost_per_item_usd",
        )
    )

    value = compute_kpis(four_verdicts(), a_run(a_call("0.02"), a_call("0.02")), spec)[0].value

    # 0.04 usd over 4 items.
    assert value == pytest.approx(0.01)


def test_cost_per_item_over_nothing_is_zero_rather_than_a_division_by_zero() -> None:
    spec = a_spec_set(
        a_spec(
            name="cost_per_item_usd",
            unit="usd",
            target=0.05,
            direction="lower_is_better",
            method="analysis",
            computation="cost_per_item_usd",
        )
    )

    assert compute_kpis([], a_run(a_call("0.02")), spec)[0].value == 0.0


def test_every_snapshot_carries_the_project_sample_size_and_method() -> None:
    spec = a_spec_set(a_spec(method="inspection"))

    snapshot = compute_kpis(four_verdicts(), a_run(), spec, measured_at=AT)[0]

    assert snapshot.project == "example"
    assert snapshot.sample_size == 4
    assert snapshot.method == "inspection"
    assert snapshot.measured_at == AT


def test_metrics_are_measured_in_the_order_the_project_declared_them() -> None:
    spec = a_spec_set(a_spec(name="second"), a_spec(name="first"))

    names = [snapshot.metric_name for snapshot in compute_kpis([], a_run(), spec)]

    assert names == ["second", "first"]


# --- what a spec refuses to be ----------------------------------------------------------


def test_a_label_metric_without_a_positive_label_is_refused() -> None:
    with pytest.raises(ValidationError, match="positive_label"):
        a_spec(name="f", computation="f1")


def test_an_unknown_computation_is_refused() -> None:
    with pytest.raises(ValidationError):
        a_spec(computation="vibes")


def test_an_roi_metric_without_roi_inputs_is_refused() -> None:
    with pytest.raises(ValidationError, match="roi block"):
        a_spec_set(
            a_spec(
                name="payback",
                unit="months",
                target=12,
                direction="lower_is_better",
                method="analysis",
                computation="payback_periods",
            )
        )


def test_duplicate_kpi_names_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate KPI name"):
        a_spec_set(a_spec(name="same"), a_spec(name="same"))


def test_an_unknown_field_in_a_spec_is_refused() -> None:
    # A typo'd key would otherwise vanish and take its meaning with it.
    with pytest.raises(ValidationError):
        a_spec(tolerence=0.1)


# --- the ROI model ----------------------------------------------------------------------


def test_hours_displaced_is_the_baseline_less_what_a_human_still_does() -> None:
    result = compute_roi(some_roi())

    # Baseline 6 + 4 = 10 hours; residual 1 hour; 9 displaced per unit, 90 per month at 10.
    assert result.baseline_hours_per_unit == 10.0
    assert result.residual_hours_per_unit == 1.0
    assert result.hours_displaced_per_unit == 9.0
    assert result.hours_displaced_per_period == 90.0


def test_cost_per_unit_is_the_residual_human_cost_plus_the_model_cost() -> None:
    result = compute_roi(some_roi(automated_model_cost_per_unit_usd="2.50"))

    # Baseline: 6 x 100 + 4 x 50 = 800. Residual: 1 x 100 = 100. Automated: 100 + 2.50.
    assert result.baseline_cost_per_unit_usd == Decimal("800.0000")
    assert result.residual_human_cost_per_unit_usd == Decimal("100.0000")
    assert result.automated_cost_per_unit_usd == Decimal("102.5000")
    assert result.saving_per_unit_usd == Decimal("697.5000")


def test_payback_is_the_implementation_cost_over_the_saving_per_period() -> None:
    result = compute_roi(some_roi())

    # Saving 700/unit x 10 units = 7000/month; 20000 / 7000 = 2.857 months.
    assert result.saving_per_period_usd == Decimal("7000.0000")
    assert result.payback_periods == pytest.approx(20000 / 7000)


def test_a_system_that_saves_nothing_never_pays_back() -> None:
    # None rather than a negative or an infinite number, which would read like a schedule.
    result = compute_roi(
        some_roi(
            residual_roles=[{"role": "manager", "hours_per_unit": 6.0, "hourly_cost_usd": "100"}],
            automated_model_cost_per_unit_usd="500",
        )
    )

    # Automated 600 + 500 = 1100 against a baseline of 800: a loss of 300 per unit.
    assert result.saving_per_unit_usd == Decimal("-300.0000")
    assert result.payback_periods is None


def test_a_payback_kpi_over_a_system_that_never_pays_back_refuses() -> None:
    spec = a_spec_set(
        a_spec(
            name="payback",
            unit="months",
            target=12,
            direction="lower_is_better",
            method="analysis",
            computation="payback_periods",
        ),
        roi=some_roi(
            residual_roles=[{"role": "manager", "hours_per_unit": 6.0, "hourly_cost_usd": "100"}],
            automated_model_cost_per_unit_usd="500",
        ),
    )

    with pytest.raises(NoPaybackError, match="never does"):
        compute_kpis([], a_run(), spec)


def test_a_break_even_system_never_pays_back_either() -> None:
    # Exactly zero saving: the boundary is "no saving", not "a loss".
    result = compute_roi(
        some_roi(
            residual_roles=[{"role": "manager", "hours_per_unit": 8.0, "hourly_cost_usd": "100"}],
        )
    )

    assert result.saving_per_unit_usd == Decimal("0.0000")
    assert result.payback_periods is None


def test_an_roi_with_no_residual_human_work_displaces_every_hour() -> None:
    result = compute_roi(some_roi(residual_roles=[]))

    assert result.hours_displaced_per_unit == 10.0
    assert result.residual_human_cost_per_unit_usd == Decimal("0")


def test_residual_hours_above_the_baseline_are_refused() -> None:
    with pytest.raises(ValidationError, match="exceed the baseline"):
        some_roi(
            residual_roles=[{"role": "manager", "hours_per_unit": 99.0, "hourly_cost_usd": "100"}]
        )


def test_every_roi_assumption_is_required_with_no_default() -> None:
    # The rule that matters: no rate, volume or implementation cost is ever supplied by
    # this module. A set of inputs that omits one cannot be built at all.
    required = {
        "unit",
        "period_label",
        "units_per_period",
        "implementation_cost_usd",
        "automated_model_cost_per_unit_usd",
        "assumptions_source",
        "baseline_roles",
    }

    assert {name for name, field in ROIInputs.model_fields.items() if field.is_required()} == (
        required
    )


def test_a_role_needs_both_its_hours_and_its_rate() -> None:
    assert {name for name, field in RoleHours.model_fields.items() if field.is_required()} == {
        "role",
        "hours_per_unit",
        "hourly_cost_usd",
    }


def test_compute_kpis_prefers_the_measured_model_cost_over_the_assumed_one() -> None:
    # The config's figure is the assumption; what the run cost is the measurement.
    spec = a_spec_set(
        a_spec(
            name="cost_per_unit",
            unit="usd",
            target=200,
            direction="lower_is_better",
            method="analysis",
            computation="automated_cost_per_unit_usd",
        ),
        roi=some_roi(automated_model_cost_per_unit_usd="999"),
    )

    value = compute_kpis(four_verdicts(), a_run(a_call("0.40")), spec)[0].value

    # 0.40 usd over 4 items = 0.10/item, on top of 100 usd of residual human time.
    assert value == pytest.approx(100.10)


def test_an_roi_metric_with_no_roi_model_is_an_error_not_a_guess() -> None:
    spec = a_spec_set(a_spec(name="hours", unit="hours", target=1, computation="accuracy"))
    unsupported = spec.model_copy(
        update={
            "kpis": [
                a_spec(
                    name="hours",
                    unit="hours",
                    target=1,
                    computation="hours_displaced_per_unit",
                    method="analysis",
                )
            ]
        }
    )

    with pytest.raises(KPIError, match="needs an ROI model"):
        compute_kpis([], a_run(), unsupported)


# --- rendering --------------------------------------------------------------------------


def a_snapshot(**overrides: object) -> KPISnapshot:
    settings: dict[str, object] = {
        "project": "example",
        "metric_name": "accuracy",
        "value": 0.9,
        "unit": "ratio",
        "target": 0.85,
        "direction": "higher_is_better",
        "measured_at": AT,
        "sample_size": 10,
        "method": "test",
    }
    settings.update(overrides)
    return KPISnapshot.model_validate(settings)


def test_a_metric_above_a_higher_is_better_target_passes() -> None:
    assert meets_target(a_snapshot(value=0.9, target=0.85))


def test_a_metric_exactly_on_target_passes() -> None:
    # The boundary is inclusive in both directions: the target is what was asked for.
    assert meets_target(a_snapshot(value=0.85, target=0.85))
    assert meets_target(a_snapshot(value=0.05, target=0.05, direction="lower_is_better"))


def test_a_lower_is_better_metric_above_its_target_fails() -> None:
    assert not meets_target(a_snapshot(value=0.09, target=0.05, direction="lower_is_better"))


def test_the_table_shows_target_actual_and_a_marker_per_row() -> None:
    table = render_markdown(
        [
            a_snapshot(metric_name="accuracy", value=0.9, target=0.85),
            a_snapshot(
                metric_name="cost",
                value=0.09,
                target=0.05,
                unit="usd",
                direction="lower_is_better",
                method="analysis",
            ),
        ]
    )
    lines = table.strip().splitlines()

    assert lines[0] == "| Metric | Target | Actual | Unit | Method | Status |"
    assert lines[2] == "| accuracy | ≥ 0.85 | 0.9 | ratio | test | PASS |"
    assert lines[3] == "| cost | ≤ 0.05 | 0.09 | usd | analysis | FAIL |"


def test_the_direction_is_visible_in_the_target_column() -> None:
    # A reader must be able to tell which way is good without consulting the config.
    assert "≥" in render_markdown([a_snapshot(direction="higher_is_better")])
    assert "≤" in render_markdown([a_snapshot(direction="lower_is_better")])


def test_rendering_no_metrics_still_produces_a_table() -> None:
    table = render_markdown([])

    assert table.startswith("| Metric | Target | Actual | Unit | Method | Status |")
    assert "_no metrics declared_" in table


def test_every_roi_figure_is_reachable_as_a_kpi() -> None:
    # One test per branch of the ROI dispatch, so a reordering cannot silently swap two.
    spec = a_spec_set(
        a_spec(
            name="per_unit",
            unit="hours",
            target=1,
            method="analysis",
            computation="hours_displaced_per_unit",
        ),
        a_spec(
            name="per_period",
            unit="hours",
            target=1,
            method="analysis",
            computation="hours_displaced_per_period",
        ),
        a_spec(
            name="automated",
            unit="usd",
            target=1000,
            direction="lower_is_better",
            method="analysis",
            computation="automated_cost_per_unit_usd",
        ),
        a_spec(
            name="saving",
            unit="usd",
            target=1,
            method="analysis",
            computation="saving_per_unit_usd",
        ),
        a_spec(
            name="payback",
            unit="months",
            target=99,
            direction="lower_is_better",
            method="analysis",
            computation="payback_periods",
        ),
        roi=some_roi(),
    )

    values = {s.metric_name: s.value for s in compute_kpis([], a_run(), spec)}

    # Baseline 10h/800 usd, residual 1h/100 usd, 10 units per period, 20000 to build.
    assert values == pytest.approx(
        {
            "per_unit": 9.0,
            "per_period": 90.0,
            "automated": 100.0,
            "saving": 700.0,
            "payback": 20000 / 7000,
        }
    )


def test_total_cost_reports_the_whole_run_not_a_rate() -> None:
    spec = a_spec_set(
        a_spec(
            name="total_cost_usd",
            unit="usd",
            target=1.0,
            direction="lower_is_better",
            method="analysis",
            computation="total_cost_usd",
        )
    )

    value = compute_kpis(four_verdicts(), a_run(a_call("0.02"), a_call("0.03")), spec)[0].value

    assert value == pytest.approx(0.05)

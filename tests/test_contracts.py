"""Unit tests for the shared contract models.

Deliberately does not touch the network, the filesystem or any environment variable.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from spine.contracts import (
    AgentRun,
    BoundingBox,
    EvidenceRef,
    KPISnapshot,
    ModelCall,
    StepTrace,
    Verdict,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def a_call(prompt: int, completion: int, cached: int, cost: str) -> ModelCall:
    return ModelCall(
        provider="anthropic",
        model="a-model",
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cost_usd=Decimal(cost),
        latency_ms=120,
        timestamp=AT,
        purpose="extract",
    )


def a_step(name: str, calls: list[ModelCall]) -> StepTrace:
    return StepTrace(
        step_name=name,
        started_at=AT,
        finished_at=AT,
        status="ok",
        model_calls=calls,
        metadata={"gold_set": "v1", "retries": 2, "ratio": 0.5, "cached": True},
    )


def a_run(steps: list[StepTrace]) -> AgentRun:
    return AgentRun(
        run_id="run-1",
        project="ri05",
        started_at=AT,
        finished_at=AT,
        steps=steps,
        status="completed",
    )


def a_run_with_status(status: str) -> AgentRun:
    return AgentRun(
        run_id="run-1",
        project="ri05",
        started_at=AT,
        finished_at=AT,
        steps=[],
        status=status,
    )


def every_model() -> list[BaseModel]:
    call = a_call(100, 20, 5, "0.001")
    return [
        call,
        a_step("extract", [call]),
        a_run([a_step("extract", [call])]),
        BoundingBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0),
        EvidenceRef(
            source_id="src-1",
            document="tender.pdf",
            page=4,
            clause="3.2.1",
            bbox=BoundingBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0),
            quote="The supplier shall provide monthly reporting.",
        ),
        Verdict(
            item_id="item-1",
            expected="yes",
            actual="yes",
            passed=True,
            score=1.0,
            rationale="Exact match.",
            judge_model=None,
        ),
        KPISnapshot(
            project="ri05",
            metric_name="extraction_accuracy",
            value=0.91,
            unit="ratio",
            target=0.9,
            direction="higher_is_better",
            measured_at=AT,
            sample_size=20,
            method="test",
        ),
    ]


# --- computed totals on AgentRun ---------------------------------------------------


def test_agent_run_totals_are_summed_from_the_steps() -> None:
    run = a_run(
        [
            a_step("extract", [a_call(100, 20, 5, "0.001"), a_call(200, 30, 0, "0.002")]),
            a_step("score", [a_call(50, 10, 7, "0.0005")]),
        ]
    )

    assert run.total_prompt_tokens == 350
    assert run.total_completion_tokens == 60
    assert run.total_cached_tokens == 12
    assert run.total_tokens == 410
    assert run.total_cost_usd == Decimal("0.0035")


def test_agent_run_totals_are_zero_without_steps() -> None:
    run = a_run([])

    assert run.total_tokens == 0
    assert run.total_cost_usd == Decimal("0")


def test_agent_run_cost_is_exact_where_binary_floats_would_drift() -> None:
    run = a_run([a_step("s", [a_call(1, 1, 0, "0.1"), a_call(1, 1, 0, "0.2")])])

    assert run.total_cost_usd == Decimal("0.3")
    assert 0.1 + 0.2 != 0.3  # the reason cost_usd is not a binary float


def test_agent_run_totals_are_serialised_but_not_stored() -> None:
    run = a_run([a_step("extract", [a_call(100, 20, 5, "0.001")])])

    assert "total_cost_usd" in run.model_dump()
    assert "total_cost_usd" not in type(run).model_fields


# --- serialisation round-trip ------------------------------------------------------


@pytest.mark.parametrize("instance", every_model(), ids=lambda m: type(m).__name__)
def test_json_round_trip_preserves_the_record(instance: BaseModel) -> None:
    restored = type(instance).model_validate_json(instance.model_dump_json())

    assert restored == instance


@pytest.mark.parametrize("instance", every_model(), ids=lambda m: type(m).__name__)
def test_python_round_trip_preserves_the_record(instance: BaseModel) -> None:
    restored = type(instance).model_validate(instance.model_dump())

    assert restored == instance


def test_money_survives_the_json_round_trip_as_an_exact_decimal() -> None:
    call = a_call(1, 1, 0, "0.30")

    restored = ModelCall.model_validate_json(call.model_dump_json())

    assert restored.cost_usd == Decimal("0.30")


# --- frozen records reject mutation ------------------------------------------------


@pytest.mark.parametrize("instance", every_model(), ids=lambda m: type(m).__name__)
def test_records_are_frozen(instance: BaseModel) -> None:
    field = next(iter(type(instance).model_fields))

    with pytest.raises(ValidationError) as excinfo:
        setattr(instance, field, None)

    # Assert on the reason: without frozen=True the assignment would simply succeed,
    # so a bare "raises" could pass for the wrong reason once validate_assignment lands.
    assert [error["type"] for error in excinfo.value.errors()] == ["frozen_instance"]


def test_a_computed_total_cannot_be_assigned() -> None:
    run = a_run([])

    with pytest.raises((AttributeError, ValidationError)):
        run.total_cost_usd = Decimal("99")


# --- field constraints and to_row ---------------------------------------------------


def test_score_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Verdict(item_id="i", expected="a", actual="a", passed=True, score=1.5, rationale="r")


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValidationError):
        a_call(1, 1, 0, "-0.01")


def test_to_row_is_flat_and_unformatted() -> None:
    snapshot = KPISnapshot(
        project="ri05",
        metric_name="cost_per_tender_usd",
        value=1.5,
        unit="usd",
        target=2.0,
        direction="lower_is_better",
        measured_at=AT,
        sample_size=20,
        method="test",
    )

    row = snapshot.to_row()

    assert row == {
        "project": "ri05",
        "metric_name": "cost_per_tender_usd",
        "value": 1.5,
        "unit": "usd",
        "target": 2.0,
        "direction": "lower_is_better",
        "measured_at": "2026-01-02T03:04:05+00:00",
        "sample_size": 20,
        "method": "test",
    }
    assert not any(isinstance(v, str) and v.startswith("$") for v in row.values())
    assert all(not isinstance(v, dict | list) for v in row.values())


# --- EvidenceRef locating fields ----------------------------------------------------


@pytest.mark.parametrize(
    "locating",
    [
        {"page": 4},
        {"clause": "3.2.1"},
        {"bbox": BoundingBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)},
        {"locator": "BOQ row 214"},
        {"locator": "Drawing SEC-004 detail B"},
        {"page": 4, "clause": "3.2.1"},
    ],
    ids=["page", "clause", "bbox", "locator-boq", "locator-drawing", "page-and-clause"],
)
def test_any_single_locating_field_is_enough(locating: dict[str, object]) -> None:
    ref = EvidenceRef(source_id="s", document="d.pdf", quote="q", **locating)

    for name, value in locating.items():
        assert getattr(ref, name) == value


def test_page_and_clause_are_optional() -> None:
    ref = EvidenceRef(source_id="s", document="d.pdf", quote="q", locator="BOQ row 214")

    assert ref.page is None
    assert ref.clause is None


def test_an_evidence_ref_that_points_nowhere_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EvidenceRef(source_id="s", document="d.pdf", quote="q")

    assert "at least one of page, clause, bbox or locator" in str(excinfo.value)


def test_an_evidence_ref_with_every_field_none_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceRef(
            source_id="s",
            document="d.pdf",
            quote="q",
            page=None,
            clause=None,
            bbox=None,
            locator=None,
        )


def test_page_zero_counts_as_a_locator() -> None:
    # Page 0 is falsy but present; the validator must test for None, not truthiness.
    ref = EvidenceRef(source_id="s", document="d.pdf", quote="q", page=0)

    assert ref.page == 0


def test_a_locator_only_ref_round_trips() -> None:
    ref = EvidenceRef(source_id="s", document="d.pdf", quote="q", locator="BOQ row 214")

    assert EvidenceRef.model_validate_json(ref.model_dump_json()) == ref


# --- split status enums -------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["running", "awaiting_review", "completed", "completed_with_errors", "failed", "cancelled"],
)
def test_a_run_accepts_every_run_status(status: str) -> None:
    assert a_run_with_status(status).status == status


def test_a_run_pausing_at_a_review_interrupt_is_representable() -> None:
    run = a_run_with_status("awaiting_review")

    assert run.status == "awaiting_review"


def test_a_run_rejects_a_step_status() -> None:
    # "skipped" and "ok" are meaningless for a run.
    for status in ("skipped", "ok"):
        with pytest.raises(ValidationError):
            a_run_with_status(status)


@pytest.mark.parametrize("status", ["ok", "failed", "skipped"])
def test_a_step_accepts_every_step_status(status: str) -> None:
    step = StepTrace(step_name="s", started_at=AT, finished_at=AT, status=status)

    assert step.status == status


def test_a_step_rejects_a_run_status() -> None:
    with pytest.raises(ValidationError):
        StepTrace(step_name="s", started_at=AT, finished_at=AT, status="awaiting_review")


# --- widened step metadata ----------------------------------------------------------


def test_metadata_carries_mixed_scalars_without_coercing_them() -> None:
    step = StepTrace(
        step_name="s",
        started_at=AT,
        finished_at=AT,
        status="ok",
        metadata={"name": "v1", "retries": 2, "ratio": 0.5, "cached": True},
    )

    assert step.metadata["name"] == "v1"
    assert step.metadata["retries"] == 2
    assert isinstance(step.metadata["retries"], int)
    assert step.metadata["ratio"] == 0.5
    assert step.metadata["cached"] is True
    # bool is a subclass of int, so assert the union did not widen True into 1.
    assert isinstance(step.metadata["cached"], bool)


def test_metadata_types_survive_the_json_round_trip() -> None:
    step = StepTrace(
        step_name="s",
        started_at=AT,
        finished_at=AT,
        status="ok",
        metadata={"name": "v1", "retries": 2, "ratio": 0.5, "cached": True},
    )

    restored = StepTrace.model_validate_json(step.model_dump_json())

    assert restored == step
    assert isinstance(restored.metadata["cached"], bool)
    assert isinstance(restored.metadata["retries"], int)


def test_metadata_rejects_a_nested_structure() -> None:
    with pytest.raises(ValidationError):
        StepTrace(
            step_name="s",
            started_at=AT,
            finished_at=AT,
            status="ok",
            metadata={"nested": {"not": "scalar"}},
        )


# --- BoundingBox convention ----------------------------------------------------------


def test_bounding_box_docstring_states_the_origin_and_axes() -> None:
    doc = BoundingBox.__doc__ or ""

    assert "top-left" in doc
    assert "downward" in doc
    assert "points" in doc


# --- a failed step cannot hide -------------------------------------------------------


def a_failed_step(name: str = "price") -> StepTrace:
    return StepTrace(
        step_name=name, started_at=AT, finished_at=AT, status="failed", error="Boom: nope"
    )


def a_run_with(steps: list[StepTrace], status: str = "completed_with_errors") -> AgentRun:
    return AgentRun(
        run_id="run-1",
        project="ri05",
        started_at=AT,
        finished_at=AT,
        steps=steps,
        status=status,
    )


def test_a_clean_run_reports_no_failed_steps() -> None:
    run = a_run_with([a_step("extract", [])], status="completed")

    assert run.has_failed_steps is False
    assert run.failed_steps == []


def test_failed_steps_are_reported_in_order() -> None:
    steps = [a_step("extract", []), a_failed_step("price"), a_failed_step("draft")]

    run = a_run_with(steps)

    assert run.has_failed_steps is True
    assert [step.step_name for step in run.failed_steps] == ["price", "draft"]


def test_a_run_with_a_failed_step_cannot_be_completed() -> None:
    with pytest.raises(ValidationError, match="completed_with_errors"):
        a_run_with([a_step("extract", []), a_failed_step()], status="completed")


def test_the_error_names_the_failed_step() -> None:
    with pytest.raises(ValidationError, match="price"):
        a_run_with([a_failed_step("price")], status="completed")


def test_completed_with_errors_is_accepted_for_a_run_that_had_a_failure() -> None:
    run = a_run_with([a_step("extract", []), a_failed_step()])

    assert run.status == "completed_with_errors"


def test_other_statuses_are_unaffected_by_a_failed_step() -> None:
    for status in ("running", "awaiting_review", "failed", "cancelled"):
        assert a_run_with([a_failed_step()], status=status).status == status


def test_has_failed_steps_is_serialised_but_failed_steps_is_not() -> None:
    dumped = a_run_with([a_failed_step()]).model_dump()

    assert dumped["has_failed_steps"] is True
    # The traces are already under "steps"; repeating them would double every record.
    assert "failed_steps" not in dumped

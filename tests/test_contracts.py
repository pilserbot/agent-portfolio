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
        metadata={"gold_set": "v1"},
    )


def a_run(steps: list[StepTrace]) -> AgentRun:
    return AgentRun(
        run_id="run-1",
        project="ri05",
        started_at=AT,
        finished_at=AT,
        steps=steps,
        status="ok",
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

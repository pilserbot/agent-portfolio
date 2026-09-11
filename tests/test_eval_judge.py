"""Unit tests for the rubric-backed judge.

Every test injects a fake judge function, so nothing here reaches a network or needs a
credential. The point under test is the Python around the model: rubric loading and
versioning, the weighted score, and the self-consistency vote.

Deliberately does not test what a real model would say — that is what
`spine.eval.calibrate` measures, and it cannot be asserted in a unit test.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from spine.contracts import ModelCall, Verdict
from spine.eval.judge import (
    DEFAULT_TIER,
    Criterion,
    Judge,
    JudgeRequest,
    JudgeResponse,
    Rubric,
    RubricError,
    build_prompt,
    load_rubric,
    load_rubric_file,
    score_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_ROOT = REPO_ROOT / "evals" / "rubrics"

RUBRIC_YAML = """
name: unit
version: 3
description: a rubric for tests
instructions: Score each criterion from 0.0 to 1.0.
pass_threshold: 0.7
criteria:
  - id: alpha
    weight: 0.5
    description: the first thing
  - id: beta
    weight: 0.5
    description: the second thing
"""


def a_rubric(**overrides: object) -> Rubric:
    values: dict[str, object] = {
        "name": "unit",
        "version": 1,
        "instructions": "Score each criterion.",
        "pass_threshold": 0.7,
        "criteria": [
            Criterion(id="alpha", description="first", weight=0.5),
            Criterion(id="beta", description="second", weight=0.5),
        ],
    }
    values.update(overrides)
    return Rubric(**values)


def write_rubric(tmp_path: Path, text: str = RUBRIC_YAML, name: str = "unit") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def responder(
    *score_sets: dict[str, float], rationale: str = "because"
) -> Callable[[JudgeRequest], tuple[JudgeResponse, ModelCall | None]]:
    """A fake judge function returning one prepared response per sample, in order."""
    prepared = list(score_sets)
    seen: list[JudgeRequest] = []

    def judge_fn(request: JudgeRequest) -> tuple[JudgeResponse, ModelCall | None]:
        seen.append(request)
        scores = prepared[min(request.sample_index, len(prepared) - 1)]
        return JudgeResponse(criterion_scores=scores, rationale=rationale), None

    judge_fn.seen = seen  # type: ignore[attr-defined]
    return judge_fn


# --- rubric loading and versioning ----------------------------------------------------------------


def test_a_rubric_loads_from_yaml(tmp_path: Path) -> None:
    rubric = load_rubric_file(write_rubric(tmp_path))

    assert rubric.name == "unit"
    assert rubric.version == 3
    assert rubric.criterion_ids == ["alpha", "beta"]
    assert rubric.pass_threshold == 0.7
    assert rubric.source_path == tmp_path / "unit.yaml"


def test_the_version_is_part_of_the_rubrics_identity(tmp_path: Path) -> None:
    rubric = load_rubric_file(write_rubric(tmp_path))

    assert rubric.label == "unit@v3"


def test_a_rubric_loads_by_name_from_a_root(tmp_path: Path) -> None:
    write_rubric(tmp_path, name="mine")

    assert load_rubric("mine", root=tmp_path).name == "unit"


def test_a_missing_rubric_says_so(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="not found"):
        load_rubric("absent", root=tmp_path)


def test_a_directory_is_not_a_rubric(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="directory"):
        load_rubric_file(tmp_path)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(RubricError, match="not valid YAML"):
        load_rubric_file(path)


def test_a_yaml_scalar_is_not_a_rubric(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("just a string\n", encoding="utf-8")

    with pytest.raises(RubricError, match="expected a YAML mapping"):
        load_rubric_file(path)


def test_a_missing_version_is_rejected(tmp_path: Path) -> None:
    # An unversioned rubric would make its scores untraceable.
    path = write_rubric(tmp_path, RUBRIC_YAML.replace("version: 3\n", ""))

    with pytest.raises(RubricError, match="version"):
        load_rubric_file(path)


def test_an_unknown_rubric_field_is_rejected(tmp_path: Path) -> None:
    path = write_rubric(tmp_path, RUBRIC_YAML + "threshold: 0.9\n")

    with pytest.raises(RubricError, match="threshold"):
        load_rubric_file(path)


def test_a_rubric_with_no_criteria_is_rejected(tmp_path: Path) -> None:
    text = RUBRIC_YAML.split("criteria:")[0] + "criteria: []\n"

    with pytest.raises(RubricError, match="criteria"):
        load_rubric_file(write_rubric(tmp_path, text))


def test_duplicate_criterion_ids_are_rejected(tmp_path: Path) -> None:
    # One would silently overwrite the other's score.
    text = RUBRIC_YAML.replace("  - id: beta", "  - id: alpha")

    with pytest.raises(RubricError, match="duplicate criterion"):
        load_rubric_file(write_rubric(tmp_path, text))


def test_a_threshold_outside_zero_to_one_is_rejected(tmp_path: Path) -> None:
    text = RUBRIC_YAML.replace("pass_threshold: 0.7", "pass_threshold: 1.5")

    with pytest.raises(RubricError, match="pass_threshold"):
        load_rubric_file(write_rubric(tmp_path, text))


def test_the_committed_example_rubric_loads() -> None:
    rubric = load_rubric("example", root=RUBRIC_ROOT)

    assert rubric.label == "example@v1"
    assert rubric.total_weight == pytest.approx(1.0)
    assert len(rubric.criteria) == 3


# --- the prompt is built from the rubric ----------------------------------------------------------


def test_the_prompt_carries_the_rubric_and_the_item() -> None:
    request = JudgeRequest(
        item_id="i1", rubric=a_rubric(version=4), expected="a requirement", output="an answer"
    )

    prompt = build_prompt(request)

    assert "unit@v4" in prompt, "the version is in the prompt, so a score is traceable"
    assert "alpha" in prompt and "beta" in prompt
    assert "a requirement" in prompt and "an answer" in prompt
    assert "Do not state whether the item passes" in prompt


def test_the_prompt_is_the_same_for_the_same_rubric_and_item() -> None:
    request = JudgeRequest(item_id="i1", rubric=a_rubric(), expected="e", output="o")

    assert build_prompt(request) == build_prompt(request)


# --- the score is computed in Python, not by the model --------------------------------------------


def test_the_weighted_score() -> None:
    # 0.5*1.0 + 0.5*0.4 = 0.7, over a total weight of 1.0.
    rubric = a_rubric()
    response = JudgeResponse(criterion_scores={"alpha": 1.0, "beta": 0.4})

    assert score_of(rubric, response) == pytest.approx(0.7)


def test_weights_need_not_sum_to_one() -> None:
    # (3*1.0 + 1*0.0) / 4 = 0.75.
    rubric = a_rubric(
        criteria=[
            Criterion(id="alpha", description="a", weight=3),
            Criterion(id="beta", description="b", weight=1),
        ]
    )
    response = JudgeResponse(criterion_scores={"alpha": 1.0, "beta": 0.0})

    assert score_of(rubric, response) == pytest.approx(0.75)


def test_an_omitted_criterion_scores_zero() -> None:
    # Silence is not credit: 0.5*1.0 + 0.5*0.0 = 0.5.
    assert score_of(a_rubric(), JudgeResponse(criterion_scores={"alpha": 1.0})) == pytest.approx(
        0.5
    )


def test_a_criterion_the_rubric_does_not_list_is_ignored() -> None:
    # A model inventing a criterion must not be able to inflate the total.
    response = JudgeResponse(criterion_scores={"alpha": 1.0, "beta": 1.0, "invented": 1.0})

    assert score_of(a_rubric(), response) == pytest.approx(1.0)


def test_a_score_outside_zero_to_one_is_rejected_not_clamped() -> None:
    with pytest.raises(ValueError, match="within 0..1"):
        JudgeResponse(criterion_scores={"alpha": 1.4})


def test_the_response_schema_has_no_pass_field() -> None:
    # The model is never asked whether the item passes; Python decides that.
    assert set(JudgeResponse.model_fields) == {"criterion_scores", "rationale"}


# --- judging one item -----------------------------------------------------------------------------


def test_a_score_at_the_threshold_passes() -> None:
    judge = Judge(a_rubric(), judge_fn=responder({"alpha": 1.0, "beta": 0.4}))

    result = judge.judge("i1", expected="e", output="o")

    assert result.verdict.score == pytest.approx(0.7)
    assert result.verdict.passed, "the threshold is inclusive"
    assert isinstance(result.verdict, Verdict)


def test_a_score_below_the_threshold_fails() -> None:
    judge = Judge(a_rubric(), judge_fn=responder({"alpha": 1.0, "beta": 0.39}))

    assert not judge.judge("i1", expected="e", output="o").verdict.passed


def test_the_verdict_is_stamped_with_the_rubric_version() -> None:
    judge = Judge(a_rubric(version=7), judge_fn=responder({"alpha": 1.0, "beta": 1.0}))

    result = judge.judge("i1", expected="e", output="o")

    assert "unit@v7" in result.verdict.rationale
    assert result.rubric_label == "unit@v7"


def test_the_written_rationale_reaches_the_verdict() -> None:
    judge = Judge(
        a_rubric(), judge_fn=responder({"alpha": 1.0, "beta": 1.0}, rationale="names the clause")
    )

    assert "names the clause" in judge.judge("i1", expected="e", output="o").verdict.rationale


def test_the_judge_model_is_recorded_when_a_call_was_made() -> None:
    from datetime import UTC, datetime
    from decimal import Decimal

    call = ModelCall(
        provider="anthropic",
        model="a-small-model",
        prompt_tokens=10,
        completion_tokens=5,
        cached_tokens=0,
        cost_usd=Decimal("0.0001"),
        latency_ms=12,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        purpose="judge",
    )

    def judge_fn(request: JudgeRequest) -> tuple[JudgeResponse, ModelCall | None]:
        return JudgeResponse(criterion_scores={"alpha": 1.0, "beta": 1.0}), call

    result = Judge(a_rubric(), judge_fn=judge_fn).judge("i1", expected="e", output="o")

    assert result.verdict.judge_model == "a-small-model"
    assert result.model_calls == [call]


def test_the_default_tier_is_small() -> None:
    # A judge runs on the cheap tier unless asked otherwise.
    assert DEFAULT_TIER == "small"
    assert Judge(a_rubric(), judge_fn=responder({"alpha": 1.0})).tier == "small"


def test_a_judge_with_neither_router_nor_function_cannot_measure() -> None:
    with pytest.raises(RubricError, match="router or a judge_fn"):
        Judge(a_rubric()).judge("i1", expected="e", output="o")


# --- self-consistency -----------------------------------------------------------------------------


def test_one_sample_is_the_default() -> None:
    judge_fn = responder({"alpha": 1.0, "beta": 1.0})
    judge = Judge(a_rubric(), judge_fn=judge_fn)

    judge.judge("i1", expected="e", output="o")

    assert len(judge_fn.seen) == 1


def test_k_samples_are_taken_and_indexed() -> None:
    judge_fn = responder(
        {"alpha": 1.0, "beta": 1.0}, {"alpha": 0.0, "beta": 0.0}, {"alpha": 1.0, "beta": 1.0}
    )
    judge = Judge(a_rubric(), samples=3, judge_fn=judge_fn)

    result = judge.judge("i1", expected="e", output="o")

    assert [request.sample_index for request in judge_fn.seen] == [0, 1, 2]
    assert result.sample_scores == [1.0, 0.0, 1.0]


def test_the_majority_carries_the_vote() -> None:
    # Two pass, one fail -> passes. The reported score is the median, 1.0, not the mean.
    judge = Judge(
        a_rubric(),
        samples=3,
        judge_fn=responder(
            {"alpha": 1.0, "beta": 1.0}, {"alpha": 0.0, "beta": 0.0}, {"alpha": 1.0, "beta": 1.0}
        ),
    )

    result = judge.judge("i1", expected="e", output="o")

    assert result.sample_passes == [True, False, True]
    assert result.verdict.passed
    assert result.verdict.score == pytest.approx(1.0)
    assert not result.agreed, "the samples disagreed, and the result says so"


def test_a_minority_pass_loses_the_vote() -> None:
    judge = Judge(
        a_rubric(),
        samples=3,
        judge_fn=responder(
            {"alpha": 1.0, "beta": 1.0}, {"alpha": 0.0, "beta": 0.0}, {"alpha": 0.0, "beta": 0.0}
        ),
    )

    result = judge.judge("i1", expected="e", output="o")

    assert not result.verdict.passed
    assert result.verdict.score == pytest.approx(0.0)


def test_a_tied_vote_does_not_pass() -> None:
    # A split jury is not a pass: the item was not shown to meet the rubric.
    judge = Judge(
        a_rubric(),
        samples=2,
        judge_fn=responder({"alpha": 1.0, "beta": 1.0}, {"alpha": 0.0, "beta": 0.0}),
    )

    result = judge.judge("i1", expected="e", output="o")

    assert result.sample_passes == [True, False]
    assert not result.verdict.passed


def test_unanimous_samples_are_reported_as_agreed() -> None:
    judge = Judge(
        a_rubric(),
        samples=3,
        judge_fn=responder({"alpha": 1.0, "beta": 1.0}),
    )

    result = judge.judge("i1", expected="e", output="o")

    assert result.agreed
    assert "unanimous" in result.verdict.rationale


def test_a_split_is_named_in_the_rationale() -> None:
    judge = Judge(
        a_rubric(),
        samples=2,
        judge_fn=responder({"alpha": 1.0, "beta": 1.0}, {"alpha": 0.0, "beta": 0.0}),
    )

    assert "split" in judge.judge("i1", expected="e", output="o").verdict.rationale


def test_the_median_ignores_one_wild_sample() -> None:
    # Scores 1.0, 0.9, 0.0: the median is 0.9, the mean would be 0.633.
    judge = Judge(
        a_rubric(),
        samples=3,
        judge_fn=responder(
            {"alpha": 1.0, "beta": 1.0},
            {"alpha": 1.0, "beta": 0.8},
            {"alpha": 0.0, "beta": 0.0},
        ),
    )

    result = judge.judge("i1", expected="e", output="o")

    assert result.verdict.score == pytest.approx(0.9)


def test_fewer_than_one_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Judge(a_rubric(), samples=0, judge_fn=responder({"alpha": 1.0}))

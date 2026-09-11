"""A model-backed judge that scores an output against an explicit, versioned rubric.

The rubric is a YAML file under `evals/rubrics/`, not a string in this module, so a score
can always be traced to the exact wording that produced it. The version is stamped into
every Verdict.

The division of labour matters and is the whole reason this module can exist inside
CLAUDE.md's rules: the model reads prose and fills in a per-criterion score. It is never
asked whether the item passes. The weighted total, the pass/fail against the rubric's
threshold, and the majority vote across self-consistency samples are all computed here in
Python from the numbers the model returned.

`judge_model` on the Verdicts from this module is set, unlike those from
`spine.eval.checkers`. That is the honest signal that a model was involved, and it is what
`spine.eval.calibrate` exists to hold to account: a judge is an instrument, and an
uncalibrated instrument is a guess with a number attached.

Deliberately does not: decide anything a rubric did not ask about, retry a low score into a
higher one, or fall back to a different model. It also does not cache: two runs of the same
item are two measurements, which is what self-consistency needs.
"""

import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from spine.contracts import ModelCall, Verdict

DEFAULT_RUBRIC_ROOT = Path("evals/rubrics")
DEFAULT_TIER: Literal["large", "small"] = "small"
DEFAULT_SAMPLES = 1

__all__ = [
    "DEFAULT_RUBRIC_ROOT",
    "DEFAULT_SAMPLES",
    "DEFAULT_TIER",
    "Criterion",
    "Judge",
    "JudgeRequest",
    "JudgeResponse",
    "JudgeResult",
    "Rubric",
    "RubricError",
    "SupportsStructured",
    "build_prompt",
    "load_rubric",
    "score_of",
]


class SupportsStructured(Protocol):
    """The slice of `spine.router.Router` a judge needs.

    A protocol rather than the concrete class so that importing this module does not drag
    in a provider SDK: a rubric can be loaded, and a judge run against an injected
    function, without litellm ever being imported.
    """

    def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        purpose: str,
        tier: str,
    ) -> tuple[BaseModel, ModelCall]:
        """Extract a typed object from a prompt."""
        ...


class RubricError(Exception):
    """A rubric could not be loaded, or does not describe a usable instrument."""


class Criterion(BaseModel):
    """One thing a rubric asks the judge to score, and how much it counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(gt=0.0, description="Relative weight; weights need not sum to 1.")


class Rubric(BaseModel):
    """A versioned scoring instrument, loaded from YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(ge=1, description="Bump on any change to criteria, weights or wording.")
    description: str = ""
    instructions: str = Field(min_length=1)
    pass_threshold: float = Field(ge=0.0, le=1.0)
    criteria: list[Criterion] = Field(min_length=1)
    source_path: Path | None = None

    @model_validator(mode="after")
    def _criterion_ids_are_unique(self) -> "Rubric":
        """Refuse duplicate criterion ids: one would silently overwrite the other's score."""
        seen = [criterion.id for criterion in self.criteria]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate criterion id(s): {', '.join(duplicates)}")
        return self

    @property
    def label(self) -> str:
        """How this rubric identifies itself in a Verdict or a report."""
        return f"{self.name}@v{self.version}"

    @property
    def criterion_ids(self) -> list[str]:
        """The criteria this rubric scores, in rubric order."""
        return [criterion.id for criterion in self.criteria]

    @property
    def total_weight(self) -> float:
        """The sum of the criterion weights, used to normalise a score into 0..1."""
        return sum(criterion.weight for criterion in self.criteria)


def load_rubric(name: str, *, root: Path = DEFAULT_RUBRIC_ROOT) -> Rubric:
    """Load a rubric by name from the rubric directory."""
    return load_rubric_file(root / f"{name}.yaml")


def load_rubric_file(path: Path) -> Rubric:
    """Load a rubric from a YAML file, failing loudly and specifically."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RubricError(f"rubric not found: {path}") from error
    except IsADirectoryError as error:
        raise RubricError(f"rubric path is a directory: {path}") from error

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RubricError(f"{path}: not valid YAML: {error}") from error

    if not isinstance(payload, Mapping):
        raise RubricError(f"{path}: expected a YAML mapping, got {type(payload).__name__}")

    try:
        rubric = Rubric.model_validate({**payload, "source_path": path})
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or '(root)'}: {problem['msg']}"
            for problem in error.errors()
        )
        raise RubricError(f"{path}: {problems}") from error
    return rubric


class JudgeRequest(BaseModel):
    """One item put to the judge, with the rubric it is to be scored against."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    rubric: Rubric
    expected: str = Field(description="The requirement or criteria the output should meet.")
    output: str = Field(description="The output being judged.")
    sample_index: int = Field(default=0, ge=0, description="Which self-consistency sample.")


class JudgeResponse(BaseModel):
    """The only thing the model produces: a score per criterion, and why.

    Note what is absent: no pass/fail and no total. Those are computed from these numbers
    by `score_of`, in Python, against the rubric's threshold.
    """

    model_config = ConfigDict(extra="forbid")

    criterion_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="after")
    def _scores_are_in_range(self) -> "JudgeResponse":
        """Reject a score outside 0..1 rather than clamping it out of sight."""
        bad = {
            key: value for key, value in self.criterion_scores.items() if not 0.0 <= value <= 1.0
        }
        if bad:
            raise ValueError(f"criterion scores must be within 0..1; got {bad}")
        return self


class JudgeResult(BaseModel):
    """A judged item: the Verdict, and the samples it was computed from."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    rubric_label: str
    sample_scores: list[float] = Field(default_factory=list)
    sample_passes: list[bool] = Field(default_factory=list)
    responses: list[JudgeResponse] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)

    @property
    def agreed(self) -> bool:
        """Whether every self-consistency sample reached the same pass decision."""
        return len(set(self.sample_passes)) <= 1


def build_prompt(request: JudgeRequest) -> str:
    """Render the judge prompt from the rubric, deterministically.

    Built here rather than stored as a template so that the rubric file stays the single
    source of the wording, and so the same rubric always produces the same prompt.
    """
    rubric = request.rubric
    lines = [
        f"You are scoring one item against rubric {rubric.label}.",
        "",
        rubric.instructions.strip(),
        "",
        "Criteria:",
    ]
    for criterion in rubric.criteria:
        lines.append(f"- {criterion.id} (weight {criterion.weight:g}): {criterion.description}")
    lines += [
        "",
        "Requirement or expected content:",
        request.expected.strip(),
        "",
        "Output to judge:",
        request.output.strip(),
        "",
        (
            "Return a score from 0.0 to 1.0 for each criterion id above, and a rationale "
            "naming the wording that decided each score. Do not state whether the item "
            "passes: that is computed from your scores."
        ),
    ]
    return "\n".join(lines)


def score_of(rubric: Rubric, response: JudgeResponse) -> float:
    """The weighted score in 0..1 for one response.

    A criterion the model omitted scores zero: silence is not credit. Scores for ids the
    rubric does not list are ignored, so a model inventing a criterion cannot inflate a
    total.
    """
    weighted = sum(
        criterion.weight * response.criterion_scores.get(criterion.id, 0.0)
        for criterion in rubric.criteria
    )
    return weighted / rubric.total_weight


def _majority_pass(passes: Sequence[bool]) -> bool:
    """The majority pass decision, with a tie counted as a failure.

    A split jury is not a pass: if the samples cannot agree, the honest reading is that the
    item was not shown to meet the rubric.
    """
    if not passes:
        return False
    return sum(passes) * 2 > len(passes)


type JudgeFn = Callable[[JudgeRequest], tuple[JudgeResponse, ModelCall | None]]


class Judge:
    """Scores outputs against one rubric, through the router.

    State is genuinely held: the rubric, the router and the sampling settings.

    Inject `judge_fn` to run without a network — it receives a `JudgeRequest` and returns a
    `JudgeResponse` with the `ModelCall` that produced it, which is exactly what the router
    path does.
    """

    def __init__(
        self,
        rubric: Rubric,
        *,
        router: SupportsStructured | None = None,
        samples: int = DEFAULT_SAMPLES,
        tier: Literal["large", "small"] = DEFAULT_TIER,
        judge_fn: JudgeFn | None = None,
    ) -> None:
        """Build a judge for one rubric."""
        if samples < 1:
            raise ValueError(f"samples must be at least 1; got {samples}")
        self.rubric = rubric
        self.samples = samples
        self.tier = tier
        self._router = router
        self._judge_fn = judge_fn

    def judge(self, item_id: str, *, expected: str, output: str) -> JudgeResult:
        """Score one item, taking `samples` measurements and voting on the outcome."""
        responses: list[JudgeResponse] = []
        calls: list[ModelCall] = []

        for index in range(self.samples):
            request = JudgeRequest(
                item_id=item_id,
                rubric=self.rubric,
                expected=expected,
                output=output,
                sample_index=index,
            )
            response, call = self._ask(request)
            responses.append(response)
            if call is not None:
                calls.append(call)

        scores = [score_of(self.rubric, response) for response in responses]
        passes = [score >= self.rubric.pass_threshold for score in scores]
        passed = _majority_pass(passes)
        # The median, not the mean: one wild sample should not drag the reported score.
        score = statistics.median(scores)

        return JudgeResult(
            verdict=Verdict(
                item_id=item_id,
                expected=expected,
                actual=output,
                passed=passed,
                score=min(1.0, max(0.0, score)),
                rationale=self._rationale(responses, scores, passes),
                judge_model=calls[0].model if calls else None,
            ),
            rubric_label=self.rubric.label,
            sample_scores=scores,
            sample_passes=passes,
            responses=responses,
            model_calls=calls,
        )

    def _ask(self, request: JudgeRequest) -> tuple[JudgeResponse, ModelCall | None]:
        """Take one measurement, through the injected function or the router."""
        if self._judge_fn is not None:
            return self._judge_fn(request)
        if self._router is None:
            raise RubricError("a Judge needs either a router or a judge_fn to take a measurement")
        response, call = self._router.structured(
            build_prompt(request),
            JudgeResponse,
            purpose=f"judge:{self.rubric.label}",
            tier=self.tier,
        )
        if not isinstance(response, JudgeResponse):  # pragma: no cover - router contract
            raise RubricError(f"router returned {type(response).__name__}, not a JudgeResponse")
        return response, call

    def _rationale(
        self,
        responses: Sequence[JudgeResponse],
        scores: Sequence[float],
        passes: Sequence[bool],
    ) -> str:
        """Assemble the written rationale, stamped with the rubric version."""
        parts = [f"[{self.rubric.label}]"]
        if len(responses) > 1:
            agreement = "unanimous" if len(set(passes)) <= 1 else "split"
            parts.append(
                f"{len(responses)} samples, {agreement}: "
                f"scores {', '.join(f'{score:.3f}' for score in scores)}."
            )
        parts.extend(response.rationale.strip() for response in responses if response.rationale)
        return " ".join(part for part in parts if part)

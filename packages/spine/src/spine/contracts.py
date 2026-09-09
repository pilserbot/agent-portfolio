"""The Pydantic v2 models every part of this project exchanges across module boundaries.

Defines the shared vocabulary: what a model call cost, what a step did, what a run
produced, where a claim came from, whether an item passed, and what a KPI measured. These
types are records of things that already happened, so they are all frozen.

Deliberately does not: perform I/O, call a model, format anything for display (no currency
strings, no rounding for humans), or decide any of the values it carries. A `Verdict` is
constructed from a decision that deterministic Python has already made; `judge_model` only
records which model turned text into structure along the way, never which model decided
the outcome.

Note on freezing: `frozen=True` prevents attribute assignment, not mutation of a list or
dict already inside a field. Build the collections first, then construct the record.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# Money is carried as Decimal so that summing costs does not accumulate binary-float error.
# It is never formatted here: presentation belongs to the caller.
UsdAmount = Annotated[
    Decimal,
    Field(ge=0, description="Amount in usd. Unformatted; the caller decides presentation."),
]

StepStatus = Literal["ok", "failed", "skipped"]

# A run is long-lived: it can pause at a human review interrupt for days and resume, so
# it needs mid-flight states. "skipped" is meaningless for a run and is absent here.
RunStatus = Literal["running", "awaiting_review", "completed", "failed", "cancelled"]
Direction = Literal["higher_is_better", "lower_is_better"]
Method = Literal["test", "analysis", "inspection", "demonstration"]


class ModelCall(BaseModel):
    """One call to a language model, and what it cost."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cost_usd: UsdAmount
    latency_ms: int = Field(ge=0)
    timestamp: datetime
    purpose: str = Field(description='Short label for why the call was made, e.g. "extract".')


class StepTrace(BaseModel):
    """One step of a run: what it was, how it ended, and the model calls it made."""

    model_config = ConfigDict(frozen=True)

    step_name: str
    started_at: datetime
    finished_at: datetime
    status: StepStatus
    model_calls: list[ModelCall] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Free-form annotations. Scalar values only, so the record stays portable.",
    )


class AgentRun(BaseModel):
    """One end-to-end run of a project's agent, and the steps it took."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    project: str = Field(description='Which project produced the run, e.g. "ri05".')
    started_at: datetime
    finished_at: datetime
    steps: list[StepTrace] = Field(default_factory=list)
    status: RunStatus

    @computed_field(description="Sum of every model call's cost_usd across every step, in usd.")
    @property
    def total_cost_usd(self) -> Decimal:
        """Total cost of the run, summed from the steps rather than stored."""
        return sum(
            (call.cost_usd for step in self.steps for call in step.model_calls),
            Decimal("0"),
        )

    @computed_field
    @property
    def total_prompt_tokens(self) -> int:
        """Prompt tokens across every model call in every step."""
        return sum(call.prompt_tokens for step in self.steps for call in step.model_calls)

    @computed_field
    @property
    def total_completion_tokens(self) -> int:
        """Completion tokens across every model call in every step."""
        return sum(call.completion_tokens for step in self.steps for call in step.model_calls)

    @computed_field
    @property
    def total_cached_tokens(self) -> int:
        """Cached tokens across every model call in every step."""
        return sum(call.cached_tokens for step in self.steps for call in step.model_calls)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens. Cached tokens are reported separately."""
        return self.total_prompt_tokens + self.total_completion_tokens


class BoundingBox(BaseModel):
    """A rectangle on a page of the source document.

    Coordinates follow the extraction convention rather than the PDF drawing convention:
    the origin is the **top-left** corner of the page, x increases rightward and y
    increases **downward**. Units are PDF points (1/72 inch). `(x0, y0)` is the top-left
    corner of the rectangle and `(x1, y1)` the bottom-right, so `x1 >= x0` and
    `y1 >= y0`; that ordering is a convention here, not an enforced constraint. The page
    is the one named by the `EvidenceRef` carrying this box.

    Beware when reading boxes straight from a PDF library: PDF user space puts the origin
    at the bottom-left with y increasing upward, so those coordinates must be flipped
    before they are stored here.
    """

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class EvidenceRef(BaseModel):
    """A pointer back to where something came from.

    Everything this project asserts carries one of these, so any figure or verdict can be
    traced to the passage it rests on. Sources vary in what they can be addressed by — a
    tender has clauses, a drawing has none — so every locating field is optional, but at
    least one must be present: a reference that points nowhere is not evidence.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    document: str
    page: int | None = Field(default=None, ge=0)
    clause: str | None = None
    bbox: BoundingBox | None = None
    locator: str | None = Field(
        default=None,
        description="Free-form address for sources with no page or clause, "
        'e.g. "BOQ row 214" or "Drawing SEC-004 detail B".',
    )
    quote: str

    @model_validator(mode="after")
    def _require_a_locator(self) -> "EvidenceRef":
        """Reject a reference that carries no way to find the passage again."""
        if self.page is None and self.clause is None and self.bbox is None and self.locator is None:
            raise ValueError(
                "EvidenceRef needs at least one of page, clause, bbox or locator; "
                "otherwise it does not point anywhere."
            )
        return self


class Verdict(BaseModel):
    """One evaluation outcome for one item.

    `passed` and `score` are computed by deterministic Python before this record is built.
    `judge_model` records which model, if any, turned prose into structure on the way —
    never which model decided the outcome.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str
    expected: str
    actual: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    judge_model: str | None = None


class KPISnapshot(BaseModel):
    """One measurement of one metric, at one point in time."""

    model_config = ConfigDict(frozen=True)

    project: str
    metric_name: str
    value: float
    unit: str
    target: float
    direction: Direction
    measured_at: datetime
    sample_size: int = Field(ge=0)
    method: Method

    def to_row(self) -> dict[str, str | float | int]:
        """Flatten the snapshot for tabular display, without formatting any value."""
        return {
            "project": self.project,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "target": self.target,
            "direction": self.direction,
            "measured_at": self.measured_at.isoformat(),
            "sample_size": self.sample_size,
            "method": self.method,
        }

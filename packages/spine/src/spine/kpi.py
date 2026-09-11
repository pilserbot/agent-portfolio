"""KPI computation: turning typed run records into the numbers a dashboard displays.

A `KPISpec` says what one metric is — its name, unit, target, direction, the method that
establishes it, and which named computation produces it. A project declares its set of
them in `evals/projects/<project>.yaml`, so the definition of a metric lives in version
control next to the code, and a number in a report can always be traced to the spec that
asked for it.

`compute_kpis` dispatches each spec to a pure function over the Verdicts and the AgentRun.
The set of computations is a closed vocabulary, not a callable in a config file: an
evaluation harness that could be handed arbitrary code to run is no longer an instrument.

Cost figures obey one rule, and it is the reason this module cares what mode a call ran
in: **a run holding a replayed call cannot produce a cost figure unless the caller says so
explicitly.** A replayed call carries the cost the real call had when it was recorded, so a
demo would otherwise report a per-unit cost that looks measured and is not. See
`cost_basis`.

The ROI model takes every assumption as a named input — hours, hourly costs, volumes, the
implementation cost, and a free-text note saying where those numbers came from. There are
no default rates and no hardcoded figures anywhere in this module, because an ROI whose
assumptions are invisible is a sales slide rather than a measurement.

Deliberately does not: render anything beyond one markdown table, query a database, fetch a
price list, or estimate a figure with a language model. Every monetary amount, rate and
count here is arithmetic over structures the caller has already loaded. It also does not
decide whether a KPI set is the right one for a project — that judgement lives in the
project's YAML, where a human wrote it.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spine.contracts import AgentRun, Direction, KPISnapshot, Method, ModelCall, Verdict
from spine.eval import metrics

__all__ = [
    "COST_COMPUTATIONS",
    "Computation",
    "CostBasis",
    "KPIError",
    "KPISpec",
    "KPISpecSet",
    "NoPaybackError",
    "ReplayedCostError",
    "ROIInputs",
    "ROIResult",
    "RoleHours",
    "compute_kpis",
    "compute_roi",
    "cost_basis",
    "meets_target",
    "render_markdown",
]

# Every metric this module knows how to produce. A closed vocabulary on purpose: a spec
# names one of these, rather than supplying an expression for the harness to evaluate.
Computation = Literal[
    "items_evaluated",
    "accuracy",
    "mean_score",
    "precision",
    "recall",
    "f1",
    "macro_f1_by_tag",
    "failed_steps",
    "total_tokens",
    "mean_latency_ms",
    "total_cost_usd",
    "cost_per_item_usd",
    "hours_displaced_per_unit",
    "hours_displaced_per_period",
    "automated_cost_per_unit_usd",
    "saving_per_unit_usd",
    "payback_periods",
]

# The computations that state what something cost, and so must not be produced from a run
# holding replayed calls unless the caller has said so explicitly.
COST_COMPUTATIONS: frozenset[str] = frozenset(
    {
        "total_cost_usd",
        "cost_per_item_usd",
        "automated_cost_per_unit_usd",
        "saving_per_unit_usd",
        "payback_periods",
    }
)

# The computations that need an ROI block in the project config.
_ROI_COMPUTATIONS: frozenset[str] = frozenset(
    {
        "hours_displaced_per_unit",
        "hours_displaced_per_period",
        "automated_cost_per_unit_usd",
        "saving_per_unit_usd",
        "payback_periods",
    }
)

# The computations that compare a predicted label against an expected one, and so need the
# spec to say which label is the positive one.
_LABEL_COMPUTATIONS: frozenset[str] = frozenset({"precision", "recall", "f1", "macro_f1_by_tag"})

PASS_MARK = "PASS"
FAIL_MARK = "FAIL"


class KPIError(Exception):
    """A KPI could not be computed from what was supplied."""


class ReplayedCostError(KPIError):
    """A cost figure was asked for from a run whose calls were served from a cassette.

    Raised rather than answered, because the honest answers are both bad: excluding the
    replayed calls reports a cost of nothing, and including them reports what some earlier
    run spent as though this one had spent it. Either would look like a measurement. The
    caller decides, in the open, by setting `include_replayed_cost`.
    """


class NoPaybackError(KPIError):
    """A payback was asked for from assumptions under which the build never pays back."""


class CostBasis(BaseModel):
    """What a run's model calls cost, and which of them were counted.

    `total_usd` is the sum over the counted calls only. `replayed_calls` is reported even
    when they were counted, so a report can always say what it is built on.
    """

    model_config = ConfigDict(frozen=True)

    total_usd: Decimal = Field(ge=0, description="Cost of the counted calls, in usd.")
    counted_calls: int = Field(ge=0)
    billed_calls: int = Field(ge=0, description="Calls that reached a provider and spent money.")
    replayed_calls: int = Field(ge=0, description="Calls served from a cassette.")
    includes_replayed: bool = Field(description="Whether replayed calls are inside total_usd.")

    @property
    def is_measured(self) -> bool:
        """Whether every counted call actually spent the money it reports."""
        return self.replayed_calls == 0


def model_calls(run: AgentRun) -> list[ModelCall]:
    """Every model call the run made, in step order."""
    return [call for step in run.steps for call in step.model_calls]


def cost_basis(run: AgentRun, *, include_replayed: bool = False) -> CostBasis:
    """Total what a run's calls cost, refusing to guess when some of them were replayed.

    With `include_replayed=False` (the default) a run holding any replayed call raises
    `ReplayedCostError`. Not "the replayed ones are dropped": dropping them would report a
    demo as having cost nothing, which is exactly the plausible, wrong number this rule
    exists to prevent. One replayed call is enough — a partly replayed run's spend is not
    the cost of doing the work either.

    With `include_replayed=True` the recorded costs are counted, and the returned basis
    says so through `includes_replayed` and `is_measured`, so a report can disclose it.
    """
    calls = model_calls(run)
    replayed = [call for call in calls if not call.was_billed]
    billed = [call for call in calls if call.was_billed]

    if replayed and not include_replayed:
        raise ReplayedCostError(
            f"run {run.run_id!r} has {len(replayed)} of {len(calls)} model call(s) served from a "
            f"cassette (mode='replay'), so it cannot report what the work costs. Run it live, or "
            f"set include_replayed_cost on the project's KPI config to count the recorded costs "
            f"— those are what an earlier live run spent, not money spent now."
        )

    counted = calls if include_replayed else billed
    return CostBasis(
        total_usd=sum((call.cost_usd for call in counted), Decimal("0")),
        counted_calls=len(counted),
        billed_calls=len(billed),
        replayed_calls=len(replayed),
        includes_replayed=bool(replayed) and include_replayed,
    )


class RoleHours(BaseModel):
    """One role's time on one unit of work, and what an hour of it costs.

    Both numbers are required. There is no default hourly rate here or anywhere else in
    this module: a rate nobody stated is an assumption nobody can audit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    hours_per_unit: float = Field(ge=0.0)
    hourly_cost_usd: Decimal = Field(ge=0, description="Fully loaded cost of an hour, in usd.")

    @property
    def cost_per_unit_usd(self) -> Decimal:
        """What this role's time on one unit costs, in usd."""
        return (Decimal(str(self.hours_per_unit)) * self.hourly_cost_usd).quantize(
            Decimal("0.0001")
        )


class ROIInputs(BaseModel):
    """Every assumption behind an ROI figure, stated explicitly.

    `unit` names what a unit is (a tender, a bid, a document); `period_label` names the
    period volumes and payback are expressed in. `assumptions_source` is required and has
    no default: a set of ROI inputs that does not say where its numbers came from is
    indistinguishable from a set that was invented, and this model refuses to be that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str = Field(min_length=1, description='What one unit of work is, e.g. "tender".')
    period_label: str = Field(min_length=1, description='The period, e.g. "month".')
    units_per_period: float = Field(gt=0.0)
    baseline_roles: list[RoleHours] = Field(
        min_length=1, description="The human work one unit took before any automation."
    )
    residual_roles: list[RoleHours] = Field(
        default_factory=list,
        description="The human work one unit still takes with the system in place.",
    )
    implementation_cost_usd: Decimal = Field(
        ge=0, description="One-off cost of building and deploying, in usd. Drives payback."
    )
    automated_model_cost_per_unit_usd: Decimal = Field(
        ge=0,
        description="Model spend per unit, in usd. This is the assumed figure; when a run "
        "is available `compute_kpis` replaces it with what that run actually cost, because "
        "a measurement beats an assumption wherever one exists.",
    )
    assumptions_source: str = Field(
        min_length=1,
        description="Where these numbers came from. Say 'illustrative placeholder' when "
        "they are not measured — a reader must never have to guess.",
    )

    @model_validator(mode="after")
    def _residual_must_not_exceed_baseline(self) -> "ROIInputs":
        """Refuse residual hours above the baseline: that is not automation."""
        baseline = sum(role.hours_per_unit for role in self.baseline_roles)
        residual = sum(role.hours_per_unit for role in self.residual_roles)
        if residual > baseline:
            raise ValueError(
                f"residual human hours per unit ({residual:g}) exceed the baseline "
                f"({baseline:g}); an ROI model cannot displace a negative number of hours."
            )
        return self


class ROIResult(BaseModel):
    """What the ROI inputs imply, computed and nothing more.

    `payback_periods` is None when the saving is zero or negative: a system that saves
    nothing never pays back, and reporting a negative or infinite payback as a number
    would read like a schedule.
    """

    model_config = ConfigDict(frozen=True)

    unit: str
    period_label: str
    baseline_hours_per_unit: float = Field(ge=0.0)
    residual_hours_per_unit: float = Field(ge=0.0)
    hours_displaced_per_unit: float = Field(ge=0.0)
    hours_displaced_per_period: float = Field(ge=0.0)
    baseline_cost_per_unit_usd: Decimal = Field(ge=0)
    residual_human_cost_per_unit_usd: Decimal = Field(ge=0)
    automated_model_cost_per_unit_usd: Decimal = Field(ge=0)
    automated_cost_per_unit_usd: Decimal = Field(
        ge=0, description="Residual human cost plus model cost, per unit, in usd."
    )
    saving_per_unit_usd: Decimal = Field(description="Baseline minus automated; may be negative.")
    saving_per_period_usd: Decimal
    implementation_cost_usd: Decimal = Field(ge=0)
    payback_periods: float | None = Field(
        default=None, description="Periods to recover the implementation cost; None if never."
    )
    assumptions_source: str


def _hours(roles: Sequence[RoleHours]) -> float:
    """Total hours per unit across roles."""
    return sum(role.hours_per_unit for role in roles)


def _cost(roles: Sequence[RoleHours]) -> Decimal:
    """Total cost per unit across roles, in usd."""
    return sum((role.cost_per_unit_usd for role in roles), Decimal("0"))


def compute_roi(inputs: ROIInputs) -> ROIResult:
    """Turn a stated set of ROI assumptions into the figures they imply.

    Pure arithmetic: the same inputs always give the same result, and every number in the
    result traces to a field of the input.
    """
    baseline_hours = _hours(inputs.baseline_roles)
    residual_hours = _hours(inputs.residual_roles)
    displaced = baseline_hours - residual_hours

    baseline_cost = _cost(inputs.baseline_roles)
    residual_cost = _cost(inputs.residual_roles)
    automated_cost = residual_cost + inputs.automated_model_cost_per_unit_usd
    saving_per_unit = baseline_cost - automated_cost
    units = Decimal(str(inputs.units_per_period))
    saving_per_period = saving_per_unit * units

    payback: float | None = None
    if saving_per_period > 0:
        payback = float(inputs.implementation_cost_usd / saving_per_period)

    return ROIResult(
        unit=inputs.unit,
        period_label=inputs.period_label,
        baseline_hours_per_unit=baseline_hours,
        residual_hours_per_unit=residual_hours,
        hours_displaced_per_unit=displaced,
        hours_displaced_per_period=displaced * inputs.units_per_period,
        baseline_cost_per_unit_usd=baseline_cost,
        residual_human_cost_per_unit_usd=residual_cost,
        automated_model_cost_per_unit_usd=inputs.automated_model_cost_per_unit_usd,
        automated_cost_per_unit_usd=automated_cost,
        saving_per_unit_usd=saving_per_unit,
        saving_per_period_usd=saving_per_period,
        implementation_cost_usd=inputs.implementation_cost_usd,
        payback_periods=payback,
        assumptions_source=inputs.assumptions_source,
    )


class KPISpec(BaseModel):
    """One metric a project reports: what it is, what good looks like, and how it is got."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1, description='e.g. "ratio", "usd", "count", "hours".')
    target: float
    direction: Direction
    method: Method = Field(
        description="How the figure is established: test, analysis, inspection or demonstration."
    )
    computation: Computation = Field(description="Which named computation produces the value.")
    positive_label: str | None = Field(
        default=None,
        description="Which expected label counts as positive. Required by precision, "
        "recall, f1 and macro_f1_by_tag, meaningless to the rest.",
    )
    tolerance: float = Field(
        default=0.0,
        ge=0.0,
        description="How far this metric may drift the wrong way before it is a regression.",
    )
    description: str = ""

    @model_validator(mode="after")
    def _label_metrics_need_a_label(self) -> "KPISpec":
        """Refuse a classification metric with no positive label named."""
        if self.computation in _LABEL_COMPUTATIONS and not self.positive_label:
            raise ValueError(
                f"KPI {self.name!r} computes {self.computation!r}, which compares labels; "
                f"set positive_label to say which one is positive."
            )
        return self

    @property
    def needs_roi(self) -> bool:
        """Whether this metric can only be computed with an ROI model."""
        return self.computation in _ROI_COMPUTATIONS

    @property
    def is_cost(self) -> bool:
        """Whether this metric states what something cost."""
        return self.computation in COST_COMPUTATIONS


class KPISpecSet(BaseModel):
    """Everything a project reports, and the ROI assumptions behind the money in it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str = Field(min_length=1)
    kpis: list[KPISpec] = Field(min_length=1)
    roi: ROIInputs | None = None
    include_replayed_cost: bool = Field(
        default=False,
        description="Count cassette-served calls in cost figures. Opt in knowingly: the "
        "resulting number is what an earlier live run spent, not what this one did.",
    )

    @model_validator(mode="after")
    def _roi_metrics_need_roi_inputs(self) -> "KPISpecSet":
        """Refuse to declare an ROI metric with no assumptions to compute it from."""
        if self.roi is None:
            orphans = [spec.name for spec in self.kpis if spec.needs_roi]
            if orphans:
                raise ValueError(
                    f"{', '.join(orphans)} need an roi block; without one there is nothing to "
                    f"compute them from, and this module will not supply a default rate."
                )
        return self

    @model_validator(mode="after")
    def _kpi_names_are_unique(self) -> "KPISpecSet":
        """Refuse duplicate names: one metric's value would overwrite the other's."""
        names = [spec.name for spec in self.kpis]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate KPI name(s): {', '.join(duplicates)}")
        return self


def meets_target(snapshot: KPISnapshot) -> bool:
    """Whether a measurement is on the right side of its target."""
    if snapshot.direction == "higher_is_better":
        return snapshot.value >= snapshot.target
    return snapshot.value <= snapshot.target


def _mean_latency_ms(run: AgentRun) -> float:
    """Mean latency across the run's model calls, or zero when it made none."""
    calls = model_calls(run)
    return sum(call.latency_ms for call in calls) / len(calls) if calls else 0.0


def _mean_score(verdicts: Sequence[Verdict]) -> float:
    """Mean verdict score, or zero over nothing."""
    return sum(verdict.score for verdict in verdicts) / len(verdicts) if verdicts else 0.0


def _label_value(spec: KPISpec, verdicts: Sequence[Verdict], tags: dict[str, list[str]]) -> float:
    """Compute one of the label-comparing metrics."""
    # The validator above guarantees a label is present for exactly these computations.
    label = spec.positive_label or ""
    if spec.computation == "precision":
        return metrics.precision(verdicts, positive_label=label).value
    if spec.computation == "recall":
        return metrics.recall(verdicts, positive_label=label).value
    if spec.computation == "f1":
        return metrics.f1(verdicts, positive_label=label).value
    return metrics.macro_f1_by_tag(verdicts, tags, positive_label=label).macro_f1


def _roi_value(spec: KPISpec, roi: ROIResult) -> float:
    """Read one ROI figure off a computed ROI result."""
    if spec.computation == "hours_displaced_per_unit":
        return roi.hours_displaced_per_unit
    if spec.computation == "hours_displaced_per_period":
        return roi.hours_displaced_per_period
    if spec.computation == "automated_cost_per_unit_usd":
        return float(roi.automated_cost_per_unit_usd)
    if spec.computation == "saving_per_unit_usd":
        return float(roi.saving_per_unit_usd)
    # payback_periods. There is no honest float for "never": zero flatters it, a sentinel
    # is a lie, and infinity serialises to JSON null and will not round-trip back into the
    # gate. So this refuses, exactly as a replayed cost figure does.
    if roi.payback_periods is None:
        raise NoPaybackError(
            f"KPI {spec.name!r} asks how many {roi.period_label}s until the build pays for "
            f"itself, but at these assumptions it saves "
            f"{roi.saving_per_period_usd} usd per {roi.period_label} and never does. "
            f"Fix the assumptions ({roi.assumptions_source}) or stop reporting a payback."
        )
    return roi.payback_periods


def compute_kpis(
    verdicts: Sequence[Verdict],
    run: AgentRun,
    spec: KPISpecSet,
    *,
    tags_by_item: dict[str, list[str]] | None = None,
    measured_at: datetime | None = None,
) -> list[KPISnapshot]:
    """Measure every metric a project declares, in the order it declared them.

    Raises `ReplayedCostError` before producing anything when the set contains a cost
    metric and the run holds replayed calls that the config has not opted to count. It
    fails up front rather than part way, so a caller never gets a half-written report
    whose quality numbers are real and whose money numbers are missing.
    """
    at = measured_at or datetime.now(UTC)
    tags = tags_by_item or {}

    basis: CostBasis | None = None
    if any(item.is_cost for item in spec.kpis):
        basis = cost_basis(run, include_replayed=spec.include_replayed_cost)

    roi: ROIResult | None = None
    if spec.roi is not None and any(item.needs_roi for item in spec.kpis):
        measured = spec.roi.automated_model_cost_per_unit_usd
        if basis is not None and verdicts:
            # Prefer what the run actually cost per item over the config's stated figure:
            # the config value is the assumption, this is the measurement.
            measured = (basis.total_usd / Decimal(len(verdicts))).quantize(Decimal("0.000001"))
        roi = compute_roi(
            spec.roi.model_copy(update={"automated_model_cost_per_unit_usd": measured})
        )

    snapshots: list[KPISnapshot] = []
    for item in spec.kpis:
        snapshots.append(
            KPISnapshot(
                project=spec.project,
                metric_name=item.name,
                value=_value_of(item, verdicts, run, basis, roi, tags),
                unit=item.unit,
                target=item.target,
                direction=item.direction,
                measured_at=at,
                sample_size=len(verdicts),
                method=item.method,
            )
        )
    return snapshots


def _value_of(
    spec: KPISpec,
    verdicts: Sequence[Verdict],
    run: AgentRun,
    basis: CostBasis | None,
    roi: ROIResult | None,
    tags: dict[str, list[str]],
) -> float:
    """Dispatch one spec to the computation it names."""
    if spec.needs_roi:
        if roi is None:
            raise KPIError(f"KPI {spec.name!r} needs an ROI model and none was supplied.")
        return _roi_value(spec, roi)
    if spec.computation in _LABEL_COMPUTATIONS:
        return _label_value(spec, verdicts, tags)
    if spec.computation == "items_evaluated":
        return float(len(verdicts))
    if spec.computation == "accuracy":
        return metrics.accuracy(verdicts).value
    if spec.computation == "mean_score":
        return _mean_score(verdicts)
    if spec.computation == "failed_steps":
        return float(len(run.failed_steps))
    if spec.computation == "total_tokens":
        return float(run.total_tokens)
    if spec.computation == "mean_latency_ms":
        return _mean_latency_ms(run)
    if basis is None:  # pragma: no cover - every cost computation populates a basis
        raise KPIError(f"KPI {spec.name!r} is a cost metric with no cost basis.")
    if spec.computation == "total_cost_usd":
        return float(basis.total_usd)
    # cost_per_item_usd
    return float(basis.total_usd / Decimal(len(verdicts))) if verdicts else 0.0


def _format(value: float) -> str:
    """Render a number for the table without inventing precision it does not have."""
    return f"{value:,.4g}"


def render_markdown(snapshots: Sequence[KPISnapshot]) -> str:
    """Render measurements as one markdown table: target, actual, and whether it passed.

    Returns a string rather than a model because it is the presentation boundary — the
    values it formats have all been decided already, and nothing downstream reads this
    back as data.
    """
    header = [
        "| Metric | Target | Actual | Unit | Method | Status |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    if not snapshots:
        return "\n".join([*header, "| _no metrics declared_ | | | | | |"]) + "\n"

    rows = []
    for snapshot in snapshots:
        arrow = "≥" if snapshot.direction == "higher_is_better" else "≤"
        status = PASS_MARK if meets_target(snapshot) else FAIL_MARK
        rows.append(
            f"| {snapshot.metric_name} | {arrow} {_format(snapshot.target)} "
            f"| {_format(snapshot.value)} | {snapshot.unit} | {snapshot.method} | {status} |"
        )
    return "\n".join(header + rows) + "\n"

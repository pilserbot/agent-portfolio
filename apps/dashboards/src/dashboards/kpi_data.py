"""Reading a project's declared KPIs and its result history off disk, ready to render.

Everything the KPI dashboard shows is assembled here, and nothing here imports Streamlit.
That split is what lets the assembly be tested headlessly, and it keeps the rendering layer
to layout and words.

The inputs are committed files and nothing else: `evals/projects/<project>.yaml` for what a
project reports, and `evals/results/<project>_<timestamp>.json` for what it measured. No
database, no API key, no network — which is what makes the app deployable to Streamlit
Community Cloud from a clone of this repository.

Deliberately does not: compute a metric, decide whether a run passed, or price anything.
Every figure it hands the page was computed by `spine` and written to a file before this
module ever saw it; the one judgement here is ordering, and even that is read from config.
A result file it cannot parse is reported as unreadable rather than skipped, because a
dashboard quietly showing fewer runs than exist is worse than one saying it found a bad
file.
"""

import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spine.contracts import KPISnapshot
from spine.eval.projects import (
    DEFAULT_PROJECT_ROOT,
    ProjectConfig,
    ProjectError,
    load_project_config,
)
from spine.eval.run import DEFAULT_RESULTS_ROOT, EvalReport
from spine.kpi import KPISpec, ROIResult, compute_roi, meets_target

DEFAULT_HEADLINE_COUNT = 2

__all__ = [
    "DEFAULT_HEADLINE_COUNT",
    "DashboardError",
    "Headline",
    "MetricRow",
    "ProjectView",
    "ResultHistory",
    "TrendPoint",
    "TrendSeries",
    "available_projects",
    "build_project_view",
    "format_usd",
    "format_value",
    "headline_specs",
    "load_history",
    "metric_rows",
    "result_paths",
    "roi_of",
    "trend_series",
]


class DashboardError(Exception):
    """The dashboard could not assemble what it was asked to show."""


def available_projects(project_root: Path = DEFAULT_PROJECT_ROOT) -> list[str]:
    """Every project with a committed config, sorted.

    Driven by what is on disk rather than by a list in this module, so adding a project is
    adding a YAML file and nothing else.
    """
    if not project_root.is_dir():
        return []
    return sorted(path.stem for path in project_root.glob("*.yaml"))


def result_paths(project: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> list[Path]:
    """Every committed result file for a project, oldest name first.

    Matched on the `<project>_<timestamp>.json` shape the runner writes, so a project named
    `example` never picks up `example_human`'s files. The timestamps sort lexically because
    they are written as `%Y%m%dT%H%M%SZ`, which is why the name can be trusted for order
    before anything is parsed.
    """
    if not results_root.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(project)}_\d{{8}}T\d{{6}}Z$")
    return sorted(
        path for path in results_root.glob(f"{project}_*.json") if pattern.match(path.stem)
    )


class ResultHistory(BaseModel):
    """Every result a project has recorded, oldest first, and what could not be read."""

    model_config = ConfigDict(frozen=True)

    project: str
    reports: list[EvalReport] = Field(default_factory=list)
    unreadable: list[Path] = Field(
        default_factory=list,
        description="Files that did not parse. Surfaced, never skipped in silence.",
    )

    @property
    def is_empty(self) -> bool:
        """Whether the project has recorded nothing yet."""
        return not self.reports

    @property
    def latest(self) -> EvalReport | None:
        """The most recent result, by the moment it was measured."""
        return self.reports[-1] if self.reports else None


def load_history(project: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> ResultHistory:
    """Load a project's result files, ordered by when they were measured.

    Ordered by `measured_at` rather than by filename, so a file copied in out of order
    still lands where it belongs on a trend line.
    """
    reports: list[EvalReport] = []
    unreadable: list[Path] = []
    for path in result_paths(project, results_root):
        try:
            reports.append(EvalReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValidationError, ValueError):
            unreadable.append(path)
    reports.sort(key=lambda report: report.measured_at)
    return ResultHistory(project=project, reports=reports, unreadable=unreadable)


def headline_specs(config: ProjectConfig) -> tuple[list[KPISpec], bool]:
    """The metrics to show large, and whether they were declared or fallen back to.

    A project that declares nothing gets its first two metrics, and the caller is told so,
    because a reader must be able to tell an editorial choice from a default.
    """
    declared = config.kpis.headline_specs()
    if declared:
        return declared, True
    return list(config.kpis.kpis[:DEFAULT_HEADLINE_COUNT]), False


class MetricRow(BaseModel):
    """One row of the KPI table: what was asked for, what was measured, and the verdict."""

    model_config = ConfigDict(frozen=True)

    metric: str
    target: float
    value: float
    unit: str
    method: str
    direction: str
    sample_size: int = Field(ge=0)
    passed: bool
    description: str = ""

    @property
    def comparator(self) -> str:
        """The sign a reader needs to see which way is good."""
        return "≥" if self.direction == "higher_is_better" else "≤"

    @property
    def status(self) -> str:
        """The word shown in the status column."""
        return "PASS" if self.passed else "FAIL"


def metric_rows(config: ProjectConfig, report: EvalReport) -> list[MetricRow]:
    """Pair each measurement with the spec that asked for it, in declaration order.

    A measurement with no matching spec is dropped and a spec with no measurement is
    skipped: both mean the config has moved since the run, and the honest thing is to show
    what the two agree on. The count of what was dropped is the caller's to surface — see
    `ProjectView.stale_metrics`.
    """
    described = {spec.name: spec.description for spec in config.kpis.kpis}
    order = {spec.name: index for index, spec in enumerate(config.kpis.kpis)}
    rows = [
        MetricRow(
            metric=snapshot.metric_name,
            target=snapshot.target,
            value=snapshot.value,
            unit=snapshot.unit,
            method=snapshot.method,
            direction=snapshot.direction,
            sample_size=snapshot.sample_size,
            passed=meets_target(snapshot),
            description=described.get(snapshot.metric_name, ""),
        )
        for snapshot in report.snapshots
        if snapshot.metric_name in order
    ]
    return sorted(rows, key=lambda row: order[row.metric])


class TrendPoint(BaseModel):
    """One metric's value at one moment."""

    model_config = ConfigDict(frozen=True)

    measured_at: datetime
    value: float
    run_id: str
    passed: bool


class TrendSeries(BaseModel):
    """One metric's history, and the target it was held to at the end of it."""

    model_config = ConfigDict(frozen=True)

    metric: str
    unit: str
    direction: str
    target: float
    points: list[TrendPoint] = Field(default_factory=list)

    @property
    def has_history(self) -> bool:
        """Whether there is more than one point, so a line means something."""
        return len(self.points) > 1


def trend_series(history: ResultHistory) -> list[TrendSeries]:
    """One series per metric, across every result the project has recorded.

    A metric absent from an older report simply has no point there rather than a zero: a
    gap in the record is not a measurement of nothing.
    """
    ordered: list[str] = []
    points: dict[str, list[TrendPoint]] = {}
    latest: dict[str, KPISnapshot] = {}

    for report in history.reports:
        for snapshot in report.snapshots:
            if snapshot.metric_name not in points:
                ordered.append(snapshot.metric_name)
                points[snapshot.metric_name] = []
            points[snapshot.metric_name].append(
                TrendPoint(
                    measured_at=report.measured_at,
                    value=snapshot.value,
                    run_id=report.run_id,
                    passed=meets_target(snapshot),
                )
            )
            latest[snapshot.metric_name] = snapshot

    return [
        TrendSeries(
            metric=name,
            unit=latest[name].unit,
            direction=latest[name].direction,
            target=latest[name].target,
            points=points[name],
        )
        for name in ordered
    ]


def roi_of(config: ProjectConfig, report: EvalReport | None) -> ROIResult | None:
    """The project's ROI under its stated assumptions, or None when it declares none.

    Where a run measured what the work cost, that figure replaces the config's assumed one
    — the same substitution `spine.kpi.compute_kpis` makes, for the same reason: a
    measurement beats an assumption wherever one exists. Where the run may not state a
    cost, the assumption stands and the page says which it is showing.
    """
    inputs = config.kpis.roi
    if inputs is None:
        return None
    if report is not None and report.cost.basis is not None and report.items_evaluated > 0:
        per_item = report.cost.basis.total_usd / report.items_evaluated
        inputs = inputs.model_copy(update={"automated_model_cost_per_unit_usd": per_item})
    return compute_roi(inputs)


class Headline(BaseModel):
    """One of the figures shown large at the top of the page."""

    model_config = ConfigDict(frozen=True)

    row: MetricRow
    declared: bool = Field(description="False when this was chosen by fallback, not by config.")


class ProjectView(BaseModel):
    """Everything one project's page shows, assembled from files and nothing else."""

    model_config = ConfigDict(frozen=True)

    project: str
    config: ProjectConfig
    history: ResultHistory
    latest: EvalReport | None = None
    headlines: list[Headline] = Field(default_factory=list)
    rows: list[MetricRow] = Field(default_factory=list)
    trends: list[TrendSeries] = Field(default_factory=list)
    roi: ROIResult | None = None

    @property
    def has_results(self) -> bool:
        """Whether this project has anything measured to show."""
        return self.latest is not None

    @property
    def stale_metrics(self) -> list[str]:
        """Metrics the config declares that the latest run did not measure.

        Named rather than counted: the usual cause is a metric added since the last run,
        and a reader seeing the name can tell that from a metric that failed to compute.
        """
        if self.latest is None:
            return []
        measured = {snapshot.metric_name for snapshot in self.latest.snapshots}
        return [spec.name for spec in self.config.kpis.kpis if spec.name not in measured]

    @property
    def retired_metrics(self) -> list[str]:
        """Metrics the latest run measured that the config no longer declares."""
        if self.latest is None:
            return []
        declared = {spec.name for spec in self.config.kpis.kpis}
        return sorted(
            {
                snapshot.metric_name
                for snapshot in self.latest.snapshots
                if snapshot.metric_name not in declared
            }
        )


def build_project_view(
    project: str,
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> ProjectView:
    """Assemble one project's page from its config and its result history.

    A project with no results yet is not an error: the view comes back with `has_results`
    False and the config filled in, so the page can say plainly that nothing has run.
    """
    try:
        config = load_project_config(project, root=project_root)
    except ProjectError as error:
        raise DashboardError(str(error)) from error

    history = load_history(project, results_root)
    latest = history.latest
    specs, declared = headline_specs(config)
    rows = metric_rows(config, latest) if latest is not None else []
    by_name = {row.metric: row for row in rows}

    return ProjectView(
        project=project,
        config=config,
        history=history,
        latest=latest,
        headlines=[
            Headline(row=by_name[spec.name], declared=declared)
            for spec in specs
            if spec.name in by_name
        ],
        rows=rows,
        trends=trend_series(history),
        roi=roi_of(config, latest),
    )


def format_usd(amount: Decimal | float) -> str:
    """Render a monetary amount for display, to the cent or finer when it is small.

    Formatting money happens here, at the presentation edge, and nowhere in `spine` — the
    contracts carry `Decimal` precisely so a display can decide this and the records need
    not. A per-item cost of a fraction of a cent shows its real magnitude rather than
    rounding to `$0.00`, which would read as free.
    """
    value = float(amount)
    if value and abs(value) < 0.01:
        return f"${value:,.6f}"
    return f"${value:,.2f}"


def format_value(value: float, unit: str) -> str:
    """Render one metric value for display, in the terms its unit implies."""
    if unit == "usd":
        return format_usd(value)
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "ratio":
        return f"{value:.1%}"
    return f"{value:,.4g}"

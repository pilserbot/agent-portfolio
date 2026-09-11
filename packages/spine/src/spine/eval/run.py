"""Evaluation entry point: run a project's gold set, measure its KPIs, gate on regressions.

Invoked as `python -m spine.eval.run --project example --sample 20 --out eval_report.md
--json-out eval_report.json`.

Omitting `--sample` evaluates the full gold set. Three files are written: the human report
at `--out`, the machine-readable one at `--json-out` (beside `--out` under the same stem
when that flag is omitted), and a timestamped copy under `evals/results/`, so a run's
numbers survive the next run overwriting the working-tree report.

The exit code is the verdict: zero when every metric the baseline records is inside its
tolerance, one when any has regressed. `--update-baseline` writes the current results as
the new reference instead, which is how a deliberate change is accepted.

Nothing here decides whether an answer was right. The project's registered entry point
produces Verdicts and a run record, `spine.kpi` turns those into measurements, and
`spine.eval.baseline` compares them. This module only sequences that and writes files.

Deliberately does not: contain any project's evaluation logic, know what a good number is,
or compute a metric of its own. It also does not silently tolerate a missing project: a
name with no config or no registered entry point is an error naming what is available.
"""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from spine.contracts import KPISnapshot
from spine.eval import example_project  # noqa: F401 - registers the reference project
from spine.eval.baseline import (
    Baseline,
    GateVerdict,
    baseline_from_snapshots,
    check_regressions,
    readings_from_snapshots,
)
from spine.eval.datasets import load_gold_set
from spine.eval.projects import (
    DEFAULT_PROJECT_ROOT,
    EvaluationInput,
    ProjectConfig,
    evaluator_for,
    load_project_config,
)
from spine.kpi import KPIError, compute_kpis, meets_target, render_markdown

DEFAULT_BASELINE_PATH = Path("evals/baseline.json")
DEFAULT_RESULTS_ROOT = Path("evals/results")
DEFAULT_PROJECT = "example"

__all__ = [
    "DEFAULT_BASELINE_PATH",
    "DEFAULT_PROJECT",
    "DEFAULT_RESULTS_ROOT",
    "EvalReport",
    "EvalRequest",
    "KpiRow",
    "WrittenReport",
    "build_report",
    "main",
    "render_report",
    "write_report",
]


class KpiRow(BaseModel):
    """One row of the report's KPI table, flattened for the gate and for a dashboard."""

    model_config = ConfigDict(frozen=True)

    metric: str
    value: float
    unit: str
    target: float
    direction: str
    method: str
    passed: bool


class EvalRequest(BaseModel):
    """What to evaluate, and where the result goes."""

    model_config = ConfigDict(frozen=True)

    out: Path
    json_out: Path | None = Field(
        default=None,
        description="Where the machine-readable report goes. None puts it beside `out`.",
    )
    project: str = Field(default=DEFAULT_PROJECT, description="Which project is evaluated.")
    sample: int | None = Field(default=None, description="None evaluates the full gold set.")
    seed: int = Field(default=0, description="Which reproducible sample `--sample` takes.")
    project_root: Path = Field(
        default=DEFAULT_PROJECT_ROOT, description="Where project configs are read from."
    )
    results_root: Path = DEFAULT_RESULTS_ROOT
    run_id: str | None = Field(default=None, description="None derives one from the clock.")


class EvalReport(BaseModel):
    """The outcome of one evaluation run."""

    model_config = ConfigDict(frozen=True)

    project: str
    scope: Literal["sample", "full"]
    sample_size: int | None = Field(
        description="What was asked for; `items_evaluated` is what ran."
    )
    items_evaluated: int = Field(ge=0)
    gold_set: str
    run_id: str
    measured_at: datetime
    snapshots: list[KPISnapshot] = Field(default_factory=list)
    cost_note: str = Field(
        default="",
        description="Anything a reader must know before believing the money in this report.",
    )

    @computed_field(description="The measurements flattened for the gate and for a dashboard.")
    @property
    def rows(self) -> list[KpiRow]:
        """The snapshots as flat rows.

        Derived rather than stored so the table and the measurements cannot disagree. It
        is a computed field rather than a plain property because `spine.eval.gate` reads
        it back out of the written JSON.
        """
        return [
            KpiRow(
                metric=snapshot.metric_name,
                value=snapshot.value,
                unit=snapshot.unit,
                target=snapshot.target,
                direction=snapshot.direction,
                method=snapshot.method,
                passed=meets_target(snapshot),
            )
            for snapshot in self.snapshots
        ]

    @property
    def failures(self) -> list[KpiRow]:
        """The metrics that missed their target, in report order."""
        return [row for row in self.rows if not row.passed]


class WrittenReport(BaseModel):
    """Where the report was written."""

    model_config = ConfigDict(frozen=True)

    markdown_path: Path
    metrics_path: Path
    archive_path: Path


def _cost_note(config: ProjectConfig) -> str:
    """What a reader must be told about the money in this report, if anything."""
    if config.kpis.include_replayed_cost:
        return (
            "Cost figures include calls served from a cassette. They are what an earlier "
            "live run spent, not money spent by this run."
        )
    return ""


def build_report(request: EvalRequest) -> EvalReport:
    """Run a project's evaluation and measure everything it declares.

    Raises `ProjectError` when the project has no config or no registered entry point,
    `GoldSetError` when its gold set will not load, and `KPIError` when a declared metric
    cannot honestly be produced from the run — a cost asked of a replayed run, say.
    """
    config = load_project_config(request.project, root=request.project_root)
    gold = load_gold_set(config.gold_set, root=config.gold_root)
    if request.sample is not None:
        gold = gold.sample(request.sample, seed=request.seed)

    measured_at = datetime.now(UTC)
    run_id = request.run_id or f"{request.project}-{measured_at.strftime('%Y%m%dT%H%M%SZ')}"

    evaluate = evaluator_for(request.project)
    outcome = evaluate(
        EvaluationInput(project=request.project, run_id=run_id, gold=gold, config=config)
    )

    snapshots = compute_kpis(
        outcome.verdicts,
        outcome.run,
        config.kpis,
        tags_by_item=gold.tags_by_item(),
        measured_at=measured_at,
    )
    return EvalReport(
        project=request.project,
        scope="full" if request.sample is None else "sample",
        sample_size=request.sample,
        items_evaluated=len(outcome.verdicts),
        gold_set=config.gold_set,
        run_id=run_id,
        measured_at=measured_at,
        snapshots=list(snapshots),
        cost_note=_cost_note(config),
    )


def render_report(report: EvalReport, verdict: GateVerdict | None = None) -> str:
    """Render the report as the markdown posted to the pull request."""
    scope = (
        "full gold set" if report.scope == "full" else f"{report.sample_size}-item sample requested"
    )
    lines = [
        "## Evaluation report",
        "",
        f"Project: **{report.project}**",
        f"Scope: **{scope}** — {report.items_evaluated} item(s) from `{report.gold_set}`",
        f"Run: `{report.run_id}` at {report.measured_at.isoformat()}",
        "",
        render_markdown(report.snapshots).rstrip("\n"),
        "",
    ]
    if report.cost_note:
        lines += [f"> **Note on cost.** {report.cost_note}", ""]
    if report.failures:
        missed = ", ".join(row.metric for row in report.failures)
        lines += [f"{len(report.failures)} metric(s) below target: {missed}.", ""]
    else:
        lines += ["Every metric is on target.", ""]

    if verdict is not None:
        lines += _gate_lines(verdict)
    return "\n".join(lines)


def _gate_lines(verdict: GateVerdict) -> list[str]:
    """The part of the report that says how it compared against the baseline."""
    lines = ["### Against the baseline", ""]
    if verdict.compared == 0 and not verdict.missing:
        lines += ["No baseline metrics recorded yet; nothing to compare.", ""]
        return lines
    bullets = [
        f"- Baseline metric `{name}` is absent from this report." for name in verdict.missing
    ]
    bullets += [
        f"- **REGRESSION** `{regression.metric}`: {regression.observed:g} against a baseline "
        f"of {regression.baseline:g} (off by {regression.delta:g}, tolerance "
        f"{regression.tolerance:g})."
        for regression in verdict.regressions
    ]
    if bullets:
        lines += [*bullets, ""]
    lines += [
        f"Compared {verdict.compared} metric(s): {'pass' if verdict.passed else 'fail'}.",
        "",
    ]
    return lines


def write_report(request: EvalRequest, report: EvalReport, markdown: str) -> WrittenReport:
    """Write the markdown report, its machine-readable sidecar, and the timestamped archive."""
    markdown_path = request.out
    metrics_path = request.json_out or markdown_path.with_suffix(".json")
    stamp = report.measured_at.strftime("%Y%m%dT%H%M%SZ")
    archive_path = request.results_root / f"{report.project}_{stamp}.json"

    payload = report.model_dump_json(indent=2) + "\n"
    for path, content in (
        (markdown_path, markdown),
        (metrics_path, payload),
        (archive_path, payload),
    ):
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return WrittenReport(
        markdown_path=markdown_path, metrics_path=metrics_path, archive_path=archive_path
    )


def _load_baseline(path: Path) -> Baseline:
    """Read the committed baseline, treating an absent file as nothing accepted yet."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Baseline()
    return Baseline.model_validate_json(text)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface, as one object so a test can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m spine.eval.run",
        description="Evaluate a project's gold set, measure its KPIs and gate on regressions.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help='Which project to evaluate, e.g. "example".',
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate this many items. Omit to evaluate the full gold set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Which reproducible sample --sample takes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path of the markdown report to write.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path of the machine-readable report. Defaults to --out with a .json suffix.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path of the committed baseline to compare against.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory the timestamped copy of each result is archived under.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Directory project configs are read from. Defaults to evals/projects.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write these results as the new baseline instead of gating on the old one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return the process exit code."""
    args = build_parser().parse_args(argv)
    request = EvalRequest(
        out=args.out,
        json_out=args.json_out,
        project=args.project,
        sample=args.sample,
        seed=args.seed,
        project_root=args.project_root or DEFAULT_PROJECT_ROOT,
        results_root=args.results_root,
    )

    try:
        report = build_report(request)
    except KPIError as error:
        # A refusal, not a crash: the module declined to publish a misleading figure.
        print(f"Cannot report KPIs for {request.project!r}: {error}")
        return 1

    baseline = _load_baseline(args.baseline)
    verdict = check_regressions(baseline, readings_from_snapshots(report.snapshots))
    written = write_report(request, report, render_report(report, verdict))
    print(f"Wrote {written.markdown_path}, {written.metrics_path} and {written.archive_path}")

    if args.update_baseline:
        config = load_project_config(request.project, root=request.project_root)
        updated = baseline_from_snapshots(report.snapshots, tolerances=config.tolerances)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Updated the baseline at {args.baseline} with {len(updated.metrics)} metric(s).")
        return 0

    for name in verdict.missing:
        print(f"WARNING: baseline metric {name!r} is absent from the report.")
    for regression in verdict.regressions:
        print(
            f"REGRESSION: {regression.metric} is {regression.observed:g} "
            f"against a baseline of {regression.baseline:g} "
            f"(off by {regression.delta:g}, tolerance {regression.tolerance:g})."
        )
    if not baseline.metrics:
        print("No baseline metrics recorded yet; nothing to compare.")
    print(f"Compared {verdict.compared} metric(s): {'pass' if verdict.passed else 'fail'}.")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

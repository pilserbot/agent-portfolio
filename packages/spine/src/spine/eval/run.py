"""Evaluation entry point: run a gold set and write a KPI report.

Invoked as `python -m spine.eval.run --sample 20 --out eval_report.md`. Omitting
`--sample` evaluates the full gold set. Two files are written: the markdown report at
`--out`, and a machine-readable sidecar with the same stem and a `.json` suffix that
`spine.eval.gate` compares against `evals/baseline.json`.

Deliberately does not: call a model, read a gold set, or compute a real KPI. This is a
stub that fixes the CLI contract and the report shape so the CI plumbing can be built and
verified today. It also never decides whether a run passed — that verdict belongs to
`spine.eval.gate`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class KpiRow(BaseModel):
    """One row of the report's KPI table."""

    metric: str
    value: float
    unit: str


class EvalRequest(BaseModel):
    """What to evaluate, and where the result goes."""

    out: Path
    sample: int | None = Field(default=None, description="None evaluates the full gold set.")


class EvalReport(BaseModel):
    """The outcome of one evaluation run."""

    scope: Literal["sample", "full"]
    sample_size: int | None
    rows: list[KpiRow]


class WrittenReport(BaseModel):
    """Where the report was written."""

    markdown_path: Path
    metrics_path: Path


def build_report(request: EvalRequest) -> EvalReport:
    """Produce the report for a request."""
    # TODO: replace with a real run over the gold set in data/. The metric names below are
    # placeholders; they are not yet agreed with the keys evals/baseline.json will record.
    rows = [
        KpiRow(metric="items_evaluated", value=float(request.sample or 0), unit="count"),
        KpiRow(metric="extraction_accuracy", value=0.0, unit="ratio"),
        KpiRow(metric="cost_per_item_usd", value=0.0, unit="usd"),
    ]
    return EvalReport(
        scope="full" if request.sample is None else "sample",
        sample_size=request.sample,
        rows=rows,
    )


def write_report(request: EvalRequest, report: EvalReport) -> WrittenReport:
    """Write the markdown report and its machine-readable sidecar."""
    markdown_path = request.out
    metrics_path = markdown_path.with_suffix(".json")
    if markdown_path.parent != Path(""):
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    metrics_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return WrittenReport(markdown_path=markdown_path, metrics_path=metrics_path)


def _render_markdown(report: EvalReport) -> str:
    """Render a report as the markdown posted to the pull request."""
    scope = "full gold set" if report.scope == "full" else f"{report.sample_size}-item sample"
    lines = [
        "## Evaluation report",
        "",
        f"Scope: **{scope}**",
        "",
        "| Metric | Value | Unit |",
        "| --- | ---: | --- |",
    ]
    lines += [f"| {row.metric} | {row.value:g} | {row.unit} |" for row in report.rows]
    lines += [
        "",
        "_Placeholder report — the evaluation harness is not implemented yet._",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m spine.eval.run",
        description="Evaluate the gold set and write a KPI report.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate this many items. Omit to evaluate the full gold set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path of the markdown report to write.",
    )
    args = parser.parse_args(argv)

    request = EvalRequest(out=args.out, sample=args.sample)
    written = write_report(request, build_report(request))
    print(f"Wrote {written.markdown_path} and {written.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

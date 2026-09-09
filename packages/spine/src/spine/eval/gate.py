"""Regression gate: decide whether an evaluation report is acceptable.

Invoked as `python -m spine.eval.gate --report eval_report.json --baseline
evals/baseline.json`. Exits 0 when every metric the baseline records is within its
tolerance, and 1 when any has regressed beyond it.

Deliberately does not: run an evaluation, write a report, or ask a model whether a change
is acceptable. The verdict is arithmetic over the two files and nothing else. A baseline
that records no metrics is not a failure — there is simply nothing to compare yet.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from spine.eval.run import EvalReport


class BaselineMetric(BaseModel):
    """The reference value for one metric, and how far it may drift."""

    value: float
    tolerance: float = 0.0
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"


class Baseline(BaseModel):
    """The committed reference a run is measured against."""

    metrics: dict[str, BaselineMetric] = Field(default_factory=dict)


class Regression(BaseModel):
    """One metric that moved the wrong way by more than its tolerance."""

    metric: str
    baseline: float
    observed: float
    tolerance: float
    delta: float


class GateVerdict(BaseModel):
    """The outcome of comparing a report against a baseline."""

    compared: int
    regressions: list[Regression]
    missing: list[str]
    passed: bool


def evaluate_gate(baseline: Baseline, report: EvalReport) -> GateVerdict:
    """Compare a report against a baseline and return the verdict."""
    observed = {row.metric: row.value for row in report.rows}
    regressions: list[Regression] = []
    missing: list[str] = []

    for name, expected in sorted(baseline.metrics.items()):
        if name not in observed:
            # TODO: decide whether a baseline metric absent from the report should fail the
            # gate. Reported but tolerated until the metric names are settled.
            missing.append(name)
            continue
        actual = observed[name]
        if expected.direction == "higher_is_better":
            delta = expected.value - actual
        else:
            delta = actual - expected.value
        if delta > expected.tolerance:
            regressions.append(
                Regression(
                    metric=name,
                    baseline=expected.value,
                    observed=actual,
                    tolerance=expected.tolerance,
                    delta=delta,
                )
            )

    return GateVerdict(
        compared=len(baseline.metrics) - len(missing),
        regressions=regressions,
        missing=missing,
        passed=not regressions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m spine.eval.gate",
        description="Fail if an evaluation report regressed beyond tolerance.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path of the JSON sidecar written by `spine.eval.run`.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path of the committed baseline, usually evals/baseline.json.",
    )
    args = parser.parse_args(argv)

    baseline = Baseline.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    report = EvalReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    verdict = evaluate_gate(baseline, report)

    for name in verdict.missing:
        print(f"WARNING: baseline metric {name!r} is absent from the report.")
    if not baseline.metrics:
        print("No baseline metrics recorded yet; nothing to compare.")
    for regression in verdict.regressions:
        print(
            f"REGRESSION: {regression.metric} is {regression.observed:g} "
            f"against a baseline of {regression.baseline:g} "
            f"(off by {regression.delta:g}, tolerance {regression.tolerance:g})."
        )
    print(f"Compared {verdict.compared} metric(s): {'pass' if verdict.passed else 'fail'}.")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

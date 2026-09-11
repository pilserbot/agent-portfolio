"""Regression gate: decide whether an evaluation report is acceptable.

Invoked as `python -m spine.eval.gate --report eval_report.json --baseline
evals/baseline.json`. Exits 0 when every metric the baseline records is within its
tolerance, and 1 when any has regressed beyond it.

`spine.eval.run` already gates itself on the same comparison, so this command is for the
case where the report and the check are separated: a report downloaded from an earlier job,
or a baseline changed after the fact. Both read one implementation, in
`spine.eval.baseline`, so the standalone check and the runner can never disagree.

Deliberately does not: run an evaluation, write a report, or ask a model whether a change
is acceptable. The verdict is arithmetic over the two files and nothing else. A baseline
that records no metrics is not a failure — there is simply nothing to compare yet.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from spine.eval.baseline import (
    Baseline,
    BaselineMetric,
    GateVerdict,
    Regression,
    check_regressions,
    readings_from_snapshots,
)
from spine.eval.run import EvalReport

__all__ = [
    "Baseline",
    "BaselineMetric",
    "GateVerdict",
    "Regression",
    "evaluate_gate",
    "main",
]


def evaluate_gate(baseline: Baseline, report: EvalReport) -> GateVerdict:
    """Compare a written report against a baseline and return the verdict."""
    return check_regressions(baseline, readings_from_snapshots(report.snapshots))


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

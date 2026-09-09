"""Unit tests for the evaluation stub and the regression gate.

Deliberately does not touch the network, the filesystem outside pytest's tmp_path, or any
environment variable.
"""

import json
from pathlib import Path

from spine.eval.gate import Baseline, evaluate_gate
from spine.eval.run import EvalRequest, build_report, main, write_report


def test_sample_and_full_scopes_are_distinguished() -> None:
    assert build_report(EvalRequest(out=Path("r.md"), sample=20)).scope == "sample"
    assert build_report(EvalRequest(out=Path("r.md"), sample=None)).scope == "full"


def test_write_report_writes_markdown_and_a_json_sidecar(tmp_path: Path) -> None:
    request = EvalRequest(out=tmp_path / "eval_report.md", sample=20)

    written = write_report(request, build_report(request))

    assert written.markdown_path.read_text(encoding="utf-8").startswith("## Evaluation report")
    assert written.metrics_path == tmp_path / "eval_report.json"
    assert json.loads(written.metrics_path.read_text(encoding="utf-8"))["sample_size"] == 20


def test_cli_exits_zero(tmp_path: Path) -> None:
    out = tmp_path / "eval_report.md"

    assert main(["--sample", "20", "--out", str(out)]) == 0
    assert out.exists()


def test_empty_baseline_passes_the_gate() -> None:
    report = build_report(EvalRequest(out=Path("r.md"), sample=20))

    verdict = evaluate_gate(Baseline.model_validate_json("{}"), report)

    assert verdict.passed
    assert verdict.compared == 0


def test_a_drop_beyond_tolerance_is_a_regression() -> None:
    report = build_report(EvalRequest(out=Path("r.md"), sample=20))
    baseline = Baseline.model_validate(
        {"metrics": {"extraction_accuracy": {"value": 0.8, "tolerance": 0.02}}}
    )

    verdict = evaluate_gate(baseline, report)

    assert not verdict.passed
    assert [r.metric for r in verdict.regressions] == ["extraction_accuracy"]


def test_a_drift_within_tolerance_passes() -> None:
    report = build_report(EvalRequest(out=Path("r.md"), sample=20))
    baseline = Baseline.model_validate(
        {"metrics": {"extraction_accuracy": {"value": 0.01, "tolerance": 0.05}}}
    )

    assert evaluate_gate(baseline, report).passed


def test_a_baseline_metric_absent_from_the_report_is_reported_not_failed() -> None:
    report = build_report(EvalRequest(out=Path("r.md"), sample=20))
    baseline = Baseline.model_validate({"metrics": {"nonexistent": {"value": 1.0}}})

    verdict = evaluate_gate(baseline, report)

    assert verdict.passed
    assert verdict.missing == ["nonexistent"]


def test_json_out_overrides_the_default_sidecar_path(tmp_path: Path) -> None:
    request = EvalRequest(
        out=tmp_path / "eval_report.md",
        json_out=tmp_path / "elsewhere" / "metrics.json",
        sample=20,
    )

    written = write_report(request, build_report(request))

    assert written.metrics_path == tmp_path / "elsewhere" / "metrics.json"
    assert written.metrics_path.exists()
    assert not (tmp_path / "eval_report.json").exists()


def test_json_out_defaults_to_the_markdown_stem(tmp_path: Path) -> None:
    request = EvalRequest(out=tmp_path / "eval_report.md", sample=20)

    written = write_report(request, build_report(request))

    assert written.metrics_path == tmp_path / "eval_report.json"


def test_project_reaches_the_report_and_the_markdown(tmp_path: Path) -> None:
    out = tmp_path / "eval_report.md"

    assert main(["--sample", "20", "--out", str(out), "--project", "example"]) == 0

    assert "Project: **example**" in out.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))["project"] == (
        "example"
    )


def test_the_cli_accepts_exactly_the_flags_the_workflow_passes(tmp_path: Path) -> None:
    # Guards .github/workflows/ci.yml: a flag removed here turns the eval job red.
    exit_code = main(
        [
            "--sample",
            "20",
            "--out",
            str(tmp_path / "eval_report.md"),
            "--json-out",
            str(tmp_path / "eval_report.json"),
            "--project",
            "example",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "eval_report.json").exists()

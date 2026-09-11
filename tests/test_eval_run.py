"""Unit tests for the evaluation runner, the project config and the entry-point registry.

The `example` project runs end to end in these tests with no key, no network and no model —
which is the point of it existing. Its numbers are pinned by hand below, so a change to the
reference classifier fails here rather than drifting through CI unnoticed.

These tests read the repository's own config and gold set through relative paths, so they
run from the repository root — as `make test-fast` and CI both do. That is inherent rather
than incidental: a project config names its gold set relative to the root too.

Deliberately does not touch the network, any environment variable, or the repository's own
`evals/baseline.json`: every test that writes points at pytest's tmp_path.
"""

import json
from pathlib import Path

import pytest
import yaml

from spine.eval.datasets import load_gold_set
from spine.eval.example_project import classify
from spine.eval.projects import (
    ProjectError,
    evaluator_for,
    load_project_config,
    load_project_config_file,
    register_evaluator,
    registered_projects,
)
from spine.eval.run import (
    EvalRequest,
    build_parser,
    build_report,
    main,
    render_report,
    write_report,
)

# The repository's own configs and gold sets, read (never written) by these tests.
PROJECT_ROOT = Path("evals/projects")

# Hand-checked against data/gold/example.jsonl and the reference classifier:
# expected "yes" for ex-001, 003, 004, 005, 007, 008, 010; "no" for ex-002, 006, 009.
# The classifier gets ex-010 ("Service credits ... apply") wrong — it carries no modal verb.
# So 9 of 10 correct, and against positive_label "yes": TP=6, FP=0, FN=1, TN=3.
EXPECTED_ACCURACY = 0.9
EXPECTED_F1 = 2 * (1.0 * 6 / 7) / (1.0 + 6 / 7)  # P=6/6, R=6/7 -> 0.9230769...


def a_request(tmp_path: Path, **overrides: object) -> EvalRequest:
    settings: dict[str, object] = {
        "out": tmp_path / "eval_report.md",
        "project": "example",
        "project_root": PROJECT_ROOT,
        "results_root": tmp_path / "results",
        "run_id": "run-under-test",
    }
    settings.update(overrides)
    return EvalRequest.model_validate(settings)


def values_of(request: EvalRequest) -> dict[str, float]:
    return {row.metric: row.value for row in build_report(request).rows}


# --- the reference classifier -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The supplier shall provide monthly reporting.", "yes"),
        ("The contractor must hold ISO 9001.", "yes"),
        ("Sub-processors outside the UK require prior approval.", "yes"),
        ("Reports may be submitted in PDF.", "no"),
        ("The supplier is encouraged to propose improvements.", "no"),
        ("Indicative pricing is provided for guidance only.", "no"),
        ("", "no"),
    ],
)
def test_the_reference_classifier_reads_the_modal_verb(text: str, expected: str) -> None:
    assert classify(text) == expected


def test_permission_wording_beats_obligation_wording() -> None:
    # Stated in the module rather than left to regex order, so it is pinned here too.
    assert classify("The report may be required in PDF.") == "no"


# --- the project config -----------------------------------------------------------------


def test_the_example_project_config_loads() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)

    assert config.project == "example"
    assert config.gold_set == "example"
    assert [spec.name for spec in config.kpis.kpis][:2] == [
        "items_evaluated",
        "extraction_accuracy",
    ]


def test_the_example_roi_says_its_numbers_are_not_measured() -> None:
    # The marker is in the data, so it survives the file being copied into a real project.
    config = load_project_config("example", root=PROJECT_ROOT)

    assert config.kpis.roi is not None
    assert "illustrative" in config.kpis.roi.assumptions_source.lower()


def test_tolerances_are_read_off_the_declared_kpis() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)

    assert config.tolerances["extraction_accuracy"] == 0.02
    assert config.tolerances["items_evaluated"] == 0.0


def test_a_missing_project_config_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="project config not found"):
        load_project_config("nope", root=tmp_path)


def test_a_config_that_is_not_yaml_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("project: [unclosed\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="not valid YAML"):
        load_project_config_file(path)


def test_a_config_that_is_not_a_mapping_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="expected a YAML mapping"):
        load_project_config_file(path)


def test_a_config_missing_a_required_field_names_the_field(tmp_path: Path) -> None:
    path = tmp_path / "thin.yaml"
    path.write_text("project: thin\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="gold_set"):
        load_project_config_file(path)


def test_a_filename_that_disagrees_with_the_project_field_is_refused(tmp_path: Path) -> None:
    # Otherwise a copied file would quietly file its snapshots under the name it came from.
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    (tmp_path / "copied.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ProjectError, match="filename and the project field must agree"):
        load_project_config("copied", root=tmp_path)


def test_a_kpi_block_naming_another_project_is_refused(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    source["kpis"]["project"] = "somewhere-else"
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ProjectError, match="filed under the wrong name"):
        load_project_config("example", root=tmp_path)


def test_an_unknown_key_in_a_config_is_refused(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    source["gold_sett"] = "typo"
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ProjectError):
        load_project_config("example", root=tmp_path)


# --- the registry -----------------------------------------------------------------------


def test_the_example_project_registers_itself_on_import() -> None:
    assert "example" in registered_projects()
    assert evaluator_for("example") is not None


def test_an_unregistered_project_names_the_ones_that_are() -> None:
    with pytest.raises(ProjectError, match="Registered: .*example"):
        evaluator_for("never-registered")


def test_registering_the_same_entry_point_twice_is_allowed() -> None:
    evaluator = evaluator_for("example")

    register_evaluator("example", evaluator)  # idempotent: the same function, the same name.

    assert evaluator_for("example") is evaluator


def test_two_entry_points_cannot_claim_one_project() -> None:
    # Otherwise the report would depend on import order.
    def other(request: object) -> object:
        raise AssertionError("never called")

    with pytest.raises(ProjectError, match="already has a different"):
        register_evaluator("example", other)  # type: ignore[arg-type]


# --- running the example project end to end ---------------------------------------------


def test_the_example_project_produces_its_hand_checked_numbers(tmp_path: Path) -> None:
    values = values_of(a_request(tmp_path))

    assert values["items_evaluated"] == 10.0
    assert values["extraction_accuracy"] == pytest.approx(EXPECTED_ACCURACY)
    assert values["requirement_f1"] == pytest.approx(EXPECTED_F1)


def test_a_project_with_no_model_calls_costs_nothing_rather_than_refusing(tmp_path: Path) -> None:
    # Honestly zero, not unknown: there were no calls at all, replayed or otherwise.
    assert values_of(a_request(tmp_path))["cost_per_item_usd"] == 0.0


def test_the_roi_metrics_follow_from_the_configs_assumptions(tmp_path: Path) -> None:
    values = values_of(a_request(tmp_path))

    # Baseline 6.0 + 2.5 = 8.5 hours, residual 1.5 + 0.75 = 2.25, so 6.25 displaced.
    assert values["hours_displaced_per_unit"] == pytest.approx(6.25)
    # Baseline cost 6.0*70 + 2.5*95 = 657.50; residual 1.5*70 + 0.75*95 = 176.25;
    # saving 481.25/unit x 12 units = 5775/month; 24000 / 5775 = 4.1558... months.
    assert values["payback_periods"] == pytest.approx(24000 / 5775)


def test_every_declared_metric_is_measured(tmp_path: Path) -> None:
    config = load_project_config("example", root=PROJECT_ROOT)

    report = build_report(a_request(tmp_path))

    assert [row.metric for row in report.rows] == [spec.name for spec in config.kpis.kpis]


def test_the_report_records_which_gold_set_and_run_it_came_from(tmp_path: Path) -> None:
    report = build_report(a_request(tmp_path))

    assert report.gold_set == "example"
    assert report.run_id == "run-under-test"
    assert report.items_evaluated == 10


def test_a_sample_larger_than_the_gold_set_evaluates_all_of_it(tmp_path: Path) -> None:
    # This is the CI path: `--sample 20` against a ten-item set.
    report = build_report(a_request(tmp_path, sample=20))

    assert report.scope == "sample"
    assert report.sample_size == 20
    assert report.items_evaluated == 10


def test_a_sample_evaluates_only_the_sampled_items(tmp_path: Path) -> None:
    report = build_report(a_request(tmp_path, sample=4, seed=7))

    assert report.items_evaluated == 4


def test_sampling_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    first = build_report(a_request(tmp_path, sample=4, seed=7))
    second = build_report(a_request(tmp_path, sample=4, seed=7))

    assert [row.value for row in first.rows] == [row.value for row in second.rows]


def test_rows_are_derived_from_the_snapshots_so_they_cannot_disagree(tmp_path: Path) -> None:
    report = build_report(a_request(tmp_path))

    assert [row.metric for row in report.rows] == [s.metric_name for s in report.snapshots]
    assert [row.value for row in report.rows] == [s.value for s in report.snapshots]


def test_an_unknown_project_is_an_error_not_an_empty_report(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        build_report(a_request(tmp_path, project="no-such-project"))


# --- what gets written ------------------------------------------------------------------


def test_three_files_are_written(tmp_path: Path) -> None:
    request = a_request(tmp_path, json_out=tmp_path / "metrics.json")
    report = build_report(request)

    written = write_report(request, report, render_report(report))

    assert written.markdown_path == tmp_path / "eval_report.md"
    assert written.metrics_path == tmp_path / "metrics.json"
    assert written.archive_path.parent == tmp_path / "results"
    assert all(
        path.exists()
        for path in (written.markdown_path, written.metrics_path, written.archive_path)
    )


def test_json_out_defaults_to_the_markdown_stem(tmp_path: Path) -> None:
    request = a_request(tmp_path)
    report = build_report(request)

    written = write_report(request, report, render_report(report))

    assert written.metrics_path == tmp_path / "eval_report.json"


def test_json_out_overrides_the_default_sidecar_path(tmp_path: Path) -> None:
    request = a_request(tmp_path, json_out=tmp_path / "elsewhere" / "metrics.json")
    report = build_report(request)

    write_report(request, report, render_report(report))

    assert (tmp_path / "elsewhere" / "metrics.json").exists()
    assert not (tmp_path / "eval_report.json").exists()


def test_the_archive_is_named_for_the_project_and_the_moment(tmp_path: Path) -> None:
    request = a_request(tmp_path)
    report = build_report(request)

    written = write_report(request, report, render_report(report))

    assert written.archive_path.name.startswith("example_")
    assert written.archive_path.suffix == ".json"


def test_the_archive_and_the_sidecar_hold_the_same_report(tmp_path: Path) -> None:
    # The archive exists so a run's numbers survive the next run overwriting the report.
    request = a_request(tmp_path)
    report = build_report(request)

    written = write_report(request, report, render_report(report))

    assert written.archive_path.read_text(encoding="utf-8") == written.metrics_path.read_text(
        encoding="utf-8"
    )


def test_the_written_json_round_trips_back_into_a_report(tmp_path: Path) -> None:
    # The gate reads this file, so it has to parse. Pinned because an infinite or NaN value
    # would serialise to JSON null and fail here rather than in CI.
    request = a_request(tmp_path)
    report = build_report(request)
    written = write_report(request, report, render_report(report))

    reloaded = json.loads(written.metrics_path.read_text(encoding="utf-8"))

    assert [row["metric"] for row in reloaded["rows"]] == [row.metric for row in report.rows]
    assert all(isinstance(row["value"], int | float) for row in reloaded["rows"])


def test_the_markdown_shows_the_project_scope_and_a_row_per_metric(tmp_path: Path) -> None:
    report = build_report(a_request(tmp_path))

    markdown = render_report(report)

    assert "Project: **example**" in markdown
    assert "10 item(s) from `example`" in markdown
    assert "| extraction_accuracy | ≥ 0.85 | 0.9 | ratio | test | PASS |" in markdown


def test_the_markdown_says_when_every_metric_is_on_target(tmp_path: Path) -> None:
    report = build_report(a_request(tmp_path))

    assert "Every metric is on target." in render_report(report)


def test_a_project_that_counts_replayed_costs_says_so_in_the_report(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    source["kpis"]["include_replayed_cost"] = True
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    report = build_report(a_request(tmp_path, project_root=tmp_path))

    assert "not money spent by this run" in report.cost_note
    assert "Note on cost" in render_report(report)


def test_a_project_measuring_its_own_spend_adds_no_note(tmp_path: Path) -> None:
    assert build_report(a_request(tmp_path)).cost_note == ""


# --- the command line -------------------------------------------------------------------


def cli(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--project",
        "example",
        "--project-root",
        str(PROJECT_ROOT),
        "--out",
        str(tmp_path / "eval_report.md"),
        "--json-out",
        str(tmp_path / "eval_report.json"),
        "--results-root",
        str(tmp_path / "results"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        *extra,
    ]


def test_the_cli_accepts_exactly_the_flags_the_workflow_passes(tmp_path: Path) -> None:
    # Guards .github/workflows/ci.yml: a flag removed here turns the eval job red.
    parsed = build_parser().parse_args(
        [
            "--sample",
            "20",
            "--out",
            "eval_report.md",
            "--json-out",
            "eval_report.json",
            "--project",
            "example",
        ]
    )

    assert (parsed.sample, parsed.project) == (20, "example")
    assert (parsed.out, parsed.json_out) == (Path("eval_report.md"), Path("eval_report.json"))


def test_the_cli_runs_the_example_project_and_exits_zero(tmp_path: Path) -> None:
    assert main(cli(tmp_path)) == 0
    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_report.json").exists()


def test_an_absent_baseline_means_nothing_has_been_accepted_yet(tmp_path: Path) -> None:
    assert main(cli(tmp_path)) == 0


def test_update_baseline_writes_the_current_results(tmp_path: Path) -> None:
    assert main(cli(tmp_path, "--update-baseline")) == 0

    written = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert written["metrics"]["extraction_accuracy"]["value"] == pytest.approx(EXPECTED_ACCURACY)
    assert written["metrics"]["extraction_accuracy"]["tolerance"] == 0.02
    assert written["metrics"]["cost_per_item_usd"]["direction"] == "lower_is_better"


def test_a_run_gated_against_its_own_baseline_passes(tmp_path: Path) -> None:
    main(cli(tmp_path, "--update-baseline"))

    assert main(cli(tmp_path)) == 0


def test_a_regression_beyond_tolerance_exits_non_zero(tmp_path: Path) -> None:
    (tmp_path / "baseline.json").write_text(
        json.dumps({"metrics": {"extraction_accuracy": {"value": 0.99, "tolerance": 0.01}}}),
        encoding="utf-8",
    )

    assert main(cli(tmp_path)) == 1


def test_the_report_is_still_written_when_the_gate_fails(tmp_path: Path) -> None:
    # CI posts the report before it gates, so a regression must still leave one to post.
    (tmp_path / "baseline.json").write_text(
        json.dumps({"metrics": {"extraction_accuracy": {"value": 0.99, "tolerance": 0.01}}}),
        encoding="utf-8",
    )

    main(cli(tmp_path))

    assert "REGRESSION" in (tmp_path / "eval_report.md").read_text(encoding="utf-8")


def test_a_refusal_to_report_a_cost_exits_non_zero_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A replayed run asked for a cost per item: spine.kpi refuses, and the CLI reports the
    # refusal rather than writing a report with a plausible, wrong number in it.
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    register_evaluator("replayed-demo", _replayed_evaluator)
    demo = dict(source, project="replayed-demo")
    demo["kpis"] = dict(source["kpis"], project="replayed-demo")
    (tmp_path / "replayed-demo.yaml").write_text(yaml.safe_dump(demo), encoding="utf-8")

    exit_code = main(
        [
            "--project",
            "replayed-demo",
            "--project-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "eval_report.md"),
            "--results-root",
            str(tmp_path / "results"),
            "--baseline",
            str(tmp_path / "baseline.json"),
        ]
    )

    assert exit_code == 1
    assert "Cannot report KPIs" in capsys.readouterr().out
    assert not (tmp_path / "eval_report.md").exists()


def _replayed_evaluator(request: object) -> object:
    """An evaluator whose run was served entirely from a cassette."""
    from datetime import UTC, datetime
    from decimal import Decimal

    from spine.contracts import AgentRun, ModelCall, StepTrace
    from spine.eval.example_project import evaluate
    from spine.eval.projects import EvaluationInput, EvaluationOutput

    assert isinstance(request, EvaluationInput)
    outcome = evaluate(request)
    at = datetime.now(UTC)
    call = ModelCall(
        provider="anthropic",
        model="claude-sonnet-5",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=Decimal("0.02"),
        latency_ms=1,
        timestamp=at,
        purpose="extract",
        mode="replay",
    )
    step = StepTrace(
        step_name="replayed",
        started_at=at,
        finished_at=at,
        status="ok",
        model_calls=[call],
    )
    return EvaluationOutput(
        verdicts=outcome.verdicts,
        run=AgentRun(
            run_id=request.run_id,
            project=request.project,
            started_at=at,
            finished_at=at,
            steps=[step],
            status="completed",
        ),
    )


def test_the_gold_set_the_example_project_uses_is_the_committed_one() -> None:
    # If this file moved or shrank, every number pinned above would change silently.
    config = load_project_config("example", root=PROJECT_ROOT)

    gold = load_gold_set(config.gold_set, root=config.gold_root)

    assert config.gold_root == Path("data/gold")
    assert len(gold) == 10
    assert sum(1 for item in gold.items if item.expected["is_requirement"] == "yes") == 7


def test_a_config_path_that_is_a_directory_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "adirectory.yaml").mkdir()

    with pytest.raises(ProjectError, match="is a directory"):
        load_project_config("adirectory", root=tmp_path)


def test_the_markdown_names_the_metrics_that_missed_their_target(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    for spec in source["kpis"]["kpis"]:
        if spec["name"] == "extraction_accuracy":
            spec["target"] = 0.99
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    markdown = render_report(build_report(a_request(tmp_path, project_root=tmp_path)))

    assert "1 metric(s) below target: extraction_accuracy." in markdown
    assert "| extraction_accuracy | ≥ 0.99 | 0.9 | ratio | test | FAIL |" in markdown


def test_a_baseline_metric_the_report_no_longer_has_is_warned_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "baseline.json").write_text(
        json.dumps({"metrics": {"retired_metric": {"value": 1.0}}}), encoding="utf-8"
    )

    exit_code = main(cli(tmp_path))

    assert exit_code == 0
    assert "retired_metric" in capsys.readouterr().out

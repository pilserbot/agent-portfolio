"""Unit tests for the KPI dashboard's config loading and data assembly, run headlessly.

Nothing here starts Streamlit or renders a page. The dashboard is split so that everything
which decides *what* is shown lives in `dashboards.kpi_data`, which imports no UI library at
all — these tests are what that split is for.

The repository's own `evals/projects/` and `evals/results/` are read (never written), so
the committed config and result history are exercised as the deployed app would read them.
Everything else points at pytest's tmp_path.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from dashboards.kpi_data import (
    DashboardError,
    ResultHistory,
    available_projects,
    build_project_view,
    format_usd,
    format_value,
    headline_specs,
    load_history,
    metric_rows,
    result_paths,
    roi_of,
    trend_series,
)
from spine.contracts import AgentRun, KPISnapshot, ModelCall, StepTrace
from spine.eval.projects import load_project_config
from spine.eval.run import EvalReport
from spine.kpi import CostReport, cost_report

PROJECT_ROOT = Path("evals/projects")
RESULTS_ROOT = Path("evals/results")

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def a_snapshot(name: str, value: float, **overrides: object) -> KPISnapshot:
    settings: dict[str, object] = {
        "project": "demo",
        "metric_name": name,
        "value": value,
        "unit": "ratio",
        "target": 0.8,
        "direction": "higher_is_better",
        "measured_at": AT,
        "sample_size": 10,
        "method": "test",
    }
    settings.update(overrides)
    return KPISnapshot.model_validate(settings)


def a_report(
    *snapshots: KPISnapshot,
    project: str = "demo",
    at: datetime = AT,
    cost: CostReport | None = None,
    items: int = 10,
) -> EvalReport:
    return EvalReport(
        project=project,
        scope="full",
        sample_size=None,
        items_evaluated=items,
        gold_set="example",
        run_id=f"{project}-{at:%Y%m%dT%H%M%SZ}",
        measured_at=at,
        snapshots=list(snapshots),
        cost=cost or CostReport(total_calls=0, billed_calls=0, replayed_calls=0),
    )


def write_report(root: Path, report: EvalReport) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{report.project}_{report.measured_at:%Y%m%dT%H%M%SZ}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def a_call(cost: str, *, mode: str = "live", purpose: str = "extract") -> ModelCall:
    return ModelCall(
        provider="anthropic",
        model="claude-sonnet-5",
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=5,
        cost_usd=Decimal(cost),
        latency_ms=10,
        timestamp=AT,
        purpose=purpose,
        mode=mode,
    )


def a_run(*calls: ModelCall) -> AgentRun:
    return AgentRun(
        run_id="run-1",
        project="demo",
        started_at=AT,
        finished_at=AT,
        steps=[
            StepTrace(
                step_name="step",
                started_at=AT,
                finished_at=AT,
                status="ok",
                model_calls=list(calls),
            )
        ],
        status="completed",
    )


def a_project_config(root: Path, **overrides: object) -> Path:
    """Copy the committed example config into tmp_path, with fields overridden."""
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    source.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{source['project']}.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    return path


# --- discovering projects ----------------------------------------------------------------


def test_projects_come_from_the_config_directory_not_a_list_in_code() -> None:
    assert "example" in available_projects(PROJECT_ROOT)


def test_a_missing_config_directory_is_no_projects_rather_than_a_crash(tmp_path: Path) -> None:
    assert available_projects(tmp_path / "nowhere") == []


def test_a_new_yaml_file_is_all_it_takes_to_add_a_project(tmp_path: Path) -> None:
    a_project_config(tmp_path, project="brand-new")

    assert available_projects(tmp_path) == ["brand-new"]


# --- finding and loading result files ----------------------------------------------------


def test_result_files_are_matched_on_the_runners_naming(tmp_path: Path) -> None:
    write_report(tmp_path, a_report(project="demo", at=AT))
    (tmp_path / "demo_notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "demo.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in result_paths("demo", tmp_path)] == ["demo_20260102T030405Z.json"]


def test_one_project_does_not_pick_up_another_whose_name_it_prefixes(tmp_path: Path) -> None:
    # `example` must not absorb `example_human`'s files: the glob would, the pattern does not.
    write_report(tmp_path, a_report(project="example", at=AT))
    write_report(tmp_path, a_report(project="example_human", at=AT))

    assert [path.stem for path in result_paths("example", tmp_path)] == ["example_20260102T030405Z"]


def test_a_missing_results_directory_is_an_empty_history(tmp_path: Path) -> None:
    history = load_history("demo", tmp_path / "nowhere")

    assert history.is_empty
    assert history.latest is None


def test_history_is_ordered_by_when_each_run_was_measured(tmp_path: Path) -> None:
    write_report(tmp_path, a_report(project="demo", at=AT + timedelta(hours=2)))
    write_report(tmp_path, a_report(project="demo", at=AT))
    write_report(tmp_path, a_report(project="demo", at=AT + timedelta(hours=1)))

    history = load_history("demo", tmp_path)

    assert [report.measured_at for report in history.reports] == [
        AT,
        AT + timedelta(hours=1),
        AT + timedelta(hours=2),
    ]
    assert history.latest is not None
    assert history.latest.measured_at == AT + timedelta(hours=2)


def test_an_unreadable_result_is_reported_rather_than_skipped_in_silence(tmp_path: Path) -> None:
    # A dashboard quietly showing fewer runs than exist is worse than one naming a bad file.
    write_report(tmp_path, a_report(a_snapshot("accuracy", 0.9), project="demo"))
    broken = tmp_path / "demo_20260303T030303Z.json"
    broken.write_text("{not json", encoding="utf-8")

    history = load_history("demo", tmp_path)

    assert len(history.reports) == 1
    assert [path.name for path in history.unreadable] == [broken.name]


def test_a_result_file_that_parses_but_is_not_a_report_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "demo_20260303T030303Z.json").write_text(
        json.dumps({"something": "else"}), encoding="utf-8"
    )

    assert len(load_history("demo", tmp_path).unreadable) == 1


def test_the_committed_history_loads(tmp_path: Path) -> None:
    # The deployed app reads exactly these files, so a broken one fails here.
    history = load_history("example", RESULTS_ROOT)

    assert history.unreadable == []
    assert not history.is_empty


# --- headline metrics --------------------------------------------------------------------


def test_the_declared_headlines_are_used_in_declaration_order() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)

    specs, declared = headline_specs(config)

    assert declared
    assert [spec.name for spec in specs] == ["extraction_accuracy", "cost_per_item_usd"]


def test_a_project_declaring_no_headlines_falls_back_and_says_so(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    del source["kpis"]["headline"]
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")
    config = load_project_config("example", root=tmp_path)

    specs, declared = headline_specs(config)

    assert not declared
    assert [spec.name for spec in specs] == ["items_evaluated", "extraction_accuracy"]


def test_a_headline_naming_an_undeclared_metric_is_refused(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    source["kpis"]["headline"] = ["no_such_metric"]
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(Exception, match="does not report"):
        load_project_config("example", root=tmp_path)


# --- the KPI table -----------------------------------------------------------------------


def test_rows_follow_the_configs_declaration_order_not_the_reports(tmp_path: Path) -> None:
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(
        a_snapshot("requirement_f1", 0.9),
        a_snapshot("items_evaluated", 10, unit="count", target=1),
        a_snapshot("extraction_accuracy", 0.9),
    )

    rows = metric_rows(config, report)

    assert [row.metric for row in rows] == [
        "items_evaluated",
        "extraction_accuracy",
        "requirement_f1",
    ]


def test_a_row_carries_everything_the_table_column_headings_promise() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(a_snapshot("extraction_accuracy", 0.9, sample_size=10))

    row = metric_rows(config, report)[0]

    assert (row.target, row.value, row.unit, row.method) == (0.8, 0.9, "ratio", "test")
    assert (row.sample_size, row.passed, row.status, row.comparator) == (10, True, "PASS", "≥")


def test_a_lower_is_better_row_reads_the_other_way() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(
        a_snapshot("cost_per_item_usd", 0.09, target=0.05, direction="lower_is_better", unit="usd")
    )

    row = metric_rows(config, report)[0]

    assert row.comparator == "≤"
    assert row.status == "FAIL"


def test_a_measurement_the_config_no_longer_declares_is_dropped_from_the_table() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(a_snapshot("extraction_accuracy", 0.9), a_snapshot("retired_metric", 1.0))

    assert [row.metric for row in metric_rows(config, report)] == ["extraction_accuracy"]


# --- trends ------------------------------------------------------------------------------


def test_a_series_is_built_per_metric_across_every_result(tmp_path: Path) -> None:
    for offset, value in enumerate((0.7, 0.8, 0.9)):
        write_report(
            tmp_path,
            a_report(a_snapshot("accuracy", value), at=AT + timedelta(hours=offset)),
        )

    series = trend_series(load_history("demo", tmp_path))

    assert [s.metric for s in series] == ["accuracy"]
    assert [point.value for point in series[0].points] == [0.7, 0.8, 0.9]
    assert series[0].has_history


def test_a_metric_absent_from_an_older_run_has_no_point_there_rather_than_a_zero(
    tmp_path: Path,
) -> None:
    # A gap in the record is not a measurement of nothing.
    write_report(tmp_path, a_report(a_snapshot("accuracy", 0.9), at=AT))
    write_report(
        tmp_path,
        a_report(a_snapshot("accuracy", 0.95), a_snapshot("f1", 0.8), at=AT + timedelta(hours=1)),
    )

    series = {s.metric: s for s in trend_series(load_history("demo", tmp_path))}

    assert [point.value for point in series["accuracy"].points] == [0.9, 0.95]
    assert [point.value for point in series["f1"].points] == [0.8]
    assert not series["f1"].has_history


def test_a_single_run_is_not_a_trend() -> None:
    history = ResultHistory(project="demo", reports=[a_report(a_snapshot("accuracy", 0.9))])

    assert not trend_series(history)[0].has_history


def test_a_series_carries_the_most_recent_target_it_was_held_to(tmp_path: Path) -> None:
    write_report(tmp_path, a_report(a_snapshot("accuracy", 0.9, target=0.8), at=AT))
    write_report(
        tmp_path,
        a_report(a_snapshot("accuracy", 0.9, target=0.95), at=AT + timedelta(hours=1)),
    )

    assert trend_series(load_history("demo", tmp_path))[0].target == 0.95


def test_no_results_is_no_series() -> None:
    assert trend_series(ResultHistory(project="demo")) == []


# --- the cost panel's data, and the replay rule -------------------------------------------


def test_a_live_run_states_its_cost_and_breaks_it_down_by_purpose() -> None:
    report = cost_report(
        a_run(a_call("0.02", purpose="extract"), a_call("0.03", purpose="summarise"))
    )

    assert report.can_state_a_cost
    assert report.is_measured
    assert report.basis is not None
    assert report.basis.total_usd == Decimal("0.05")
    assert [(u.purpose, u.cost_usd) for u in report.by_purpose] == [
        ("extract", Decimal("0.02")),
        ("summarise", Decimal("0.03")),
    ]


def test_a_replayed_run_states_no_cost_and_says_why() -> None:
    # The rule from spine.kpi, in the shape a screen needs: the refusal is reported, not
    # raised, because a page has to render something.
    report = cost_report(a_run(a_call("0.02", mode="replay")))

    assert not report.can_state_a_cost
    assert report.basis is None
    assert "cassette" in report.refusal
    assert report.replayed_calls == 1


def test_a_replayed_run_still_reports_its_tokens() -> None:
    # Tokens count work that was really done, and stay true whether a call was served live
    # or from a cassette. Money does not.
    report = cost_report(a_run(a_call("0.02", mode="replay")))

    assert report.total_prompt_tokens == 100
    assert report.total_completion_tokens == 20
    assert [usage.cost_usd for usage in report.by_purpose] == [None]


def test_one_replayed_call_among_live_ones_withholds_the_whole_cost() -> None:
    report = cost_report(a_run(a_call("0.02"), a_call("0.03", mode="replay")))

    assert not report.can_state_a_cost
    assert (report.billed_calls, report.replayed_calls, report.total_calls) == (1, 1, 2)


def test_an_opted_in_replayed_run_states_a_cost_but_is_not_measured() -> None:
    report = cost_report(a_run(a_call("0.02", mode="replay")), include_replayed=True)

    assert report.can_state_a_cost
    assert not report.is_measured  # what the page hangs its warning banner on
    assert report.basis is not None
    assert report.basis.total_usd == Decimal("0.02")


def test_a_run_with_no_model_calls_costs_nothing_rather_than_refusing() -> None:
    report = cost_report(a_run())

    assert report.can_state_a_cost
    assert report.is_measured
    assert report.by_purpose == []


def test_purposes_are_ordered_so_the_table_does_not_reshuffle_between_reloads() -> None:
    report = cost_report(a_run(a_call("0.01", purpose="zeta"), a_call("0.01", purpose="alpha")))

    assert [usage.purpose for usage in report.by_purpose] == ["alpha", "zeta"]


# --- the ROI panel's data ------------------------------------------------------------------


def test_roi_comes_back_with_the_projects_stated_assumptions() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)

    roi = roi_of(config, None)

    assert roi is not None
    # Baseline 6.0 + 2.5 = 8.5 hours, residual 1.5 + 0.75 = 2.25, so 6.25 displaced.
    assert roi.hours_displaced_per_unit == pytest.approx(6.25)
    assert "illustrative" in roi.assumptions_source.lower()


def test_a_measured_cost_replaces_the_assumed_one() -> None:
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(
        cost=cost_report(a_run(a_call("1.00"))),
        items=10,
    )

    roi = roi_of(config, report)

    assert roi is not None
    # 1.00 usd over 10 items = 0.10/item, on top of 176.25 usd of residual human time.
    assert roi.automated_cost_per_unit_usd == pytest.approx(Decimal("176.35"))


def test_a_replayed_run_leaves_the_assumed_cost_standing() -> None:
    # No measurement exists, so the assumption is what is shown — and the page says which.
    config = load_project_config("example", root=PROJECT_ROOT)
    report = a_report(cost=cost_report(a_run(a_call("1.00", mode="replay"))), items=10)

    roi = roi_of(config, report)

    assert roi is not None
    assert (
        roi.automated_model_cost_per_unit_usd == config.kpis.roi.automated_model_cost_per_unit_usd
    )


def test_a_project_with_no_roi_block_gets_none(tmp_path: Path) -> None:
    source = yaml.safe_load((PROJECT_ROOT / "example.yaml").read_text(encoding="utf-8"))
    del source["kpis"]["roi"]
    source["kpis"]["kpis"] = [
        spec
        for spec in source["kpis"]["kpis"]
        if spec["computation"] not in {"hours_displaced_per_unit", "payback_periods"}
    ]
    source["kpis"]["headline"] = ["extraction_accuracy"]
    (tmp_path / "example.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    assert roi_of(load_project_config("example", root=tmp_path), None) is None


# --- the whole view ------------------------------------------------------------------------


def test_the_committed_example_project_assembles_end_to_end() -> None:
    view = build_project_view("example", project_root=PROJECT_ROOT, results_root=RESULTS_ROOT)

    assert view.has_results
    assert [h.row.metric for h in view.headlines] == ["extraction_accuracy", "cost_per_item_usd"]
    assert [row.metric for row in view.rows] == [spec.name for spec in view.config.kpis.kpis]
    assert view.roi is not None
    assert view.stale_metrics == []
    assert view.retired_metrics == []


def test_a_project_with_no_results_yet_says_so_rather_than_failing(tmp_path: Path) -> None:
    a_project_config(tmp_path / "projects")

    view = build_project_view(
        "example", project_root=tmp_path / "projects", results_root=tmp_path / "results"
    )

    assert not view.has_results
    assert view.headlines == []
    assert view.rows == []
    assert view.trends == []
    # Nothing was measured, so nothing is stale and nothing is retired — a project that
    # has never run is not a project whose metrics have all gone missing.
    assert view.stale_metrics == []
    assert view.retired_metrics == []
    # The config is still loaded, so the page can list what the project *would* report.
    assert len(view.config.kpis.kpis) > 0


def test_an_unknown_project_is_a_dashboard_error_naming_the_problem(tmp_path: Path) -> None:
    with pytest.raises(DashboardError, match="project config not found"):
        build_project_view("no-such-project", project_root=tmp_path, results_root=tmp_path)


def test_a_metric_added_since_the_last_run_is_named_not_silently_missing(tmp_path: Path) -> None:
    a_project_config(tmp_path / "projects")
    write_report(
        tmp_path / "results",
        a_report(a_snapshot("extraction_accuracy", 0.9), project="example"),
    )

    view = build_project_view(
        "example", project_root=tmp_path / "projects", results_root=tmp_path / "results"
    )

    assert "requirement_f1" in view.stale_metrics
    assert [row.metric for row in view.rows] == ["extraction_accuracy"]


def test_a_headline_the_latest_run_did_not_measure_is_simply_not_shown(tmp_path: Path) -> None:
    a_project_config(tmp_path / "projects")
    write_report(
        tmp_path / "results",
        a_report(a_snapshot("extraction_accuracy", 0.9), project="example"),
    )

    view = build_project_view(
        "example", project_root=tmp_path / "projects", results_root=tmp_path / "results"
    )

    assert [h.row.metric for h in view.headlines] == ["extraction_accuracy"]


def test_a_retired_metric_is_named_too(tmp_path: Path) -> None:
    a_project_config(tmp_path / "projects")
    snapshots = [
        a_snapshot(spec.name, 1.0, unit=spec.unit, target=spec.target, direction=spec.direction)
        for spec in load_project_config("example", root=PROJECT_ROOT).kpis.kpis
    ]
    write_report(
        tmp_path / "results",
        a_report(*snapshots, a_snapshot("gone_away", 1.0), project="example"),
    )

    view = build_project_view(
        "example", project_root=tmp_path / "projects", results_root=tmp_path / "results"
    )

    assert view.retired_metrics == ["gone_away"]
    assert view.stale_metrics == []


# --- display formatting --------------------------------------------------------------------


def test_a_fraction_of_a_cent_shows_its_magnitude_rather_than_rounding_to_free() -> None:
    # "$0.00" for a real per-item cost would read as free, which it is not.
    assert format_usd(Decimal("0.000123")) == "$0.000123"


def test_ordinary_amounts_show_to_the_cent() -> None:
    assert format_usd(Decimal("1234.5")) == "$1,234.50"
    assert format_usd(Decimal("0")) == "$0.00"


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (0.9231, "ratio", "92.3%"),
        (10.0, "count", "10"),
        (0.05, "usd", "$0.05"),
        (4.1558, "months", "4.156"),
    ],
)
def test_a_value_is_rendered_in_the_terms_its_unit_implies(
    value: float, unit: str, expected: str
) -> None:
    assert format_value(value, unit) == expected


# --- the Streamlit entry point itself --------------------------------------------------
#
# Imported rather than rendered. The point is that a typo, a bad import or a renamed
# helper in kpi_app.py fails here rather than on the deployed page — importing it is safe
# because `main()` runs only under `if __name__ == "__main__"`, which is how Streamlit
# executes a script and is not how this imports it.


def load_kpi_app() -> object:
    import importlib.util

    path = Path("apps/dashboards/kpi_app.py").resolve()
    spec = importlib.util.spec_from_file_location("kpi_app_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_streamlit_entry_point_imports_without_drawing_anything() -> None:
    module = load_kpi_app()

    assert callable(module.main)
    for name in ("render_headlines", "render_metric_table", "render_trends", "render_cost"):
        assert callable(getattr(module, name)), name


def test_the_entry_point_resolves_the_repository_root_from_its_own_location() -> None:
    # This is what lets it deploy: nothing depends on the working directory it is run from.
    module = load_kpi_app()

    assert (module.REPO_ROOT / "evals" / "projects" / "example.yaml").is_file()
    assert module.PROJECT_ROOT.is_dir()
    assert module.RESULTS_ROOT.is_dir()


def test_every_panel_the_page_promises_is_actually_called() -> None:
    # A render function defined and never called is a panel that silently does not appear.
    source = Path("apps/dashboards/kpi_app.py").read_text(encoding="utf-8")
    body = source.split("def render_project(", 1)[1]

    for panel in ("render_headlines(", "render_metric_table(", "render_trends(", "render_cost("):
        assert panel in body, panel


def test_the_pinned_requirements_match_what_is_installed() -> None:
    # Streamlit Community Cloud installs exactly this file, so a pin that drifts from the
    # version tested here is a deploy that behaves differently from CI.
    import pandas
    import pydantic
    import streamlit
    import yaml as pyyaml

    pins = dict(
        line.split("==")
        for line in Path("apps/dashboards/requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if "==" in line
    )

    assert pins == {
        "streamlit": streamlit.__version__,
        "pandas": pandas.__version__,
        "pydantic": pydantic.VERSION,
        "PyYAML": pyyaml.__version__,
    }

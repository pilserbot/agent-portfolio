"""The KPI dashboard: one page per project, driven entirely by that project's config.

Run it with `streamlit run apps/dashboards/kpi_app.py` from the repository root, or deploy
this file to Streamlit Community Cloud from a clone. It reads only committed files —
`evals/projects/<project>.yaml` and `evals/results/<project>_<timestamp>.json` — so there
is no database to reach, no key to supply and nothing to configure at deploy time.

There is no project-specific code here, and adding a project to this dashboard is adding a
YAML file. Which metrics exist, what they are called, what good looks like for each, which
two lead the page and what the ROI assumes are all read from that file.

Two things the page insists on, both deliberate:

- **A run whose calls were replayed does not get a cost figure.** It gets a banner saying
  why. The rule lives in `spine.kpi`; this page shows the refusal rather than working
  around it, because a cost-per-unit from a demo is exactly the plausible, wrong number
  the rule exists to prevent.
- **The ROI assumptions are on screen beside the ROI.** Hours, rates, volumes and where
  the numbers came from, in the open. An ROI whose assumptions live only in a config file
  is a claim rather than a calculation, and the point of showing them is that a reader can
  disagree with them.

Deliberately does not: compute any figure it displays, reach a verdict, or write anything.
Every number here was computed by `spine` and written to a file before this page opened;
this module is layout and words. It also does not cache across runs or hold state — a
reload reads the files again, which is what makes a fresh result appear.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The monorepo's packages are not pip-installed on Streamlit Community Cloud, which installs
# only what requirements.txt names. Putting the sources on the path keeps that file down to
# the four libraries this page actually needs, instead of pulling in every provider SDK
# `spine` depends on for work this page never does.
REPO_ROOT = Path(__file__).resolve().parents[2]
for source in (REPO_ROOT / "packages" / "spine" / "src", REPO_ROOT / "apps" / "dashboards" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from dashboards.kpi_data import (  # noqa: E402 - must follow the sys.path setup above
    DashboardError,
    ProjectView,
    TrendSeries,
    available_projects,
    build_project_view,
    format_usd,
    format_value,
)
from spine.eval.projects import DEFAULT_PROJECT_ROOT  # noqa: E402 - as above
from spine.eval.run import DEFAULT_RESULTS_ROOT  # noqa: E402 - as above
from spine.kpi import CostReport, ROIInputs, ROIResult  # noqa: E402 - as above

PROJECT_ROOT = REPO_ROOT / DEFAULT_PROJECT_ROOT
RESULTS_ROOT = REPO_ROOT / DEFAULT_RESULTS_ROOT


def _comparator(direction: str) -> str:
    """The sign that shows which way a target is good."""
    return "≥" if direction == "higher_is_better" else "≤"


def render_headlines(view: ProjectView) -> None:
    """The two figures a reader should see first, large, with their targets and verdicts."""
    if not view.headlines:
        return
    columns = st.columns(len(view.headlines))
    for column, headline in zip(columns, view.headlines, strict=True):
        row = headline.row
        target = format_value(row.target, row.unit)
        with column:
            st.metric(
                label=row.metric.replace("_", " "),
                value=format_value(row.value, row.unit),
                delta=f"{row.status} · target {row.comparator} {target}",
                delta_color="normal" if row.passed else "inverse",
            )
    if not view.headlines[0].declared:
        st.caption(
            "No headline metrics are declared for this project, so the first two it reports "
            "are shown. Name them under `headline:` in the project config to choose."
        )


def render_metric_table(view: ProjectView) -> None:
    """Every metric the project reports, in the order the project declared them."""
    st.subheader("KPIs")
    frame = pd.DataFrame(
        [
            {
                "Metric": row.metric,
                "Target": f"{row.comparator} {format_value(row.target, row.unit)}",
                "Actual": format_value(row.value, row.unit),
                "Unit": row.unit,
                "Method": row.method,
                "Sample size": row.sample_size,
                "Status": row.status,
            }
            for row in view.rows
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")

    if view.stale_metrics:
        st.info(
            "Declared but not measured in the latest run: "
            f"{', '.join(view.stale_metrics)}. Re-run the evaluation to fill these in."
        )
    if view.retired_metrics:
        st.info(
            f"Measured by the latest run but no longer declared: {', '.join(view.retired_metrics)}."
        )


def render_trends(view: ProjectView) -> None:
    """One chart per metric, across every result the project has recorded."""
    st.subheader("Trend")
    drawable = [series for series in view.trends if series.has_history]
    if not drawable:
        # One point draws an empty grid, which reads as a broken chart rather than as a
        # short history. The number is in the table above; this says why there is no line.
        st.caption(
            f"Only {len(view.history.reports)} result recorded so far. A trend needs at "
            f"least two — run the evaluation again and this fills in."
        )
        return
    for series in drawable:
        _render_one_trend(series)


def _render_one_trend(series: TrendSeries) -> None:
    """One metric's history, with its target drawn alongside so the line has a meaning.

    The x axis is one position per run, labelled with when that run happened, rather than a
    continuous time axis. Runs are what a KPI moves between, and spacing them by elapsed
    time makes the labels unreadable whenever two runs land close together — as any two on
    the same afternoon do. The label carries the date, so nothing about when is hidden; what
    the axis does not claim is that the gaps between runs are to scale.
    """
    frame = pd.DataFrame(
        {
            "run": [f"{point.measured_at:%m-%d %H:%M:%S}" for point in series.points],
            series.metric: [point.value for point in series.points],
            "target": [series.target for _ in series.points],
        }
    ).set_index("run")
    arrow = _comparator(series.direction)
    st.caption(f"**{series.metric}** ({series.unit}) — target {arrow} {series.target:g}")
    st.line_chart(frame, height=280, x_label="run (oldest first)", y_label=series.unit)


def render_cost(report: CostReport, note: str) -> None:
    """What the run spent and consumed — or why no figure may be shown for it."""
    st.subheader("Cost")

    if not report.can_state_a_cost:
        st.error(
            "**No cost can be shown for this run.**\n\n"
            f"{report.replayed_calls} of {report.total_calls} model calls were served from a "
            "cassette rather than a provider. Dropping them would report this run as having "
            "cost nothing; counting them would report what an earlier run spent as though "
            "this one had spent it. Both look like measurements, so neither is shown.\n\n"
            "Run the evaluation live, or set `include_replayed_cost` in the project config "
            "to state the recorded costs knowingly."
        )
    elif not report.is_measured:
        st.warning(
            f"**These costs include {report.replayed_calls} replayed call(s).** "
            + (note or "They are what an earlier live run spent, not money spent by this run.")
        )

    basis = report.basis
    left, middle, right = st.columns(3)
    left.metric("Total run cost", format_usd(basis.total_usd) if basis else "—")
    middle.metric("Model calls", f"{report.total_calls:,}")
    right.metric(
        "Prompt / completion tokens",
        f"{report.total_prompt_tokens:,} / {report.total_completion_tokens:,}",
    )

    if not report.by_purpose:
        st.caption("This run made no model calls, so there is nothing to break down.")
        return

    st.caption("By purpose")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Purpose": usage.purpose,
                    "Calls": usage.calls,
                    "Prompt tokens": usage.prompt_tokens,
                    "Completion tokens": usage.completion_tokens,
                    "Cached tokens": usage.cached_tokens,
                    "Replayed": usage.replayed_calls,
                    "Cost": format_usd(usage.cost_usd) if usage.cost_usd is not None else "—",
                }
                for usage in report.by_purpose
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if not report.can_state_a_cost:
        st.caption(
            "Tokens are shown because they count work that was really done, whether a call "
            "was served live or from a cassette. Money is not."
        )


def render_roi(roi: ROIResult, inputs: ROIInputs, measured_cost: bool) -> None:
    """The ROI, and every assumption it rests on, side by side.

    The assumptions are not in an expander. A reader who sees the result and not the inputs
    has been told a conclusion; a reader who sees both has been shown a calculation.
    """
    st.subheader("Return on investment")
    st.warning(f"**These figures rest on stated assumptions.** {inputs.assumptions_source}")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Result**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Figure": f"Hours displaced per {roi.unit}",
                        "Value": f"{roi.hours_displaced_per_unit:g} h",
                    },
                    {
                        "Figure": f"Hours displaced per {roi.period_label}",
                        "Value": f"{roi.hours_displaced_per_period:g} h",
                    },
                    {
                        "Figure": f"Human cost per {roi.unit}, before",
                        "Value": format_usd(roi.baseline_cost_per_unit_usd),
                    },
                    {
                        "Figure": f"Cost per {roi.unit}, after",
                        "Value": format_usd(roi.automated_cost_per_unit_usd),
                    },
                    {
                        "Figure": f"Saving per {roi.period_label}",
                        "Value": format_usd(roi.saving_per_period_usd),
                    },
                    {
                        "Figure": "Payback",
                        "Value": (
                            f"{roi.payback_periods:.1f} {roi.period_label}s"
                            if roi.payback_periods is not None
                            else "never at these assumptions"
                        ),
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if roi.payback_periods is None:
            st.error(
                "**This does not pay back.** The saving is zero or negative under these "
                "assumptions, so there is no number of months that recovers the build cost."
            )

    with right:
        st.markdown("**Assumptions**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Role": role.role,
                        "Stage": stage,
                        "Hours": role.hours_per_unit,
                        "Rate": format_usd(role.hourly_cost_usd),
                    }
                    for stage, roles in (
                        ("before", inputs.baseline_roles),
                        ("after", inputs.residual_roles),
                    )
                    for role in roles
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"- Volume: **{inputs.units_per_period:g} {inputs.unit}s** per "
            f"{inputs.period_label}\n"
            f"- Implementation cost: **{format_usd(inputs.implementation_cost_usd)}**, one-off\n"
            f"- Model cost per {inputs.unit}: "
            f"**{format_usd(roi.automated_model_cost_per_unit_usd)}** "
            f"({'measured from the latest run' if measured_cost else 'assumed'})"
        )


def render_project(view: ProjectView) -> None:
    """One project's whole page."""
    st.title(f"{view.project} — KPIs")
    if view.config.description:
        st.caption(view.config.description)

    if view.history.unreadable:
        st.error(
            "These result files could not be read and are not included below: "
            + ", ".join(path.name for path in view.history.unreadable)
        )

    if not view.has_results:
        st.info(
            f"**No results yet for `{view.project}`.** This project declares "
            f"{len(view.config.kpis.kpis)} KPI(s) but nothing has been measured against them. "
            f"Run `python -m spine.eval.run --project {view.project} --out eval_report.md` "
            f"and commit the file it writes under `evals/results/`."
        )
        st.subheader("Declared KPIs")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": spec.name,
                        "Target": f"{_comparator(spec.direction)} {spec.target:g}",
                        "Unit": spec.unit,
                        "Method": spec.method,
                    }
                    for spec in view.config.kpis.kpis
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        return

    latest = view.latest
    assert latest is not None  # has_results is exactly this check
    scope = "full gold set" if latest.scope == "full" else f"{latest.sample_size}-item sample"
    st.caption(
        f"Latest run `{latest.run_id}` · {latest.measured_at:%Y-%m-%d %H:%M UTC} · "
        f"{latest.items_evaluated} item(s) from `{latest.gold_set}` ({scope}) · "
        f"{len(view.history.reports)} run(s) recorded"
    )

    render_headlines(view)
    st.divider()
    render_metric_table(view)
    st.divider()
    render_trends(view)
    st.divider()
    render_cost(latest.cost, latest.cost_note)
    st.divider()
    if view.roi is not None and view.config.kpis.roi is not None:
        render_roi(view.roi, view.config.kpis.roi, latest.cost.basis is not None)
    else:
        st.subheader("Return on investment")
        st.caption(
            "This project declares no ROI assumptions. Add an `roi:` block to its config to "
            "show one — every rate, volume and hour count has to be stated, by design."
        )


def main() -> None:
    """Draw the page for whichever project the sidebar has selected."""
    st.set_page_config(page_title="Agent portfolio KPIs", page_icon="📊", layout="wide")

    projects = available_projects(PROJECT_ROOT)
    if not projects:
        st.title("Agent portfolio KPIs")
        st.error(
            f"No project configs found under `{DEFAULT_PROJECT_ROOT}`. Each project is one "
            "YAML file there; see `evals/projects/README.md`."
        )
        return

    with st.sidebar:
        st.header("Project")
        project = st.selectbox("Project", projects, label_visibility="collapsed")
        st.caption(
            "Every project with a config file appears here. Nothing about any of them is "
            "written into this app."
        )

    try:
        view = build_project_view(project, project_root=PROJECT_ROOT, results_root=RESULTS_ROOT)
    except DashboardError as error:
        st.title(f"{project} — KPIs")
        st.error(f"This project's config could not be read.\n\n```\n{error}\n```")
        return

    render_project(view)


# Streamlit executes this file as a script with __name__ set to "__main__", so this is the
# real entry point, and importing the module (as the tests do) draws nothing.
if __name__ == "__main__":
    main()

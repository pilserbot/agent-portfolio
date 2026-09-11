# dashboards

Streamlit dashboards over the portfolio's evaluation results and KPIs.

## `kpi_app.py` — the KPI dashboard

```bash
make install
uv run streamlit run apps/dashboards/kpi_app.py
```

One page per project, and **no project-specific code in it**. Which metrics exist, what
each is called, what good looks like, which two lead the page and what the ROI assumes are
all read from `evals/projects/<project>.yaml`. Adding a project to this dashboard is adding
a YAML file; the sidebar picks up anything with a config.

It reads two committed things and nothing else — the project configs, and the result files
under `evals/results/` that `spine.eval.run` writes. No database, no API key, no network.

### Two panels that exist to be awkward

- **Cost.** A run whose model calls were served from a cassette gets no cost figure at all,
  only a banner saying why. That is the rule from `spine.kpi.cost_basis`, shown rather than
  worked around: dropping the replayed calls would report a demo as having cost nothing,
  and counting them would report an earlier run's spend as this one's. Tokens are still
  shown, because they count work that was really done either way. A project that has opted
  into `include_replayed_cost` does get a figure, under a warning saying what it is.
- **ROI.** The assumptions sit beside the result, not behind an expander: hours, rates,
  volumes, the implementation cost, and where the numbers came from. A reader who sees the
  conclusion and not the inputs has been told something; a reader who sees both has been
  shown a calculation, and can disagree with it.

### Deploying to Streamlit Community Cloud

Point it at this repository, `apps/dashboards/kpi_app.py`, with
`apps/dashboards/requirements.txt` as the dependency file. Nothing else is needed — no
secrets, no environment variables.

That file pins four libraries and none of them is a provider SDK. `spine` is not
pip-installed; `kpi_app.py` puts `packages/spine/src` on `sys.path` instead, which imports
`spine.kpi` and `spine.eval` without pulling in litellm, langgraph or langfuse for work
this page never does.

### Testing

`tests/test_dashboard_kpi.py` runs headlessly and starts no server. Everything that decides
*what* is displayed lives in `dashboards.kpi_data`, which imports no UI library at all;
`kpi_app.py` is layout and words over it.

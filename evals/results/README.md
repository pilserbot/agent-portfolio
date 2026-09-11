# evals/results

One JSON file per evaluation run, written by `spine.eval.run` as
`<project>_<timestamp>.json`. It is the same machine-readable report the `--json-out` flag
produces, kept under a name the next run will not overwrite.

**These files are committed.** They are the history the KPI dashboard
(`apps/dashboards/kpi_app.py`) draws its trend charts from, and committing them is what
lets that dashboard deploy to Streamlit Community Cloud and show something: it reads the
repository and nothing else — no database, no API key.

So each one is a record, not scratch output. Commit the runs that mean something — a run
on a change you are proposing, a run that moved a metric — and delete the ones that do not
rather than letting every local experiment accumulate. `evals/baseline.json` remains the
separate, deliberate statement of what is currently *accepted*, written only by
`--update-baseline`.

CI also uploads each report as a build artifact, which is the copy to reach for when a run
failed and you want to see why.

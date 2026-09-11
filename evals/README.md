# evals

Evaluation baselines and reports produced by `spine`'s evaluation harness.

A baseline is the committed reference a change is measured against; a report is the
output of one run over a gold set in `data/`. Both are deterministic artefacts: they are
computed in Python from typed structures, never written by a language model directly.

- `projects/` — one YAML per project: its gold set, its KPIs and their targets, tolerances
  and ROI assumptions. Read `projects/README.md` before adding one.
- `rubrics/` — versioned scoring instruments for `spine.eval.judge`.
- `calibration/` — what `spine.eval.calibrate` concluded about a judge.
- `results/` — one file per evaluation run, git-ignored.
- `baseline.json` — the numbers currently accepted, written only by
  `python -m spine.eval.run --update-baseline`.

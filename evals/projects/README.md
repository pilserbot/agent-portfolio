# evals/projects

One YAML file per project, named for the project it configures. It declares:

- **`gold_set`** — which file under `data/gold/` the project is measured on.
- **`kpis`** — every metric the project reports: name, unit, target, direction, the method
  that establishes it (test, analysis, inspection or demonstration), and the named
  computation that produces it. `spine.kpi.Computation` is a closed vocabulary; a config
  cannot supply code for the harness to run.
- **`tolerance`** per metric — how far it may drift the wrong way before
  `spine.eval.run` calls it a regression and exits non-zero.
- **`roi`** — the assumptions behind any money the project quotes: roles, hours per unit,
  hourly costs, volumes, and the one-off implementation cost. `assumptions_source` is
  required and has no default, because an ROI whose provenance is invisible is a sales
  slide rather than a measurement.

A project also needs a registered evaluation entry point — see
`spine.eval.projects.register_evaluator`, and `spine.eval.example_project` for the shape.
The config and the code are separate on purpose: the config says what is measured, and a
reviewer can read it without reading Python.

`example.yaml` is the reference. Copy its shape; do not copy its ROI numbers, which are
labelled in the file itself as illustrative placeholders.

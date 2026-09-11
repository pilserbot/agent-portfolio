"""The evaluation foundation: gold sets, deterministic checks, and the metrics over them.

- `spine.eval.datasets` loads a gold set from disk into typed records and samples it
  reproducibly.
- `spine.eval.checkers` turns one answer into one `Verdict` using ordinary Python.
- `spine.eval.metrics` aggregates Verdicts into precision, recall, F1, accuracy, macro-F1
  by tag, mean absolute percentage error and expected calibration error.
- `spine.eval.judge` scores an output against a versioned rubric, and
  `spine.eval.calibrate` measures whether that judge agrees with a human well enough to be
  used at all.
- `spine.eval.projects` holds what a project declares about its own evaluation and where
  its entry point is registered; `spine.eval.example_project` is the reference one.
- `spine.eval.run` runs a project and measures the KPIs it declares;
  `spine.eval.baseline` compares those measurements against the committed reference, and
  `spine.eval.gate` is that comparison as a standalone command.

Nothing in this package reaches a network or asks a model whether an answer was right. A
model may have produced the answer under test; deciding whether it was correct is
arithmetic over typed structures, here and nowhere else.

Deliberately does not: hold domain knowledge about what a good tender response looks like.
A gold set supplies that, and a project supplies the gold set.
"""

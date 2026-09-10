"""The evaluation foundation: gold sets, deterministic checks, and the metrics over them.

- `spine.eval.datasets` loads a gold set from disk into typed records and samples it
  reproducibly.
- `spine.eval.checkers` turns one answer into one `Verdict` using ordinary Python.
- `spine.eval.metrics` aggregates Verdicts into precision, recall, F1, accuracy, macro-F1
  by tag, mean absolute percentage error and expected calibration error.
- `spine.eval.run` produces a report; `spine.eval.gate` decides whether that report is
  acceptable against the committed baseline.

Nothing in this package reaches a network or asks a model whether an answer was right. A
model may have produced the answer under test; deciding whether it was correct is
arithmetic over typed structures, here and nowhere else.

Deliberately does not: hold domain knowledge about what a good tender response looks like.
A gold set supplies that, and a project supplies the gold set.
"""

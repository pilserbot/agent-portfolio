"""Evaluation entry points: running a gold set and gating a change on the result.

`spine.eval.run` produces a report; `spine.eval.gate` decides whether that report is
acceptable against the committed baseline.

Deliberately does not: hold the evaluation logic itself (that belongs to
`spine.evaluation`), or let either step reach a verdict with a language model.
"""

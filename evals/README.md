# evals

Evaluation baselines and reports produced by `spine`'s evaluation harness.

A baseline is the committed reference a change is measured against; a report is the
output of one run over a gold set in `data/`. Both are deterministic artefacts: they are
computed in Python from typed structures, never written by a language model directly.

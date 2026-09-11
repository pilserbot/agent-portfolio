# evals/rubrics

Judge rubrics, as YAML. One file per rubric, loaded by `spine.eval.judge.load_rubric`.

A rubric lives here rather than in code for one reason: a score is only meaningful if you
can say exactly what it was scored against. Every rubric carries a `version`, and the
version is stamped into every Verdict the judge produces.

**Bump `version` on any change to the criteria, the weights or the wording.** A changed
rubric is a different instrument; scores from two versions are not comparable, and a
calibration measured against version 1 says nothing about version 2.

The model never decides whether an item passes. It scores each criterion; the weighted
total and the pass/fail are computed in Python from those numbers and this file's
`pass_threshold`.

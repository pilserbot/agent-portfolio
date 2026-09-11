# evals/results

One JSON file per evaluation run, written by `spine.eval.run` as
`<project>_<timestamp>.json`. It is the same machine-readable report the `--json-out` flag
produces, kept under a name the next run will not overwrite.

**The files themselves are git-ignored.** A run's numbers matter while you are comparing
them; the record that survives is `evals/baseline.json`, which is committed deliberately by
`--update-baseline`. Committing every run's output would turn a review of a one-line change
into a review of a hundred numbers nobody chose.

CI uploads the report as a build artifact for the same reason.

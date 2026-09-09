# Contributing

## Two CI jobs, and why

`.github/workflows/ci.yml` deliberately splits into two jobs rather than one.

**`test`** runs on every push and every pull request. It lints, format-checks, and runs
`pytest -m "not integration" --cov`. It uses no secrets and needs no network beyond
installing packages. It is the job that must be green for a change to be reviewable.

**`eval`** runs on pull requests and on pushes to `main`. It reads `ANTHROPIC_API_KEY`,
`DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` from
repository secrets, runs `pytest -m integration`, then runs the evaluation and gates the
result against `evals/baseline.json`.

The reason for the split is the development sandbox: **it has no network access and no
environment variables.** Anything that needs an API key or a database therefore cannot run
during development at all — it can only run in CI. So the test suite is partitioned to
match:

- Unit tests make no network calls and need no environment variables. They live directly
  in `tests/` and run everywhere, including the sandbox.
- Anything needing an API key or a database lives in `tests/integration/` and is marked
  `@pytest.mark.integration`. It runs only in the `eval` job.

`make test-fast` runs exactly the first set, and must pass in a sandbox with no network
and no environment variables. That is the contract the split exists to protect.

## The `eval` job skips rather than fails

When `ANTHROPIC_API_KEY` is empty, every step in `eval` is skipped and the job reports
success. This keeps the pipeline green for contributors without secrets — most importantly
pull requests from forks, which GitHub never gives secrets to. A skipped `eval` is not a
passing evaluation; it means the evaluation did not run, and a maintainer re-runs it from a
branch in this repository before merging anything that could move the metrics.

Because the `secrets` context is not available in `if:` expressions, the guard is a first
step that reads the secret and publishes a plain output the later steps condition on.

## Sample versus full gold set

Pull requests evaluate a 20-item sample, to keep the feedback loop short. The full gold set
runs when:

- the pull request carries the `full-eval` label, or
- the trigger is a push to `main`.

Add the label and re-run the job when a change is likely to move the metrics.

## The evaluation report

`python -m spine.eval.run --sample 20 --out eval_report.md` writes two files: the markdown
report at `--out`, and a machine-readable sidecar with the same stem and a `.json` suffix.
Omit `--sample` to evaluate the full gold set.

The markdown is posted to the pull request as a **sticky** comment (via
`marocchino/sticky-pull-request-comment`, keyed on the header `eval-report`), so re-runs
update the same comment instead of piling up new ones. The JSON sidecar is what
`python -m spine.eval.gate` reads: it compares each metric against `evals/baseline.json`
and exits non-zero when one has regressed beyond its tolerance. The comment is posted
before the gate runs, so a regression still leaves the report on the pull request.

`evals/baseline.json` is `{}` today, so nothing is compared yet. Its shape is:

```json
{
  "metrics": {
    "extraction_accuracy": {
      "value": 0.82,
      "tolerance": 0.02,
      "direction": "higher_is_better"
    }
  }
}
```

Both `spine.eval.run` and `spine.eval.gate` are stubs: they establish the CLI contract and
the file shapes so the CI plumbing works today. The real harness lands later.

## Before you push

```bash
make fmt        # apply ruff format and the safe lint fixes
make lint       # ruff check + ruff format --check, both must pass clean
make test-fast  # unit tests: no network, no environment variables
```

And read [CLAUDE.md](CLAUDE.md) — the conventions there are binding, in particular that a
language model is used only to turn unstructured text into typed structures, and every
status, score, ranking, verdict or monetary figure is computed by deterministic Python.

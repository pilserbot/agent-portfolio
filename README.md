# agent-portfolio

Real-life agentic workflow case studies, built as a single Python monorepo. Two shared
libraries carry the reusable machinery — `spine` for evaluation, model routing and KPI
computation, and `req_core` for domain-agnostic requirement extraction — and the
applications on top of them are thin. The architecture rests on one rule: a language
model is used only to turn unstructured text into typed Pydantic structures, and every
status, score, ranking, verdict and monetary figure is then computed by deterministic
Python over those structures. That keeps the parts that must be auditable out of the
model's hands and under test.

## Repo map

```
packages/spine/          shared evaluation, model routing and KPI library  (import: spine)
packages/req_core/       domain-agnostic requirement extraction            (import: req_core)
apps/ri05_tender/        tender response engine, FastAPI                   (scaffold)
apps/dashboards/         Streamlit dashboards                              (scaffold)
data/                    seed data and gold sets, committed to the repo
evals/                   evaluation baselines and reports
demo/cassettes/          recorded model calls, so a demo runs with no API key
tests/                   pytest suites; tests/integration/ needs keys or a database
.github/workflows/ci.yml two jobs: `test` (no secrets) and `eval` (secrets, gold set)
CLAUDE.md                the conventions every change follows
CONTRIBUTING.md          why CI is split in two, and how the evaluation job behaves
.env.example             every configuration variable, with placeholder values
```

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make install     # create the workspace venv and install every package
cp .env.example .env
make test-fast   # unit tests: no network, no environment variables
```

## Demo mode

Any deployed demo can run with **no API key and no network**. The router records real model
calls to a cassette, and replays them later:

```bash
SPINE_MODE=record ...   # calls the provider, and writes demo/cassettes/<project>/<run_id>.jsonl
SPINE_MODE=replay ...   # serves those answers back, making no provider call at all
SPINE_MODE=live  ...    # the default: normal behaviour, nothing recorded
```

Cassettes are committed to the repository on purpose: that is what lets a reviewer with a
fresh clone, or a demo deployed with no credentials, run the real code paths end to end.
A replayed answer is one a real model actually gave — nothing is invented, and a call with
no recording is an error naming the fingerprint rather than a plausible substitute.

Matching is by fingerprint: a hash of the model, the tier, the purpose and the
whitespace-normalised prompt. Reformatting a prompt is therefore free, but changing its
words is a different call and needs re-recording.

```bash
python -m spine.replay list              # every cassette, its calls and recorded cost
python -m spine.replay verify <cassette> # check one is readable, consistent and complete
```

In replay mode the daily spend cap is not enforced, because nothing is spent, and the
replayed calls stay out of the persisted spend ledger while still appearing in the run's
own record — so a demo still shows what the work would have cost.

## Make targets

| Target           | What it does                                              |
| ---------------- | --------------------------------------------------------- |
| `make install`   | `uv sync --all-packages`                                    |
| `make lint`      | `ruff check` and `ruff format --check`                      |
| `make fmt`       | apply `ruff format` and the safe `ruff check --fix` fixes   |
| `make test`      | full suite with coverage                                    |
| `make test-fast` | `pytest -m "not integration"`                               |

## Conventions

See [CLAUDE.md](CLAUDE.md), and [CONTRIBUTING.md](CONTRIBUTING.md) for how that shapes
CI. The short version: Pydantic v2 models across every module
boundary, models only for text-to-structure, deterministic Python for every number,
module docstrings that say what the module does *and* does not do, full type hints, and
a clean `ruff`.

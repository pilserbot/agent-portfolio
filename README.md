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
tests/                   pytest suites; tests/integration/ needs keys or a database
CLAUDE.md                the conventions every change follows
.env.example             every configuration variable, with placeholder values
```

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make install     # create the workspace venv and install every package
cp .env.example .env
make test-fast   # unit tests: no network, no environment variables
```

## Make targets

| Target           | What it does                                              |
| ---------------- | --------------------------------------------------------- |
| `make install`   | `uv sync --all-packages`                                    |
| `make lint`      | `ruff check` and `ruff format --check`                      |
| `make fmt`       | apply `ruff format` and the safe `ruff check --fix` fixes   |
| `make test`      | full suite with coverage                                    |
| `make test-fast` | `pytest -m "not integration"`                               |

## Conventions

See [CLAUDE.md](CLAUDE.md). The short version: Pydantic v2 models across every module
boundary, models only for text-to-structure, deterministic Python for every number,
module docstrings that say what the module does *and* does not do, full type hints, and
a clean `ruff`.

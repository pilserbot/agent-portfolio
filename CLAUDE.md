# Conventions for this repository

These conventions are binding. Every change made in this repository — by a human or by
Claude — follows them.

## Architecture

- **Every function crossing a module boundary takes and returns a Pydantic v2 model.**
  Primitives and loose dicts are fine inside a module; they never cross its edge.

- **Language models are used ONLY to convert unstructured text into typed structures.**
  Every status, score, ranking, verdict or monetary figure is computed by deterministic
  Python code operating on those structures. This is the core architectural rule. If a
  number or a judgement can be traced back to a model's free-form output, it is a bug.

- Prefer small pure functions. Introduce a class only when state is genuinely held.

- Every module has a docstring stating what it does **and** what it deliberately does
  not do.

- Full type hints on everything.

## Testing

- Unit tests make no network calls and need no environment variables.

- Anything requiring an API key or a database lives in `tests/integration/` and is marked
  `@pytest.mark.integration`.

- `make test-fast` must pass in a sandbox with no network and no environment variables.
  It runs `pytest -m "not integration"`.

## Tooling

- Python 3.12, `uv` workspace. One `pyproject.toml` per package, plus the root workspace
  file.

- `ruff check` and `ruff format --check` must pass clean. `make lint` runs both;
  `make fmt` applies them.

## Libraries and secrets

- **Never invent a library API.** If unsure of a signature, write the smallest wrapper you
  can verify and leave a `TODO` comment naming the uncertainty.

- Never commit real secret values anywhere. Every configuration variable is listed in
  `.env.example` with a placeholder; `.env` is git-ignored.

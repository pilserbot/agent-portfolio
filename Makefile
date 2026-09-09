.PHONY: install lint fmt test test-fast

# Create the workspace virtualenv and install every package plus the dev tooling.
install:
	uv sync --all-packages

# Lint and format check. Must pass clean before anything is committed.
lint:
	uv run ruff check .
	uv run ruff format --check .

# Apply formatting and the safe lint fixes.
fmt:
	uv run ruff format .
	uv run ruff check --fix .

# Full suite, including integration tests, with coverage.
test:
	uv run pytest --cov --cov-report=term-missing

# Unit tests only: no network, no environment variables, no database.
test-fast:
	uv run pytest -m "not integration"

# tests

- `tests/` — unit tests. No network, no environment variables, no database. These run in
  `make test-fast`.
- `tests/integration/` — anything needing an API key, a database or the network. Every
  test in here carries `@pytest.mark.integration` and is excluded from `make test-fast`.

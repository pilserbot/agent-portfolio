"""TEMPORARY: a deliberately failing test, to confirm CI actually reports a red build.

Deliberately does not test anything about this project. Delete this file once the failing
run has been observed — it will fail every `make test-fast` and every CI run until then.
"""


def test_ci_reports_failures() -> None:
    assert 1 == 2

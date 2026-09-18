"""Fixtures shared by the live integration tests.

`live_extraction` is session-scoped and lazy, as pytest fixtures are: on a pull request
without the `full-eval` label every test that requests it is skipped, so nothing is
instantiated and nothing is spent. On a labelled run it produces one extraction pass, and
both the extraction gate and the findings run read that one result.

Extraction is deterministic given the corpus and the policy — the same three documents, the
same segmenter, the same modality convention — so running it twice in one job produced the
same requirements twice at a cost of 14 calls, about $0.89 and roughly ten minutes. That
second pass is most of what pushed the eval job past its wall-clock limit.

Deliberately does not: assert anything, or hold a Router across tests. The record the
fixture returns carries frozen `ModelCall` records instead, so no test's assertions can
depend on another test's spending.
"""

import pytest
from live_extraction import LiveExtraction, extract_once


@pytest.fixture(scope="session")
def live_extraction(tmp_path_factory: pytest.TempPathFactory) -> LiveExtraction:
    """Extract the Kessler Point corpus once, live, for the whole session."""
    return extract_once(tmp_path_factory.mktemp("live_extraction") / "ledger.json")

"""One real end-to-end call through the router, against a live provider.

Needs ANTHROPIC_API_KEY and a network, so it is marked `integration` and excluded from
`make test-fast`. It exists to catch what the fakes cannot: that the configured model name
resolves, that credentials work, and that a real response carries the usage counts the
cost computation depends on.

Deliberately does not: assert anything about what the model says. The reply's content is
not the router's contract; the record of the call is. It also writes its ledger to a
temporary path so a test run never touches the project's .spend directory.
"""

import os
from decimal import Decimal
from pathlib import Path

import pytest

from spine.router import Router, RouterConfig


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_a_real_call_returns_text_and_a_priced_record(tmp_path: Path) -> None:
    router = Router(
        RouterConfig.from_env().model_copy(update={"ledger_path": tmp_path / "ledger.json"})
    )

    text, call = router.complete(
        "Reply with the single word: ready.",
        purpose="integration-smoke",
        tier="small",
        max_tokens=16,
    )

    assert text.strip(), "a live call must return some text"
    assert call.model == router.config.model_small
    assert call.prompt_tokens > 0
    assert call.completion_tokens > 0
    assert call.cost_usd > Decimal("0")
    assert call.latency_ms > 0
    assert call.purpose == "integration-smoke"

    # The ledger recorded the call and persisted the day's total.
    assert router.ledger.calls == (call,)
    assert router.ledger.total_for(call.timestamp.date()) == call.cost_usd
    assert (tmp_path / "ledger.json").exists()

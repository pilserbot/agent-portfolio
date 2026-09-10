"""Unit tests for the model router.

Every test injects a fake completion function, so nothing here reaches a network or needs
a credential. Cost is still computed by litellm's bundled price map, which is the real
code path.

Deliberately does not: exercise a real provider. That lives in
tests/integration/test_router_live.py.
"""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import litellm
import pytest
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from pydantic import BaseModel, Field

from spine.router import (
    DEFAULT_MODEL_LARGE,
    DEFAULT_MODEL_SMALL,
    MAX_ATTEMPTS,
    CostLedger,
    PricingUnavailable,
    Router,
    RouterConfig,
    SpendCapExceeded,
    calls_by_purpose,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def a_response(
    *,
    model: str = DEFAULT_MODEL_LARGE,
    content: str = "hello",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    cached_tokens: int = 0,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
        ),
    )


class Recorder:
    """A fake completion function that records how it was called."""

    def __init__(self, *responses: ModelResponse | Exception) -> None:
        """Queue the responses (or exceptions) to hand back, in order."""
        self.queue = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        item = self.queue.pop(0) if self.queue else a_response()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def call_count(self) -> int:
        return len(self.calls)


def a_router(tmp_path: Path, fake: Recorder, **config: object) -> Router:
    settings = {"ledger_path": tmp_path / "ledger.json", **config}
    return Router(
        RouterConfig(**settings),
        completion_fn=fake,
        sleep_fn=lambda _seconds: None,
        now_fn=lambda: AT,
    )


# --- configuration -------------------------------------------------------------------


def test_config_falls_back_to_defaults_when_the_environment_is_empty() -> None:
    config = RouterConfig.from_env({})

    assert config.model_large == DEFAULT_MODEL_LARGE
    assert config.model_small == DEFAULT_MODEL_SMALL
    assert config.daily_spend_cap_usd == Decimal("10.00")


def test_config_reads_the_environment() -> None:
    config = RouterConfig.from_env(
        {"MODEL_LARGE": "big", "MODEL_SMALL": "small", "DAILY_SPEND_CAP_USD": "2.50"}
    )

    assert (config.model_large, config.model_small) == ("big", "small")
    assert config.daily_spend_cap_usd == Decimal("2.50")


def test_an_empty_environment_variable_is_treated_as_unset() -> None:
    assert RouterConfig.from_env({"MODEL_LARGE": ""}).model_large == DEFAULT_MODEL_LARGE


def test_the_tier_selects_the_model() -> None:
    config = RouterConfig(model_large="big", model_small="small")

    assert config.model_for("large") == "big"
    assert config.model_for("small") == "small"


# --- cost computation ----------------------------------------------------------------


def test_cost_is_computed_from_the_provider_token_counts(tmp_path: Path) -> None:
    fake = Recorder(a_response(prompt_tokens=1000, completion_tokens=200))
    router = a_router(tmp_path, fake)

    _text, call = router.complete("hi", purpose="extract")

    # Compute the expectation independently from the published per-token prices, rather
    # than by calling the same helper the router uses.
    prices = litellm.model_cost[DEFAULT_MODEL_LARGE]
    expected = Decimal(str(prices["input_cost_per_token"] * 1000)) + Decimal(
        str(prices["output_cost_per_token"] * 200)
    )
    assert call.cost_usd == pytest.approx(expected)
    assert call.cost_usd > Decimal("0")
    assert call.prompt_tokens == 1000
    assert call.completion_tokens == 200


def test_the_call_record_is_fully_populated(tmp_path: Path) -> None:
    fake = Recorder(a_response(cached_tokens=400))
    router = a_router(tmp_path, fake)

    text, call = router.complete("hi", purpose="extract", tier="small")

    assert text == "hello"
    assert call.provider == "anthropic"
    assert call.model == DEFAULT_MODEL_SMALL
    assert call.cached_tokens == 400
    assert call.purpose == "extract"
    assert call.timestamp == AT
    assert call.latency_ms >= 0


def test_a_bigger_call_costs_more(tmp_path: Path) -> None:
    small = a_router(tmp_path / "a", Recorder(a_response(prompt_tokens=1000)))
    large = a_router(tmp_path / "b", Recorder(a_response(prompt_tokens=100_000)))

    _t1, cheap = small.complete("hi", purpose="extract")
    _t2, dear = large.complete("hi", purpose="extract")

    assert dear.cost_usd > cheap.cost_usd


def test_an_unpriceable_model_raises_rather_than_recording_zero(tmp_path: Path) -> None:
    fake = Recorder(a_response(model="totally-made-up-model"))
    router = a_router(tmp_path, fake, model_large="totally-made-up-model")

    with pytest.raises(PricingUnavailable):
        router.complete("hi", purpose="extract")


# --- the ledger ----------------------------------------------------------------------


def test_the_ledger_accumulates_across_calls(tmp_path: Path) -> None:
    fake = Recorder(a_response(), a_response(), a_response())
    router = a_router(tmp_path, fake)

    costs = [router.complete("hi", purpose="extract")[1].cost_usd for _ in range(3)]

    assert len(router.ledger.calls) == 3
    assert router.ledger.session_total() == sum(costs)
    assert router.ledger.total_for(AT.date()) == sum(costs)


def test_the_ledger_total_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ledger.json"
    fake = Recorder(a_response(), a_response())
    router = Router(RouterConfig(ledger_path=path), completion_fn=fake, now_fn=lambda: AT)

    router.complete("hi", purpose="extract")
    router.complete("hi", purpose="extract")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["days"][AT.date().isoformat()]["call_count"] == 2
    assert Decimal(persisted["days"][AT.date().isoformat()]["total_usd"]) == (
        router.ledger.session_total()
    )


def test_a_second_router_sees_the_first_router_spend(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    first = Router(RouterConfig(ledger_path=path), completion_fn=Recorder(), now_fn=lambda: AT)
    first.complete("hi", purpose="extract")

    second = Router(RouterConfig(ledger_path=path), completion_fn=Recorder(), now_fn=lambda: AT)

    assert second.ledger.total_for(AT.date()) == first.ledger.session_total()
    assert second.ledger.calls == ()


def test_a_missing_or_corrupt_ledger_file_reads_as_empty(tmp_path: Path) -> None:
    missing = CostLedger(tmp_path / "absent.json")
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not json", encoding="utf-8")

    assert missing.total_for(AT.date()) == Decimal("0")
    assert CostLedger(corrupt_path).total_for(AT.date()) == Decimal("0")


def test_spend_is_totalled_per_purpose(tmp_path: Path) -> None:
    router = a_router(tmp_path, Recorder())
    router.complete("hi", purpose="extract")
    router.complete("hi", purpose="extract")
    router.complete("hi", purpose="summarise")

    totals = calls_by_purpose(router.ledger.calls)

    assert set(totals) == {"extract", "summarise"}
    assert totals["extract"] == totals["summarise"] * 2


# --- the hard spend cap ---------------------------------------------------------------


def test_the_cap_raises_before_any_call_is_made(tmp_path: Path) -> None:
    fake = Recorder()
    router = a_router(tmp_path, fake, daily_spend_cap_usd=Decimal("0"))

    with pytest.raises(SpendCapExceeded):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 0, "the cap must refuse before the provider is called"


def test_the_cap_stops_calls_once_the_day_total_reaches_it(tmp_path: Path) -> None:
    fake = Recorder(*[a_response(prompt_tokens=1_000_000) for _ in range(5)])
    router = a_router(tmp_path, fake, daily_spend_cap_usd=Decimal("5.00"))

    made = 0
    with pytest.raises(SpendCapExceeded):
        for _ in range(5):
            router.complete("hi", purpose="extract")
            made += 1

    assert 0 < made < 5
    assert fake.call_count == made
    assert router.ledger.total_for(AT.date()) >= Decimal("5.00")


def test_the_cap_also_guards_structured_calls(tmp_path: Path) -> None:
    fake = Recorder()
    router = a_router(tmp_path, fake, daily_spend_cap_usd=Decimal("0"))

    class Shape(BaseModel):
        name: str

    with pytest.raises(SpendCapExceeded):
        router.structured("hi", Shape, purpose="extract")

    assert fake.call_count == 0


def test_the_cap_counts_spend_a_previous_process_recorded(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"days": {AT.date().isoformat(): {"total_usd": "99.00", "call_count": 7}}}),
        encoding="utf-8",
    )
    fake = Recorder()
    router = a_router(tmp_path, fake, daily_spend_cap_usd=Decimal("10.00"))

    with pytest.raises(SpendCapExceeded):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 0


# --- retries -------------------------------------------------------------------------


def rate_limit() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="slow down", llm_provider="anthropic", model=DEFAULT_MODEL_LARGE
    )


def test_a_rate_limit_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    fake = Recorder(rate_limit(), rate_limit(), a_response())
    router = a_router(tmp_path, fake)

    text, call = router.complete("hi", purpose="extract")

    assert text == "hello"
    assert fake.call_count == 3
    assert call.cost_usd > Decimal("0")


def test_retries_stop_at_the_maximum_and_reraise(tmp_path: Path) -> None:
    fake = Recorder(*[rate_limit() for _ in range(MAX_ATTEMPTS + 2)])
    router = a_router(tmp_path, fake)

    with pytest.raises(litellm.RateLimitError):
        router.complete("hi", purpose="extract")

    assert fake.call_count == MAX_ATTEMPTS


def test_backoff_waits_grow_and_are_jittered(tmp_path: Path) -> None:
    waits: list[float] = []
    fake = Recorder(rate_limit(), rate_limit(), rate_limit(), a_response())
    router = Router(
        RouterConfig(ledger_path=tmp_path / "ledger.json"),
        completion_fn=fake,
        sleep_fn=waits.append,
        now_fn=lambda: AT,
    )

    router.complete("hi", purpose="extract")

    assert len(waits) == 3
    # Full jitter: each wait is drawn from [0, window) where the window doubles each time.
    for attempt, wait in enumerate(waits):
        assert 0.0 <= wait <= 0.5 * 2**attempt


def test_a_bad_request_is_not_retried(tmp_path: Path) -> None:
    error = litellm.BadRequestError(
        message="nope", llm_provider="anthropic", model=DEFAULT_MODEL_LARGE
    )
    fake = Recorder(error, a_response())
    router = a_router(tmp_path, fake)

    with pytest.raises(litellm.BadRequestError):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 1, "a bad request must not be retried"


def test_a_failed_call_costs_nothing_and_is_not_recorded(tmp_path: Path) -> None:
    fake = Recorder(*[rate_limit() for _ in range(MAX_ATTEMPTS)])
    router = a_router(tmp_path, fake)

    with pytest.raises(litellm.RateLimitError):
        router.complete("hi", purpose="extract")

    assert router.ledger.calls == ()
    assert router.ledger.total_for(AT.date()) == Decimal("0")


# --- structured extraction ------------------------------------------------------------


class Person(BaseModel):
    """A tiny schema for the structured-output tests."""

    name: str
    age: int = Field(ge=0)


def json_response(payload: dict[str, object]) -> ModelResponse:
    return a_response(content=json.dumps(payload))


def test_structured_returns_a_validated_object(tmp_path: Path) -> None:
    fake = Recorder(json_response({"name": "Ada", "age": 36}))
    router = a_router(tmp_path, fake)

    person, call = router.structured("Extract", Person, purpose="extract")

    assert isinstance(person, Person)
    assert (person.name, person.age) == ("Ada", 36)
    assert call.purpose == "extract"
    assert call.cost_usd > Decimal("0")


def test_structured_retries_when_validation_fails(tmp_path: Path) -> None:
    fake = Recorder(
        json_response({"name": "Ada", "age": -5}),
        json_response({"name": "Ada", "age": 36}),
    )
    router = a_router(tmp_path, fake)

    person, _call = router.structured("Extract", Person, purpose="extract")

    assert person.age == 36
    assert fake.call_count == 2


def test_the_validation_error_is_fed_back_into_the_prompt(tmp_path: Path) -> None:
    fake = Recorder(
        json_response({"name": "Ada", "age": -5}),
        json_response({"name": "Ada", "age": 36}),
    )
    router = a_router(tmp_path, fake)

    router.structured("Extract", Person, purpose="extract")

    retry_messages = json.dumps(fake.calls[-1]["messages"])
    assert "greater than or equal to 0" in retry_messages


def test_structured_gives_up_after_two_retries(tmp_path: Path) -> None:
    fake = Recorder(*[json_response({"name": "Ada", "age": -5}) for _ in range(6)])
    router = a_router(tmp_path, fake)

    with pytest.raises(Exception, match="(?i)retr|validation"):
        router.structured("Extract", Person, purpose="extract")

    assert fake.call_count == 3, "one attempt plus two validation retries"


def test_a_structured_call_is_recorded_in_the_ledger(tmp_path: Path) -> None:
    fake = Recorder(json_response({"name": "Ada", "age": 36}))
    router = a_router(tmp_path, fake)

    _person, call = router.structured("Extract", Person, purpose="extract")

    assert router.ledger.calls == (call,)


# --- prompt caching -------------------------------------------------------------------


def test_the_prompt_is_marked_cacheable_for_a_provider_that_supports_it(
    tmp_path: Path,
) -> None:
    fake = Recorder()
    router = a_router(tmp_path, fake)

    router.complete("a long context", purpose="extract")

    content = fake.calls[0]["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[0]["text"] == "a long context"


def test_caching_can_be_turned_off(tmp_path: Path) -> None:
    fake = Recorder()
    router = a_router(tmp_path, fake)

    router.complete("hi", purpose="extract", cache_prompt=False)

    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_no_cache_marker_for_a_provider_that_does_not_support_it(tmp_path: Path) -> None:
    fake = Recorder(a_response(model="openai/gpt-4o-mini"))
    router = a_router(tmp_path, fake, model_large="openai/gpt-4o-mini")

    router.complete("hi", purpose="extract")

    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_extra_kwargs_reach_the_provider(tmp_path: Path) -> None:
    fake = Recorder()
    router = a_router(tmp_path, fake)

    router.complete("hi", purpose="extract", temperature=0.0, max_tokens=64)

    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["max_tokens"] == 64

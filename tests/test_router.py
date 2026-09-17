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
    is_retryable,
    strip_code_fence,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def a_response(
    *,
    model: str = DEFAULT_MODEL_LARGE,
    content: str = "hello",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
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
            # litellm's own naming: `cached_tokens` is the READ count and
            # `cache_write_tokens` the creation count. The fixture spells them out so a
            # test that swapped them would read as obviously wrong.
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=cache_read_tokens,
                cache_write_tokens=cache_creation_tokens,
            ),
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
    fake = Recorder(a_response(cache_creation_tokens=900, cache_read_tokens=400))
    router = a_router(tmp_path, fake)

    text, call = router.complete("hi", purpose="extract", tier="small")

    assert text == "hello"
    assert call.provider == "anthropic"
    assert call.model == DEFAULT_MODEL_SMALL
    assert call.cache_creation_tokens == 900
    assert call.cache_read_tokens == 400
    assert call.purpose == "extract"
    assert call.timestamp == AT
    assert call.latency_ms >= 0


def test_a_cache_write_is_not_reported_as_a_cache_read(tmp_path: Path) -> None:
    """A write costs ~1.25x and a read ~0.1x, so reading one as the other inverts the sign.

    A cold call can only ever write: there is nothing yet to read from. A counter that
    covered both would show a non-zero figure here and invite the conclusion that caching
    was paying off, when what actually happened was a surcharge.
    """
    fake = Recorder(a_response(cache_creation_tokens=5000, cache_read_tokens=0))
    router = a_router(tmp_path, fake)

    _, call = router.complete("hi", purpose="extract", tier="small")

    assert call.cache_creation_tokens == 5000
    assert call.cache_read_tokens == 0
    assert call.cache_activity == "writing"


def test_a_cache_read_is_not_reported_as_a_cache_write(tmp_path: Path) -> None:
    """The other direction, so neither field can be quietly reading the other's source."""
    fake = Recorder(a_response(cache_creation_tokens=0, cache_read_tokens=5000))
    router = a_router(tmp_path, fake)

    _, call = router.complete("hi", purpose="extract", tier="small")

    assert call.cache_creation_tokens == 0
    assert call.cache_read_tokens == 5000
    assert call.cache_activity == "reading"


def test_a_call_that_writes_and_reads_says_so(tmp_path: Path) -> None:
    fake = Recorder(a_response(cache_creation_tokens=300, cache_read_tokens=5000))
    router = a_router(tmp_path, fake)

    _, call = router.complete("hi", purpose="extract", tier="small")

    assert call.cache_activity == "writing_and_reading"


def test_a_provider_that_reports_no_cache_counts_at_all(tmp_path: Path) -> None:
    """Neither field may invent a number when the provider supplied none."""
    fake = Recorder(a_response())
    router = a_router(tmp_path, fake)

    _, call = router.complete("hi", purpose="extract", tier="small")

    assert call.cache_creation_tokens == 0
    assert call.cache_read_tokens == 0
    assert call.cache_activity == "none"


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


def auth_failure() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message="invalid x-api-key", llm_provider="anthropic", model=DEFAULT_MODEL_LARGE
    )


def service_unavailable() -> litellm.ServiceUnavailableError:
    """The exact shape that killed a live pass: a 503 whose text mentions credentials.

    litellm classifies it as ServiceUnavailableError, which is the right call — sixteen
    other calls on the same key succeeded either side of it. What made it fatal was the
    wrapper, not the type.
    """
    return litellm.ServiceUnavailableError(
        message="AnthropicException - credential validation failed",
        llm_provider="anthropic",
        model=DEFAULT_MODEL_LARGE,
    )


def wrapped(inner: Exception) -> Exception:
    """`inner`, re-raised from a wrapper the retryable set does not list.

    This is what `instructor` does: it catches the provider's exception, exhausts its own
    validation retries and raises InstructorRetryException *from* it. A plain RuntimeError
    stands in for that here so the test pins the unwrapping rather than instructor's type.
    """
    outer = RuntimeError("all retry attempts exhausted")
    # Exactly what `raise outer from inner` sets, built here because this is an expression.
    outer.__cause__ = inner
    return outer


def test_a_transient_fault_inside_a_wrapper_is_still_retried(tmp_path: Path) -> None:
    """The fault that killed a full pass: a 503 wearing a type the retry set never listed."""
    fake = Recorder(wrapped(service_unavailable()), a_response())
    router = a_router(tmp_path, fake)

    text, _call = router.complete("hi", purpose="extract")

    assert text == "hello"
    assert fake.call_count == 2


def test_an_auth_failure_inside_a_wrapper_still_fails_immediately(tmp_path: Path) -> None:
    """Unwrapping must not turn a dead key into four attempts against a dead key."""
    fake = Recorder(*[wrapped(auth_failure()) for _ in range(MAX_ATTEMPTS + 2)])
    router = a_router(tmp_path, fake)

    with pytest.raises(RuntimeError):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 1, "an auth failure is answered once, not four times"


def test_a_bare_auth_failure_is_not_retried(tmp_path: Path) -> None:
    fake = Recorder(*[auth_failure() for _ in range(MAX_ATTEMPTS + 2)])
    router = a_router(tmp_path, fake)

    with pytest.raises(litellm.AuthenticationError):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 1


def test_a_fault_that_is_neither_transient_nor_fatal_is_not_retried(tmp_path: Path) -> None:
    """An unrecognised error propagates on the first attempt, as it did before unwrapping."""
    fake = Recorder(*[ValueError("something else entirely") for _ in range(MAX_ATTEMPTS)])
    router = a_router(tmp_path, fake)

    with pytest.raises(ValueError, match="something else"):
        router.complete("hi", purpose="extract")

    assert fake.call_count == 1


def test_is_retryable_reads_the_whole_chain() -> None:
    assert is_retryable(service_unavailable()) is True
    assert is_retryable(wrapped(service_unavailable())) is True
    assert is_retryable(auth_failure()) is False
    assert is_retryable(wrapped(auth_failure())) is False
    assert is_retryable(ValueError("unknown")) is False
    # A fatal link beats a transient one wherever the two meet.
    assert is_retryable(wrapped(wrapped(auth_failure()))) is False


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


# --- fenced JSON ------------------------------------------------------------------------
#
# A model asked for JSON sometimes wraps the whole answer in a markdown fence. It is a known
# structured-output failure and it killed a live pass. The fence is stripped before
# validation; nothing else about the body is touched, so a body that is not valid JSON must
# still fail exactly as it did before.


def fenced(payload: str, *, tag: str = "") -> ModelResponse:
    """A response whose entire content is one markdown fence around `payload`."""
    return a_response(content=f"```{tag}\n{payload}\n```")


def test_json_inside_a_bare_fence_still_validates(tmp_path: Path) -> None:
    fake = Recorder(fenced(json.dumps({"name": "Ada", "age": 36})))
    router = a_router(tmp_path, fake)

    person, _call = router.structured("Extract", Person, purpose="extract")

    assert (person.name, person.age) == ("Ada", 36)
    assert fake.call_count == 1, "a fence must not cost a validation retry"


def test_json_inside_a_language_tagged_fence_still_validates(tmp_path: Path) -> None:
    fake = Recorder(fenced(json.dumps({"name": "Ada", "age": 36}), tag="json"))
    router = a_router(tmp_path, fake)

    person, _call = router.structured("Extract", Person, purpose="extract")

    assert (person.name, person.age) == ("Ada", 36)
    assert fake.call_count == 1


def test_malformed_json_still_fails_even_inside_a_fence(tmp_path: Path) -> None:
    """The fence is unwrapped; the broken body underneath is not repaired.

    This is the test that keeps the strip honest. A parser that patched up the JSON would
    make this pass, and would thereby hide every real contract failure behind a guess.
    """
    fake = Recorder(*[fenced('{"name": "Ada", "age":') for _ in range(6)])
    router = a_router(tmp_path, fake)

    with pytest.raises(Exception, match="(?i)retr|json|validation"):
        router.structured("Extract", Person, purpose="extract")

    assert fake.call_count == 3, "one attempt plus two validation retries, as for any bad body"


def test_malformed_json_with_no_fence_is_unchanged_and_still_fails(tmp_path: Path) -> None:
    fake = Recorder(*[a_response(content='{"name": "Ada", "age":') for _ in range(6)])
    router = a_router(tmp_path, fake)

    with pytest.raises(Exception, match="(?i)retr|json|validation"):
        router.structured("Extract", Person, purpose="extract")

    assert fake.call_count == 3


def test_strip_code_fence_leaves_everything_that_is_not_one_whole_fence() -> None:
    """Only a response that IS a fence is unwrapped. Anything else is returned untouched."""
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('  ```\n{"a": 1}\n```  ') == '{"a": 1}'
    # A fence around part of a longer answer is not packaging, so it is left alone rather
    # than having prose silently discarded around it.
    assert strip_code_fence('Here you go:\n```json\n{"a": 1}\n```') == (
        'Here you go:\n```json\n{"a": 1}\n```'
    )
    # An unterminated fence is not a fence.
    assert strip_code_fence('```json\n{"a": 1}') == '```json\n{"a": 1}'


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

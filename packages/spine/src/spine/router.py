"""The single choke point every model call in this project passes through.

Resolves a tier to a concrete model, enforces a hard daily spend cap before any request
leaves the process, retries transient failures with exponential backoff and jitter, prices
each call deterministically from the provider's token counts, and records the result as a
`ModelCall`. `structured` is the only sanctioned way to turn text into a typed object.

Deliberately does not: decide anything about the content it moves. It does not judge,
score, rank or summarise, and nothing it returns is a verdict — a model produces text or
fills a schema, and every downstream status, score or figure is computed by Python from
those structures. It also does not format money, retry a refusal or a bad request (only
transient faults), or fall back to a different model when one fails: a silent downgrade
would make cost and quality unattributable.
"""

import json
import os
import random
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import instructor
import litellm
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import ModelCall
from spine.replay import CallRequest, CallResponse, CassetteEntry, ReplaySession
from spine.telemetry import current_run, traced

Tier = Literal["large", "small"]

DEFAULT_MODEL_LARGE = "claude-sonnet-5"
DEFAULT_MODEL_SMALL = "claude-haiku-4-5-20251001"
DEFAULT_DAILY_SPEND_CAP_USD = Decimal("10.00")
DEFAULT_LEDGER_PATH = Path(".spend/ledger.json")

# Where a cassette entry is filed when a call happens outside any run context.
DEFAULT_CASSETTE_PROJECT = "unknown"
DEFAULT_CASSETTE_RUN = "adhoc"

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 30.0
STRUCTURED_VALIDATION_RETRIES = 2

# Faults worth retrying: the request was well-formed and may succeed unchanged. A refusal,
# a bad request or an auth failure is not here on purpose — retrying those just burns quota.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
)


class RouterError(Exception):
    """Base class for every failure this module raises on purpose."""


class SpendCapExceeded(RouterError):
    """The call was refused because it would breach the daily spend cap.

    Raised before the request is made. This is a control, not a warning: there is no
    override argument, and the caller cannot proceed by ignoring it.
    """


class ReplayKindMismatch(RouterError):
    """A cassette entry exists for the fingerprint but was recorded from the other call kind."""


class PricingUnavailable(RouterError):
    """A call succeeded but its cost could not be computed.

    A spend cap that silently records unpriced calls as free is not a cap, so an unknown
    price is an error rather than a zero.
    """


class RouterConfig(BaseModel):
    """Where the router sends calls, and what it is allowed to spend."""

    model_config = ConfigDict(frozen=True)

    model_large: str = DEFAULT_MODEL_LARGE
    model_small: str = DEFAULT_MODEL_SMALL
    daily_spend_cap_usd: Decimal = Field(
        default=DEFAULT_DAILY_SPEND_CAP_USD,
        ge=0,
        description="Hard ceiling on spend per calendar day, in usd.",
    )
    ledger_path: Path = DEFAULT_LEDGER_PATH

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RouterConfig":
        """Build a config from environment variables, falling back to the defaults."""
        source = os.environ if env is None else env
        return cls(
            model_large=source.get("MODEL_LARGE") or DEFAULT_MODEL_LARGE,
            model_small=source.get("MODEL_SMALL") or DEFAULT_MODEL_SMALL,
            daily_spend_cap_usd=Decimal(
                source.get("DAILY_SPEND_CAP_USD") or DEFAULT_DAILY_SPEND_CAP_USD
            ),
            ledger_path=Path(source.get("SPEND_LEDGER_PATH") or DEFAULT_LEDGER_PATH),
        )

    def model_for(self, tier: Tier) -> str:
        """Resolve a tier to the configured model name."""
        return self.model_large if tier == "large" else self.model_small


class DailyTotal(BaseModel):
    """What one calendar day has cost so far."""

    model_config = ConfigDict(frozen=True)

    day: date
    total_usd: Decimal = Field(ge=0, description="Spend on this day so far, in usd.")
    call_count: int = Field(ge=0)


class CostLedger:
    """Accumulates model calls in memory and persists a running daily total to disk.

    State is genuinely held here: the in-memory records for this process, and a per-day
    total shared with every other process writing the same file.
    """

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        """Open a ledger backed by a JSON file, which need not exist yet."""
        self.path = path
        self._calls: list[ModelCall] = []

    @property
    def calls(self) -> tuple[ModelCall, ...]:
        """Every call recorded by this process, in the order they were made."""
        return tuple(self._calls)

    def record(self, call: ModelCall, *, persist: bool = True) -> DailyTotal:
        """Add a call to the ledger and return the day's new total.

        `persist=False` keeps the call in memory without writing it to the shared file:
        a replayed call belongs in the run's record but costs nothing, and writing it
        would walk a real daily cap toward its limit on money nobody spent. A call the
        record itself says was replayed is never persisted, whatever the caller asked
        for — the cap is a control, so the check belongs on this side of the call too.
        """
        self._calls.append(call)
        day = call.timestamp.astimezone(UTC).date()
        totals = self._read_totals()
        previous = totals.get(day.isoformat(), {"total_usd": "0", "call_count": 0})
        updated = DailyTotal(
            day=day,
            total_usd=Decimal(str(previous["total_usd"])) + call.cost_usd,
            call_count=int(previous["call_count"]) + 1,
        )
        if not persist or not call.was_billed:
            return updated
        totals[day.isoformat()] = {
            "total_usd": str(updated.total_usd),
            "call_count": updated.call_count,
        }
        self._write_totals(totals)
        return updated

    def total_for(self, day: date) -> Decimal:
        """Spend recorded against a day, across every process that wrote this ledger."""
        entry = self._read_totals().get(day.isoformat())
        return Decimal("0") if entry is None else Decimal(str(entry["total_usd"]))

    def session_total(self) -> Decimal:
        """Spend recorded by this process alone, whatever day each call landed on."""
        return sum((call.cost_usd for call in self._calls), Decimal("0"))

    def _read_totals(self) -> dict[str, dict[str, Any]]:
        """Load the persisted day totals, treating an unreadable file as empty."""
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return loaded.get("days", {}) if isinstance(loaded, dict) else {}

    def _write_totals(self, totals: dict[str, dict[str, Any]]) -> None:
        """Persist the day totals, creating the directory if it is missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"days": totals}, indent=2) + "\n", encoding="utf-8")


def _sleep_for(attempt: int, rng: random.Random) -> float:
    """Full-jitter backoff: a random wait inside an exponentially growing window."""
    window = min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)
    return rng.uniform(0.0, window)


def _cached_tokens(response: ModelResponse) -> int:
    """Read cached prompt tokens off a response, tolerating providers that omit them."""
    details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0)


def _supports_prompt_caching(model: str) -> bool:
    """Whether the provider takes an explicit cache_control marker on a message block."""
    # TODO: replace with a capability lookup once litellm exposes one that covers cache
    # control specifically; `supports_prompt_caching` exists but is not verified here.
    return "claude" in model.lower() or model.lower().startswith("anthropic/")


def _build_messages(prompt: str, *, model: str, cache_prompt: bool) -> list[dict[str, Any]]:
    """Wrap a prompt as a message list, marking it cacheable where that is supported."""
    if not (cache_prompt and _supports_prompt_caching(model)):
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
        }
    ]


class Router:
    """Routes every model call in the project, and refuses the ones that cost too much.

    State is genuinely held: the configuration, the ledger and the retry clock. Inject
    `completion_fn` to run without a network — the default is `litellm.completion`.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        ledger: CostLedger | None = None,
        completion_fn: Callable[..., ModelResponse] | None = None,
        replay: ReplaySession | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        rng: random.Random | None = None,
    ) -> None:
        """Build a router, defaulting every collaborator to its real implementation."""
        self.config = config or RouterConfig.from_env()
        self.ledger = ledger or CostLedger(self.config.ledger_path)
        self._completion_fn = completion_fn or litellm.completion
        self.replay = replay or ReplaySession.from_env()
        self._sleep = sleep_fn
        self._now = now_fn
        self._rng = rng or random.Random()

    @traced("router.complete", "model_call", kind="generation")
    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        tier: Tier = "large",
        cache_prompt: bool = True,
        **kwargs: object,
    ) -> tuple[str, ModelCall]:
        """Send a prompt and return the reply text with the record of what it cost.

        In replay mode the answer comes from a cassette and no request is made.
        """
        model = self.config.model_for(tier)
        request = CallRequest(model=model, tier=tier, purpose=purpose, prompt=prompt)

        if self.replay.is_replaying:
            entry = self._replayed(request)
            if entry.response.text is None:
                raise ReplayKindMismatch(
                    f"cassette entry {entry.fingerprint} was recorded from a structured call; "
                    f"complete() cannot serve it."
                )
            return entry.response.text, entry.response.call

        self._guard_spend_cap(model)
        messages = _build_messages(prompt, model=model, cache_prompt=cache_prompt)

        started = time.monotonic()
        response = self._with_backoff(
            lambda: self._completion_fn(model=model, messages=messages, **kwargs)
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        call = self._record(response, model=model, purpose=purpose, latency_ms=latency_ms)
        text = self._text_of(response)
        self._maybe_record_cassette(request, CallResponse(text=text, call=call))
        return text, call

    @traced("router.structured", "extraction", kind="generation")
    def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        purpose: str,
        tier: Tier = "large",
        cache_prompt: bool = True,
        **kwargs: object,
    ) -> tuple[BaseModel, ModelCall]:
        """Extract a typed object from a prompt, re-prompting on a validation failure.

        Instructor feeds each validation error back into the conversation and asks again,
        up to `STRUCTURED_VALIDATION_RETRIES` times after the first attempt.
        """
        model = self.config.model_for(tier)
        request = CallRequest(model=model, tier=tier, purpose=purpose, prompt=prompt)

        if self.replay.is_replaying:
            entry = self._replayed(request)
            if entry.response.structured_json is None:
                raise ReplayKindMismatch(
                    f"cassette entry {entry.fingerprint} was recorded from a text call; "
                    f"structured() cannot serve it."
                )
            return schema.model_validate_json(entry.response.structured_json), entry.response.call

        self._guard_spend_cap(model)
        messages = _build_messages(prompt, model=model, cache_prompt=cache_prompt)
        client = instructor.from_litellm(self._completion_fn, mode=instructor.Mode.JSON)

        started = time.monotonic()
        obj, response = self._with_backoff(
            lambda: client.chat.completions.create_with_completion(
                model=model,
                messages=messages,
                response_model=schema,
                max_retries=STRUCTURED_VALIDATION_RETRIES,
                **kwargs,
            )
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        call = self._record(response, model=model, purpose=purpose, latency_ms=latency_ms)
        self._maybe_record_cassette(
            request, CallResponse(structured_json=obj.model_dump_json(), call=call)
        )
        return obj, call

    def _run_identity(self) -> tuple[str, str]:
        """Which run a cassette entry belongs to, from the ambient telemetry context."""
        run = current_run()
        if run is None:
            return DEFAULT_CASSETTE_PROJECT, DEFAULT_CASSETTE_RUN
        return run.project, run.run_id

    def _replayed(self, request: CallRequest) -> CassetteEntry:
        """Serve a call from a cassette, recording it in memory but spending nothing.

        The spend cap is not consulted: a replayed call costs nothing, so refusing it
        would only stop a demo that was never going to spend.

        The served call is stamped `mode="replay"` rather than handed back with the mode
        it was recorded under. Without that, a replayed call is indistinguishable from a
        live one in the run record, and a KPI computed from a demo would be plausible and
        wrong. The cost figure is left exactly as recorded: it is what the call cost when
        it was really made, which is worth knowing — it is simply not money spent now.
        """
        project, run_id = self._run_identity()
        entry = self.replay.lookup(request, project=project, run_id=run_id)
        served = entry.response.model_copy(
            update={"call": entry.response.call.model_copy(update={"mode": "replay"})}
        )
        entry = entry.model_copy(update={"response": served})
        self.ledger.record(entry.response.call, persist=False)
        return entry

    def _maybe_record_cassette(self, request: CallRequest, response: CallResponse) -> None:
        """Append a live call to the run's cassette when recording."""
        if not self.replay.is_recording:
            return
        project, run_id = self._run_identity()
        self.replay.record(request, response, project=project, run_id=run_id)

    def _guard_spend_cap(self, model: str) -> None:
        """Refuse the call outright if today's spend has reached the cap."""
        today = self._now().astimezone(UTC).date()
        spent = self.ledger.total_for(today)
        if spent >= self.config.daily_spend_cap_usd:
            raise SpendCapExceeded(
                f"Daily spend cap reached: {spent} of {self.config.daily_spend_cap_usd} usd "
                f"already spent on {today.isoformat()}; refusing to call {model}."
            )

    def _with_backoff[T](self, operation: Callable[[], T]) -> T:
        """Run an operation, retrying transient faults with exponentially jittered waits."""
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return operation()
            except TRANSIENT_ERRORS as error:
                last_error = error
                if attempt == MAX_ATTEMPTS - 1:
                    break
                self._sleep(_sleep_for(attempt, self._rng))
        raise last_error if last_error else RouterError("retry loop ended without an error")

    def _record(
        self, response: ModelResponse, *, model: str, purpose: str, latency_ms: int
    ) -> ModelCall:
        """Price a response, append it to the ledger and return the record."""
        usage = getattr(response, "usage", None)
        call = ModelCall(
            provider=self._provider_of(model),
            model=model,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=_cached_tokens(response),
            cost_usd=self._price(response, model),
            latency_ms=latency_ms,
            timestamp=self._now(),
            purpose=purpose,
            # "record" and "live" both reached a provider and spent money; the distinction
            # is only that one was also written to a cassette.
            mode=self.replay.mode,
        )
        self.ledger.record(call)
        return call

    def _price(self, response: ModelResponse, model: str) -> Decimal:
        """Compute what a response cost, in usd, from the provider's own token counts."""
        try:
            cost = litellm.completion_cost(completion_response=response, model=model)
        except Exception as error:  # noqa: BLE001 - litellm raises bare Exception when unmapped
            raise PricingUnavailable(
                f"litellm could not price a call to {model!r}: {error}"
            ) from error
        return Decimal(str(cost))

    @staticmethod
    def _provider_of(model: str) -> str:
        """Read the provider off a model name, defaulting to what litellm infers."""
        if "/" in model:
            return model.split("/", 1)[0]
        return "anthropic" if "claude" in model.lower() else "unknown"

    @staticmethod
    def _text_of(response: ModelResponse) -> str:
        """Pull the assistant's text out of a completion response."""
        choices: Sequence[Any] | None = getattr(response, "choices", None)
        if not choices:
            return ""
        return getattr(getattr(choices[0], "message", None), "content", "") or ""


def calls_by_purpose(calls: Iterable[ModelCall]) -> dict[str, Decimal]:
    """Total spend per purpose label, in usd."""
    totals: dict[str, Decimal] = {}
    for call in calls:
        totals[call.purpose] = totals.get(call.purpose, Decimal("0")) + call.cost_usd
    return totals

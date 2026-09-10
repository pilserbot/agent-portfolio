"""Langfuse tracing for the router and the graph, with a no-op when it is not configured.

Wraps a call or a node in a Langfuse observation recording inputs, outputs, token counts,
cost and latency, and groups every observation of one run under a single trace keyed by
`run_id` and tagged with the project.

Observability is optional infrastructure, never a dependency of the work: with
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY absent this module is a no-op, and even when
configured a telemetry failure is swallowed rather than propagated. A missing dashboard
must never fail a bid.

Redaction works two ways: a value under a key that names a credential is replaced, and a
value shaped like one — an Anthropic key, a Langfuse key, a URL carrying userinfo — is
replaced wherever it appears, including inside a longer string. The residual limitation is
real and worth stating: a credential that matches none of those patterns and arrives with
no telling key name, such as a bare token passed positionally, is only length-truncated and
will otherwise be sent. Keep secrets in named arguments, and add a pattern here when a new
credential shape enters the project.

Deliberately does not: decide, retry, or alter what it observes — a traced function
returns exactly what it would have returned untraced, and an exception raised inside it
propagates unchanged. It does not flush on every call (the SDK batches in the background,
so tracing never blocks a run), and it does not send payloads verbatim: values are redacted
for secrets and truncated past TEXT_LIMIT characters before they leave the process.
"""

import functools
import logging
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spine.contracts import ModelCall, StepTrace

logger = logging.getLogger(__name__)

TEXT_LIMIT = 4000
MAX_DEPTH = 6

# Mapping keys whose values never leave this process. Matched case-insensitively as
# substrings, so "anthropic_api_key" and "X-Api-Token" are both caught.
SECRET_KEY_HINTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "auth",
    "credential",
    "cookie",
    "session_key",
)
REDACTED = "[redacted]"

# Secrets that arrive without a telling key name — a bare positional argument, or one
# quoted inside a longer sentence — are caught by shape instead. Matched anywhere in a
# string, not just when the string is the whole value.
SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic API key.
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    # Langfuse secret and public keys.
    re.compile(r"[sp]k-lf-[A-Za-z0-9_\-]+"),
    # Any URL carrying credentials, e.g. postgresql://user:password@host:5432/db. The
    # userinfo is required, so an ordinary https://host/path is left alone.
    re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s:/@]+:[^\s/@]+@\S+"),
)

type ObservationKind = Literal["span", "generation"]

__all__ = [
    "REDACTED",
    "TEXT_LIMIT",
    "LangfuseTracer",
    "Observation",
    "RunContext",
    "TelemetryConfig",
    "configure",
    "current_run",
    "flush",
    "get_tracer",
    "set_tracer",
    "redact",
    "scrub_secrets",
    "shutdown",
    "trace_run",
    "traced",
    "truncate",
]


class TelemetryConfig(BaseModel):
    """Credentials for the Langfuse project, if there are any."""

    model_config = ConfigDict(frozen=True)

    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "TelemetryConfig":
        """Read Langfuse credentials from the environment. Absent values stay None."""
        source = os.environ if env is None else env
        return cls(
            public_key=source.get("LANGFUSE_PUBLIC_KEY") or None,
            secret_key=source.get("LANGFUSE_SECRET_KEY") or None,
            host=source.get("LANGFUSE_HOST") or None,
        )

    @property
    def is_configured(self) -> bool:
        """Whether both keys are present. The host is optional: the SDK has a default."""
        return bool(self.public_key and self.secret_key)


class RunContext(BaseModel):
    """Which run the observations being recorded belong to."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    project: str


_current_run: ContextVar[RunContext | None] = ContextVar("spine_current_run", default=None)


def scrub_secrets(text: str) -> str:
    """Replace anything shaped like a credential, wherever it appears in the text."""
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def truncate(text: str, limit: int = TEXT_LIMIT) -> str:
    """Shorten text past the limit, leaving a marker that says what was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} of {len(text)} characters]"


def _is_secret_key(key: str) -> bool:
    """Whether a mapping key names something that must not leave the process."""
    lowered = key.lower()
    return any(hint in lowered for hint in SECRET_KEY_HINTS)


def redact(value: object, *, limit: int = TEXT_LIMIT, _depth: int = 0) -> object:
    """Strip secrets and truncate long text, recursively, before anything is sent.

    Values under a key that names a credential become `[redacted]`; strings longer than
    `limit` are truncated with a visible marker. Structures deeper than MAX_DEPTH are
    summarised rather than walked, so a cyclic or pathological payload cannot hang a run.
    """
    if _depth > MAX_DEPTH:
        return f"[depth limit reached: {type(value).__name__}]"
    if isinstance(value, str):
        return truncate(scrub_secrets(value), limit)
    if isinstance(value, bool | int | float | type(None)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, BaseModel):
        return redact(value.model_dump(mode="json"), limit=limit, _depth=_depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _is_secret_key(str(key))
                else redact(item, limit=limit, _depth=_depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [redact(item, limit=limit, _depth=_depth + 1) for item in value]
    return truncate(scrub_secrets(repr(value)), limit)


class Tracer:
    """The no-op tracer, and the base every real tracer narrows.

    Every method here does nothing and returns nothing, which is exactly the behaviour
    required when no credentials are configured.
    """

    @property
    def is_enabled(self) -> bool:
        """Whether observations actually reach a backend."""
        return False

    @contextmanager
    def run(self, context: RunContext) -> Iterator[None]:
        """Group everything recorded inside under one trace for this run."""
        yield

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        run: RunContext | None,
        payload: object,
    ) -> Iterator["Observation"]:
        """Open one observation. The yielded handle absorbs everything written to it.

        `payload` and anything passed to `finish` have already been through `redact`, so
        an implementation may forward them as they are.
        """
        yield Observation()

    def flush(self) -> None:
        """Send anything buffered. Safe to call when nothing is configured."""

    def shutdown(self) -> None:
        """Flush and stop. Safe to call when nothing is configured."""


class Observation:
    """A handle for finishing one observation. The no-op version discards everything."""

    def finish(
        self,
        *,
        output: object = None,
        call: ModelCall | None = None,
        step: StepTrace | None = None,
        latency_ms: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Record the result of the traced work."""


class LangfuseObservation(Observation):
    """Writes an observation's result to a live Langfuse span."""

    def __init__(self, span: object) -> None:
        """Wrap the SDK span this observation writes to."""
        self._span = span

    def finish(
        self,
        *,
        output: object = None,
        call: ModelCall | None = None,
        step: StepTrace | None = None,
        latency_ms: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Write outputs, usage, cost and latency onto the span, swallowing any failure."""
        fields: dict[str, object] = {"output": output}
        metadata: dict[str, object] = {}

        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms
        if call is not None:
            fields["model"] = call.model
            fields["usage_details"] = {
                "input": call.prompt_tokens,
                "output": call.completion_tokens,
                "cache_read_input_tokens": call.cached_tokens,
            }
            fields["cost_details"] = {"total": float(call.cost_usd)}
            metadata["provider"] = call.provider
            metadata["purpose"] = call.purpose
            metadata["latency_ms"] = call.latency_ms
        if step is not None:
            metadata["step_status"] = step.status
            if step.error:
                metadata["step_error"] = truncate(step.error)
        if error is not None:
            fields["level"] = "ERROR"
            fields["status_message"] = truncate(f"{type(error).__name__}: {error}")

        if metadata:
            fields["metadata"] = metadata

        # Telemetry must never be the reason a run fails.
        with suppress(Exception):
            self._span.update(**fields)


class LangfuseTracer(Tracer):
    """Records observations to a Langfuse project.

    State is genuinely held: one SDK client with its background exporter thread.
    """

    def __init__(self, client: object) -> None:
        """Wrap an already-constructed Langfuse client."""
        self._client = client

    @property
    def is_enabled(self) -> bool:
        """Observations reach a backend."""
        return True

    @property
    def client(self) -> object:
        """The underlying SDK client, for callers that need to read traces back."""
        return self._client

    @contextmanager
    def run(self, context: RunContext) -> Iterator[None]:
        """Tag everything recorded inside with the run's name and project."""
        from langfuse import propagate_attributes

        try:
            manager = propagate_attributes(
                trace_name=f"{context.project}:{context.run_id}",
                tags=[context.project],
                metadata={"run_id": context.run_id, "project": context.project},
                session_id=context.run_id,
            )
        except Exception:  # noqa: BLE001 - telemetry never breaks the caller
            logger.debug("langfuse: could not propagate run attributes", exc_info=True)
            yield
            return
        with manager:
            yield

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        run: RunContext | None,
        payload: object,
    ) -> Iterator[Observation]:
        """Open a Langfuse observation, pinned to the run's trace when there is one."""
        from langfuse.types import TraceContext

        trace_context = None
        if run is not None:
            # A deterministic id from the run id, so observations recorded days apart in
            # different processes still land on the same trace.
            with suppress(Exception):
                trace_context = TraceContext(
                    trace_id=type(self._client).create_trace_id(seed=run.run_id)
                )
        try:
            manager = self._client.start_as_current_observation(
                name=name,
                as_type=kind,
                input=payload,
                trace_context=trace_context,
            )
        except Exception:  # noqa: BLE001 - telemetry never breaks the caller
            logger.debug("langfuse: could not open observation %s", name, exc_info=True)
            yield Observation()
            return
        with manager as span:
            yield LangfuseObservation(span)

    def flush(self) -> None:
        """Send anything buffered, ignoring transport failures."""
        with suppress(Exception):
            self._client.flush()

    def shutdown(self) -> None:
        """Flush and stop the exporter, ignoring transport failures."""
        with suppress(Exception):
            self._client.shutdown()


_tracer: Tracer | None = None


def configure(config: TelemetryConfig | None = None) -> Tracer:
    """Build the process-wide tracer and return it.

    Returns the no-op tracer when the credentials are absent, or when constructing the
    client fails for any reason.
    """
    global _tracer
    settings = config or TelemetryConfig.from_env()
    if not settings.is_configured:
        logger.debug("langfuse: no credentials configured; telemetry is a no-op")
        _tracer = Tracer()
        return _tracer
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            host=settings.host,
        )
    except Exception:  # noqa: BLE001 - an unusable backend must not stop the project
        logger.warning("langfuse: client could not be created; telemetry is a no-op")
        _tracer = Tracer()
        return _tracer
    _tracer = LangfuseTracer(client)
    return _tracer


def get_tracer() -> Tracer:
    """The process-wide tracer, configured from the environment on first use."""
    return _tracer if _tracer is not None else configure()


def set_tracer(tracer: Tracer) -> None:
    """Replace the process-wide tracer. Used by tests to inject a recorder."""
    global _tracer
    _tracer = tracer


def current_run() -> RunContext | None:
    """The run observations are currently being attributed to, if any."""
    return _current_run.get()


@contextmanager
def trace_run(run_id: str, project: str) -> Iterator[RunContext]:
    """Attribute everything recorded inside to one run, under a single trace."""
    context = RunContext(run_id=run_id, project=project)
    token = _current_run.set(context)
    try:
        with get_tracer().run(context):
            yield context
    finally:
        _current_run.reset(token)


def _run_from(args: Sequence[object]) -> RunContext | None:
    """Find the run being processed: the ambient context, else a state argument.

    The fallback matters because LangGraph may run a node on a worker thread that never
    inherited the context variable. A graph state carries the run id itself, so grouping
    survives that.
    """
    ambient = _current_run.get()
    if ambient is not None:
        return ambient
    for candidate in args:
        run_id = getattr(candidate, "run_id", None)
        project = getattr(candidate, "project", None)
        if isinstance(run_id, str) and isinstance(project, str):
            return RunContext(run_id=run_id, project=project)
    return None


def _find_model_call(result: object, _depth: int = 0) -> ModelCall | None:
    """Pull a ModelCall out of a return value, however it was packaged."""
    if isinstance(result, ModelCall):
        return result
    if _depth > 2:
        return None
    if isinstance(result, tuple | list):
        for item in result:
            found = _find_model_call(item, _depth + 1)
            if found is not None:
                return found
    return None


def _find_step(result: object) -> StepTrace | None:
    """Pull the StepTrace a graph node recorded out of its state update."""
    if isinstance(result, StepTrace):
        return result
    if isinstance(result, Mapping):
        steps = result.get("steps")
        if isinstance(steps, Sequence):
            for item in reversed(list(steps)):
                if isinstance(item, StepTrace):
                    return item
    return None


def traced[**P, R](
    name: str,
    purpose: str = "",
    *,
    kind: ObservationKind = "span",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record one observation per call, with its inputs, outputs, usage, cost and latency.

    Works on a router method and on a graph node alike: the inputs are the call's
    arguments and the outputs whatever it returns, with a `ModelCall` or `StepTrace` in
    the result supplying model, tokens and cost. Pass `kind="generation"` for a model call
    so Langfuse prices it as one.

    `purpose` is the default label; a runtime `purpose=` keyword on the wrapped call wins,
    since the router takes its purpose per call rather than per function.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer()
            if not tracer.is_enabled:
                return fn(*args, **kwargs)

            call_purpose = kwargs.get("purpose") or purpose
            # `self` is the router instance, not an input worth sending.
            reportable = args[1:] if args and hasattr(args[0], "__dict__") else args
            payload = redact(
                {
                    "purpose": call_purpose,
                    "args": list(reportable),
                    "kwargs": dict(kwargs),
                }
            )

            started = time.monotonic()
            with tracer.observation(
                name=name, kind=kind, run=_run_from(args), payload=payload
            ) as observation:
                try:
                    result = fn(*args, **kwargs)
                except BaseException as error:
                    observation.finish(
                        latency_ms=int((time.monotonic() - started) * 1000), error=error
                    )
                    raise
                observation.finish(
                    output=redact(result),
                    call=_find_model_call(result),
                    step=_find_step(result),
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
                return result

        return wrapper

    return decorate


def flush() -> None:
    """Send anything the tracer has buffered."""
    get_tracer().flush()


def shutdown() -> None:
    """Flush and stop the tracer. Call once at process exit."""
    get_tracer().shutdown()

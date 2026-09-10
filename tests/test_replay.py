"""Unit tests for record-and-replay of model calls.

Nothing here reaches a network. The recording tests use a fake completion function; the
replay tests use one that raises if it is called at all, which is how "no network calls"
is proved rather than asserted.

Deliberately does not: exercise a real provider. Recording against one is what the demo
workflow does, not what a test should.
"""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from litellm.types.utils import ModelResponse, Usage
from pydantic import BaseModel, Field

from spine.replay import (
    CallRequest,
    CallResponse,
    Cassette,
    CassetteEntry,
    CassetteMiss,
    ReplayError,
    ReplaySession,
    list_cassettes,
    main,
    normalise_prompt,
    verify,
)
from spine.router import (
    DEFAULT_MODEL_LARGE,
    ModelCall,
    ReplayKindMismatch,
    Router,
    RouterConfig,
)
from spine.telemetry import trace_run

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
PROJECT = "ri05"
RUN = "run-1"


class Person(BaseModel):
    """A tiny schema for the structured-call tests."""

    name: str
    age: int = Field(ge=0)


def a_response(content: str = "the answer", prompt_tokens: int = 1000) -> ModelResponse:
    return ModelResponse(
        model=DEFAULT_MODEL_LARGE,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens, completion_tokens=200, total_tokens=prompt_tokens + 200
        ),
    )


class Counter:
    """A completion function that records how often it was called."""

    def __init__(self, content: str = "the answer") -> None:
        """Answer every call with the same content."""
        self.content = content
        self.calls = 0

    def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls += 1
        return a_response(self.content)


def explodes(**kwargs: object) -> ModelResponse:
    """A completion function that fails the test if anything calls it."""
    raise AssertionError("a provider call was made; replay must never reach the network")


def a_router(tmp_path: Path, completion_fn: object, session: ReplaySession) -> Router:
    return Router(
        RouterConfig(ledger_path=tmp_path / "ledger.json"),
        completion_fn=completion_fn,
        replay=session,
        now_fn=lambda: AT,
    )


def sessions(tmp_path: Path) -> tuple[ReplaySession, ReplaySession, ReplaySession]:
    root = tmp_path / "cassettes"
    return (
        ReplaySession(mode="live", root=root),
        ReplaySession(mode="record", root=root),
        ReplaySession(mode="replay", root=root),
    )


# --- mode selection ---------------------------------------------------------------------


def test_the_default_mode_is_live() -> None:
    session = ReplaySession.from_env({})

    assert session.mode == "live"
    assert session.is_live and not session.is_recording and not session.is_replaying


@pytest.mark.parametrize("mode", ["live", "record", "replay"])
def test_each_mode_is_read_from_the_environment(mode: str) -> None:
    assert ReplaySession.from_env({"SPINE_MODE": mode}).mode == mode


@pytest.mark.parametrize("raw", ["RECORD", " replay ", "Live"])
def test_the_mode_is_case_and_space_insensitive(raw: str) -> None:
    assert ReplaySession.from_env({"SPINE_MODE": raw}).mode == raw.strip().lower()


def test_an_unrecognised_mode_falls_back_to_live() -> None:
    # A typo must not silently disable the provider or start writing cassettes.
    assert ReplaySession.from_env({"SPINE_MODE": "reply"}).mode == "live"


def test_an_empty_mode_falls_back_to_live() -> None:
    assert ReplaySession.from_env({"SPINE_MODE": ""}).mode == "live"


def test_the_cassette_root_is_configurable() -> None:
    session = ReplaySession.from_env({"SPINE_CASSETTE_ROOT": "/tmp/elsewhere"})

    assert session.root == Path("/tmp/elsewhere")


def test_the_cassette_path_is_project_and_run_scoped(tmp_path: Path) -> None:
    session = ReplaySession(mode="record", root=tmp_path)

    assert session.cassette_path("ri05", "run-7") == tmp_path / "ri05" / "run-7.jsonl"


# --- the fingerprint --------------------------------------------------------------------


def a_request(**overrides: str) -> CallRequest:
    values = {
        "model": "a-model",
        "tier": "large",
        "purpose": "extract",
        "prompt": "Summarise the tender.",
        **overrides,
    }
    return CallRequest(**values)


def test_the_fingerprint_is_stable_across_calls() -> None:
    assert a_request().fingerprint == a_request().fingerprint


def test_the_fingerprint_ignores_cosmetic_whitespace() -> None:
    spaced = a_request(prompt="  Summarise   the\n\ttender.  ")

    assert spaced.fingerprint == a_request().fingerprint


def test_normalise_prompt_collapses_and_strips() -> None:
    assert normalise_prompt("  a   b \n c  ") == "a b c"


@pytest.mark.parametrize("field", ["model", "tier", "purpose", "prompt"])
def test_the_fingerprint_changes_with_each_input(field: str) -> None:
    assert a_request(**{field: "something-else"}).fingerprint != a_request().fingerprint


def test_case_is_significant() -> None:
    # Two prompts differing only in case may get different answers, so they are not one call.
    assert a_request(prompt="summarise the tender.").fingerprint != a_request().fingerprint


def test_the_fingerprint_is_short_and_hexadecimal() -> None:
    value = a_request().fingerprint

    assert len(value) == 16
    assert int(value, 16) >= 0


# --- record then replay -------------------------------------------------------------------


def test_record_then_replay_reproduces_identical_output(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    counter = Counter("the recorded answer")

    with trace_run(RUN, PROJECT):
        recorded_text, recorded_call = a_router(tmp_path, counter, recording).complete(
            "Summarise the tender.", purpose="extract"
        )

    assert counter.calls == 1

    with trace_run(RUN, PROJECT):
        replayed_text, replayed_call = a_router(tmp_path, explodes, replaying).complete(
            "Summarise the tender.", purpose="extract"
        )

    assert replayed_text == recorded_text == "the recorded answer"
    assert replayed_call == recorded_call
    assert replayed_call.cost_usd == recorded_call.cost_usd
    assert replayed_call.prompt_tokens == recorded_call.prompt_tokens


def test_replay_makes_no_provider_call_at_all(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    counter = Counter()

    with trace_run(RUN, PROJECT):
        a_router(tmp_path, counter, recording).complete("Summarise.", purpose="extract")
    assert counter.calls == 1

    with trace_run(RUN, PROJECT):
        # `explodes` raises on any call, so reaching a provider fails the test outright.
        a_router(tmp_path, explodes, replaying).complete("Summarise.", purpose="extract")


def test_replay_matches_a_reformatted_prompt(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)

    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter("answer"), recording).complete("A  b\nc", purpose="extract")

    with trace_run(RUN, PROJECT):
        text, _call = a_router(tmp_path, explodes, replaying).complete(
            "   A b   c   ", purpose="extract"
        )

    assert text == "answer"


def test_replay_finds_a_cassette_recorded_under_another_run_id(tmp_path: Path) -> None:
    # A demo replays under a fresh run id; the recording was made under a different one.
    _live, recording, replaying = sessions(tmp_path)

    with trace_run("run-recorded", PROJECT):
        a_router(tmp_path, Counter("answer"), recording).complete("Summarise.", purpose="extract")

    with trace_run("run-fresh", PROJECT):
        text, _call = a_router(tmp_path, explodes, replaying).complete(
            "Summarise.", purpose="extract"
        )

    assert text == "answer"


def test_a_structured_call_records_and_replays(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)

    def json_completion(**kwargs: object) -> ModelResponse:
        return a_response(json.dumps({"name": "Ada", "age": 36}))

    with trace_run(RUN, PROJECT):
        recorded, recorded_call = a_router(tmp_path, json_completion, recording).structured(
            "Extract the person.", Person, purpose="extract"
        )

    with trace_run(RUN, PROJECT):
        replayed, replayed_call = a_router(tmp_path, explodes, replaying).structured(
            "Extract the person.", Person, purpose="extract"
        )

    assert isinstance(replayed, Person)
    # Compared by value: instructor attaches the raw response as a private attribute, so
    # the object it returns is never == to one freshly validated from JSON.
    assert replayed.model_dump() == recorded.model_dump() == {"name": "Ada", "age": 36}
    assert replayed_call == recorded_call


def test_a_text_recording_cannot_serve_a_structured_call(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)

    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter("plain text"), recording).complete("Ask.", purpose="extract")

    with trace_run(RUN, PROJECT), pytest.raises(ReplayKindMismatch, match="structured"):
        a_router(tmp_path, explodes, replaying).structured("Ask.", Person, purpose="extract")


def test_the_cassette_lands_where_the_project_and_run_say(tmp_path: Path) -> None:
    _live, recording, _replaying = sessions(tmp_path)

    with trace_run("run-7", "ri05"):
        a_router(tmp_path, Counter(), recording).complete("Summarise.", purpose="extract")

    expected = tmp_path / "cassettes" / "ri05" / "run-7.jsonl"
    assert expected.exists()
    assert len(expected.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_a_recorded_entry_carries_tokens_and_cost(tmp_path: Path) -> None:
    _live, recording, _replaying = sessions(tmp_path)

    with trace_run(RUN, PROJECT):
        _text, call = a_router(tmp_path, Counter(), recording).complete(
            "Summarise.", purpose="extract"
        )

    cassette = recording.cassette_for(PROJECT, RUN)
    entry = cassette.entries[0]
    assert entry.response.call.prompt_tokens == call.prompt_tokens
    assert entry.response.call.completion_tokens == call.completion_tokens
    assert entry.response.call.cost_usd == call.cost_usd
    assert entry.request.model == DEFAULT_MODEL_LARGE
    assert entry.request.purpose == "extract"


# --- live mode is untouched ----------------------------------------------------------------


def test_live_mode_calls_the_provider_and_writes_no_cassette(tmp_path: Path) -> None:
    live, _recording, _replaying = sessions(tmp_path)
    counter = Counter()

    with trace_run(RUN, PROJECT):
        text, _call = a_router(tmp_path, counter, live).complete("Summarise.", purpose="extract")

    assert counter.calls == 1
    assert text == "the answer"
    assert not (tmp_path / "cassettes").exists()


def test_record_mode_still_calls_the_provider(tmp_path: Path) -> None:
    _live, recording, _replaying = sessions(tmp_path)
    counter = Counter()

    with trace_run(RUN, PROJECT):
        a_router(tmp_path, counter, recording).complete("Summarise.", purpose="extract")

    assert counter.calls == 1, "recording is live behaviour plus a write"


# --- a miss says what to do -------------------------------------------------------------------


def test_a_missing_fingerprint_raises_a_clear_error(tmp_path: Path) -> None:
    _live, _recording, replaying = sessions(tmp_path)

    with trace_run(RUN, PROJECT), pytest.raises(CassetteMiss) as excinfo:
        a_router(tmp_path, explodes, replaying).complete("Never recorded.", purpose="extract")

    message = str(excinfo.value)
    expected = CallRequest(
        model=DEFAULT_MODEL_LARGE, tier="large", purpose="extract", prompt="Never recorded."
    ).fingerprint
    assert expected in message, "the error names the fingerprint that was wanted"
    assert "purpose='extract'" in message, "and the inputs that produced it"
    assert "SPINE_MODE=record" in message, "and how to fix it"


def test_a_miss_lists_where_it_looked(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter(), recording).complete("Recorded.", purpose="extract")

    with trace_run(RUN, PROJECT), pytest.raises(CassetteMiss, match=r"run-1\.jsonl"):
        a_router(tmp_path, explodes, replaying).complete("Not recorded.", purpose="extract")


def test_a_miss_with_no_cassettes_at_all_says_the_file_is_missing(tmp_path: Path) -> None:
    # "Nothing was ever recorded" and "recorded, but not this call" need different fixes.
    _live, _recording, replaying = sessions(tmp_path)

    with trace_run(RUN, PROJECT), pytest.raises(CassetteMiss, match=r"run-1\.jsonl \(missing\)"):
        a_router(tmp_path, explodes, replaying).complete("Nothing here.", purpose="extract")


def test_a_miss_against_an_existing_cassette_is_not_marked_missing(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter(), recording).complete("Recorded.", purpose="extract")

    with trace_run(RUN, PROJECT), pytest.raises(CassetteMiss) as excinfo:
        a_router(tmp_path, explodes, replaying).complete("Not recorded.", purpose="extract")

    assert "(missing)" not in str(excinfo.value)


# --- the ledger and the spend cap under replay --------------------------------------------------


def test_a_replayed_call_is_in_the_run_record_but_not_the_persisted_spend(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter(), recording).complete("Summarise.", purpose="extract")

    ledger_path = tmp_path / "replay-ledger.json"
    router = Router(
        RouterConfig(ledger_path=ledger_path),
        completion_fn=explodes,
        replay=replaying,
        now_fn=lambda: AT,
    )
    with trace_run(RUN, PROJECT):
        _text, call = router.complete("Summarise.", purpose="extract")

    assert router.ledger.calls == (call,), "the run record still shows what it would have cost"
    assert not ledger_path.exists(), "but no spend was written against the daily cap"
    assert router.ledger.total_for(AT.date()) == Decimal("0")


def test_replay_is_not_blocked_by_an_exhausted_spend_cap(tmp_path: Path) -> None:
    _live, recording, replaying = sessions(tmp_path)
    with trace_run(RUN, PROJECT):
        a_router(tmp_path, Counter(), recording).complete("Summarise.", purpose="extract")

    router = Router(
        RouterConfig(ledger_path=tmp_path / "capped.json", daily_spend_cap_usd=Decimal("0")),
        completion_fn=explodes,
        replay=replaying,
        now_fn=lambda: AT,
    )
    with trace_run(RUN, PROJECT):
        text, _call = router.complete("Summarise.", purpose="extract")

    assert text == "the answer", "a cap on money cannot stop a demo that spends none"


# --- the cassette file ------------------------------------------------------------------------


def an_entry(prompt: str = "Summarise.", fingerprint: str | None = None) -> CassetteEntry:
    request = a_request(prompt=prompt)
    call = ModelCall(
        provider="anthropic",
        model="a-model",
        prompt_tokens=10,
        completion_tokens=2,
        cached_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ms=5,
        timestamp=AT,
        purpose="extract",
    )
    return CassetteEntry(
        fingerprint=fingerprint or request.fingerprint,
        request=request,
        response=CallResponse(text="answer", call=call),
        recorded_at=AT,
    )


def test_a_missing_cassette_reads_as_empty(tmp_path: Path) -> None:
    assert Cassette(tmp_path / "absent.jsonl").entries == ()


def test_an_unparsable_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(an_entry().model_dump_json() + "\n{ not json\n", encoding="utf-8")

    assert len(Cassette(path).entries) == 1


def test_appending_survives_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "c.jsonl"
    Cassette(path).append(an_entry("first"))
    Cassette(path).append(an_entry("second"))

    assert len(Cassette(path).entries) == 2


# --- the command line ---------------------------------------------------------------------------


def test_list_reports_cassettes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    Cassette(tmp_path / "ri05" / "run-1.jsonl").append(an_entry())

    assert main(["list", "--root", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "ri05" in output
    assert "run-1" in output
    assert "1 cassette(s), 1 recorded call(s)." in output


def test_list_on_an_empty_root_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--root", str(tmp_path / "nothing")]) == 0
    assert "No cassettes found." in capsys.readouterr().out


def test_list_summarises_cost(tmp_path: Path) -> None:
    Cassette(tmp_path / "ri05" / "run-1.jsonl").append(an_entry())

    summaries = list_cassettes(tmp_path)

    assert len(summaries) == 1
    assert summaries[0].project == "ri05"
    assert summaries[0].run_id == "run-1"
    assert summaries[0].total_cost_usd == pytest.approx(0.001)


def test_verify_passes_a_good_cassette(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "c.jsonl"
    Cassette(path).append(an_entry("one"))
    Cassette(path).append(an_entry("two"))

    assert main(["verify", str(path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_catches_a_fingerprint_that_does_not_match_its_request(tmp_path: Path) -> None:
    # Recomputing rather than trusting is what catches a hand-edited or stale cassette.
    path = tmp_path / "c.jsonl"
    Cassette(path).append(an_entry(fingerprint="0000000000000000"))

    report = verify(path)

    assert report.mismatched_fingerprints == ["0000000000000000"]
    assert not report.is_valid


def test_verify_catches_a_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    Cassette(path).append(an_entry("same"))
    Cassette(path).append(an_entry("same"))

    report = verify(path)

    assert len(report.duplicate_fingerprints) == 1
    assert not report.is_valid


def test_verify_catches_an_unreadable_line(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(an_entry().model_dump_json() + "\nnot json at all\n", encoding="utf-8")

    report = verify(path)

    assert report.unreadable_lines == [2]
    assert report.entries == 1
    assert not report.is_valid


def test_verify_exits_non_zero_on_a_bad_cassette(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    Cassette(path).append(an_entry(fingerprint="0000000000000000"))

    assert main(["verify", str(path)]) == 1


def test_verify_on_a_missing_file_reports_rather_than_crashes(tmp_path: Path) -> None:
    assert main(["verify", str(tmp_path / "absent.jsonl")]) == 2

    with pytest.raises(ReplayError, match="not found"):
        verify(tmp_path / "absent.jsonl")


def test_a_recorded_cassette_verifies_clean(tmp_path: Path) -> None:
    # The real round trip: what the router writes must pass the checker unaided.
    _live, recording, _replaying = sessions(tmp_path)
    with trace_run(RUN, PROJECT):
        router = a_router(tmp_path, Counter(), recording)
        router.complete("First question.", purpose="extract")
        router.complete("Second question.", purpose="extract")

    report = verify(recording.cassette_path(PROJECT, RUN))

    assert report.entries == 2
    assert report.is_valid

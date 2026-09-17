"""Persisting a live run's ledger before anything else can fail.

Shared by every integration test that spends money. An earlier version of the extraction
test made sixteen calls and threw the result away because the reporting object it was
building raised on an unrelated field, so this is deliberately dependency-light: it sums
`ModelCall` records with plain arithmetic, constructs nothing that can raise, writes a JSON
snapshot to a durable directory, and prints the per-purpose totals and the cost to stdout.

A partial run that died on call nine still records the eight calls it paid for. It lives
here rather than in one test module because the second live pass has exactly the same
obligation and copying it would let the two drift.

Deliberately does not: assert anything, or go through `spine.kpi`, `AgentRun` or any other
reporting type. The whole reason it exists is that a reporting type raised and took a
paid-for measurement with it.
"""

import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from spine.contracts import CacheActivity, ModelCall
    from spine.router import Router

# Where a snapshot is written. Durable rather than a pytest tmp_path, because the point of
# the file is to outlive the process that made the calls; CI uploads the directory.
ARTIFACT_DIR_ENV = "RI05_ARTIFACT_DIR"
DEFAULT_ARTIFACT_DIR = Path(".artifacts")


class CallRecord(BaseModel):
    """One call, kept individually because the totals hid the thing worth seeing.

    A per-call row is what distinguishes "each page cost what its text costs" from "a few
    pages were re-sent several times". Summed figures cannot tell those apart, and the first
    live run could not answer the question because only totals were kept.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    purpose: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cost_usd: Decimal
    latency_ms: int = Field(ge=0)


class PurposeTotals(BaseModel):
    """What one purpose label consumed, summed straight from the ledger."""

    model_config = ConfigDict(frozen=True)

    purpose: str
    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cost_usd: Decimal


class LedgerSnapshot(BaseModel):
    """Everything the router's ledger knows, captured before anything else can fail.

    Built from `ModelCall` records with plain sums and nothing else. It deliberately does
    not go through `spine.kpi`, `AgentRun` or any other reporting type: the whole reason
    this exists is that the reporting object raised and took a paid-for measurement with it.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    taken_at: datetime
    calls: int = Field(ge=0)
    billed_calls: int = Field(ge=0)
    replayed_calls: int = Field(ge=0)
    models: list[str] = Field(default_factory=list)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cost_usd: Decimal
    by_purpose: list[PurposeTotals] = Field(default_factory=list)
    per_call: list[CallRecord] = Field(default_factory=list)

    @property
    def cache_activity(self) -> "CacheActivity":
        """Which half of prompt caching this pass did, across every call in it.

        The same four-way branch as `ModelCall.cache_activity`, restated rather than
        imported for the reason the module docstring gives: this file must stay able to
        report a paid-for run even if something in `spine` raises. Four lines of arithmetic
        is the price of that, and the wording of the conclusion lives in `describe_cache`.
        """
        if self.cache_creation_tokens and self.cache_read_tokens:
            return "writing_and_reading"
        if self.cache_creation_tokens:
            return "writing"
        if self.cache_read_tokens:
            return "reading"
        return "none"

    def describe_cache(self) -> str:
        """One line naming what the cache actually did, with no verdict attached.

        Says which of the two things happened. It deliberately does not say that "caching
        is engaged", because that sentence is true of a pass that only ever wrote — and a
        pass that only ever writes is paying the ~1.25x write premium on every call and
        collecting the ~0.1x read discount on none of them. The earlier version of this
        line drew exactly that conclusion from one conflated counter.
        """
        writes, reads = self.cache_creation_tokens, self.cache_read_tokens
        verdicts = {
            "none": (
                "no cache activity of either kind — nothing was marked cacheable, or every "
                "marked prefix fell below the model's minimum cacheable length"
            ),
            "writing": (
                "WRITES ONLY, NO READS — every call paid the write premium and none "
                "collected the read discount, which is what a cacheable block that varies "
                "on every call looks like"
            ),
            "reading": "reads with no new writes — served entirely from an existing cache",
            "writing_and_reading": (
                "both — a prefix was written and later re-read. Whether that is a saving "
                "depends on why the prefix came round again: reuse across calls is the "
                "point of caching, but the same page re-sent inside one call is a discount "
                "on work that should not have happened. The counts do not say which"
            ),
        }
        return (
            f"CACHE: {writes} token(s) written, {reads} token(s) read — "
            f"{verdicts[self.cache_activity]}."
        )

    def describe(self) -> str:
        """The snapshot as a block of stdout, so it lands in the CI log unaided."""
        lines = [
            "",
            f"### Ledger snapshot — {self.label}",
            "",
            f"{self.calls} call(s) ({self.billed_calls} billed, {self.replayed_calls} "
            f"replayed) to {', '.join(self.models) or '(none)'} at "
            f"{self.taken_at.isoformat(timespec='seconds')}.",
            "",
            "| Purpose | Calls | Prompt | Completion | Cache writes | Cache reads | Cost (USD) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| `{item.purpose}` | {item.calls} | {item.prompt_tokens} | "
            f"{item.completion_tokens} | {item.cache_creation_tokens} | "
            f"{item.cache_read_tokens} | {item.cost_usd:.6f} |"
            for item in self.by_purpose
        ]
        lines += [
            f"| **total** | **{self.calls}** | **{self.prompt_tokens}** | "
            f"**{self.completion_tokens}** | **{self.cache_creation_tokens}** | "
            f"**{self.cache_read_tokens}** | **{self.cost_usd:.6f}** |",
            "",
            f"LEDGER TOTAL: ${self.cost_usd:.6f} over {self.prompt_tokens} prompt and "
            f"{self.completion_tokens} completion tokens.",
            self.describe_cache(),
            "",
            "| # | Purpose | Prompt | Completion | Cache writes | Cache reads | Cost (USD) | ms |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {call.index} | `{call.purpose}` | {call.prompt_tokens} | "
            f"{call.completion_tokens} | {call.cache_creation_tokens} | "
            f"{call.cache_read_tokens} | {call.cost_usd:.6f} | {call.latency_ms} |"
            for call in self.per_call
        ]
        lines.append("")
        return "\n".join(lines)


def snapshot_of(calls: "tuple[ModelCall, ...] | list[ModelCall]", *, label: str) -> LedgerSnapshot:
    """Sum a ledger's calls into a snapshot. Arithmetic only — nothing here can raise."""
    grouped: dict[str, list[ModelCall]] = {}
    for call in calls:
        grouped.setdefault(call.purpose, []).append(call)

    return LedgerSnapshot(
        label=label,
        taken_at=datetime.now(UTC),
        calls=len(calls),
        billed_calls=sum(1 for call in calls if call.was_billed),
        replayed_calls=sum(1 for call in calls if not call.was_billed),
        models=sorted({call.model for call in calls}),
        prompt_tokens=sum(call.prompt_tokens for call in calls),
        completion_tokens=sum(call.completion_tokens for call in calls),
        cache_creation_tokens=sum(call.cache_creation_tokens for call in calls),
        cache_read_tokens=sum(call.cache_read_tokens for call in calls),
        cost_usd=sum((call.cost_usd for call in calls), Decimal("0")),
        per_call=[
            CallRecord(
                index=index,
                purpose=call.purpose,
                model=call.model,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                cache_creation_tokens=call.cache_creation_tokens,
                cache_read_tokens=call.cache_read_tokens,
                cost_usd=call.cost_usd,
                latency_ms=call.latency_ms,
            )
            for index, call in enumerate(calls)
        ],
        by_purpose=[
            PurposeTotals(
                purpose=purpose,
                calls=len(group),
                prompt_tokens=sum(call.prompt_tokens for call in group),
                completion_tokens=sum(call.completion_tokens for call in group),
                cache_creation_tokens=sum(call.cache_creation_tokens for call in group),
                cache_read_tokens=sum(call.cache_read_tokens for call in group),
                cost_usd=sum((call.cost_usd for call in group), Decimal("0")),
            )
            for purpose, group in sorted(grouped.items())
        ],
    )


def artifact_dir() -> Path:
    """Where snapshots are written, created if it is not there."""
    directory = Path(os.environ.get(ARTIFACT_DIR_ENV) or DEFAULT_ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def persist_ledger(router: "Router", *, label: str) -> LedgerSnapshot:
    """Write and print the ledger. Called in a `finally`, so it must never raise.

    The money is already spent by the time this runs. Anything that could throw here — a
    reporting model, a schema that moved, a directory that is not writable — would throw the
    measurement away a second time, so the only failure this tolerates is one it swallows
    after having printed.
    """
    snapshot = snapshot_of(router.ledger.calls, label=label)
    print(snapshot.describe())
    try:
        path = artifact_dir() / f"ledger_{label}_{snapshot.taken_at:%Y%m%dT%H%M%SZ}.json"
        path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Ledger snapshot written to {path}")
    except OSError as error:  # pragma: no cover - only on an unwritable filesystem
        print(f"Could not write the ledger snapshot ({error}); the figures above stand.")
    return snapshot

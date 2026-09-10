"""Record and replay of model calls, so a demo can run with no API key and no network.

In `record` mode every router call is appended to a cassette under
`demo/cassettes/<project>/<run_id>.jsonl`, carrying the request fingerprint, the response
and the call's tokens and cost. In `replay` mode the router answers from the cassette and
never reaches a provider. In `live` mode — the default — none of this is in the way.

Cassettes are meant to be committed: they are what lets a deployed demo, or a reviewer with
a fresh clone, run the real code paths end to end with no credentials at all.

Deliberately does not: fake a model. A replayed answer is one a real model actually gave,
recorded verbatim; nothing here invents a response, and a fingerprint with no recorded
answer is an error rather than a plausible substitute. It also does not replay anything but
model calls — a database or an HTTP call elsewhere in a run is not covered — and it does
not redact: a cassette is a verbatim record, so record only what may be committed.
"""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import ModelCall

type ReplayMode = Literal["live", "record", "replay"]

DEFAULT_CASSETTE_ROOT = Path("demo/cassettes")
FINGERPRINT_LENGTH = 16
_WHITESPACE = re.compile(r"\s+")

__all__ = [
    "DEFAULT_CASSETTE_ROOT",
    "CallRequest",
    "CallResponse",
    "Cassette",
    "CassetteMiss",
    "CassetteEntry",
    "CassetteSummary",
    "ReplayError",
    "ReplayMode",
    "ReplaySession",
    "VerifyReport",
    "fingerprint_of",
    "normalise_prompt",
]


class ReplayError(Exception):
    """Base class for every failure this module raises on purpose."""


class CassetteMiss(ReplayError):
    """Replay was asked for a call the cassette does not hold.

    Carries the fingerprint and the inputs that produced it, because the whole point of a
    miss is to tell you exactly what to record.
    """


def normalise_prompt(prompt: str) -> str:
    """Collapse whitespace so cosmetic reformatting does not change a fingerprint.

    Runs of whitespace become one space and the ends are stripped. Case is left alone: two
    prompts differing only in case are different prompts, and may get different answers.
    """
    return _WHITESPACE.sub(" ", prompt).strip()


class CallRequest(BaseModel):
    """What identifies one model call, for matching a recording to a replay."""

    model_config = ConfigDict(frozen=True)

    model: str
    tier: str
    purpose: str
    prompt: str

    @property
    def fingerprint(self) -> str:
        """The stable id of this request."""
        return fingerprint_of(self)


def fingerprint_of(request: CallRequest) -> str:
    """Hash model, tier, purpose and the normalised prompt into a stable short id.

    Stable across processes and machines: the input is canonical JSON with sorted keys, so
    it does not depend on dict ordering, and the prompt is normalised first.
    """
    canonical = json.dumps(
        {
            "model": request.model,
            "tier": request.tier,
            "purpose": request.purpose,
            "prompt": normalise_prompt(request.prompt),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


class CallResponse(BaseModel):
    """What a model call returned, and what it cost.

    `text` holds a `complete` reply; `structured_json` holds the JSON of a `structured`
    one. Exactly one of them is set.
    """

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    structured_json: str | None = None
    call: ModelCall


class CassetteEntry(BaseModel):
    """One recorded call: what was asked, what came back, and when."""

    model_config = ConfigDict(frozen=True)

    fingerprint: str
    request: CallRequest
    response: CallResponse
    recorded_at: datetime


class CassetteSummary(BaseModel):
    """What one cassette file holds, for listing."""

    model_config = ConfigDict(frozen=True)

    path: Path
    project: str
    run_id: str
    entries: int
    total_cost_usd: float = Field(description="Sum of the recorded call costs, in usd.")


class VerifyReport(BaseModel):
    """The outcome of checking a cassette."""

    model_config = ConfigDict(frozen=True)

    path: Path
    entries: int
    unreadable_lines: list[int] = Field(default_factory=list)
    mismatched_fingerprints: list[str] = Field(default_factory=list)
    duplicate_fingerprints: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the cassette is complete and self-consistent."""
        return not (
            self.unreadable_lines or self.mismatched_fingerprints or self.duplicate_fingerprints
        )


class Cassette:
    """One JSONL file of recorded calls.

    State is genuinely held: the entries loaded from disk, and the file they append to.
    """

    def __init__(self, path: Path) -> None:
        """Open a cassette at a path, which need not exist yet."""
        self.path = path
        self._entries: list[CassetteEntry] = []
        self._by_fingerprint: dict[str, CassetteEntry] = {}
        self._loaded = False

    @property
    def entries(self) -> tuple[CassetteEntry, ...]:
        """Every entry in the file, in recorded order."""
        self._ensure_loaded()
        return tuple(self._entries)

    def _ensure_loaded(self) -> None:
        """Read the file once, tolerating its absence."""
        if self._loaded:
            return
        self._loaded = True
        for entry in read_entries(self.path):
            self._entries.append(entry)
            self._by_fingerprint.setdefault(entry.fingerprint, entry)

    def find(self, fingerprint: str) -> CassetteEntry | None:
        """The first entry recorded for a fingerprint, or None."""
        self._ensure_loaded()
        return self._by_fingerprint.get(fingerprint)

    def append(self, entry: CassetteEntry) -> CassetteEntry:
        """Add an entry to the file and to the in-memory index."""
        self._ensure_loaded()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        self._entries.append(entry)
        self._by_fingerprint.setdefault(entry.fingerprint, entry)
        return entry


def read_entries(path: Path) -> Iterator[CassetteEntry]:
    """Yield the entries of a cassette, skipping what cannot be parsed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            yield CassetteEntry.model_validate_json(line)
        except ValueError:
            continue


class ReplaySession(BaseModel):
    """Which mode the process is in, and where its cassettes live."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode: ReplayMode = "live"
    root: Path = DEFAULT_CASSETTE_ROOT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ReplaySession":
        """Read SPINE_MODE, falling back to live when it is unset or unrecognised."""
        source = os.environ if env is None else env
        raw = (source.get("SPINE_MODE") or "live").strip().lower()
        mode: ReplayMode = raw if raw in ("live", "record", "replay") else "live"
        root = Path(source.get("SPINE_CASSETTE_ROOT") or DEFAULT_CASSETTE_ROOT)
        return cls(mode=mode, root=root)

    @property
    def is_live(self) -> bool:
        """Whether calls go to a provider without being recorded."""
        return self.mode == "live"

    @property
    def is_recording(self) -> bool:
        """Whether calls go to a provider and are written to a cassette."""
        return self.mode == "record"

    @property
    def is_replaying(self) -> bool:
        """Whether calls are served from a cassette and never leave the process."""
        return self.mode == "replay"

    def cassette_path(self, project: str, run_id: str) -> Path:
        """Where one run's cassette lives."""
        return self.root / project / f"{run_id}.jsonl"

    def cassette_for(self, project: str, run_id: str) -> Cassette:
        """The cassette for one run."""
        return Cassette(self.cassette_path(project, run_id))

    def cassettes_for_project(self, project: str) -> list[Cassette]:
        """Every cassette recorded for a project, oldest path first."""
        directory = self.root / project
        if not directory.is_dir():
            return []
        return [Cassette(path) for path in sorted(directory.glob("*.jsonl"))]

    def lookup(self, request: CallRequest, *, project: str, run_id: str) -> CassetteEntry:
        """Find a recorded answer, or say precisely what is missing and how to get it.

        Falls back to the project's other cassettes, so a demo replays even when the run
        id differs from the one that was recorded — which it almost always will.
        """
        wanted = request.fingerprint
        candidates = [self.cassette_for(project, run_id), *self.cassettes_for_project(project)]
        searched: list[Path] = []
        for cassette in candidates:
            if cassette.path in searched:
                continue
            searched.append(cassette.path)
            found = cassette.find(wanted)
            if found is not None:
                return found
        looked_in = ", ".join(
            f"{path}{'' if path.is_file() else ' (missing)'}" for path in searched
        )
        raise CassetteMiss(
            f"No recorded call for fingerprint {wanted} "
            f"(model={request.model!r}, tier={request.tier!r}, purpose={request.purpose!r}). "
            f"Searched: {looked_in}. "
            f"Record it with SPINE_MODE=record, or check the prompt still matches: the "
            f"fingerprint covers model, tier, purpose and the whitespace-normalised prompt."
        )

    def record(
        self,
        request: CallRequest,
        response: CallResponse,
        *,
        project: str,
        run_id: str,
        now: datetime | None = None,
    ) -> CassetteEntry:
        """Append one call to the run's cassette."""
        entry = CassetteEntry(
            fingerprint=request.fingerprint,
            request=request,
            response=response,
            recorded_at=now or datetime.now(UTC),
        )
        return self.cassette_for(project, run_id).append(entry)


def summarise(path: Path) -> CassetteSummary:
    """Describe one cassette file without loading it twice."""
    entries = list(read_entries(path))
    return CassetteSummary(
        path=path,
        project=path.parent.name,
        run_id=path.stem,
        entries=len(entries),
        total_cost_usd=float(sum(entry.response.call.cost_usd for entry in entries)),
    )


def list_cassettes(root: Path = DEFAULT_CASSETTE_ROOT) -> list[CassetteSummary]:
    """Every cassette under a root, sorted by path."""
    if not root.is_dir():
        return []
    return [summarise(path) for path in sorted(root.rglob("*.jsonl"))]


def verify(path: Path) -> VerifyReport:
    """Check a cassette is readable, self-consistent and free of duplicates.

    Recomputes each fingerprint from the stored request rather than trusting the recorded
    one, so a cassette edited by hand, or written by an older fingerprint rule, is caught.
    """
    unreadable: list[int] = []
    mismatched: list[str] = []
    seen: dict[str, int] = {}
    entries = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, IsADirectoryError) as error:
        raise ReplayError(f"cassette not found: {path}") from error

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = CassetteEntry.model_validate_json(line)
        except ValueError:
            unreadable.append(number)
            continue
        entries += 1
        if fingerprint_of(entry.request) != entry.fingerprint:
            mismatched.append(entry.fingerprint)
        seen[entry.fingerprint] = seen.get(entry.fingerprint, 0) + 1

    return VerifyReport(
        path=path,
        entries=entries,
        unreadable_lines=unreadable,
        mismatched_fingerprints=mismatched,
        duplicate_fingerprints=sorted(key for key, count in seen.items() if count > 1),
    )


def _render_list(summaries: Sequence[CassetteSummary]) -> str:
    """Render the cassette listing as a table."""
    if not summaries:
        return "No cassettes found."
    lines = [f"{'PROJECT':<16} {'RUN':<28} {'CALLS':>6} {'COST_USD':>10}  PATH"]
    for item in summaries:
        lines.append(
            f"{item.project:<16} {item.run_id:<28} {item.entries:>6} "
            f"{item.total_cost_usd:>10.5f}  {item.path}"
        )
    total = sum(item.entries for item in summaries)
    lines.append(f"{len(summaries)} cassette(s), {total} recorded call(s).")
    return "\n".join(lines)


def _render_verify(report: VerifyReport) -> str:
    """Render a verification report."""
    lines = [f"{report.path}: {report.entries} entry(ies)"]
    for number in report.unreadable_lines:
        lines.append(f"  unreadable line {number}")
    for fingerprint in report.mismatched_fingerprints:
        lines.append(f"  fingerprint does not match its request: {fingerprint}")
    for fingerprint in report.duplicate_fingerprints:
        lines.append(f"  duplicate fingerprint: {fingerprint}")
    lines.append("  OK" if report.is_valid else "  FAILED")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m spine.replay",
        description="Inspect and check the cassettes a demo replays from.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    listing = subcommands.add_parser("list", help="List cassettes and their recorded calls.")
    listing.add_argument(
        "--root", type=Path, default=DEFAULT_CASSETTE_ROOT, help="Directory to search."
    )

    checking = subcommands.add_parser("verify", help="Check one cassette for completeness.")
    checking.add_argument("cassette", type=Path, help="Path of the .jsonl cassette.")

    args = parser.parse_args(argv)

    if args.command == "list":
        print(_render_list(list_cassettes(args.root)))
        return 0

    try:
        report = verify(args.cassette)
    except ReplayError as error:
        print(str(error))
        return 2
    print(_render_verify(report))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

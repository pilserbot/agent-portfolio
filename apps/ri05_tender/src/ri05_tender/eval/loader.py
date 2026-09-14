"""Reading a tender's answer key into typed gold items.

`load_gold(package)` takes the `TenderPackage` the tender loader produced and reads
`package.gold_path`. It refuses a package with no answer key rather than returning nothing:
scoring an unlabelled tender is a caller mistake, and an empty run would report a recall of
zero over nothing, which reads like a result.

Every line is validated. A malformed one stops the load naming the file, the line number and
what was wrong — a gold set that silently dropped items would change every metric computed
from it, in the flattering direction, since a dropped item is one the system can no longer
be seen to have missed.

Items marked `scored: false` load like any other and are handed back separately. They are
excluded from every denominator, and the count of them is part of the report: a shrinking
denominator nobody mentioned is how a score improves without the system improving.

Deliberately does not: call a model, reach a network, or judge anything. It reads one file
into records. It is also the only module here that opens the gold set — the tender loader
never does, and nothing downstream needs to.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ri05_tender.eval.models import GoldItem
from ri05_tender.tender.models import TenderPackage

__all__ = [
    "GoldSet",
    "GoldSetError",
    "NoGoldSetError",
    "load_gold",
    "load_gold_file",
]


class GoldSetError(Exception):
    """A gold set could not be read, or is not internally consistent."""


class NoGoldSetError(GoldSetError):
    """Scoring was asked for against a tender that carries no answer key.

    Its own class because it is not a bad file — it is a caller mistake, and the caller
    usually wants to catch exactly this one and say "this tender is not labelled" rather
    than "the gold set is broken".
    """


class GoldSet(BaseModel):
    """A tender's answer key: what is scored, and what was deliberately set aside."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    source_path: Path
    scored: list[GoldItem] = Field(default_factory=list)
    excluded: list[GoldItem] = Field(
        default_factory=list,
        description="Items carrying scored=false. Loaded, never counted, always reported.",
    )

    @property
    def all_items(self) -> list[GoldItem]:
        """Every item the file held, in file order across both lists."""
        return sorted([*self.scored, *self.excluded], key=lambda item: item.id)

    def __len__(self) -> int:
        """How many items count toward a metric."""
        return len(self.scored)

    def item(self, gold_id: str) -> GoldItem | None:
        """One scored item by id, or None."""
        for candidate in self.scored:
            if candidate.id == gold_id:
                return candidate
        return None


def load_gold(package: TenderPackage) -> GoldSet:
    """Read the answer key beside a loaded tender.

    Raises `NoGoldSetError` when the package has none. That is the production shape of a
    tender, so the error says what to do about it rather than merely reporting absence.
    """
    if not package.has_gold or package.gold_path is None:
        raise NoGoldSetError(
            f"tender {package.name!r} at {package.root_path} has no answer key, so there is "
            f"nothing to score against. A real tender arrives unlabelled — that is the "
            f"production path, and it is the pipeline that runs on it, not this harness. To "
            f"score a tender, label it first: add gold/gold_set.jsonl beneath the folder."
        )
    return load_gold_file(package.gold_path, tender_name=package.name)


def load_gold_file(path: Path, *, tender_name: str | None = None) -> GoldSet:
    """Read a gold set from a JSONL file, failing loudly and specifically on any bad line."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise GoldSetError(f"gold set not found: {path}") from error
    except IsADirectoryError as error:
        raise GoldSetError(f"gold set path is a directory: {path}") from error

    scored: list[GoldItem] = []
    excluded: list[GoldItem] = []
    seen: dict[str, int] = {}

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = _parse_line(line, path=path, number=number)
        if item.id in seen:
            raise GoldSetError(
                f"{path}:{number}: duplicate gold id {item.id!r}, first seen on line "
                f"{seen[item.id]}. Two items answering to one id would let a single finding "
                f"be credited twice."
            )
        seen[item.id] = number
        (scored if item.scored else excluded).append(item)

    return GoldSet(
        tender_name=tender_name or path.parent.parent.name,
        source_path=path,
        scored=scored,
        excluded=excluded,
    )


def _parse_line(line: str, *, path: Path, number: int) -> GoldItem:
    """Turn one line into a GoldItem, naming the line and the item when it cannot be done."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise GoldSetError(f"{path}:{number}: not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise GoldSetError(f"{path}:{number}: expected a JSON object, got {type(payload).__name__}")

    try:
        return GoldItem.model_validate(payload)
    except ValidationError as error:
        # The id is pulled straight from the payload rather than the model, because the
        # model is exactly what failed to build — and "which item" is the first thing
        # anyone asks of a bad severity or tier.
        identifier = payload.get("id", "(no id)")
        raise GoldSetError(
            f"{path}:{number}: gold item {identifier!r} is not valid: {_describe(error)}"
        ) from error


def _describe(error: ValidationError) -> str:
    """Render a validation error compactly enough for one line of a message."""
    parts = []
    for problem in error.errors():
        location = ".".join(str(piece) for piece in problem["loc"]) or "(root)"
        parts.append(f"{location}: {problem['msg']}")
    return "; ".join(parts)

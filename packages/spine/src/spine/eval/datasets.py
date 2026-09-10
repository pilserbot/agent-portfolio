"""Loading gold sets from disk into typed records, and sampling them reproducibly.

A gold set is a JSONL file at `data/gold/<name>.jsonl`, one item per line. Every line is
validated as it is read, and a malformed one stops the load with the file, the line number
and what was wrong — a gold set that silently drops items would quietly change every metric
computed from it.

Sampling is by seed and stable: the same seed and size select the same items on any machine
and any Python release, so `--sample 20` means the same twenty items in CI as on a laptop.

Deliberately does not: score anything, call a model, or reach a network. It reads files and
returns records. What counts as a correct answer is `spine.eval.checkers`' business, and how
those answers aggregate is `spine.eval.metrics`'.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

DEFAULT_GOLD_ROOT = Path("data/gold")

__all__ = [
    "DEFAULT_GOLD_ROOT",
    "GoldItem",
    "GoldSet",
    "GoldSetError",
    "load_gold_file",
    "load_gold_set",
]


class GoldSetError(Exception):
    """A gold set could not be loaded, or is not internally consistent."""


class GoldItem(BaseModel):
    """One labelled example: what goes in, what should come out, and how it is grouped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    expected: dict[str, JsonValue] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class GoldSet(BaseModel):
    """A named collection of gold items, and where it was read from."""

    model_config = ConfigDict(frozen=True)

    name: str
    items: list[GoldItem] = Field(default_factory=list)
    source_path: Path | None = None

    def __len__(self) -> int:
        """How many items the set holds."""
        return len(self.items)

    def tags_by_item(self) -> dict[str, list[str]]:
        """Each item's tags, keyed by item id, for grouping metrics by tag."""
        return {item.item_id: list(item.tags) for item in self.items}

    def with_tag(self, tag: str) -> "GoldSet":
        """The subset carrying a tag, as a gold set in its own right."""
        return self.model_copy(update={"items": [i for i in self.items if tag in i.tags]})

    def sample(self, size: int, *, seed: int | str = 0) -> "GoldSet":
        """Take a reproducible subset of the set.

        Selection is by hashing the seed with each item id and taking the lowest digests,
        rather than by shuffling. Python's `random` is not guaranteed stable across
        releases, and a sample that quietly changes between versions would move every
        metric measured against it. A hash is stable everywhere.

        A size at or above the set's own is the whole set, so `--sample 20` on a smaller
        set is not an error.
        """
        if size < 0:
            raise GoldSetError(f"sample size must not be negative; got {size}")
        if size >= len(self.items):
            return self
        ordered = sorted(self.items, key=lambda item: _sample_key(item.item_id, seed))
        chosen = {item.item_id for item in ordered[:size]}
        # Keep the file's own order, so a sampled report reads like the gold set.
        return self.model_copy(update={"items": [i for i in self.items if i.item_id in chosen]})


def _sample_key(item_id: str, seed: int | str) -> str:
    """The stable sort key one item gets under one seed."""
    return hashlib.sha256(f"{seed}\x00{item_id}".encode()).hexdigest()


def load_gold_set(name: str, *, root: Path = DEFAULT_GOLD_ROOT) -> GoldSet:
    """Load the gold set named `name` from the gold-set directory."""
    return load_gold_file(root / f"{name}.jsonl", name=name)


def load_gold_file(path: Path, *, name: str | None = None) -> GoldSet:
    """Load a gold set from a JSONL file, failing loudly on any bad line.

    Blank lines are skipped. Everything else must parse as JSON and validate as a GoldItem,
    and item ids must be unique: a duplicate would let one item's result overwrite another's.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise GoldSetError(f"gold set not found: {path}") from error
    except IsADirectoryError as error:
        raise GoldSetError(f"gold set path is a directory: {path}") from error

    items: list[GoldItem] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = _parse_line(line, path=path, number=number)
        if item.item_id in seen:
            raise GoldSetError(
                f"{path}:{number}: duplicate item_id {item.item_id!r}, "
                f"first seen on line {seen[item.item_id]}"
            )
        seen[item.item_id] = number
        items.append(item)

    return GoldSet(name=name or path.stem, items=items, source_path=path)


def _parse_line(line: str, *, path: Path, number: int) -> GoldItem:
    """Turn one line into a GoldItem, naming the line when it cannot be done."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise GoldSetError(f"{path}:{number}: not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise GoldSetError(f"{path}:{number}: expected a JSON object, got {type(payload).__name__}")
    try:
        return GoldItem.model_validate(payload)
    except ValidationError as error:
        raise GoldSetError(f"{path}:{number}: {_describe(error)}") from error


def _describe(error: ValidationError) -> str:
    """Render a validation error compactly enough for one line of a message."""
    parts = []
    for problem in error.errors():
        location = ".".join(str(piece) for piece in problem["loc"]) or "(root)"
        parts.append(f"{location}: {problem['msg']}")
    return "; ".join(parts)


def merge(sets: Iterable[GoldSet], *, name: str) -> GoldSet:
    """Combine gold sets into one, refusing ids that collide between them."""
    items: list[GoldItem] = []
    origin: dict[str, str] = {}
    for gold in sets:
        for item in gold.items:
            if item.item_id in origin:
                raise GoldSetError(
                    f"duplicate item_id {item.item_id!r} in {gold.name!r}, "
                    f"already provided by {origin[item.item_id]!r}"
                )
            origin[item.item_id] = gold.name
            items.append(item)
    return GoldSet(name=name, items=items)


def expected_values(items: Sequence[GoldItem], field: str) -> dict[str, JsonValue]:
    """One expected field across items, keyed by item id, skipping items without it."""
    return {item.item_id: item.expected[field] for item in items if field in item.expected}

"""Unit tests for gold-set loading and sampling.

Reads files under pytest's tmp_path and the committed example set. No network.

Deliberately does not test what the items mean — only that they load exactly, fail loudly,
and sample reproducibly.
"""

import json
from pathlib import Path

import pytest

from spine.eval.datasets import (
    DEFAULT_GOLD_ROOT,
    GoldItem,
    GoldSet,
    GoldSetError,
    expected_values,
    load_gold_file,
    load_gold_set,
    merge,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def an_item(item_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "item_id": item_id,
        "inputs": {"text": "a clause"},
        "expected": {"is_requirement": "yes"},
        "tags": ["requirement"],
    }
    payload.update(overrides)
    return payload


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


# --- loading ---------------------------------------------------------------------------


def test_a_well_formed_file_loads(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("a"), an_item("b")])

    gold = load_gold_file(path)

    assert len(gold) == 2
    assert [item.item_id for item in gold.items] == ["a", "b"]
    assert gold.name == "g"
    assert gold.source_path == path


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(f"{json.dumps(an_item('a'))}\n\n   \n{json.dumps(an_item('b'))}\n", "utf-8")

    assert len(load_gold_file(path)) == 2


def test_an_empty_file_loads_as_an_empty_set(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [])

    assert len(load_gold_file(path)) == 0


def test_fields_default_when_absent(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [{"item_id": "a"}])

    item = load_gold_file(path).items[0]

    assert item.inputs == {} and item.expected == {} and item.tags == []


def test_load_by_name_uses_the_gold_root(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "mine.jsonl", [an_item("a")])

    gold = load_gold_set("mine", root=tmp_path)

    assert gold.name == "mine"


# --- failing loudly ----------------------------------------------------------------------


def test_a_line_that_is_not_json_names_its_line_number(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("a"), "{ not json", an_item("c")])

    with pytest.raises(GoldSetError) as excinfo:
        load_gold_file(path)

    message = str(excinfo.value)
    assert ":2:" in message, "the failing line is named"
    assert "not valid JSON" in message


def test_a_line_missing_a_required_field_names_the_field_and_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("a"), {"inputs": {}}])

    with pytest.raises(GoldSetError) as excinfo:
        load_gold_file(path)

    assert ":2:" in str(excinfo.value)
    assert "item_id" in str(excinfo.value)


def test_an_unknown_field_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    # A typo'd field name would otherwise vanish and quietly change what is measured.
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("a", expcted={"oops": 1})])

    with pytest.raises(GoldSetError, match="expcted"):
        load_gold_file(path)


def test_an_empty_item_id_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("")])

    with pytest.raises(GoldSetError, match="item_id"):
        load_gold_file(path)


def test_a_json_array_line_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [[1, 2, 3]])

    with pytest.raises(GoldSetError, match="expected a JSON object"):
        load_gold_file(path)


def test_a_duplicate_item_id_names_both_lines(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [an_item("a"), an_item("b"), an_item("a")])

    with pytest.raises(GoldSetError) as excinfo:
        load_gold_file(path)

    message = str(excinfo.value)
    assert ":3:" in message and "first seen on line 1" in message


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(GoldSetError, match="not found"):
        load_gold_file(tmp_path / "absent.jsonl")


def test_a_directory_is_not_a_gold_set(tmp_path: Path) -> None:
    with pytest.raises(GoldSetError, match="directory"):
        load_gold_file(tmp_path)


# --- sampling ----------------------------------------------------------------------------


def a_set(count: int) -> GoldSet:
    return GoldSet(name="s", items=[GoldItem(item_id=f"i-{n:02d}") for n in range(count)])


def test_sampling_is_reproducible_for_a_seed() -> None:
    gold = a_set(50)

    first = gold.sample(10, seed=42)
    second = gold.sample(10, seed=42)

    assert [i.item_id for i in first.items] == [i.item_id for i in second.items]
    assert len(first) == 10


def test_sampling_is_stable_against_a_known_selection() -> None:
    # Pinned so a change to the selection rule shows up as a failure here rather than as a
    # quiet shift in every metric measured against a sampled set.
    gold = a_set(20)

    chosen = [item.item_id for item in gold.sample(5, seed=7).items]

    assert chosen == ["i-01", "i-04", "i-11", "i-13", "i-14"]


def test_a_different_seed_selects_differently() -> None:
    gold = a_set(50)

    assert [i.item_id for i in gold.sample(10, seed=1).items] != [
        i.item_id for i in gold.sample(10, seed=2).items
    ]


def test_a_string_seed_works_too() -> None:
    gold = a_set(20)

    assert gold.sample(5, seed="run-1").items == gold.sample(5, seed="run-1").items


def test_a_sample_keeps_the_files_own_order() -> None:
    gold = a_set(20)

    ids = [item.item_id for item in gold.sample(6, seed=3).items]

    assert ids == sorted(ids), "a sampled report should read in gold-set order"


def test_a_sample_larger_than_the_set_returns_everything() -> None:
    gold = a_set(10)

    assert len(gold.sample(20, seed=1)) == 10


def test_a_sample_of_zero_is_empty() -> None:
    assert len(a_set(10).sample(0, seed=1)) == 0


def test_a_negative_sample_size_is_rejected() -> None:
    with pytest.raises(GoldSetError, match="negative"):
        a_set(10).sample(-1, seed=1)


def test_sampling_an_empty_set_is_empty() -> None:
    assert len(a_set(0).sample(5, seed=1)) == 0


# --- helpers -----------------------------------------------------------------------------


def test_tags_by_item_maps_ids_to_tags() -> None:
    gold = GoldSet(
        name="s",
        items=[GoldItem(item_id="a", tags=["x", "y"]), GoldItem(item_id="b", tags=[])],
    )

    assert gold.tags_by_item() == {"a": ["x", "y"], "b": []}


def test_with_tag_narrows_the_set() -> None:
    gold = GoldSet(
        name="s",
        items=[GoldItem(item_id="a", tags=["x"]), GoldItem(item_id="b", tags=["y"])],
    )

    assert [i.item_id for i in gold.with_tag("x").items] == ["a"]


def test_merge_combines_sets() -> None:
    left = GoldSet(name="l", items=[GoldItem(item_id="a")])
    right = GoldSet(name="r", items=[GoldItem(item_id="b")])

    combined = merge([left, right], name="both")

    assert [i.item_id for i in combined.items] == ["a", "b"]
    assert combined.name == "both"


def test_merge_refuses_colliding_ids() -> None:
    left = GoldSet(name="l", items=[GoldItem(item_id="a")])
    right = GoldSet(name="r", items=[GoldItem(item_id="a")])

    with pytest.raises(GoldSetError, match="duplicate"):
        merge([left, right], name="both")


# --- the committed example set ---------------------------------------------------------------


def test_the_example_gold_set_loads() -> None:
    gold = load_gold_set("example", root=REPO_ROOT / DEFAULT_GOLD_ROOT)

    assert len(gold) == 10
    assert len({item.item_id for item in gold.items}) == 10
    assert all(item.inputs and item.expected and item.tags for item in gold.items)


def test_the_example_gold_set_has_both_labels() -> None:
    # Precision and recall need both classes present, or they measure nothing.
    gold = load_gold_set("example", root=REPO_ROOT / DEFAULT_GOLD_ROOT)

    labels = {item.expected["is_requirement"] for item in gold.items}
    assert labels == {"yes", "no"}


def test_sampling_the_example_set_is_reproducible() -> None:
    gold = load_gold_set("example", root=REPO_ROOT / DEFAULT_GOLD_ROOT)

    assert [i.item_id for i in gold.sample(4, seed=1).items] == [
        i.item_id for i in gold.sample(4, seed=1).items
    ]


def test_expected_values_collects_one_field_across_items() -> None:
    items = [
        GoldItem(item_id="a", expected={"cap": 100}),
        GoldItem(item_id="b", expected={"cap": 200}),
        GoldItem(item_id="c", expected={"other": 1}),
    ]

    # Items without the field are skipped rather than defaulted.
    assert expected_values(items, "cap") == {"a": 100, "b": 200}

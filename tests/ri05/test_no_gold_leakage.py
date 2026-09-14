"""A static scan proving no pipeline module can reach the answer key.

`test_tender_loader.py` already watches every file the process opens during a load. That
catches a gold file actually being read — but only along a code path the tests execute. A
reference sitting in a branch no test reaches is invisible to it, and that is exactly where
a leak would survive review: in the error handler nobody exercises, in the debug helper
somebody added on a Friday.

So this is the complementary half. It parses every module of `ri05_tender` **except its
`eval` package** and fails on any reference to gold material at all — reachable or not,
executed or not:

- an import of the RI-05 `eval` package, which is the only code allowed to read the key
- an attribute chain naming that package
- a string literal that denotes a path with a `gold` segment, a `DEFECT_LEDGER*` file, or
  the eval package

`eval` is excluded from the walk because reading the answer key is its entire job: the
scoring harness is not the pipeline, and the two must not be able to see each other.

**Exactly one allowance.** The tender loader may name the answer key once, to test whether
it exists so `has_gold` can be set. That is one string literal at one line in one file, and
it is written out below in full. If this list ever needs a second entry, the guard is being
worked around and the change belongs in review, not in the allowlist.

Docstrings are not scanned. A docstring cannot open a file, and the layout sketches in this
package's own module docstrings describe the folder shape on purpose. Comments are not in
the AST at all, so they are skipped for the same reason.

Deliberately does not: run anything from the package it scans. It reads source and parses
it, so a module that would leak on import is caught before it is imported.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

PACKAGE_ROOT = Path("apps/ri05_tender/src/ri05_tender")
EXCLUDED_SUBPACKAGE = "eval"

EVAL_PACKAGE_DOTTED = "ri05_tender.eval"
EVAL_PACKAGE_PATH = "ri05_tender/eval"
GOLD_SEGMENT = "gold"
DEFECT_LEDGER = re.compile(r"^defect_ledger", re.IGNORECASE)

# The whole allowance. One file, one literal: the relative path the tender loader tests for
# existence to decide `has_gold`. Adding to this list is not a fix.
ALLOWED_LITERALS: frozenset[tuple[str, str]] = frozenset(
    {("tender/loader.py", "gold/gold_set.jsonl")}
)


class Leak(BaseModel):
    """One reference to gold material found in a module that must not have any."""

    model_config = ConfigDict(frozen=True)

    path: str
    line: int
    token: str
    kind: str

    def __str__(self) -> str:
        """The finding as a reviewer needs to read it: file, line, exact token."""
        return f"{PACKAGE_ROOT / self.path}:{self.line}: {self.kind} references {self.token!r}"


def python_files(root: Path, *, exclude: str) -> list[Path]:
    """Every module under a package except those inside one named sub-package."""
    return sorted(
        path for path in root.rglob("*.py") if exclude not in path.relative_to(root).parts
    )


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """The id() of every docstring constant, so the scan can skip them."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _names_gold_path(literal: str) -> bool:
    """Whether a string literal denotes a path into gold material.

    Segment equality, not substring: a sentence that happens to contain the word is prose,
    while `gold/gold_set.jsonl` is an address. The distinction is what keeps the scan
    strict without making every error message a violation.
    """
    segments = [segment.strip().lower() for segment in re.split(r"[/\\]", literal)]
    if GOLD_SEGMENT in segments:
        return True
    return any(DEFECT_LEDGER.match(segment) for segment in segments if segment)


def _names_eval_package(text: str) -> bool:
    """Whether some text names the RI-05 eval package."""
    return EVAL_PACKAGE_DOTTED in text or EVAL_PACKAGE_PATH in text


def _dotted(node: ast.Attribute) -> str:
    """An attribute chain rendered back as dotted text, best effort."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def scan_source(source: str, *, relative_path: str) -> Iterator[Leak]:
    """Every reference to gold material in one module's source."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_eval_package(alias.name) or GOLD_SEGMENT in alias.name.split("."):
                    yield Leak(
                        path=relative_path, line=node.lineno, token=alias.name, kind="import"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _names_eval_package(module) or GOLD_SEGMENT in module.split("."):
                yield Leak(path=relative_path, line=node.lineno, token=module, kind="import")
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if _names_eval_package(dotted):
                yield Leak(
                    path=relative_path, line=node.lineno, token=dotted, kind="attribute chain"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            if (relative_path, node.value) in ALLOWED_LITERALS:
                continue
            if _names_gold_path(node.value) or _names_eval_package(node.value):
                yield Leak(
                    path=relative_path, line=node.lineno, token=node.value, kind="string literal"
                )


def scan_package() -> list[Leak]:
    """Scan the shipped pipeline package, eval excluded."""
    leaks: list[Leak] = []
    for path in python_files(PACKAGE_ROOT, exclude=EXCLUDED_SUBPACKAGE):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        leaks.extend(scan_source(path.read_text(encoding="utf-8"), relative_path=relative))
    return leaks


# --- the scan itself ------------------------------------------------------------------


def test_no_pipeline_module_references_gold_material() -> None:
    leaks = scan_package()

    assert not leaks, "gold material referenced outside the eval package:\n" + "\n".join(
        f"  {leak}" for leak in leaks
    )


def test_the_scan_actually_reads_the_package() -> None:
    # A scan that found no files would pass this suite while checking nothing.
    files = python_files(PACKAGE_ROOT, exclude=EXCLUDED_SUBPACKAGE)

    assert len(files) >= 4
    assert "tender/loader.py" in {f.relative_to(PACKAGE_ROOT).as_posix() for f in files}


def test_the_eval_package_is_excluded_and_does_reference_gold() -> None:
    # Two things at once: eval is outside the walk, and it is outside because it genuinely
    # names the answer key — so the exclusion is load-bearing, not decorative.
    walked = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_files(PACKAGE_ROOT, exclude=EXCLUDED_SUBPACKAGE)
    }
    assert not any(name.startswith("eval/") for name in walked)

    eval_loader = (PACKAGE_ROOT / "eval" / "loader.py").read_text(encoding="utf-8")
    assert list(scan_source(eval_loader, relative_path="eval/loader.py"))


# --- the one allowance ------------------------------------------------------------------


def test_the_allowance_is_exactly_one_literal() -> None:
    # Its size is the point. If this number ever goes up, the guard is being worked around.
    assert len(ALLOWED_LITERALS) == 1
    assert ALLOWED_LITERALS == frozenset({("tender/loader.py", "gold/gold_set.jsonl")})


def test_the_allowed_literal_is_only_tested_for_existence() -> None:
    # The loader may ask whether the answer key is there. It may not open it.
    source = (PACKAGE_ROOT / "tender" / "loader.py").read_text(encoding="utf-8")

    assert "GOLD_SET_RELATIVE_PATH" in source
    assert "is_file()" in source
    for forbidden in ("read_text", "open(", "read_bytes", "load_gold", "json.loads"):
        assert forbidden not in source.split("def load_tender(", 1)[1], forbidden


def test_the_allowance_does_not_cover_the_same_literal_elsewhere() -> None:
    # It is scoped to one file, so the same string in another module is still a leak.
    leaks = list(scan_source('PATH = "gold/gold_set.jsonl"\n', relative_path="tender/models.py"))

    assert [leak.token for leak in leaks] == ["gold/gold_set.jsonl"]


# --- positive controls: the guard must be able to fail ------------------------------------


@pytest.mark.parametrize(
    ("source", "expected_token"),
    [
        ('PATH = "data/tenders/x/gold/gold_set.jsonl"\n', "data/tenders/x/gold/gold_set.jsonl"),
        ('NAME = "gold"\n', "gold"),
        ('LEDGER = "DEFECT_LEDGER_v3.xlsx"\n', "DEFECT_LEDGER_v3.xlsx"),
        ('LEDGER = "defect_ledger.csv"\n', "defect_ledger.csv"),
        ('MODULE = "ri05_tender.eval.loader"\n', "ri05_tender.eval.loader"),
        ("import ri05_tender.eval.loader\n", "ri05_tender.eval.loader"),
        ("from ri05_tender.eval import loader\n", "ri05_tender.eval"),
        ("x = ri05_tender.eval.loader.load_gold\n", "ri05_tender.eval.loader.load_gold"),
    ],
)
def test_a_forbidden_reference_is_detected(source: str, expected_token: str) -> None:
    leaks = list(scan_source(source, relative_path="tender/somewhere.py"))

    assert expected_token in {leak.token for leak in leaks}


def test_a_forbidden_reference_in_a_real_file_is_detected(tmp_path: Path) -> None:
    # The positive control the guard exists for: a temporary module carrying a leak must be
    # caught, so the scan cannot silently stop working.
    package = tmp_path / "pkg"
    (package / "tender").mkdir(parents=True)
    (package / "tender" / "sneaky.py").write_text(
        '"""A module docstring mentioning gold/ harmlessly."""\n\nSECRET = "gold/gold_set.jsonl"\n',
        encoding="utf-8",
    )

    files = python_files(package, exclude=EXCLUDED_SUBPACKAGE)
    leaks = [
        leak
        for path in files
        for leak in scan_source(
            path.read_text(encoding="utf-8"), relative_path=path.relative_to(package).as_posix()
        )
    ]

    assert [(leak.line, leak.token) for leak in leaks] == [(3, "gold/gold_set.jsonl")]


def test_a_leak_names_the_file_the_line_and_the_token() -> None:
    leak = next(iter(scan_source('X = "gold/x.jsonl"\n', relative_path="tender/loader.py")))

    rendered = str(leak)
    assert "tender/loader.py" in rendered
    assert ":1:" in rendered
    assert "'gold/x.jsonl'" in rendered


# --- what the scan must NOT flag ------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        '"""Layout: documents/ and optionally gold/ beside it."""\n',  # a docstring
        'MESSAGE = "A tender is a folder containing documents/ and optionally an answer key."\n',
        'NAME = "GOLD_SET_RELATIVE_PATH"\n',  # an identifier, not a path
        'LABEL = "Gold set: "\n',  # prose, not a path segment
        'NOTE = "has_gold is True but gold_path is None"\n',  # the sanctioned interface
        "package.has_gold\n",
        "package.gold_path\n",
        "from spine.eval import metrics\n",  # a different project's eval package
    ],
)
def test_prose_and_the_sanctioned_interface_are_not_flagged(source: str) -> None:
    # A guard that fires on every mention of the word would be turned off within a week.
    assert not list(scan_source(source, relative_path="tender/models.py"))

"""No detector may be written against the answers.

The existing gold-leakage guard stops a module *reading* the answer key. It cannot stop a
developer typing `if clause == "TS-B.25"` or `if value == -40` into a detector, and that is
the failure that matters more, because it does not look like cheating. It looks like a
detector. It passes review, it passes its own unit tests, and it produces a recall figure
that is a number about the developer rather than about the system — a measurement of how
many answers somebody had already seen.

So this parses every source file under the detector packages and fails on:

- **a clause reference** in any of this tender's numbering schemes — `TS-B.25`, `IF-4.2`,
  `CS-1.3`, `ITB-9.2`, `GCC-11.3`, `SOW-9.4`, `A-2.1`, `DR-1.4`, `GN-09`. A detector has no
  business naming a clause in any document, let alone this one.
- **a distinctive numeric value** from the demo-set findings — the pixel density, the
  temperature, the PoE budget, the focal length, the range, the milestone days. A detector
  that knows the number knows the answer.

**What is deliberately NOT on the value list**, and why it matters: 0, 1, 2, 3, 10, 100 and
their kind. They are the constants of ordinary code — a list index, a percentage, a default
— and a guard that fired on `range(2)` would be switched off within a week, taking the rest
of its coverage with it. The list holds values distinctive enough that their appearance in a
detector is evidence rather than coincidence, and that judgement is stated here so it can be
argued with rather than discovered.

**The two halves are scanned differently, and the asymmetry is deliberate.**

A *clause reference* is scanned everywhere, prose included. A detector whose docstring cites
a clause of this tender was written with the answer sheet open, and that is true whether the
citation sits in an expression or in an explanation.

A *numeric value* is scanned in code literals and in non-docstring strings only. Docstrings
are prose, and prose has numbers in it that mean nothing: the first draft of this guard
flagged `"neither is '12 more' than the other"` in a paragraph explaining why IP ratings are
not arithmetic. Numbers are also matched on word boundaries, because the same draft read the
project's own name, `RI-05`, as the value -5. A guard that cries wolf at its own
documentation is a guard somebody deletes, so the rule is: a number keyed into code is
evidence; a number in a sentence is not.

Positive controls: a module carrying each shape must be caught, so the guard cannot quietly
stop working.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

# Every package a detector or its configuration may live in. Adding a detector elsewhere
# without adding the path here would put it outside the guard, which is why the test below
# asserts the roots actually contain the detectors.
DETECTOR_ROOTS: tuple[Path, ...] = (
    Path("packages/req_core/src/req_core/detectors"),
    Path("packages/req_core/src/req_core/claims.py"),
    Path("packages/req_core/src/req_core/findings.py"),
    Path("packages/req_core/src/req_core/engine.py"),
    Path("apps/ri05_tender/src/ri05_tender/detect"),
)

# This tender's clause numbering, as a shape rather than as a list of clauses. Matching the
# scheme rather than the instances is what makes the guard hold for a clause nobody has
# planted yet.
CLAUSE_REFERENCE = re.compile(
    r"\b(?:TS|IF|CS|ITB|GCC|SOW|DR|GN|BOQ)-[A-Z]?\.?\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)

# Values that appear in the demo-set findings. Each is distinctive enough that a detector
# containing it is a detector that had been shown the answer:
#
#   80      px/m, the pixel-density requirement          74.8    px/m, what the lens achieves
#   55      m, the range it is computed at               12      mm, the specified focal length
#   -40     °C, the camera's rated minimum               -5      °C, the site minimum
#   15.4    W, the PoE budget                            8200    LF of perimeter
#   510/540/600  the milestone days that do not add up
#   1.06    $M saved by the alternative                  15.9/14.8/15.0  $M, the bid arithmetic
#   98      %, the phantom row's accuracy                298     the matrix row count
DEMO_VALUES: frozenset[float] = frozenset(
    {
        80.0,
        74.8,
        55.0,
        12.0,
        -40.0,
        -5.0,
        15.4,
        8200.0,
        510.0,
        540.0,
        600.0,
        1.06,
        15.9,
        14.8,
        15.0,
        98.0,
        298.0,
    }
)

# Values excluded from DEMO_VALUES on purpose — see the module docstring. Kept as a named
# set so the decision is visible and a later reader can challenge it.
ORDINARY_CONSTANTS: frozenset[float] = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0})

# A number standing on its own, not a digit run inside an identifier. Without the lookaround
# this reads the project's own name, `RI-05`, as the value -5, and `IP66` as 66.
STANDALONE_NUMBER = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?(?![\w.])")


class Leak(BaseModel):
    """One piece of this tender's answers found inside a detector."""

    model_config = ConfigDict(frozen=True)

    path: str
    line: int
    token: str
    kind: str

    def __str__(self) -> str:
        """The finding as a reviewer needs to read it: file, line, exact token."""
        return f"{self.path}:{self.line}: {self.kind} {self.token!r}"


def _docstring_ids(tree: ast.Module) -> set[int]:
    """The id() of every docstring constant, so the numeric scan can skip them."""
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


def detector_files(roots: tuple[Path, ...] = DETECTOR_ROOTS) -> list[Path]:
    """Every source file under the detector packages."""
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
        else:
            found.extend(sorted(root.rglob("*.py")))
    return found


def _numbers_in(node: ast.AST) -> Iterator[tuple[int, float]]:
    """Every numeric literal, with negation folded in so -40 reads as -40 and not as 40."""
    for item in ast.walk(node):
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
            operand = item.operand
            if isinstance(operand, ast.Constant) and isinstance(operand.value, int | float):
                if not isinstance(operand.value, bool):
                    yield item.lineno, -float(operand.value)
        elif isinstance(item, ast.Constant) and isinstance(item.value, int | float):
            if not isinstance(item.value, bool):
                yield item.lineno, float(item.value)


def scan_source(source: str, *, path: str) -> list[Leak]:
    """Every clause reference and every demo value in one module's source."""
    leaks: list[Leak] = []
    tree = ast.parse(source)

    docstrings = _docstring_ids(tree)

    # Clause references, in code and in prose alike. A detector's docstring citing a clause
    # of this tender is a detector written with the answer sheet open.
    for number, line in enumerate(source.splitlines(), start=1):
        for match in CLAUSE_REFERENCE.finditer(line):
            leaks.append(
                Leak(path=path, line=number, token=match.group(0), kind="clause reference")
            )

    for line_number, value in _numbers_in(tree):
        if value in DEMO_VALUES:
            leaks.append(
                Leak(
                    path=path,
                    line=line_number,
                    token=f"{value:g}",
                    kind="numeric value from the demo set",
                )
            )

    for item in ast.walk(tree):
        if not (isinstance(item, ast.Constant) and isinstance(item.value, str)):
            continue
        if id(item) in docstrings:
            continue
        for candidate in STANDALONE_NUMBER.findall(item.value):
            try:
                number = float(candidate)
            except ValueError:  # pragma: no cover - the pattern cannot produce this
                continue
            if number in DEMO_VALUES:
                leaks.append(
                    Leak(
                        path=path,
                        line=item.lineno,
                        token=candidate,
                        kind="demo-set value inside a string",
                    )
                )
    return leaks


def scan_detectors() -> list[Leak]:
    """Scan every detector source file."""
    return [
        leak
        for path in detector_files()
        for leak in scan_source(path.read_text(encoding="utf-8"), path=str(path))
    ]


# --- the scan itself ------------------------------------------------------------------


def test_no_detector_names_a_clause_or_knows_an_answer() -> None:
    leaks = scan_detectors()

    assert not leaks, (
        "a detector contains something from this tender's answers. A recall figure produced "
        "by detectors written against the answers is a number about the developer, not about "
        "the system:\n" + "\n".join(f"  {leak}" for leak in leaks)
    )


def test_the_scan_actually_covers_the_detectors() -> None:
    # A scan that found no files would pass the suite while checking nothing.
    files = {path.name for path in detector_files()}

    for detector in ("tolerance.py", "modality.py", "atomicity.py", "verifiability.py"):
        assert detector in files, detector
    for detector in ("ordinal.py", "margin.py"):
        assert detector in files, detector
    assert "config.py" in files  # the application's severity and scale configuration


def test_every_registered_detector_lives_under_a_scanned_root() -> None:
    # The guard is only as wide as its roots. Asked of each detector directly rather than
    # guessed from its name: a detector added outside the scan is exactly the case that must
    # fail here, and a name-matching heuristic would miss the one that was renamed to hide.
    import inspect

    from req_core.detectors import DETECTORS

    scanned = {path.resolve() for path in detector_files()}
    for name, detector in DETECTORS.items():
        where = Path(inspect.getfile(detector)).resolve()
        assert where in scanned, (
            f"detector {name!r} is defined in {where}, which is not under any scanned root "
            f"{DETECTOR_ROOTS}. Either it lives outside the guard, or the roots need "
            f"extending — the first is the dangerous one."
        )


# --- positive controls: the guard must be able to fail ------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('CLAUSE = "TS-B.25"\n', "TS-B.25"),
        ('CLAUSE = "IF-4.2"\n', "IF-4.2"),
        ('CLAUSE = "GN-09"\n', "GN-09"),
        ("# the rule from GCC-11.3\n", "GCC-11.3"),
        ('"""Worked example: see SOW-9.4."""\n', "SOW-9.4"),
    ],
)
def test_a_clause_reference_is_caught(source: str, expected: str) -> None:
    leaks = scan_source(source, path="detectors/example.py")

    assert expected in {leak.token for leak in leaks}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("THRESHOLD = 74.8\n", "74.8"),
        ("MINIMUM_C = -40\n", "-40"),
        ("BUDGET_W = 15.4\n", "15.4"),
        ("REQUIRED_PX_PER_M = 80\n", "80"),
        ('NOTE = "not less than 80 px/m"\n', "80"),
    ],
)
def test_a_demo_value_is_caught(source: str, expected: str) -> None:
    leaks = scan_source(source, path="detectors/example.py")

    assert expected in {leak.token for leak in leaks}


def test_a_real_file_carrying_a_leak_is_caught(tmp_path: Path) -> None:
    # The control the guard exists for: a temporary detector must be caught, so the scan
    # cannot silently stop working.
    package = tmp_path / "detectors"
    package.mkdir()
    (package / "sneaky.py").write_text(
        '"""A detector."""\n\n\ndef detect(context: object) -> list[object]:\n'
        '    """Fire on the known case."""\n'
        '    if context == "TS-B.25":\n'
        "        return [74.8]\n"
        "    return []\n",
        encoding="utf-8",
    )

    leaks = [
        leak
        for path in detector_files((package,))
        for leak in scan_source(path.read_text(encoding="utf-8"), path=str(path))
    ]

    assert {leak.kind for leak in leaks} == {"clause reference", "numeric value from the demo set"}
    assert {leak.token for leak in leaks} == {"TS-B.25", "74.8"}


def test_a_leak_names_the_file_the_line_and_the_token() -> None:
    leak = scan_source("X = 1\nY = 74.8\n", path="detectors/example.py")[0]

    rendered = str(leak)
    assert "detectors/example.py" in rendered
    assert ":2:" in rendered
    assert "'74.8'" in rendered


# --- what the guard must NOT flag -----------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "LIMIT = 0\n",
        "for index in range(2):\n    pass\n",
        "CONFIDENCE = 0.85\n",
        "THRESHOLD = 0.6\n",
        'SCALES = frozenset({"ip", "nema"})\n',
        '"""A clause that constrains a property without bounding it."""\n',
        "PERCENT = 100\n",
        'UNIT = "px/m"\n',  # the unit is generic; the number beside it was the answer
    ],
)
def test_ordinary_code_is_not_flagged(source: str) -> None:
    # A guard that fires on `range(2)` gets switched off, and takes its real coverage with it.
    assert not scan_source(source, path="detectors/example.py")


def test_the_excluded_constants_are_stated_rather_than_implied() -> None:
    # The judgement about which numbers are too ordinary to scan for is part of the guard's
    # contract, so it is written down and asserted rather than left in someone's head.
    assert not (DEMO_VALUES & ORDINARY_CONSTANTS)
    assert 0.0 in ORDINARY_CONSTANTS and 100.0 in ORDINARY_CONSTANTS
    assert 74.8 in DEMO_VALUES

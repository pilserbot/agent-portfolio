"""The threshold, and the detector shapes it silently makes unreportable.

`CONFIDENCE_THRESHOLD` and each detector's confidence table were chosen independently and
neither author saw the other's number. Where a shape scores below the threshold the result
is a finding that can never be asserted — not a bug in either decision, but a consequence
of the pair, and one that reads as a detector failure if nobody writes it down.

`SUB_THRESHOLD_SHAPES` writes it down. The test below keeps it from going stale: it reads
the confidence literals out of the detector sources and fails if any value below the
threshold is not named in the catalogue. Add a shape at 0.3 and this breaks until somebody
says what its zero does not mean.

Deliberately does not: assert that any particular confidence is right, or that the
threshold is. Both are judgements, and a test that pinned them to a gold-set outcome would
be the tuning `tests/test_no_domain_leakage.py` exists to prevent.
"""

import ast
from pathlib import Path

import pytest

from req_core import detectors as detector_package
from req_core.detectors import DETECTORS
from ri05_tender.detect.config import (
    CONFIDENCE_THRESHOLD,
    ORDINAL_SCALES,
    SEVERITY_POLICY,
    SUB_THRESHOLD_SHAPES,
)

DETECTOR_ROOT = Path(detector_package.__file__).parent


def _assigned_floats(value: ast.expr) -> set[float]:
    """Every float literal in one assignment's value.

    Both branches of `0.7 if has_unit else 0.5` and both values of a lookup table, because a
    regex reading up to the first number would have missed exactly the shape this catalogue
    exists for. Dict keys are skipped: in `{2: 0.7, 3: 0.8}` the 2 and 3 are child counts,
    not confidences, and taking floats only excludes them anyway.
    """
    sources = value.values if isinstance(value, ast.Dict) else [value]
    return {
        node.value
        for source in sources
        for node in ast.walk(source)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }


def confidences_by_module() -> dict[str, set[float]]:
    """Every literal confidence each detector module can assign.

    Read from the source rather than by running the detectors, so a shape reachable only on
    an input no test happens to supply is still counted. A value assigned to `confidence`,
    passed as `confidence=`, or held by a module constant whose name says so.
    """
    found: dict[str, set[float]] = {}
    for path in sorted(DETECTOR_ROOT.glob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        values: set[float] = set()
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "confidence":
                values |= _assigned_floats(node.value)
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if any(name == "confidence" or "CONFIDENCE" in name for name in names):
                    values |= _assigned_floats(node.value)
        if values:
            found[path.stem] = values
    return found


def test_every_detector_module_states_its_confidences_as_literals() -> None:
    """The scan below is only a guard if it can see every value."""
    modules = confidences_by_module()
    assert len(modules) == len(DETECTORS), (
        f"scanned {sorted(modules)} against {sorted(DETECTORS)} registered detectors — a "
        f"module whose confidences are computed rather than written as literals is one this "
        f"guard cannot read, and its sub-threshold shapes would go unnamed."
    )


def test_every_sub_threshold_confidence_is_named_in_the_catalogue() -> None:
    """The guard itself: a shape below the threshold must say what its zero does not mean."""
    below: set[tuple[str, float]] = set()
    for module, values in confidences_by_module().items():
        below |= {(module, value) for value in values if value < CONFIDENCE_THRESHOLD}

    # The module file is named after the mechanism, the detector after the finding; map one
    # to the other through the registry rather than by guessing at the spelling.
    detector_of_module = {
        Path(function.__code__.co_filename).stem: name for name, function in DETECTORS.items()
    }
    documented = {(shape.detector, shape.confidence) for shape in SUB_THRESHOLD_SHAPES}
    scanned = {(detector_of_module[module], value) for module, value in below}

    assert scanned == documented, (
        f"confidences below {CONFIDENCE_THRESHOLD} found in the detectors: {sorted(scanned)}; "
        f"named in SUB_THRESHOLD_SHAPES: {sorted(documented)}. A shape that can never be "
        f"asserted and is not written down reads as a detector that cannot find the defect."
    )


def test_no_documented_shape_is_actually_reportable() -> None:
    """A shape at or above the threshold does not belong in a list of what cannot be said."""
    for shape in SUB_THRESHOLD_SHAPES:
        assert shape.confidence < CONFIDENCE_THRESHOLD, shape.detector


def test_every_documented_shape_names_a_registered_detector() -> None:
    for shape in SUB_THRESHOLD_SHAPES:
        assert shape.detector in DETECTORS, shape.detector


def test_each_shape_says_what_the_zero_does_not_mean() -> None:
    """A catalogue of confidences without consequences is a list nobody reads."""
    for shape in SUB_THRESHOLD_SHAPES:
        assert len(shape.shape) > 20, shape.detector
        assert len(shape.consequence) > 40, shape.detector


def test_the_threshold_itself_is_not_pinned_to_an_outcome() -> None:
    """A sanity bound, not a target. It exists so an accidental 0.0 or 1.0 fails loudly."""
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0


def test_the_severity_policy_and_scales_are_this_project_s_and_complete() -> None:
    assert SEVERITY_POLICY.name
    for name in DETECTORS:
        assert SEVERITY_POLICY.severity_for(name)
    assert ORDINAL_SCALES.scales


@pytest.mark.parametrize("shape", SUB_THRESHOLD_SHAPES, ids=lambda s: s.detector)
def test_a_shape_cannot_be_constructed_at_or_above_the_threshold(shape: object) -> None:
    """The model refuses it, so the catalogue cannot drift above the line it describes."""
    from pydantic import ValidationError

    from ri05_tender.detect.config import SubThresholdShape

    with pytest.raises(ValidationError):
        SubThresholdShape(
            detector=shape.detector,  # type: ignore[attr-defined]
            confidence=CONFIDENCE_THRESHOLD,
            shape="a shape at exactly the threshold, which is reportable",
            consequence="nothing, because a finding at the threshold is asserted like any other",
        )

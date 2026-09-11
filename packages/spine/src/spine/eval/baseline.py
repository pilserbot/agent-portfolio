"""The committed reference a run is measured against, and the comparison against it.

A baseline records, for each metric, the value last accepted, how far it may drift before
that counts as a regression, and which direction is the good one. `check_regressions` is
the whole verdict: arithmetic over two structures, with no tolerance for cleverness.

This lives apart from both `spine.eval.run` and `spine.eval.gate` so the runner can gate
itself and the standalone gate command can read a written report, without either importing
the other.

Deliberately does not: run an evaluation, write a file, or decide what a good tolerance is.
A baseline recording no metrics is not a failure — it means nothing has been accepted yet,
and there is genuinely nothing to compare.
"""

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import Direction, KPISnapshot

# A drift this close to the tolerance counts as being inside it. Without it the check
# turns on the last bit of a float: 0.9 - 0.85 is 0.050000000000000044, so a metric sitting
# exactly on a declared tolerance of 0.05 would fail the gate, and re-deriving the same
# baseline from the same run would not settle it.
_BOUNDARY_REL_TOL = 1e-9
_BOUNDARY_ABS_TOL = 1e-12

__all__ = [
    "Baseline",
    "BaselineMetric",
    "GateVerdict",
    "MetricReading",
    "MetricReadings",
    "Regression",
    "baseline_from_snapshots",
    "check_regressions",
    "readings_from_snapshots",
]


class BaselineMetric(BaseModel):
    """The reference value for one metric, and how far it may drift."""

    value: float
    tolerance: float = 0.0
    direction: Direction = "higher_is_better"


class Baseline(BaseModel):
    """The committed reference a run is measured against."""

    metrics: dict[str, BaselineMetric] = Field(default_factory=dict)


class MetricReading(BaseModel):
    """One observed value, named."""

    model_config = ConfigDict(frozen=True)

    metric: str
    value: float


class MetricReadings(BaseModel):
    """What a run observed, in the form the comparison consumes."""

    model_config = ConfigDict(frozen=True)

    readings: list[MetricReading] = Field(default_factory=list)

    def by_name(self) -> dict[str, float]:
        """The readings keyed by metric name."""
        return {reading.metric: reading.value for reading in self.readings}


class Regression(BaseModel):
    """One metric that moved the wrong way by more than its tolerance."""

    metric: str
    baseline: float
    observed: float
    tolerance: float
    delta: float


class GateVerdict(BaseModel):
    """The outcome of comparing observations against a baseline."""

    compared: int
    regressions: list[Regression]
    missing: list[str]
    passed: bool


def _exceeds(delta: float, tolerance: float) -> bool:
    """Whether a drift is genuinely beyond a tolerance rather than on its boundary."""
    return delta > tolerance and not math.isclose(
        delta, tolerance, rel_tol=_BOUNDARY_REL_TOL, abs_tol=_BOUNDARY_ABS_TOL
    )


def check_regressions(baseline: Baseline, readings: MetricReadings) -> GateVerdict:
    """Compare observations against a baseline and return the verdict.

    The direction recorded on each baseline metric decides which way is the wrong way, so
    a cost that went up and an accuracy that went down are both caught by one rule. A drift
    that lands exactly on the tolerance is inside it — see `_exceeds` for why that needs
    saying.
    """
    observed = readings.by_name()
    regressions: list[Regression] = []
    missing: list[str] = []

    for name, expected in sorted(baseline.metrics.items()):
        if name not in observed:
            # TODO: decide whether a baseline metric absent from the report should fail the
            # gate. Reported but tolerated until the metric names are settled.
            missing.append(name)
            continue
        actual = observed[name]
        if expected.direction == "higher_is_better":
            delta = expected.value - actual
        else:
            delta = actual - expected.value
        if _exceeds(delta, expected.tolerance):
            regressions.append(
                Regression(
                    metric=name,
                    baseline=expected.value,
                    observed=actual,
                    tolerance=expected.tolerance,
                    delta=delta,
                )
            )

    return GateVerdict(
        compared=len(baseline.metrics) - len(missing),
        regressions=regressions,
        missing=missing,
        passed=not regressions,
    )


def readings_from_snapshots(snapshots: Sequence[KPISnapshot]) -> MetricReadings:
    """Reduce measurements to the name/value pairs the comparison needs."""
    return MetricReadings(
        readings=[
            MetricReading(metric=snapshot.metric_name, value=snapshot.value)
            for snapshot in snapshots
        ]
    )


def baseline_from_snapshots(
    snapshots: Sequence[KPISnapshot],
    *,
    tolerances: dict[str, float] | None = None,
) -> Baseline:
    """Turn a run's measurements into the baseline a later run is held to.

    The direction comes from each snapshot, so accepting a new baseline cannot quietly
    flip which way a metric is supposed to move. Tolerances come from the project config,
    because how much drift is acceptable is a decision, not an observation.
    """
    allowed = tolerances or {}
    return Baseline(
        metrics={
            snapshot.metric_name: BaselineMetric(
                value=snapshot.value,
                tolerance=allowed.get(snapshot.metric_name, 0.0),
                direction=snapshot.direction,
            )
            for snapshot in snapshots
        }
    )

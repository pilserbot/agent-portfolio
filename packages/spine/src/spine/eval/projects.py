"""What a project declares about its own evaluation, and how the runner finds its code.

A project's configuration is a YAML file at `evals/projects/<project>.yaml`: which gold
set it is measured on, which KPIs it reports, the tolerances its regression gate uses, and
the named ROI assumptions behind any money it quotes. Nothing here is defaulted silently —
a missing or malformed file stops the run naming the file and the problem, because a report
built from a config nobody can see is not a measurement.

The runner also needs code: `register_evaluator` puts a project's entry point into a
registry keyed by name, and `evaluator_for` looks it up. An evaluator takes an
`EvaluationInput` and returns an `EvaluationOutput` — Verdicts and the run record — so the
runner never needs to know anything about the domain it just evaluated.

Deliberately does not: import any project's code on your behalf. A project registers itself
when its module is imported, and the runner imports the modules it knows about; a name that
was never registered is an error naming the names that were, rather than a guess. It also
does not run anything, score anything or write anything.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spine.contracts import AgentRun, Verdict
from spine.eval.datasets import DEFAULT_GOLD_ROOT, GoldSet
from spine.kpi import KPISpecSet

DEFAULT_PROJECT_ROOT = Path("evals/projects")

__all__ = [
    "DEFAULT_PROJECT_ROOT",
    "EvaluationInput",
    "EvaluationOutput",
    "Evaluator",
    "ProjectConfig",
    "ProjectError",
    "evaluator_for",
    "load_project_config",
    "load_project_config_file",
    "register_evaluator",
    "registered_projects",
]


class ProjectError(Exception):
    """A project's configuration or entry point could not be found or understood."""


class ProjectConfig(BaseModel):
    """Everything a project declares about how it is evaluated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: str = Field(min_length=1)
    description: str = ""
    gold_set: str = Field(min_length=1, description="Name of the file under data/gold/.")
    gold_root: Path = DEFAULT_GOLD_ROOT
    kpis: KPISpecSet
    source_path: Path | None = None

    @property
    def tolerances(self) -> dict[str, float]:
        """How far each metric may drift before it counts as a regression."""
        return {spec.name: spec.tolerance for spec in self.kpis.kpis}


def load_project_config(name: str, *, root: Path = DEFAULT_PROJECT_ROOT) -> ProjectConfig:
    """Load a project's configuration by name."""
    return load_project_config_file(root / f"{name}.yaml", expected_name=name)


def load_project_config_file(path: Path, *, expected_name: str | None = None) -> ProjectConfig:
    """Load a project configuration from YAML, failing loudly and specifically."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ProjectError(f"project config not found: {path}") from error
    except IsADirectoryError as error:
        raise ProjectError(f"project config path is a directory: {path}") from error

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ProjectError(f"{path}: not valid YAML: {error}") from error

    if not isinstance(payload, Mapping):
        raise ProjectError(f"{path}: expected a YAML mapping, got {type(payload).__name__}")

    try:
        config = ProjectConfig.model_validate({**payload, "source_path": path})
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or '(root)'}: {problem['msg']}"
            for problem in error.errors()
        )
        raise ProjectError(f"{path}: {problems}") from error

    if expected_name is not None and config.project != expected_name:
        raise ProjectError(
            f"{path}: declares project {config.project!r} but was loaded as {expected_name!r}; "
            f"the filename and the project field must agree."
        )
    if config.kpis.project != config.project:
        raise ProjectError(
            f"{path}: kpis.project is {config.kpis.project!r} but the project is "
            f"{config.project!r}; a snapshot would be filed under the wrong name."
        )
    return config


class EvaluationInput(BaseModel):
    """What a project's evaluation entry point is handed."""

    model_config = ConfigDict(frozen=True)

    project: str
    run_id: str
    gold: GoldSet
    config: ProjectConfig


class EvaluationOutput(BaseModel):
    """What a project's evaluation entry point returns: the scores and the run record."""

    model_config = ConfigDict(frozen=True)

    verdicts: list[Verdict] = Field(default_factory=list)
    run: AgentRun


type Evaluator = Callable[[EvaluationInput], EvaluationOutput]

_REGISTRY: dict[str, Evaluator] = {}


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    """Register a project's evaluation entry point under its name.

    Re-registering the same name is refused: two entry points claiming one project would
    make the report depend on import order.
    """
    if name in _REGISTRY and _REGISTRY[name] is not evaluator:
        raise ProjectError(f"project {name!r} already has a different evaluation entry point.")
    _REGISTRY[name] = evaluator


def evaluator_for(name: str) -> Evaluator:
    """The registered entry point for a project, or an error naming what is registered."""
    try:
        return _REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ProjectError(
            f"no evaluation entry point registered for project {name!r}. Registered: {known}. "
            f"A project registers itself when its module is imported."
        ) from error


def registered_projects() -> list[str]:
    """Every project name with a registered entry point, sorted."""
    return sorted(_REGISTRY)

"""The reference evaluation entry point, so the harness runs end to end today.

The `example` project classifies each clause in `data/gold/example.jsonl` as a requirement
or not, using an ordinary rule over modal verbs, and scores the answer with
`spine.eval.checkers.exact_match`. It exists to exercise the whole pipeline — config, gold
set, entry point, KPIs, report, regression gate — with no key, no network and no model, so
CI and a reviewer on a plane both get the same numbers.

It is a **reference implementation, not a product**. The classifier is a handful of
keywords and gets one of ten items wrong, which is on purpose: an example that scored
perfectly would make precision and recall degenerate and hide a broken metric. A real
project registers its own entry point and does the work properly.

Deliberately does not: call a model, reach a network, read anything outside the gold set it
is given, or pretend to be a serious extractor. It also does not decide its own KPIs — those
are declared in `evals/projects/example.yaml`, like every other project's.
"""

import re
from datetime import UTC, datetime

from spine.contracts import AgentRun, StepTrace, Verdict
from spine.eval.checkers import exact_match
from spine.eval.projects import EvaluationInput, EvaluationOutput, register_evaluator

PROJECT = "example"
EXPECTED_FIELD = "is_requirement"
CLAUSE_FIELD = "text"

# Wording that makes a clause binding, and wording that explicitly does not. Deliberately
# crude: see the module docstring.
_OBLIGATION = re.compile(r"\b(shall|must|required to|requires|require)\b", re.IGNORECASE)
_PERMISSION = re.compile(r"\b(may|encouraged|optional|guidance only|not binding)\b", re.IGNORECASE)


def classify(text: str) -> str:
    """Say whether a clause reads as binding: "yes" or "no".

    Permission wording wins over obligation wording, so "may be required" reads as
    optional. That is a judgement call, stated here rather than buried in the regex order.
    """
    if _PERMISSION.search(text):
        return "no"
    return "yes" if _OBLIGATION.search(text) else "no"


def evaluate(request: EvaluationInput) -> EvaluationOutput:
    """Classify every gold item and score it, returning the Verdicts and a run record."""
    started = datetime.now(UTC)
    verdicts: list[Verdict] = []

    for item in request.gold.items:
        expected = str(item.expected.get(EXPECTED_FIELD, ""))
        actual = classify(str(item.inputs.get(CLAUSE_FIELD, "")))
        verdicts.append(exact_match(expected, actual, item_id=item.item_id))

    finished = datetime.now(UTC)
    # One step, no model calls: the classifier is a regex. The empty `model_calls` is what
    # makes this project's cost figures honestly zero rather than unknown.
    step = StepTrace(
        step_name="classify_clauses",
        started_at=started,
        finished_at=finished,
        status="ok",
        model_calls=[],
        metadata={"items": len(request.gold.items), "classifier": "rule-based"},
    )
    run = AgentRun(
        run_id=request.run_id,
        project=request.project,
        started_at=started,
        finished_at=finished,
        steps=[step],
        status="completed",
    )
    return EvaluationOutput(verdicts=verdicts, run=run)


register_evaluator(PROJECT, evaluate)

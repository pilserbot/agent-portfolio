"""The real extraction pass over Kessler Point Documents 4, 5 and 6, and the gate.

Needs ANTHROPIC_API_KEY and a network, so it is marked `integration` and excluded from
`make test-fast`. This is the first step in the project that calls a model in anger, and it
exists to prove three things the offline tests cannot:

1. **The anchor assertion holds against a model that really ran.** Every requirement the
   pass produced resolves to a page that actually contains its quoted span. The offline
   tests prove the arithmetic; this proves it on output nobody scripted.
2. **The gate measures something.** The extracted clause set is compared against Document 9,
   the Compliance Matrix, which extraction never held. Both directions are printed in full.
3. **The run's cost is a measurement, not an estimate.** Reported from the router's own
   ledger, with tokens broken out by purpose, through `spine.kpi.cost_report` rather than a
   second sum written here.

It writes its ledger to a temporary path, so a run never walks the project's real daily
spend cap toward its limit.

Deliberately does not: assert what the model said about any particular clause. How a clause
should be split is a judgement, and pinning one here would be pinning today's answer as the
contract. What is asserted is everything deterministic around it — the anchors, the lineage,
the scope, and that nothing was invented.
"""

import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from req_core.anchors import verify
from req_core.extraction import EXTRACT_PURPOSE, extract_requirements
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.extract.gate import (
    MATRIX_DOCUMENT_ID,
    compare,
    extraction_corpus,
    render_markdown,
    verify_scope,
)
from ri05_tender.tender.loader import load_tender
from spine.contracts import AgentRun, StepTrace
from spine.kpi import cost_report
from spine.router import Router, RouterConfig

KESSLER_POINT = Path("data/tenders/kessler_point")

# What the deterministic layer finds, and what the package's own index claims for these three
# documents. Pinned so a live run that reads fewer clauses fails here rather than quietly
# reporting a smaller, better-looking gate.
EXPECTED_CLAUSES = 301

# The estimate this run was approved against: ~18k prompt and ~21k completion tokens over 16
# calls at Sonnet 5's $2/$10 per million. A real run an order of magnitude above that is not
# a surprise to absorb — it is a bug in the batching, and it should fail the test.
COST_CEILING_USD = Decimal("3.00")


def router_completion(router: Router):  # noqa: ANN201 - returns a StructuredCompletion closure
    """Bind a router into the `StructuredCompletion` shape `req_core` takes.

    This adapter is the whole of the coupling between the extractor and this project's model
    routing. `req_core` holds no client, no key and no retry policy; it holds a callable.
    """

    def complete(prompt: str, schema: type[BaseModel], *, purpose: str) -> BaseModel:
        obj, _call = router.structured(prompt, schema, purpose=purpose, tier="large")
        return obj

    return complete


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_a_real_extraction_pass_anchors_every_requirement_and_clears_the_gate(
    tmp_path: Path,
) -> None:
    package = load_tender(KESSLER_POINT)
    corpus = extraction_corpus(package)

    # The corpus physically cannot hold the matrix — `SourceCorpus` refuses it — so this is a
    # restatement of a guarantee already made, placed here because the live run is the one
    # where it would matter most.
    assert MATRIX_DOCUMENT_ID not in corpus.document_ids
    assert corpus.withheld == {MATRIX_DOCUMENT_ID}

    router = Router(
        RouterConfig.from_env().model_copy(update={"ledger_path": tmp_path / "ledger.json"})
    )
    started = datetime.now(UTC)

    result = extract_requirements(
        corpus,
        router_completion(router),
        policy=ITB_2_1_POLICY,
        style=KESSLER_POINT_CLAUSE_STYLE,
    )

    finished = datetime.now(UTC)
    verify_scope(result, corpus)

    # --- 1. the hard assertion, on output nobody scripted --------------------------------
    anchors = verify(result.requirements, corpus)
    assert anchors.ok, anchors.describe()
    assert anchors.checked == len(result.requirements)

    # --- what the deterministic layer must still be doing ---------------------------------
    assert result.clauses_read == EXPECTED_CLAUSES
    assert len(result.clause_ids) == EXPECTED_CLAUSES
    assert result.policy_name == ITB_2_1_POLICY.name
    # Every child names a parent that is really in the set, and no id was invented.
    by_id = {requirement.requirement_id: requirement for requirement in result.requirements}
    for requirement in result.requirements:
        if requirement.parent_id is not None:
            assert requirement.parent_id in by_id
            assert requirement.requirement_id.startswith(requirement.parent_id)
    assert len(by_id) == len(result.requirements), "requirement ids must be unique"

    # --- 2. the gate ---------------------------------------------------------------------
    comparison = compare(result, package)

    # --- 3. the cost, from the ledger rather than from arithmetic written here ------------
    run = AgentRun(
        run_id=f"extract-{uuid.uuid4().hex[:8]}",
        project="ri05",
        started_at=started,
        finished_at=finished,
        steps=[
            StepTrace(
                step_name="req_core.extract",
                started_at=started,
                finished_at=finished,
                status="ok",
                model_calls=list(router.ledger.calls),
            )
        ],
        status="ok",
    )
    costs = cost_report(run)

    assert costs.basis is not None, costs.refusal
    assert costs.basis.is_measured, "a live pass must not be reporting replayed calls"
    assert costs.total_calls > 0
    assert {usage.purpose for usage in costs.by_purpose} == {EXTRACT_PURPOSE}

    # Printed, and also written beside the ledger so a CI run can upload it: pytest swallows
    # stdout on a passing test, and the report is the reason to make the run at all.
    report = _render(result, comparison, run, costs, anchors)
    print(report)
    (tmp_path / "extraction_gate.md").write_text(report, encoding="utf-8")

    assert "extraction gate" in report
    assert costs.basis.total_usd < COST_CEILING_USD, (
        f"the pass cost ${costs.basis.total_usd} against a ceiling of ${COST_CEILING_USD}. "
        f"The estimate was ~$0.25 over 16 calls; an order of magnitude above that is a "
        f"batching bug, not a surprise to absorb."
    )


def _render(result, comparison, run, costs, anchors) -> str:  # noqa: ANN001 - a local formatter
    """The whole run as one block, for a human reading the test output."""
    lines = [
        "",
        render_markdown(comparison, result),
        "### Anchors",
        "",
        anchors.describe(),
        "",
        "### Cost, from the router's ledger",
        "",
        f"Run `{run.run_id}` — {costs.total_calls} call(s), "
        f"{costs.billed_calls} billed, {costs.replayed_calls} replayed.",
        "",
        "| Purpose | Calls | Prompt | Completion | Cached | Cost (USD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for usage in costs.by_purpose:
        cost = "—" if usage.cost_usd is None else f"{usage.cost_usd:.6f}"
        lines.append(
            f"| `{usage.purpose}` | {usage.calls} | {usage.prompt_tokens} | "
            f"{usage.completion_tokens} | {usage.cached_tokens} | {cost} |"
        )
    assert costs.basis is not None
    lines += [
        "",
        f"**Total: ${costs.basis.total_usd:.4f}** over "
        f"{run.total_prompt_tokens} prompt and {run.total_completion_tokens} completion "
        f"tokens ({run.total_cached_tokens} cached).",
        "",
    ]
    return "\n".join(lines)

"""The one live extraction pass a labelled run makes, and the record of what it produced.

Extraction is deterministic given the corpus and the policy: the same three documents, the
same segmenter, the same modality convention. Running it twice in one job produced the same
requirements twice and cost 14 calls, about $0.89 and roughly ten minutes the second time —
which is most of what pushed the eval job into its wall-clock limit. So it runs once, through the
session-scoped fixture in `conftest`, and both the extraction gate and the findings run
read that one result.

`calls` travels on the record rather than the `Router` that made them. The router is a live
object with a spend cap and a ledger path; handing it between tests would let one test's
assertions depend on another test's spending. The calls are frozen records and cannot.

Deliberately does not: assert anything, detect, score, define the fixture, or persist
anything but the ledger it is obliged to persist. What the extraction means is the gate
test's business; this only makes sure it is produced exactly once.
"""

from datetime import UTC, datetime
from pathlib import Path

from ledger import LedgerSnapshot, persist_ledger
from pydantic import BaseModel, ConfigDict, Field

from req_core.contracts import ExtractionResult
from req_core.corpus import SourceCorpus
from req_core.extraction import StructuredCompletion, extract_requirements
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.extract.gate import extraction_corpus
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage
from spine.contracts import ModelCall
from spine.router import Router, RouterConfig

KESSLER_POINT = Path("data/tenders/kessler_point")

# Extraction is one call per page that carries a clause. Asserted by the tests that use
# this, so a batching regression fails on a number rather than on a bill.
EXPECTED_EXTRACTION_CALLS = 14


class LiveExtraction(BaseModel):
    """One real extraction pass over the real tender, and everything it produced."""

    model_config = ConfigDict(frozen=True)

    package: TenderPackage
    corpus: SourceCorpus
    result: ExtractionResult
    snapshot: LedgerSnapshot
    calls: list[ModelCall] = Field(
        description="The pass's model calls, as records. Carried instead of the Router that "
        "made them so no test's assertions can depend on another test's spending."
    )
    started_at: datetime
    finished_at: datetime


def router_completion(router: Router) -> StructuredCompletion:
    """Bind a router into the `StructuredCompletion` shape `req_core` takes.

    This adapter is the whole of the coupling between the extractor and this project's model
    routing. `req_core` holds no client, no key and no retry policy; it holds a callable.
    """

    def complete(prompt: str, schema: type[BaseModel], *, purpose: str) -> BaseModel:
        obj, _call = router.structured(prompt, schema, purpose=purpose, tier="large")
        return obj

    return complete


def extract_once(ledger_path: Path) -> LiveExtraction:
    """Run extraction, live, exactly once, and record everything it produced.

    The ledger is persisted the instant extraction returns or raises, before this record is
    built. A helper that raised while assembling its return value would otherwise throw away
    calls that were already paid for, which is the failure this project has already had once
    and does not intend to have again.
    """
    package = load_tender(KESSLER_POINT)
    corpus = extraction_corpus(package)
    router = Router(RouterConfig.from_env().model_copy(update={"ledger_path": ledger_path}))

    started = datetime.now(UTC)
    try:
        result = extract_requirements(
            corpus,
            router_completion(router),
            policy=ITB_2_1_POLICY,
            style=KESSLER_POINT_CLAUSE_STYLE,
        )
    finally:
        snapshot = persist_ledger(router, label="full_pass")
    finished = datetime.now(UTC)

    return LiveExtraction(
        package=package,
        corpus=corpus,
        result=result,
        snapshot=snapshot,
        calls=list(router.ledger.calls),
        started_at=started,
        finished_at=finished,
    )

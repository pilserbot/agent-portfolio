"""The RI-05 scoring harness: how a run over a labelled tender is measured.

The pipeline this measures does not exist yet, and that is deliberate — the measurement is
defined first, so the pipeline is built against a target rather than the target being drawn
around whatever the pipeline happened to do. Everything here is exercised against fixtures.

- `models` — `GoldItem`, mirroring one line of a tender's answer key, and `Finding`, the
  shape the pipeline will emit.
- `loader` — reads a `TenderPackage`'s answer key, and refuses a tender that has none.
- `matcher` — decides, deterministically and one-to-one, which findings answer which gold
  items, ranking candidate pairs by completeness before overlap. Greedy, and defined as
  greedy so the figure is reproducible. No model call anywhere in it.
- `metrics` — recall overall and by class, tier and severity; severity-weighted recall;
  implicit recovery; both precisions; the no-bid gate. Every ratio is computed by
  `spine.eval.metrics` through a thin adapter rather than reimplemented here.
- `adjudication` — an unmatched finding is put to a human, because the gold set records what
  was planted, not every defect in the package.
- `report` — the score card as markdown.

Deliberately does not: produce a finding, call a model, or reach a network. It measures.
"""

__all__ = ["adjudication", "loader", "matcher", "metrics", "models", "report"]

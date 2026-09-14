"""Rendering a score card as the markdown a human reads.

The order is deliberate. The header states how many items were scored **and how many were
excluded**, because a denominator that shrank without anyone saying so is the easiest way
for a score to improve while the system does not. Then the headline figures — with UNSTATED
and DISPLACED recovery on separate rows, each carrying its own denominator, because they
were one row until rev 3 and that row was measuring two different things — then recall
broken out three ways, then the no-bid gate on a line of its own, then the outcomes that
are neither a find nor a clean miss: PARTIAL, OUTPUT_MISS, DUPLICATE, and the findings
still waiting on a human.

The gate gets its own line and the word PASS or FAIL because it is not a rate and must not
be read as one. Everything else on the page can be traded off; that cannot.

Deliberately does not: compute anything. Every number here was decided by `metrics` before
this module saw it, and the one judgement made here is what to put first.
"""

from ri05_tender.eval.metrics import Breakdown, ScoreCard

__all__ = ["render_markdown"]


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]], aligns: str) -> list[str]:
    """A markdown table, or a single row saying there was nothing to show."""
    rule = "| " + " | ".join("---:" if a == "r" else "---" for a in aligns) + " |"
    lines = ["| " + " | ".join(header) + " |", rule]
    if not rows:
        return [*lines, "| " + " | ".join(["_none_", *([""] * (len(header) - 1))]) + " |"]
    return [*lines, *("| " + " | ".join(row) + " |" for row in rows)]


def _breakdown_rows(breakdowns: list[Breakdown]) -> list[tuple[str, ...]]:
    """One row per group: the key, the recall, and the counts behind it."""
    return [(item.key, f"{item.value:.1%}", f"{item.found} / {item.total}") for item in breakdowns]


def render_markdown(result: ScoreCard) -> str:
    """Render a score card for a pull request comment or a report file."""
    excluded = (
        f"{result.excluded_items} excluded (`scored: false`)"
        if result.excluded_items
        else "none excluded"
    )
    lines = [
        f"## RI-05 scoring — {result.tender_name}",
        "",
        f"**{result.scored_items} scored gold item(s)**, {excluded}. "
        f"{result.findings} finding(s) reported.",
        "",
        *_table(
            ("Metric", "Value", "Of"),
            [
                (
                    "Recall (overall)",
                    f"{result.recall_overall:.1%}",
                    f"{result.matches} / {result.scored_items}",
                ),
                ("Recall (severity-weighted)", f"{result.recall_weighted:.1%}", "by weight"),
                (
                    "UNSTATED recovery",
                    f"{result.unstated_recovery:.1%}",
                    f"{result.unstated_total} item(s)",
                ),
                (
                    "DISPLACED recovery",
                    f"{result.displaced_recovery:.1%}",
                    f"{result.displaced_total} item(s)",
                ),
                (
                    "Precision (strict)",
                    f"{result.precisions.strict:.1%}",
                    f"{result.precisions.strict_support} finding(s)",
                ),
                (
                    "Precision (adjudicated)",
                    f"{result.precisions.adjudicated:.1%}",
                    f"{result.precisions.adjudicated_support} finding(s)",
                ),
            ],
            "lrr",
        ),
        "",
        "Both precisions are shown because the gold set records what was *planted*, not "
        "every defect in the package: strict counts every unmatched finding against the "
        "system, adjudicated counts only the ones a human called wrong.",
        "",
        "UNSTATED and DISPLACED recovery are two rows and never one. UNSTATED items are "
        "written nowhere in the package; DISPLACED items are written out plainly, in a "
        "drawing note or an annex nobody reads. Reading finds the second and cannot find "
        "the first, so an average over both says less the more it moves.",
        "",
        "### Recall by severity",
        "",
        *_table(("Severity", "Recall", "Found"), _breakdown_rows(result.by_severity), "lrr"),
        "",
        "### Recall by tier",
        "",
        *_table(("Tier", "Recall", "Found"), _breakdown_rows(result.by_tier), "lrr"),
        "",
        "### Recall by class",
        "",
        *_table(("Class", "Recall", "Found"), _breakdown_rows(result.by_class), "lrr"),
        "",
        "### No-bid gate",
        "",
        f"**{result.passed_gate}** — {result.no_bid_found} of {result.no_bid_total} scored "
        f"`no_bid` item(s) matched.",
        "",
        _gate_note(result),
        "",
        "### Everything else",
        "",
        *_table(
            ("Outcome", "Count", "What it means"),
            [
                ("MATCH", str(result.matches), "found, with every required output"),
                (
                    "OUTPUT_MISS",
                    str(result.output_misses),
                    "right defect, a required output missing — not counted as found",
                ),
                (
                    "PARTIAL",
                    str(result.partials),
                    "right clause, wrong kind of finding — not counted as found",
                ),
                (
                    "DUPLICATE",
                    str(result.duplicates),
                    "restates a defect another finding already answered — counts against "
                    "strict precision, never adjudicated",
                ),
                ("MISS", str(result.misses), "no finding was credited with it"),
                (
                    "UNMATCHED",
                    str(result.unmatched),
                    "cites nothing in the key — awaiting adjudication",
                ),
                (
                    "Pending adjudication",
                    str(result.precisions.pending),
                    "unmatched findings nobody has ruled on yet",
                ),
                (
                    "Confirmed new",
                    str(result.precisions.true_new),
                    "real defects the key had not planted",
                ),
            ],
            "lrl",
        ),
        "",
    ]
    return "\n".join(lines)


def _gate_note(result: ScoreCard) -> str:
    """The sentence under the gate, which says what the verdict means rather than repeating it."""
    if result.no_bid_total == 0:
        return (
            "_This tender plants no `no_bid` items, so the gate passes without asserting "
            "anything. It is not evidence the system would refuse a tender it should._"
        )
    if result.no_bid_gate:
        return "_Every tender the system should refuse, it found grounds to refuse._"
    return (
        "_The gate is not a rate: a bid going out on a tender the system should have "
        "refused is not offset by anything else on this page._"
    )

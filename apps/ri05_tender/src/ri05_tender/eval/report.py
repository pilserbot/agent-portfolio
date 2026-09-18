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

from ri05_tender.detect.config import CONFIDENCE_THRESHOLD, SUB_THRESHOLD_SHAPES
from ri05_tender.eval.metrics import AbstentionCost, Breakdown, RecallPair, ScoreCard

__all__ = ["render_markdown"]


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]], aligns: str) -> list[str]:
    """A markdown table, or a single row saying there was nothing to show."""
    rule = "| " + " | ".join("---:" if a == "r" else "---" for a in aligns) + " |"
    lines = ["| " + " | ".join(header) + " |", rule]
    if not rows:
        return [*lines, "| " + " | ".join(["_none_", *([""] * (len(header) - 1))]) + " |"]
    return [*lines, *("| " + " | ".join(row) + " |" for row in rows)]


def _keyed(pairs: list[RecallPair]) -> dict[str, RecallPair]:
    """The pairs by group key, so a breakdown row can find its own."""
    return {pair.key: pair for pair in pairs}


def _group_header(name: str, cost: AbstentionCost | None) -> tuple[str, ...]:
    """The header for a breakdown table, widened when there is a delta to show."""
    base = (name, "Recall", "Found")
    return base if cost is None else (*base, "If asserted", "Δ")


def _pad(row: tuple[str, ...], width: int) -> tuple[str, ...]:
    """Fill a row out to the table's width with em dashes.

    Precision has no if-asserted reading: scoring the withheld findings would change the
    denominator as well as the numerator, and a precision computed over findings the run
    declined to make is a number about a run that did not happen. The cells stay empty
    rather than being filled with a figure that looks comparable and is not.
    """
    return row + ("—",) * (width - len(row))


def _breakdown_rows(
    breakdowns: list[Breakdown], pairs: dict[str, RecallPair] | None = None
) -> list[tuple[str, ...]]:
    """One row per group: the key, the recall, the counts, and what abstention cost it.

    With `pairs`, every group gains the reading it would have had with no threshold and the
    delta between the two. That delta per group is what says which detector's zero is a
    detector that cannot find the defect and which is one that found it and declined.
    """
    rows: list[tuple[str, ...]] = []
    for item in breakdowns:
        row = (item.key, f"{item.value:.1%}", f"{item.found} / {item.total}")
        if pairs is None:
            rows.append(row)
            continue
        pair = pairs.get(item.key)
        rows.append(
            row + ("—", "—")
            if pair is None
            else row + (f"{pair.value_if_asserted:.1%}", f"{pair.delta:+.1%}")
        )
    return rows


def render_markdown(result: ScoreCard) -> str:
    """Render a score card for a pull request comment or a report file."""
    excluded = (
        f"{result.excluded_items} excluded (`scored: false`)"
        if result.excluded_items
        else "none excluded"
    )
    cost = result.abstention

    def recall_row(label: str, value: float, of: str, pair: object | None) -> tuple[str, ...]:
        """One recall row, with the if-asserted reading and the delta when there is one."""
        if cost is None:
            return (label, f"{value:.1%}", of)
        if pair is None:
            return (label, f"{value:.1%}", of, "—", "—")
        return (
            label,
            f"{value:.1%}",
            of,
            f"{pair.value_if_asserted:.1%}",
            f"{pair.delta:+.1%}",
        )

    header = (
        ("Metric", "Value", "Of") if cost is None else ("Metric", "Value", "Of", "If asserted", "Δ")
    )
    align = "lrr" if cost is None else "lrrrr"

    lines = [
        f"## RI-05 scoring — {result.tender_name}",
        "",
        f"**{result.scored_items} scored gold item(s)**, {excluded}. "
        f"{result.findings} finding(s) reported.",
        "",
        *_table(
            header,
            [
                recall_row(
                    "Recall (overall)",
                    result.recall_overall,
                    f"{result.matches} / {result.scored_items}",
                    cost.overall if cost else None,
                ),
                *(
                    []
                    if result.addressable is None
                    else [
                        recall_row(
                            "Recall over addressable items",
                            result.addressable.recall,
                            f"{result.addressable.found} / {result.addressable.total}",
                            cost.addressable if cost else None,
                        ),
                        recall_row(
                            "— of those, reachable today",
                            result.addressable.recall_reachable,
                            f"{result.addressable.found} / {result.addressable.reachable_total}",
                            None,
                        ),
                    ]
                ),
                (
                    ("Recall (severity-weighted)", f"{result.recall_weighted:.1%}", "by weight")
                    if cost is None
                    else (
                        "Recall (severity-weighted)",
                        f"{result.recall_weighted:.1%}",
                        "by weight",
                        f"{cost.weighted_value_if_asserted:.1%}",
                        f"{cost.weighted_delta:+.1%}",
                    )
                ),
                recall_row(
                    "UNSTATED recovery",
                    result.unstated_recovery,
                    f"{result.unstated_total} item(s)",
                    cost.unstated if cost else None,
                ),
                recall_row(
                    "DISPLACED recovery",
                    result.displaced_recovery,
                    f"{result.displaced_total} item(s)",
                    cost.displaced if cost else None,
                ),
                _pad(
                    (
                        "Precision (strict)",
                        f"{result.precisions.strict:.1%}",
                        f"{result.precisions.strict_support} finding(s)",
                    ),
                    len(header),
                ),
                _pad(
                    (
                        "Precision (adjudicated)",
                        f"{result.precisions.adjudicated:.1%}",
                        f"{result.precisions.adjudicated_support} finding(s)",
                    ),
                    len(header),
                ),
            ],
            align,
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
        *_table(
            _group_header("Severity", cost),
            _breakdown_rows(result.by_severity, _keyed(cost.by_severity) if cost else None),
            align,
        ),
        "",
        "### Recall by tier",
        "",
        *_table(
            _group_header("Tier", cost),
            _breakdown_rows(result.by_tier, _keyed(cost.by_tier) if cost else None),
            align,
        ),
        "",
        "### Recall by class",
        "",
        *_table(
            _group_header("Class", cost),
            _breakdown_rows(result.by_class, _keyed(cost.by_class) if cost else None),
            align,
        ),
        "",
        *_abstention_section(result),
        *_addressable_section(result),
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


def _addressable_section(result: ScoreCard) -> list[str]:
    """The ceiling behind the recall figure, and a row for every item inside it.

    Printed only when the caller said which finding types and outputs are wired up. A
    recall of 0.0% over 186 items and a recall of 0.0% over the 32 the detectors can even
    speak to are different statements, and the second is the one that says whether the
    detectors are failing to find planted defects or were never able to answer them.
    """
    card = result.addressable
    if card is None:
        return []

    blocked = card.blocked_on_outputs
    outputs = ", ".join(f"`{name}`" for name in card.emitted_outputs) or "**none**"
    lines = [
        "### What the detectors could answer at all",
        "",
        f"Of **{card.scored_total}** scored item(s), **{card.total}** are *addressable*: "
        f"their classes map to a finding type one of the registered detectors emits "
        f"({', '.join(f'`{name}`' for name in card.emitted_finding_types)}). The other "
        f"{card.scored_total - card.total} cannot be answered by any detector that exists, "
        f"so they are in recall's denominator and out of this one.",
        "",
        f"Outputs any registered detector carries: {outputs}. **{blocked}** of the "
        f"{card.total} addressable item(s) demand an output nothing emits and cannot reach "
        f"MATCH however well the clause is read; **{card.reachable_total}** demand nothing "
        f"that is missing, and those are the MATCH ceiling as the system stands.",
        "",
        "_A recall figure whose ceiling is not stated beside it is not interpretable. "
        "These two denominators do not replace the overall one — the product is measured "
        "against every planted defect — they say which part of the gap is this step's._",
        "",
        "#### Every addressable item",
        "",
        *_table(
            ("Item", "Classes", "Severity", "Outcome", "What blocked it"),
            [
                (
                    f"`{item.gold_id}`",
                    ", ".join(item.classes),
                    item.severity,
                    item.outcome.upper(),
                    item.blocked_by or "—",
                )
                for item in card.items
            ],
            "lllll",
        ),
        "",
        "_PARTIAL here is not one of the matcher's gold-item outcomes: for recall the item "
        "is a miss. It is named so the reason stays visible — a finding cited the clause "
        "and said the wrong kind of thing about it, which is a different failure from "
        "nothing citing the clause at all._",
        "",
    ]
    return lines


def _abstention_section(result: ScoreCard) -> list[str]:
    """What the confidence threshold cost, per group, as a number.

    A recall of zero has two completely different causes and they are indistinguishable
    from the figure alone: the detectors never located the defect, or they located it and
    the threshold withheld the finding. The delta separates them. Where it is zero, the
    detectors did not find it; where it is positive, they did and declined to say so.
    """
    cost = result.abstention
    if cost is None:
        return []

    moved = [pair for pair in cost.by_class if pair.recovered]
    lines = [
        "### What abstention cost",
        "",
        f"**{cost.abstained_findings}** of {cost.asserted_findings + cost.abstained_findings} "
        f"finding(s) — {cost.abstention_rate:.1%} — were below the confidence threshold and "
        f"were routed to a human rather than asserted. Scoring them alongside the asserted "
        f"ones moves overall recall by **{cost.overall.delta:+.1%}** "
        f"({cost.overall.recovered:+d} item(s)).",
        "",
        "_Nothing in that second figure is credited to the run. It is the price of the "
        "abstention policy, in recall, and it exists so a zero can be read: where the delta "
        "is zero the detectors did not find the defect, and where it is positive they found "
        "it and declined to assert it._",
        "",
    ]
    if moved:
        lines += [
            "Classes where a withheld finding would have changed the count:",
            "",
            *_table(
                ("Class", "Recall", "If asserted", "Δ", "Items recovered"),
                [
                    (
                        pair.key,
                        f"{pair.value:.1%}",
                        f"{pair.value_if_asserted:.1%}",
                        f"{pair.delta:+.1%}",
                        f"{pair.recovered:+d} of {pair.total}",
                    )
                    for pair in moved
                ],
                "lrrrr",
            ),
            "",
        ]
    else:
        lines += [
            "_No withheld finding would have matched any scored item. The abstention policy "
            "cost this run nothing in recall, which is a measurement and not a guarantee._",
            "",
        ]

    lines += [
        "#### Shapes the threshold makes unreportable",
        "",
        f"These detector shapes score below the threshold of {CONFIDENCE_THRESHOLD:.2f} and "
        f"therefore can never be asserted. Two independently reasonable numbers — a detector "
        f"author's judgement that a shape is worth a glance rather than an assertion, and "
        f"this project's judgement about where assertion begins — combine into an outcome "
        f"neither intended.",
        "",
        *_table(
            ("Detector", "Confidence", "Shape", "What the zero does not mean"),
            [
                (f"`{shape.detector}`", f"{shape.confidence:.2f}", shape.shape, shape.consequence)
                for shape in SUB_THRESHOLD_SHAPES
            ],
            "lrll",
        ),
        "",
        "_A class whose only planted defect has one of these shapes scores zero recall by "
        "construction. That is a stated limitation of the two settings together, not a "
        "detector failing to see the clause — and neither setting is moved to improve a "
        "figure, because a threshold tuned against the answer key measures the developer._",
        "",
    ]
    return lines

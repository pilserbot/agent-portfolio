"""Streamlit dashboards over the portfolio's evaluation results and KPIs.

`dashboards.kpi_data` reads a project's declared KPIs and its result history off disk and
assembles what a page shows. `apps/dashboards/kpi_app.py` is the Streamlit entry point over
it — the split keeps every decision about *what* is displayed testable without a browser,
and keeps the UI module to layout and words.

Deliberately does not: compute any figure it displays, or reach a verdict of its own. Every
number shown here is produced by deterministic code in `spine`, written to a file, and
merely presented. Nothing in this package holds domain knowledge about any project either:
what a project reports is read from its config.
"""

__version__ = "0.1.0"

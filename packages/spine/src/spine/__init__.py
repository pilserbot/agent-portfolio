"""Shared evaluation, model routing and KPI library for the agent portfolio.

Exposes the cross-cutting machinery every application in this repository reuses:
evaluation runs, model selection and KPI computation, all over Pydantic v2 models.

Deliberately does not: hold any domain knowledge (tenders, requirements, customers),
call language models to produce a judgement, or persist anything. Scores, rankings and
verdicts are computed here by deterministic Python; storage belongs to the applications.
"""

__version__ = "0.1.0"

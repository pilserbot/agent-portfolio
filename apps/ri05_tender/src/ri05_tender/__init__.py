"""Tender response engine (FastAPI application).

Will expose the HTTP surface that ingests a tender, extracts its requirements through
`req_core`, and assembles a scored response using `spine`.

Deliberately does not: reimplement extraction, evaluation, routing or KPI logic — those
live in the shared packages. This package holds transport, configuration and wiring only.
Empty for now: no endpoints, no business logic.
"""

__version__ = "0.1.0"

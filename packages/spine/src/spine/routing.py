"""Model routing: choosing which model handles a unit of work, and under which budget.

Deliberately does not: call any model provider, hold API credentials, or make the routing
decision with a model. Routing is a deterministic function of the typed request and the
configured policy; credentials stay in the environment and are read by the caller.
"""

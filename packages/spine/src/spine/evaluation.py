"""Evaluation primitives: running a gold set through a system and scoring the result.

Deliberately does not: define what "good" means for any specific domain, fetch gold sets
from the network, or ask a language model to grade an answer. Grading is deterministic
Python over typed structures; domain-specific rubrics are supplied by the caller.
"""

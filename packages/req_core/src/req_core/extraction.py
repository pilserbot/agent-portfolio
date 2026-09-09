"""Extraction pass: unstructured text to typed requirement structures.

This is the one place in the package where a language model is used, and it is used for
exactly one thing: converting prose into structures.

Deliberately does not: assign statuses, scores, priorities or monetary figures to what it
extracts, and does not let the model decide anything beyond the shape of the text. Every
downstream verdict is deterministic Python operating on the models returned here.
"""

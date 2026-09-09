"""KPI computation: turning typed run records into the numbers a dashboard displays.

Deliberately does not: render anything, query a database, or estimate a figure with a
language model. Every monetary amount, rate and count here is arithmetic over structures
the caller has already loaded.
"""

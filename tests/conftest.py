"""Test-suite setup shared by every test.

Pins litellm to its bundled price map so importing it makes no network call, keeping the
unit suite hermetic and fast. Set before any test imports litellm, which reads this on
import.

Deliberately does not: provide credentials, fixtures that reach a network, or any default
that a test could mistake for real configuration.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

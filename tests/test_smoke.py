"""Smoke test: the workspace packages import and expose a version.

Deliberately does not touch the network, the filesystem or any environment variable.
"""

import dashboards
import req_core
import ri05_tender
import spine


def test_workspace_packages_are_importable() -> None:
    assert spine.__version__
    assert req_core.__version__
    assert ri05_tender.__version__
    assert dashboards.__version__

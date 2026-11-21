"""Session setup for the examples tests."""

from __future__ import annotations

import os

# The tests import component entrypoints (speediedan/rte and others) from the components cache, which interpretune
# refuses to execute without an explicit opt-in. These are this repository's own first-party components.
os.environ.setdefault("IT_TRUST_REMOTE_CODE", "1")

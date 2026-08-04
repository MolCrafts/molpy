"""Regression: dual public names deleted in 0.12 API cleanup."""

from __future__ import annotations

import inspect

from molpy.builder.ambertools import AmberResult
from molpy.core.atomistic import Atomistic
import molpy.pack as pack

assert "forcefield" in AmberResult.__dataclass_fields__
assert "ff" not in AmberResult.__dataclass_fields__
assert not hasattr(AmberResult, "ff")

sig = inspect.signature(Atomistic.get_topo)
assert "entity_type" not in sig.parameters
assert "link_type" not in sig.parameters

assert not hasattr(pack, "get_packer")
print("ok release-0-12-molpy-01-api-cleanup")

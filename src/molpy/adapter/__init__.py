"""Adapters for synchronizing MolPy internal structures with external representations.

Adapters are responsible for *data synchronization only* (in-memory conversion and/or
file artifact read/write). Adapters MUST NOT execute external binaries; execution
belongs in wrappers (see :mod:`molpy.wrapper`) and/or higher-level compute nodes.
"""

from .base import Adapter

# Optional RDKit adapter
try:  # pragma: no cover
    from .rdkit import MP_ID, RDKitAdapter

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing
    _HAS_RDKIT = False
    MP_ID = None  # type: ignore[assignment]
    RDKitAdapter = None  # type: ignore[assignment]

__all__ = [
    "Adapter",
]

if _HAS_RDKIT:
    __all__ += ["RDKitAdapter", "MP_ID"]

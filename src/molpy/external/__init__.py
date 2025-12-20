"""Compatibility facade for external integrations.

During the external-layer refactor, canonical import paths are:
- Adapters: :mod:`molpy.adapter`
- Wrappers: :mod:`molpy.wrapper`

This package is retained temporarily to reduce user breakage. New code should
prefer importing directly from `molpy.adapter` / `molpy.wrapper`.
"""

from molpy.adapter import Adapter
from molpy.wrapper import AntechamberWrapper, TLeapWrapper, Wrapper

# RDKit is an optional dependency. Keep imports guarded so that `import molpy`
# works without rdkit installed.
try:  # pragma: no cover
    from molpy.adapter import MP_ID, RDKitAdapter
    from .rdkit_compute import Generate3D, OptimizeGeometry

    _HAS_RDKIT = True
except ModuleNotFoundError:
    _HAS_RDKIT = False
    MP_ID = None  # type: ignore[assignment]
    RDKitAdapter = None  # type: ignore[assignment]
    Generate3D = None  # type: ignore[assignment]
    OptimizeGeometry = None  # type: ignore[assignment]

__all__ = ["Adapter", "Wrapper", "AntechamberWrapper", "TLeapWrapper"]

if _HAS_RDKIT:
    __all__ += ["RDKitAdapter", "MP_ID", "Generate3D", "OptimizeGeometry"]

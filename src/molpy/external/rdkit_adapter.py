"""Compatibility shim for the refactor.

Prefer importing from :mod:`molpy.adapter`.
"""

try:  # pragma: no cover
    from molpy.adapter.rdkit import MP_ID, RDKitAdapter

    __all__ = ["RDKitAdapter", "MP_ID"]
except ModuleNotFoundError:  # rdkit missing
    MP_ID = None  # type: ignore[assignment]
    RDKitAdapter = None  # type: ignore[assignment]
    __all__ = ["RDKitAdapter", "MP_ID"]

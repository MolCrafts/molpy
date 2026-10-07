"""Adapters synchronise a molpy structure with an external library's objects.

Adapters do *data synchronisation only* — in-memory conversion and/or file
artifact read/write. They MUST NOT execute external binaries; execution belongs
in :mod:`molpy.wrapper`. Those are the two bridging patterns, and MolPy keeps a
worked example of each: :class:`RdkitAdapter` here, and the AmberTools CLI
wrappers (antechamber, parmchk2, prepgen, tleap) on the wrapper side.

**An example is not a dependency.** RDKit is an optional extra: importing molpy
never requires it, no molpy code path routes through it, and this package is the
only place in the source tree allowed to import it. Everything MolPy needs for
itself is native — 3D embedding is :class:`molpy.conformer.Conformer` (native
ETKDGv3 + MMFF94 cleanup), perception is :mod:`molpy.perceive`'s ``perceive_*`` /
``assign_*`` functions, SMILES is ``mp.io.smiles.SmilesIr``, ring facts are
``RingSet``, and GAFF typing
is antechamber delegation. Reach for the adapter to use *RDKit's* algorithms,
not to do something molpy already does.

One adapter exemplar is enough, so RDKit is the only one.
"""

from ._adapter import Adapter

# Optional RDKit adapter — the worked example of this pattern.
try:  # pragma: no cover - depends on whether the optional extra is installed
    from ._rdkit import MP_ID, RdkitAdapter

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing, which is the normal case
    _HAS_RDKIT = False
    MP_ID = None  # type: ignore[assignment]
    RdkitAdapter = None  # type: ignore[assignment]

__all__ = ["Adapter"]

if _HAS_RDKIT:
    __all__ += ["RdkitAdapter", "MP_ID"]

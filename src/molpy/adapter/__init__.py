"""Adapters synchronise a molpy structure with an external library's objects.

Adapters do *data synchronisation only* — in-memory conversion and/or file
artifact read/write. They MUST NOT execute external binaries; execution belongs
in :mod:`molpy.wrapper`. Those are the two bridging patterns, and MolPy keeps
one worked example of each: :class:`RDKitAdapter` here, and the Packmol packer
(:mod:`molpy.pack.packer.packmol`) on the wrapper side.

**An example is not a dependency.** RDKit is an optional extra: importing molpy
never requires it, no molpy code path routes through it, and this package is the
only place in the source tree allowed to import it. Everything MolPy needs for
itself is native — 3D embedding is :class:`molpy.conformer.Conformer` (molrs
ETKDGv3 + MMFF94 cleanup), perception is :class:`molpy.Perceive`, SMILES is
:class:`molrs.io.SmilesIR`, ring facts are :class:`molrs.perceive.RingInfo`, and GAFF typing
is antechamber delegation. Reach for the adapter to use *RDKit's* algorithms,
not to do something molpy already does.

One adapter exemplar is enough, so RDKit is the only one.
"""

from .base import Adapter

# Optional RDKit adapter — the worked example of this pattern.
try:  # pragma: no cover - depends on whether the optional extra is installed
    from .rdkit import MP_ID, RDKitAdapter

    _HAS_RDKIT = True
except ModuleNotFoundError:  # rdkit missing, which is the normal case
    _HAS_RDKIT = False
    MP_ID = None  # type: ignore[assignment]
    RDKitAdapter = None  # type: ignore[assignment]

__all__ = ["Adapter"]

if _HAS_RDKIT:
    __all__ += ["RDKitAdapter", "MP_ID"]

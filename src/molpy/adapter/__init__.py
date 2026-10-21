"""Adapters convert a molpy structure to and from an external library's objects.

An adapter is a pair of pure functions, ``to_<lib>`` and ``from_<lib>``: each
builds a new object and leaves its input untouched. Adapters do *in-memory
conversion only*; they MUST NOT execute external binaries, which belongs in
:mod:`molpy.wrapper`. Those are the two bridging patterns, and MolPy keeps a
worked example of each: :func:`to_rdkit` / :func:`from_rdkit` here, and the
AmberTools CLI wrappers (antechamber, parmchk2, prepgen, tleap) on the wrapper
side.

**An example is not a dependency.** RDKit is an optional extra: importing molpy
— or this package — never requires it, no molpy code path routes through it,
and this package is the only place in the source tree allowed to import it.
Reading one of its names without RDKit installed raises ``ImportError``.
Everything MolPy needs for itself is native — 3D embedding is
:class:`molpy.conformer.Conformer` (native ETKDGv3 + MMFF94 cleanup),
perception is :mod:`molpy.perceive`'s ``perceive_*`` / ``assign_*`` functions,
SMILES is ``mp.io.smiles.SmilesIr``, ring facts are ``RingSet``, and GAFF typing
is ``mp.ff.typifier.AtdTypifier`` + ``GaffTypifier`` (AmberTools is needed only
for AM1-BCC charges). Reach for the adapter to use *RDKit's* algorithms,
not to do something molpy already does.

One adapter exemplar is enough, so RDKit is the only one.
"""

from __future__ import annotations

__all__ = ["PROP_PREFIX", "from_rdkit", "generate_3d", "to_rdkit"]

_RDKIT_MISSING: ModuleNotFoundError | None = None
try:
    from molpy.adapter._rdkit import PROP_PREFIX, from_rdkit, generate_3d, to_rdkit
except ModuleNotFoundError as exc:
    if exc.name != "rdkit":  # a broken install, not an absent optional extra
        raise
    _RDKIT_MISSING = exc


def __getattr__(name: str) -> object:
    if name in __all__ and _RDKIT_MISSING is not None:
        raise ImportError(
            f"molpy.adapter.{name} needs RDKit, which is not installed: "
            "pip install 'molcrafts-molpy[rdkit]'"
        ) from _RDKIT_MISSING
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

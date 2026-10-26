"""Canonical keys, projected from the compiled Rust tables — ``molrs::core::keys``.

Column keys and their ordered groups (``COORDS``, …), frame-meta keys,
molecular-graph keys (``PORTS``, ``FRAG_ID``, ``EQUIV_CLASS``, …) and the
LAMMPS frame-meta keys. Every scalar is a :class:`Key`; a group is a list of
:class:`Key`. A key added to the Rust tables appears here with no edit.
"""

from molpy import _core

__all__ = sorted(name for name in dir(_core.keys) if not name.startswith("_"))

globals().update({name: getattr(_core.keys, name) for name in __all__})

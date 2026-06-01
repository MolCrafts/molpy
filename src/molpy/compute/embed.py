"""3D coordinate generation — molrs-backed (main trunk).

``Generate3D`` is a :class:`~molpy.compute.base.Compute` operator that wraps
the molrs ``generate_3d`` Rust pipeline (fragment build → distance geometry →
energy minimization → rotor/stereo guard). It takes a :class:`molpy.Atomistic`
heavy-atom graph and returns a fresh structure with generated coordinates;
the input is never mutated.

This replaces the former RDKit-backed ``compute/rdkit.py``. The RDKit adapter
(:mod:`molpy.adapter.rdkit`), which also hosts the optional RDKit ``Generate3D``
/ ``OptimizeGeometry`` operators, remains available as an external backend, but
molrs is the trunk.

Note
----
molrs exposes only the full ``generate_3d`` pipeline (which already includes
energy minimization); it does not expose a standalone "optimize an existing
geometry" entry point. A geometry-only optimizer therefore lives in the
optional RDKit adapter (:mod:`molpy.adapter.rdkit`), not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molpy.embed import generate_3d as _embed_generate_3d

from .base import Compute

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


class Generate3D(Compute["Atomistic", "Atomistic"]):
    """Generate 3D coordinates for a molecular graph via molrs.

    Parameters
    ----------
    speed : str
        Quality preset: ``"fast"``, ``"medium"`` (default), or ``"better"``.
    add_hydrogens : bool
        Add explicit hydrogens before embedding (default ``True``).
    seed : int | None
        Deterministic RNG seed; ``None`` uses an arbitrary seed.

    Examples
    --------
    >>> from molpy.parser import parse_molecule
    >>> mol = parse_molecule("CCO")
    >>> mol_3d = Generate3D(seed=42)(mol)
    >>> len(list(mol_3d.atoms))   # heavy atoms + added hydrogens
    9
    """

    def __init__(
        self,
        speed: str = "medium",
        add_hydrogens: bool = True,
        seed: int | None = None,
    ):
        super().__init__(speed=speed, add_hydrogens=add_hydrogens, seed=seed)
        self.speed = speed
        self.add_hydrogens = bool(add_hydrogens)
        self.seed = seed

    def _compute(self, input: "Atomistic") -> "Atomistic":
        out, _report = _embed_generate_3d(
            input,
            speed=self.speed,
            add_hydrogens=self.add_hydrogens,
            rng_seed=self.seed,
        )
        return out

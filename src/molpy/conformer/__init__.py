"""3D conformer generation for molpy molecules (native-backed).

:class:`Conformer` subclasses the native ``Conformer``; :meth:`Conformer.generate`
guards against an empty molecule and returns the native result directly
(``molpy.Atomistic is molrs.Atomistic``). The heavy lifting — fragment / distance-geometry build, energy
minimisation, rotor search, stereo guard — runs inside the native core.

The report types are inherited verbatim from the native core (re-exported here, not
re-declared). The optional RDKit backend (:mod:`molpy.adapter.rdkit`) remains
available as a separate external adapter.
"""

from __future__ import annotations

import molrs

# molpy inherits the molrs report types directly; it does not re-declare them.
from molrs.conformer import ConformerReport, ConformerStageReport

from molpy.core.atomistic import Atomistic

__all__ = ["Conformer", "ConformerReport", "ConformerStageReport"]


class Conformer(molrs.conformer.Conformer):
    """3D conformer generator for molpy molecules.

    Subclasses the native ``Conformer``; the constructor parameters
    (``speed``, ``add_hydrogens``, ``seed``) are inherited unchanged. Only the
    empty-molecule guard in :meth:`generate` is added.

    Examples:
        >>> import molpy as mp
        >>> mol = mp.io.read_smiles("CCO")
        >>> mol_3d, report = Conformer(seed=42).generate(mol)
        >>> mol_3d.n_atoms   # heavy atoms + added hydrogens
        9
    """

    def generate(self, mol: Atomistic) -> tuple[Atomistic, ConformerReport]:
        """Generate 3D coordinates, returning a fresh ``Atomistic``.

        ``mol`` is the native graph, so the inherited Rust generator embeds it
        directly and its result is returned as is. The core
        reads the canonical integer ``"formal_charge"`` key for valence filling
        (``[N+]`` / ``[N-]`` hydrogen counts); the parsers emit that key, so a
        charged input must already carry it. The core clones the graph internally,
        so the input is not mutated.

        Args:
            mol: Input molecular graph. Element symbols and bond orders are
                required; coordinates may be missing.

        Returns:
            A tuple of the generated structure (an ``Atomistic``) and the
            per-stage :class:`~molpy.conformer.ConformerReport`.

        Raises:
            ValueError: If ``mol`` has no atoms.
        """
        if mol.n_atoms == 0:
            raise ValueError("cannot generate 3D structure for empty molecule")
        return super().generate(mol)

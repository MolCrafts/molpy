"""3D conformer generation for molpy molecules (molrs-backed).

:class:`Conformer` subclasses :class:`molrs.Conformer` and overrides
:meth:`Conformer.generate` to marshal :class:`molpy.Atomistic` across the molrs
boundary: it folds formal charges into the form the molrs pipeline expects, runs
the inherited Rust generator, and re-adopts the result as a molpy graph-backed
``Atomistic``. The heavy lifting — fragment / distance-geometry build, energy
minimisation, rotor search, stereo guard — runs inside molrs.

The report types are inherited verbatim from molrs (re-exported here, not
re-declared). The optional RDKit backend (:mod:`molpy.adapter.rdkit`) remains
available as a separate external adapter.
"""

from __future__ import annotations

import molrs

# molpy inherits the molrs report types directly; it does not re-declare them.
from molrs import ConformerReport, ConformerStageReport

from molpy.core.atomistic import Atomistic

__all__ = ["Conformer", "ConformerReport", "ConformerStageReport"]


class Conformer(molrs.Conformer):
    """3D conformer generator for molpy molecules.

    Subclasses :class:`molrs.Conformer`; the constructor parameters
    (``speed``, ``add_hydrogens``, ``seed``) are inherited unchanged. Only the
    molpy-side marshalling in :meth:`generate` is added.

    Examples:
        >>> from molpy.parser import parse_molecule
        >>> mol = parse_molecule("CCO")
        >>> mol_3d, report = Conformer(seed=42).generate(mol)
        >>> mol_3d.n_atoms   # heavy atoms + added hydrogens
        9
    """

    def generate(self, mol: Atomistic) -> tuple[Atomistic, ConformerReport]:
        """Generate 3D coordinates, returning a fresh molpy ``Atomistic``.

        ``mol`` is already a molrs graph (``Atomistic`` is-a ``molrs.Graph``), so
        the inherited Rust generator embeds it directly — no translation. molrs
        reads the canonical integer ``"formal_charge"`` key for valence filling
        (``[N+]`` / ``[N-]`` hydrogen counts); the parsers emit that key, so a
        charged input must already carry it. molrs clones the graph internally,
        so the input is not mutated.

        Args:
            mol: Input molecular graph. Element symbols and bond orders are
                required; coordinates may be missing.

        Returns:
            A tuple of the generated structure (a molpy ``Atomistic``) and the
            per-stage :class:`molrs.ConformerReport`.

        Raises:
            ValueError: If ``mol`` has no atoms.
        """
        if mol.n_atoms == 0:
            raise ValueError("cannot generate 3D structure for empty molecule")
        out_graph, report = super().generate(mol)
        return Atomistic.adopt(out_graph), report

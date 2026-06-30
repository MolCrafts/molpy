"""Native MMFF94 typifier + force field, backed by molrs (RDKit-free).

MMFF94 is the Merck Molecular Force Field — Halgren, T. A., *J. Comput. Chem.*
**17**, 490-519 (1996); the static variant MMFF94s is Halgren, *J. Comput. Chem.*
**20**, 720-729 (1999). Energies are in kcal/mol and lengths in angstrom (molrs
MMFF convention).

This is the native path that needs no RDKit: typify an :class:`Atomistic` into a
``to_potentials``-ready :class:`molrs.Frame`, take the MMFF94
:class:`~molpy.core.forcefield.ForceField`, and minimize the frame with
:class:`molpy.optimize.LBFGS` driving :class:`~molpy.optimize.ForceFieldPotential`.

Example:
    >>> from molpy.typifier import MMFFTypifier
    >>> from molpy.optimize import LBFGS, ForceFieldPotential
    >>> typifier = MMFFTypifier()
    >>> frame = typifier.typify(mol)               # to_potentials-ready molrs.Frame
    >>> ff = typifier.forcefield()                 # molpy ForceField (MMFF94)
    >>> result = LBFGS(ForceFieldPotential(ff)).run(frame, fmax=0.05)
"""

from __future__ import annotations

import molrs

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField

_VARIANTS = ("MMFF94", "MMFF94s")


class MMFFTypifier:
    """MMFF94 atom-type assigner and force-field provider, backed by molrs.

    Args:
        variant: Force-field variant, ``"MMFF94"`` (default) or ``"MMFF94s"``.

    Raises:
        ValueError: If ``variant`` is not a recognized MMFF variant.
        NotImplementedError: For ``"MMFF94s"`` — the native molrs typifier
            currently provides only MMFF94; MMFF94s support is pending in molrs.
    """

    def __init__(self, variant: str = "MMFF94") -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown MMFF variant {variant!r}; choose from {list(_VARIANTS)}"
            )
        if variant == "MMFF94s":
            raise NotImplementedError(
                "MMFF94s is not yet available via the native molrs MMFFTypifier "
                "(only MMFF94); use variant='MMFF94'."
            )
        self.variant = variant
        self._inner = molrs.MMFFTypifier()

    def typify(self, mol: Atomistic) -> "molrs.Frame":
        """Assign MMFF94 types and return a ``to_potentials``-ready Frame.

        The returned frame is assembly-complete — the stretch-bend reference
        lengths are merged onto the angles block and the intramolecular pair
        list is inserted — so it can be passed straight to
        :meth:`ForceField.to_potentials` (energies in kcal/mol).

        Args:
            mol: Molecule (:class:`Atomistic`) carrying element symbols, bonds,
                and 3D coordinates (angstrom).

        Returns:
            A typed :class:`molrs.Frame`, ready for force-field compilation.

        Raises:
            ValueError: If the molecule cannot be MMFF-typed (MMFF parameters may
                not be available for this molecule).
        """
        try:
            return self._inner.typify(mol)
        except ValueError as exc:
            raise ValueError(
                f"MMFF94 typing failed: {exc}. "
                "MMFF parameters may not be available for this molecule."
            ) from exc

    def forcefield(self) -> ForceField:
        """Return the MMFF94 :class:`~molpy.core.forcefield.ForceField`.

        The force field is molpy's public ``ForceField`` (re-exported from molrs)
        and is consumed by :class:`~molpy.optimize.ForceFieldPotential`.
        """
        return self._inner.forcefield()

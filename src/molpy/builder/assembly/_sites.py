"""Mark reaction sites on a molecule — the first of the three assembly inputs.

A site is an ordinary atom carrying :data:`~molpy.core.fields.SITE`. This class
owns one :class:`~molpy.core.atomistic.Atomistic` and names atoms on it so a
reaction SMARTS ``%label`` predicate can find them. It is not a port system:
labels are unordered names, not heads and tails.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from molpy.core import fields

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom, Atomistic


class SiteMap:
    """Name the atoms on ``struct`` that a reaction may bind.

    Example::

        SiteMap(eo).label_elements("O", "a", "b")
        # hydroxyl oxygens are now SITE "a" and "b"
    """

    def __init__(self, struct: Atomistic) -> None:
        self._struct = struct

    @property
    def struct(self) -> Atomistic:
        return self._struct

    def label(self, atom: Atom, name: str) -> SiteMap:
        """Set ``atom[fields.SITE] = name`` and return ``self`` for chaining."""
        atom[fields.SITE] = name
        return self

    def label_elements(self, element: str, *names: str) -> list[Atom]:
        """Mark atoms of ``element`` (in handle order) with successive site names.

        Returns the marked atoms. Raises if there are fewer matching atoms than
        names — silent reuse of the same atom would corrupt the reaction map.
        """
        if not names:
            raise ValueError("label_elements needs at least one site name")
        matches = [
            atom for atom in self._struct.atoms if atom.get(fields.ELEMENT) == element
        ]
        if len(matches) < len(names):
            raise ValueError(
                f"need {len(names)} atoms with element {element!r}, "
                f"found {len(matches)}"
            )
        marked: list[Atom] = []
        for atom, name in zip(matches, names, strict=False):
            atom[fields.SITE] = name
            marked.append(atom)
        return marked

    def label_atoms(self, atoms: Sequence[Atom], *names: str) -> list[Atom]:
        """Mark an explicit atom list with successive site names."""
        if len(atoms) < len(names):
            raise ValueError(f"need {len(names)} atoms, got {len(atoms)}")
        if not names:
            raise ValueError("label_atoms needs at least one site name")
        marked: list[Atom] = []
        for atom, name in zip(atoms, names, strict=False):
            atom[fields.SITE] = name
            marked.append(atom)
        return marked

    def every_nth(
        self,
        atoms: Sequence[Atom],
        step: int,
        site: str,
        *,
        leaving: str | None = None,
        fold_charge: bool = True,
    ) -> list[Atom]:
        """Mark ``atoms[0::step]`` with ``site``; optionally prepare leaving H.

        When ``leaving`` is set, each marked atom's lowest-handle hydrogen
        neighbour is labelled ``leaving``. If ``fold_charge`` is true and both
        atoms carry charge, the hydrogen's charge is added onto the site atom and
        zeroed on the hydrogen so a reaction that deletes it conserves net charge.

        Returns the marked site atoms.
        """
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        marked: list[Atom] = []
        for atom in atoms[::step]:
            atom[fields.SITE] = site
            marked.append(atom)
            if leaving is not None:
                self._prepare_leaving(atom, leaving, fold_charge=fold_charge)
        return marked

    def prepare_leaving_hydrogens(
        self,
        site: str,
        leaving: str = "h",
        *,
        fold_charge: bool = True,
    ) -> int:
        """For every atom with SITE ``site``, mark one neighbour H as ``leaving``.

        Returns how many leaving hydrogens were prepared.
        """
        n = 0
        for atom in self._struct.atoms:
            if atom.get(fields.SITE) != site:
                continue
            self._prepare_leaving(atom, leaving, fold_charge=fold_charge)
            n += 1
        return n

    def _prepare_leaving(
        self, site_atom: Atom, leaving: str, *, fold_charge: bool
    ) -> Atom:
        hydrogens = [
            nbr
            for nbr in self._struct.get_neighbors(site_atom)
            if nbr.get(fields.ELEMENT) == "H"
        ]
        if not hydrogens:
            raise ValueError(
                f"atom {site_atom.handle} has site label but no hydrogen neighbour "
                "to mark as leaving group"
            )
        hydrogen = min(hydrogens, key=lambda a: a.handle)
        if fold_charge:
            hq = hydrogen.get(fields.CHARGE)
            sq = site_atom.get(fields.CHARGE)
            if hq is not None and sq is not None:
                # Stash original H charge so unreacted sites can be thawed exactly.
                hydrogen["q0"] = float(hq)
                site_atom[fields.CHARGE] = float(sq) + float(hq)
                hydrogen[fields.CHARGE] = 0.0
        hydrogen[fields.SITE] = leaving
        return hydrogen

    def clear(self, atoms: Iterable[Atom] | None = None) -> SiteMap:
        """Remove site labels (set empty string) from ``atoms`` or the whole struct."""
        targets = list(self._struct.atoms) if atoms is None else list(atoms)
        for atom in targets:
            if atom.get(fields.SITE) is not None:
                atom[fields.SITE] = ""
        return self

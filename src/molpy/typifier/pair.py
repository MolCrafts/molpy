"""Pair (nonbonded) typifier — force-field agnostic."""

from molpy.core.atomistic import Atom
from molpy.core.forcefield import ForceField, PairType

from .base import TypifierBase


class PairTypifier(TypifierBase):
    """Pair (nonbonded) typifier."""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield, strict)
        self._build_pair_table()

    def _build_pair_table(self) -> None:
        self._pair_table = {}
        for pair_type in self.ff.get_types(PairType):
            self._pair_table[pair_type.name] = pair_type

    def typify(self, elem: Atom) -> Atom:
        """Assign nonbonded parameters to atom."""
        atom = elem
        atom_type = atom.get("type", None)

        if atom_type is None:
            if self.strict:
                raise ValueError(
                    f"Atom must have 'type' attribute before pair typification: {atom}"
                )
            return atom

        pair_type = self._pair_table.get(atom_type)

        if pair_type:
            atom.update(**pair_type.params.kwargs)
        elif self.strict:
            raise ValueError(f"No pair type found for atom type: {atom_type}")

        return atom

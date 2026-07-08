"""Read a SMILES / BigSMILES-monomer string into a 3D, named ``Atomistic``.

Unlike the file readers in this package the *source* is the notation string
itself, so :class:`SmilesReader` does not extend :class:`DataReader`; it keeps
the same ``.read()`` idiom. Living in ``io`` (a boundary layer) lets it depend
on ``parser`` + ``adapter`` + ``core`` without inverting the dependency graph
— ``core`` never imports outward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


class SmilesReader:
    """Turn a SMILES (or BigSMILES ``{...}`` monomer) string into an Atomistic.

    Auto-detects monomer notation (a leading ``{``), embeds a 3D conformer via
    RDKit, optionally derives angle/dihedral topology, and assigns a unique
    ``name`` to every atom (required by PDB export and the AmberTools wrappers).

    Example:
        >>> tfsi = SmilesReader("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
        ...                     add_hydrogens=False).read()
        >>> cat = SmilesReader("{[][<]CC(C)(C(=O)OCC[N+](C)(C)C)[>][]}").read()
    """

    def __init__(
        self,
        smiles: str,
        *,
        add_hydrogens: bool = True,
        optimize: bool = True,
        gen_topo: bool = False,
        name_atoms: bool = True,
    ) -> None:
        self.smiles = smiles
        self.add_hydrogens = add_hydrogens
        self.optimize = optimize
        self.gen_topo = gen_topo
        self.name_atoms = name_atoms

    def read(self) -> "Atomistic":
        """Parse, embed in 3D, (optionally) build topology, and name atoms."""
        from molpy.adapter import RDKitAdapter
        from molpy.parser import parse_molecule, parse_monomer

        smiles = self.smiles
        parse = parse_monomer if smiles.lstrip().startswith("{") else parse_molecule
        mol = RDKitAdapter(parse(smiles)).generate_3d(
            add_hydrogens=self.add_hydrogens, optimize=self.optimize
        )
        if self.gen_topo:
            mol = mol.get_topo(gen_angle=True, gen_dihe=True)
        if self.name_atoms:
            for idx, atom in enumerate(mol.atoms, start=1):
                if atom.get("name") is None:
                    atom["name"] = f"{atom.get('element', 'X')}{idx}"
        return mol

"""RDKit adapter for MolPy.

This module provides bidirectional conversion between RDKit Chem.Mol objects
and MolPy Atomistic structures, as well as utilities for 3D coordinate generation,
molecular visualization, and structure optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import ChiralType

from molpy import Atomistic
from molpy.parser.smiles import SmilesGraphIR
from molpy.core.wrapper import Wrapper

# Monomer is no longer used, removed from imports


# TypeVar for generic wrapper support
T = TypeVar("T", bound=Atomistic)


# Stable property tag for bidirectional atom mapping
# Stores entity ID for reliable round-trip conversion
MP_ID = "mp_id"

# Bond order mappings between MolPy and RDKit
BOND_ORDER_TO_RDKIT: dict[float, Chem.BondType] = {
    1.0: Chem.BondType.SINGLE,
    2.0: Chem.BondType.DOUBLE,
    3.0: Chem.BondType.TRIPLE,
    1.5: Chem.BondType.AROMATIC,  # Approximate; true aromaticity determined by RDKit
}

RDKIT_TO_BOND_ORDER: dict[Chem.BondType, float] = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}


# =============================================================================
#   Low-level RDKit helpers
# =============================================================================


def _rdkit_bond_type(order: float) -> Chem.BondType:
    """Convert MolPy bond order to RDKit BondType."""
    return BOND_ORDER_TO_RDKIT.get(float(order), Chem.BondType.SINGLE)


def _order_from_rdkit(bt: Chem.BondType) -> float:
    """Convert RDKit BondType to MolPy bond order."""
    return RDKIT_TO_BOND_ORDER.get(bt, 1.0)


def _ensure_2d(mol: Chem.Mol) -> None:
    """Compute 2D coordinates in-place if not present.

    Uses CoordGen for better layout of complex structures like polymers.
    """
    if not mol.GetNumConformers():
        rdDepictor.SetPreferCoordGen(True)
        # Use CoordGen for better layout, especially for large molecules
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            # Fallback to standard 2D coordinate generation
            AllChem.Compute2DCoords(mol)


def _normalize_symbol(symbol: str | None) -> str:
    """Normalize an atomic symbol (SMILES-style) to RDKit-style symbol."""
    if not symbol:
        return "C"
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _build_atom_id_index(mon_atoms: list[Any]) -> dict[int, Any]:
    """Index Atomistic atoms by their 'id' property."""
    by_id: dict[int, Any] = {}
    for atom in mon_atoms:
        atom_id = atom.get("id")
        if atom_id is not None:
            by_id[int(atom_id)] = atom
    return by_id


# =============================================================================
#   Conversion Functions: SmilesIR <-> Chem.Mol <-> Atomistic
# =============================================================================


def smilesir_to_mol(ir: SmilesIR) -> Chem.Mol:
    """Convert SmilesIR to RDKit Mol.

    Converts MolPy's internal SMILES representation to an RDKit molecule,
    preserving aromaticity, charges, stereochemistry, and explicit hydrogens.
    """
    mol = Chem.RWMol()
    atom_map: dict[int, int] = {}
    aromatic_flags: dict[int, bool] = {}

    for atom_ir in ir.atoms:
        symbol = atom_ir.element or "C"
        # Check aromatic flag or lowercase symbol
        is_aromatic = atom_ir.aromatic or (symbol.islower() if symbol else False)
        rd_symbol = _normalize_symbol(symbol.upper() if symbol else "C")

        rdkit_atom = Chem.Atom(rd_symbol)
        if is_aromatic:
            rdkit_atom.SetIsAromatic(True)

        if atom_ir.charge is not None:
            rdkit_atom.SetFormalCharge(int(atom_ir.charge))

        # Check extras for isotope
        isotope = atom_ir.extras.get("isotope")
        if isotope is not None:
            rdkit_atom.SetIsotope(int(isotope))

        # Check hydrogens attribute
        if atom_ir.hydrogens is not None:
            rdkit_atom.SetNumExplicitHs(int(atom_ir.hydrogens))

        # Check extras for chiral
        chiral = atom_ir.extras.get("chiral")
        if isinstance(chiral, str):
            if chiral == "@":
                rdkit_atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CCW)
            elif chiral == "@@":
                rdkit_atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)

        idx = mol.AddAtom(rdkit_atom)
        key = id(atom_ir)
        atom_map[key] = idx
        aromatic_flags[key] = is_aromatic

    bond_type_map: dict[str, Chem.BondType] = {
        "-": Chem.BondType.SINGLE,
        "=": Chem.BondType.DOUBLE,
        "#": Chem.BondType.TRIPLE,
        ":": Chem.BondType.AROMATIC,
    }

    for bond_ir in ir.bonds:
        start_key = id(bond_ir.atom_i)
        end_key = id(bond_ir.atom_j)
        begin = atom_map.get(start_key)
        end = atom_map.get(end_key)
        if begin is None or end is None or begin == end:
            continue
        if mol.GetBondBetweenAtoms(begin, end) is not None:
            continue

        # Convert bond order to symbol
        bond_order = bond_ir.order
        if bond_order == 2:
            bond_symbol = "="
        elif bond_order == 3:
            bond_symbol = "#"
        elif bond_order == "ar":
            bond_symbol = ":"
        else:
            bond_symbol = "-"
        # If both atoms are aromatic and no explicit aromatic bond is set, use aromatic bond
        start_is_aromatic = aromatic_flags.get(start_key, False)
        end_is_aromatic = aromatic_flags.get(end_key, False)
        if bond_symbol == "-" and start_is_aromatic and end_is_aromatic:
            bond_symbol = ":"

        bond_type = bond_type_map.get(bond_symbol, Chem.BondType.SINGLE)
        mol.AddBond(begin, end, bond_type)

    Chem.SanitizeMol(mol)
    return mol.GetMol()


def smilesir_to_atomistic(ir: SmilesGraphIR) -> Atomistic:
    """Convert SmilesIR to Atomistic structure via RDKit."""
    wrapper = RDKitWrapper.from_mol(smilesir_to_mol(ir))
    return wrapper.core


def atomistic_to_mol(atomistic: Atomistic | Wrapper[Atomistic]) -> Chem.Mol:
    """Convert Atomistic structure to RDKit Mol."""
    mol = Chem.RWMol()
    atom_map: dict[int, int] = {}  # id(atom) -> rdkit_idx
    rdkit_atoms = []

    # Add atoms
    for i, atom in enumerate(atomistic.atoms):
        symbol = atom.get("symbol")
        rd_atom = Chem.Atom(symbol)

        charge = atom.get("charge")
        if charge is not None:
            rd_atom.SetFormalCharge(int(charge))

        atom_id = atom.get("id", i)
        rd_atom.SetIntProp(MP_ID, int(atom_id))

        # idx = mol.AddAtom(rd_atom)
        # atom_map[id(atom)] = idx
        rdkit_atoms.append(rd_atom)

    rdkit_atoms = sorted(rdkit_atoms, key=lambda x: x.GetIntProp(MP_ID))
    for i, rd_atom in enumerate(rdkit_atoms):
        idx = mol.AddAtom(rd_atom)
        atom_map[id(atomistic.atoms[i])] = idx

    # Add bonds
    for bond in atomistic.bonds:
        begin_idx = atom_map.get(id(bond.itom))
        end_idx = atom_map.get(id(bond.jtom))
        if begin_idx is None or end_idx is None:
            continue

        # Prefer explicit numeric 'order' attribute. Some workflows set a
        # human-readable bond 'type' (e.g. 'CT-CT') instead of a numeric order.
        # Be robust: try 'order' first, then attempt to coerce 'type' to float,
        # otherwise fall back to single bond (1.0).
        order_val = bond.get("order")
        if order_val is None:
            type_val = bond.get("type", 1.0)
            try:
                order = float(type_val)
            except Exception:
                order = 1.0
        else:
            try:
                order = float(order_val)
            except Exception:
                order = 1.0

        bt = _rdkit_bond_type(order)
        mol.AddBond(begin_idx, end_idx, bt)

    # Chem.SanitizeMol(mol)
    return mol.GetMol()


def mol_to_atomistic(mol: Chem.Mol) -> Atomistic:
    """Convert RDKit Mol to Atomistic structure."""
    atomistic = Atomistic()
    atom_map: dict[int, Any] = {}  # rdkit_idx -> Atom

    # Add atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()

        props: dict[str, Any] = {
            "symbol": symbol,
            "atomic_num": atomic_num,
        }

        if atom.GetFormalCharge() != 0:
            props["charge"] = atom.GetFormalCharge()

        if atom.HasProp(MP_ID):
            props["id"] = atom.GetIntProp(MP_ID)
        else:
            props["id"] = atom.GetIdx()

        new_atom = atomistic.def_atom(**props)
        atom_map[atom.GetIdx()] = new_atom

    # Add bonds
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        a1 = atom_map.get(begin_idx)
        a2 = atom_map.get(end_idx)
        if a1 is None or a2 is None:
            continue

        order = _order_from_rdkit(bond.GetBondType())
        atomistic.def_bond(a1, a2, order=order)

    # Transfer coordinates if conformer exists
    # Use atom_map to correctly map RDKit indices to Atomistic atoms
    # This ensures coordinates are assigned to the correct atoms even if
    # the order in atomistic.atoms differs from RDKit Mol order
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        for rdkit_idx in range(conf.GetNumAtoms()):
            atom = atom_map.get(rdkit_idx)
            if atom is None:
                continue
            pos = conf.GetAtomPosition(rdkit_idx)
            atom["x"] = float(pos.x)
            atom["y"] = float(pos.y)
            atom["z"] = float(pos.z)
            # Also set xyz tuple for convenience
            atom["xyz"] = [float(pos.x), float(pos.y), float(pos.z)]

    return atomistic


# =============================================================================
#   AtomMapper: Bidirectional Atom Mapping
# =============================================================================


class AtomMapper:
    """Manages bidirectional atom mapping between RDKit Mol and Atomistic atoms.

    This class handles the correspondence between RDKit atom indices and
    MolPy Atomistic atom entities, supporting operations like hydrogen addition
    where atom counts may differ.
    """

    def __init__(self, mol: Chem.Mol, atomistic_atoms: list[Any]):
        """Initialize atom mapper.

        Args:
            mol: RDKit molecule
            atomistic_atoms: List of Atomistic atom entities
        """
        self.mol = mol
        self.atomistic_atoms = atomistic_atoms
        self._forward_map: dict[int, Any] | None = None  # rdkit_idx -> atom
        self._reverse_map: dict[int, int] | None = None  # id(atom) -> rdkit_idx
        self.build_mapping()

    def build_mapping(self) -> dict[int, Any]:
        """Build atom mapping from RDKit indices to Atomistic atoms.

        Returns:
            Dictionary mapping RDKit atom index to Atomistic atom entity
        """
        if self._forward_map is not None:
            return self._forward_map

        atom_map: dict[int, Any] = {}
        atom_by_id = self._build_atom_id_index()

        mol_atoms = list(self.mol.GetAtoms())
        n_mol = len(mol_atoms)
        n_atomistic = len(self.atomistic_atoms)

        # Simple case: same atom count, map one-to-one by index
        if n_mol == n_atomistic:
            for rd_idx in range(n_mol):
                atom_map[rd_idx] = self.atomistic_atoms[rd_idx]
            self._forward_map = atom_map
            self._reverse_map = {id(v): k for k, v in atom_map.items()}
            return atom_map

        # Fallback: heavy-atom-based mapping
        heavy_atoms = [
            atom for atom in self.atomistic_atoms if atom.get("atomic_num", 0) != 1
        ]
        heavy_pos = 0

        for rd_idx, rd_atom in enumerate(mol_atoms):
            if rd_atom.GetAtomicNum() == 1:
                continue  # Skip H for map

            ent: Any | None = None

            # Try MP_ID tag first
            if rd_atom.HasProp(MP_ID):
                hid = int(rd_atom.GetIntProp(MP_ID))
                ent = atom_by_id.get(hid)

            # Fallback to position-based mapping
            if ent is None and heavy_pos < len(heavy_atoms):
                ent = heavy_atoms[heavy_pos]
                heavy_pos += 1

            if ent is not None:
                atom_map[rd_idx] = ent

        self._forward_map = atom_map
        self._reverse_map = {id(v): k for k, v in atom_map.items()}
        return atom_map

    def _build_atom_id_index(self) -> dict[int, Any]:
        """Build index of Atomistic atoms by their 'id' property.

        Returns:
            Dictionary mapping atom ID to atom entity
        """
        by_id: dict[int, Any] = {}
        for atom in self.atomistic_atoms:
            atom_id = atom.get("id")
            if atom_id is not None:
                by_id[int(atom_id)] = atom
        return by_id

    def ensure_tags(self) -> None:
        """Ensure MP_ID tags are set on RDKit atoms for reliable mapping."""
        for i, rd_atom in enumerate(self.mol.GetAtoms()):
            if rd_atom.GetAtomicNum() == 1:
                continue  # Skip hydrogens
            if rd_atom.HasProp(MP_ID):
                continue
            if i < len(self.atomistic_atoms):
                ent = self.atomistic_atoms[i]
                ent_id = ent.get("id", i)
                rd_atom.SetIntProp(MP_ID, int(ent_id))

    def get_atomistic_atom(self, rdkit_idx: int) -> Any | None:
        """Get Atomistic atom for RDKit atom index.

        Args:
            rdkit_idx: RDKit atom index

        Returns:
            Atomistic atom entity or None if not found
        """
        if self._forward_map is None:
            self.build_mapping()
        return self._forward_map.get(rdkit_idx)

    def get_rdkit_index(self, atomistic_atom: Any) -> int | None:
        """Get RDKit atom index for Atomistic atom.

        Args:
            atomistic_atom: Atomistic atom entity

        Returns:
            RDKit atom index or None if not found
        """
        if self._reverse_map is None:
            self.build_mapping()
        return self._reverse_map.get(id(atomistic_atom))

    def rebuild(self) -> None:
        """Rebuild mapping after structural changes."""
        self._forward_map = None
        self._reverse_map = None
        self.build_mapping()


# =============================================================================
#   RDKitWrapper: Bidirectional RDKit Mol <-> MolPy Atomistic Bridge
# =============================================================================


class RDKitWrapper(Wrapper[T], Generic[T]):
    """Bridge associating RDKit Mol with MolPy Atomistic.

    Generic wrapper that preserves the type of the inner Atomistic structure.
    Supports Atomistic structures.

    Type Parameters:
        T: Type of the inner structure, must be Atomistic or a subclass

    Examples:
        >>> wrapper: RDKitWrapper[Atomistic] = RDKitWrapper.from_smiles("CCO")
        >>> wrapper: RDKitWrapper[Atomistic] = RDKitWrapper.from_atomistic(struct)
    """

    __slots__ = ("_atom_mapper", "_mol")

    def __init__(
        self,
        inner: T | Wrapper[T],
        mol: Chem.Mol | None = None,
        atom_map: dict[int, Any] | None = None,
        **props: Any,
    ) -> None:
        """Initialize wrapper with core object."""
        super().__init__(inner, **props)

        # Store / build RDKit molecule
        if mol is None:
            mol = atomistic_to_mol(inner)
        object.__setattr__(self, "_mol", mol)

        # Create atom mapper
        atom_mapper = AtomMapper(mol, list(self.unwrap().atoms))
        object.__setattr__(self, "_atom_mapper", atom_mapper)

        # Ensure MP_ID tags are set
        atom_mapper.ensure_tags()

    # ------------------------------------------------------------------
    #   Small internal helpers
    # ------------------------------------------------------------------

    def _core_atoms(self) -> list[Any]:
        """Return a list of Atomistic atoms (convenience wrapper)."""
        core = self.unwrap()
        return list(core.atoms)

    def _rebuild_atom_mapper(self) -> None:
        """Rebuild atom mapper after structural changes."""
        atom_mapper = AtomMapper(self._mol, list(self.unwrap().atoms))
        object.__setattr__(self, "_atom_mapper", atom_mapper)
        atom_mapper.ensure_tags()

    # ------------------------------------------------------------------
    #   Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> RDKitWrapper[Atomistic]:
        """Create RDKitWrapper from a Chem.Mol.

        Args:
            mol: RDKit molecule

        Returns:
            RDKitWrapper containing an Atomistic structure
        """
        atomistic = mol_to_atomistic(mol)
        wrapper = cls(atomistic, mol=mol)
        wrapper.sync_from_mol()
        return wrapper

    @classmethod
    def from_atomistic(cls, atomistic: T | Wrapper[T]) -> RDKitWrapper[T]:
        """Create RDKitWrapper from an Atomistic structure.

        Preserves the type of the input structure.

        Args:
            atomistic: Atomistic structure or wrapper

        Returns:
            RDKitWrapper with the same inner type as the input
        """
        return cls(atomistic)

    # ------------------------------------------------------------------
    #   Properties
    # ------------------------------------------------------------------

    @property
    def mol(self) -> Chem.Mol:
        """Get the RDKit molecule."""
        return self._mol

    def with_mol(self, mol: Chem.Mol) -> RDKitWrapper[T]:
        """Return a new RDKitWrapper with the given Chem.Mol."""
        return RDKitWrapper(self.unwrap(), mol=mol)

    def sync_from_mol(self) -> None:
        """Synchronize Atomistic structure from RDKit Mol.

        Syncs everything: atoms, bonds, and coordinates.
        Updates existing atoms, adds new atoms, and rebuilds bonds.
        Preserves atom references (important for port markers on atoms).

        Uses atom_mapper to correctly map RDKit indices to Atomistic atoms,
        ensuring coordinates and properties are assigned to the correct atoms
        even if the order differs.
        """
        inner = self.unwrap()
        mol = self._mol

        # Rebuild atom mapper to ensure it's up to date
        self._rebuild_atom_mapper()
        atom_map = self._atom_mapper.build_mapping()

        # Get conformer for coordinates if available
        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None

        # Build list of existing atoms
        existing_atoms = list(inner.atoms)
        old_atom_count = len(existing_atoms)

        # Create mapping from RDKit index to Atomistic atom
        rdkit_to_atom: dict[int, Any] = {}

        # Update existing atoms using atom_mapper for correct mapping
        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            atom = atom_map.get(rdkit_idx)

            # If atom not in mapping, it's a new atom (e.g., added hydrogens)
            if atom is None:
                # Create new atom
                props: dict[str, Any] = {
                    "symbol": rd_atom.GetSymbol(),
                    "atomic_num": rd_atom.GetAtomicNum(),
                }

                # Use MP_ID if available, otherwise use RDKit index
                if rd_atom.HasProp(MP_ID):
                    props["id"] = rd_atom.GetIntProp(MP_ID)
                else:
                    props["id"] = rdkit_idx

                if rd_atom.GetFormalCharge() != 0:
                    props["charge"] = rd_atom.GetFormalCharge()

                # Get coordinates if available
                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    props["x"] = float(pos.x)
                    props["y"] = float(pos.y)
                    props["z"] = float(pos.z)
                    props["xyz"] = [float(pos.x), float(pos.y), float(pos.z)]

                # Add new atom to atomistic
                new_atom = inner.def_atom(**props)
                rdkit_to_atom[rdkit_idx] = new_atom
            else:
                # Update existing atom
                # Sync all properties from RDKit atom
                atom["symbol"] = rd_atom.GetSymbol()
                atom["atomic_num"] = rd_atom.GetAtomicNum()

                # Preserve existing id, or set from RDKit if not set
                if atom.get("id") is None:
                    if rd_atom.HasProp(MP_ID):
                        atom["id"] = rd_atom.GetIntProp(MP_ID)
                    else:
                        atom["id"] = rdkit_idx

                # Update charge if present
                if rd_atom.GetFormalCharge() != 0:
                    atom["charge"] = rd_atom.GetFormalCharge()

                # Update coordinates if available
                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    atom["x"] = float(pos.x)
                    atom["y"] = float(pos.y)
                    atom["z"] = float(pos.z)

                rdkit_to_atom[rdkit_idx] = atom

        # Rebuild bonds - remove all existing bonds and add from RDKit mol
        existing_bonds = list(inner.bonds)
        if existing_bonds:
            inner.remove_link(*existing_bonds)

        # Add all bonds from RDKit mol
        for rd_bond in mol.GetBonds():
            begin_idx = rd_bond.GetBeginAtomIdx()
            end_idx = rd_bond.GetEndAtomIdx()

            atom1 = rdkit_to_atom.get(begin_idx)
            atom2 = rdkit_to_atom.get(end_idx)

            if atom1 is not None and atom2 is not None:
                order = RDKIT_TO_BOND_ORDER.get(rd_bond.GetBondType(), 1.0)
                inner.def_bond(atom1, atom2, order=order)

    def sync_to_mol(self):
        """Synchronize data from MolPy `Atomistic` into RDKit `Mol`."""
        mon_atoms = self._core_atoms()
        n_atoms = len(mon_atoms)
        if n_atoms == 0:
            return

        # Get or create conformer
        if self._mol.GetNumConformers() == 0:
            conf = Chem.Conformer(n_atoms)
            self._mol.AddConformer(conf, assignId=True)
        else:
            conf = self._mol.GetConformer()

        for a in self._mol.GetAtoms():
            ridx = a.GetIdx()
            if a.GetAtomicNum() == 1:
                continue

            ent = self._atom_mapper.get_atomistic_atom(ridx)
            if ent is None and ridx < len(mon_atoms):
                ent = mon_atoms[ridx]
            if ent is None:
                continue

            x = ent.get("x", 0.0)
            y = ent.get("y", 0.0)
            z = ent.get("z", 0.0)
            conf.SetAtomPosition(ridx, (float(x), float(y), float(z)))

    # ------------------------------------------------------------------
    #   Convenience factories / helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_smiles(cls, smi: str) -> RDKitWrapper[Atomistic]:
        """Create RDKitWrapper directly from a SMILES string.

        Args:
            smi: SMILES string

        Returns:
            RDKitWrapper containing an Atomistic structure
        """
        mol = Chem.MolFromSmiles(smi)
        return cls.from_mol(mol)

    def to_smiles(self, canonical: bool = True) -> str:
        """Return SMILES for the wrapped RDKit molecule."""
        return Chem.MolToSmiles(Chem.Mol(self._mol), canonical=canonical)

    # ------------------------------------------------------------------
    #   Visualization
    # ------------------------------------------------------------------

    def draw(
        self,
        *,
        size: tuple[int, int] = (320, 260),
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        title: str | None = None,
        show_atomistic_idx: bool = False,
        show_rdkit_idx: bool = False,
        show_explicit_H: bool = False,
    ) -> str:
        """Generate 2D molecular structure drawing as SVG."""
        _ensure_2d(self._mol)
        # dm = Chem.Mol(self._mol)
        # if show_explicit_H:
        #     dm = Chem.AddHs(dm)
        #     _ensure_2d(dm)
        # else:
        #     dm = Chem.RemoveHs(dm)
        #     _ensure_2d(dm)

        w, h = size
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        # Improved padding and spacing for better layout
        opts.padding = 0.15
        opts.additionalAtomLabelPadding = 0.08
        opts.fixedFontSize = -1
        opts.minFontSize = 10
        opts.maxFontSize = 14
        opts.bondLineWidth = 2.5
        opts.addAtomIndices = bool(show_rdkit_idx)
        opts.addStereoAnnotation = False
        opts.explicitMethyl = False
        # Better rendering for large molecules
        opts.scaleBondWidth = True
        opts.scaleHighlightBondWidth = True
        if show_explicit_H:
            mol = Chem.AddHs(self._mol)
        else:
            mol = Chem.RemoveHs(self._mol)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms or [],
            highlightBonds=highlight_bonds or [],
            legend=(title or ""),
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg

    # ------------------------------------------------------------------
    #   Hydrogen management
    # ------------------------------------------------------------------

    def add_hydrogens(self) -> RDKitWrapper[T]:
        """Add explicit hydrogens to the molecule.

        This method:
        1. Adds hydrogens to the RDKit Mol (heavy atoms keep same indices)
        2. Syncs all atoms, bonds, and coordinates to Atomistic via sync_from_mol
        3. Preserves all existing atom references (including port markers)

        Returns:
            Self for method chaining
        """
        # Ensure properties are calculated before AddHs
        self._mol.UpdatePropertyCache(strict=False)

        # Add hydrogens to RDKit molecule
        # IMPORTANT: AddHs preserves heavy atom indices, new H atoms are appended
        mol_h = Chem.AddHs(self._mol, addCoords=True)

        # If original had conformer, ensure hydrogenated version has it too
        if self._mol.GetNumConformers() > 0 and mol_h.GetNumConformers() == 0:
            mol_h.AddConformer(self._mol.GetConformer(), assignId=True)

        # Sanitize the molecule to ensure RingInfo and other properties are initialized
        try:
            Chem.SanitizeMol(mol_h)
        except Exception:
            pass  # Sanitization may fail for some structures, but we can continue

        # Update mol reference
        object.__setattr__(self, "_mol", mol_h)

        # Sync all changes to Atomistic structure
        self.sync_from_mol()

        # Rebuild atom mapper
        self._rebuild_atom_mapper()

        return self

    def remove_hydrogens(self) -> RDKitWrapper[T]:
        """Remove explicit hydrogens from the molecule.

        This method:
        1. Removes hydrogens from the RDKit Mol
        2. Removes hydrogen atoms from the existing Atomistic structure
        3. Preserves all existing heavy atom references (including Monomer ports)
        4. Rebuilds atom mapping

        Returns:
            Self for method chaining
        """
        # Get reference to inner structure
        inner = self.unwrap()

        # Remove hydrogens from RDKit molecule
        mol_no_h = Chem.RemoveHs(self._mol, updateExplicitCount=True)

        # Remove hydrogen atoms from inner structure
        # Collect hydrogen atoms to remove
        h_atoms = [atom for atom in inner.atoms if atom.get("atomic_num") == 1]
        for h_atom in h_atoms:
            inner.remove_entity(h_atom)

        # Update mol reference
        object.__setattr__(self, "_mol", mol_no_h)
        self._rebuild_atom_mapper()

        return self

    # ------------------------------------------------------------------
    #   Coordinate generation
    # ------------------------------------------------------------------

    def generate_2d_coords(self) -> RDKitWrapper[T]:
        """Generate 2D coordinates for visualization.

        Returns:
            Self for method chaining
        """
        _ensure_2d(self._mol)
        self.sync_from_mol()
        return self

    def generate_3d_coords(self, random_seed: int = 42) -> RDKitWrapper[T]:
        """Generate 3D coordinates using RDKit ETKDG.

        Args:
            random_seed: Random seed for reproducible coordinate generation

        Returns:
            Self for method chaining
        """
        if self._mol.GetNumAtoms() == 0:
            raise ValueError("Cannot generate 3D coordinates for empty molecule")

        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(random_seed)
        params.useRandomCoords = True

        embed_result = AllChem.EmbedMolecule(self._mol, params)  # type: ignore[attr-defined]
        if embed_result == -1:
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(self._mol, params)  # type: ignore[attr-defined]
            if embed_result == -1:
                raise RuntimeError("ETKDG embedding failed")

        self.sync_from_mol()
        return self

    # ------------------------------------------------------------------
    #   Geometry optimization
    # ------------------------------------------------------------------

    def optimize_geometry(
        self,
        *,
        max_iters: int = 200,
        force_field: str = "MMFF",
    ) -> RDKitWrapper[T]:
        """Optimize molecular geometry using force field minimization.

        Requires existing 3D coordinates. Call generate_3d_coords() first if needed.

        Args:
            max_iters: Maximum number of optimization iterations
            force_field: Force field to use ('MMFF' or 'UFF')

        Returns:
            Self for method chaining
        """
        if self._mol.GetNumConformers() == 0:
            raise ValueError("No conformer found. Call generate_3d_coords() first.")

        if force_field == "MMFF":
            AllChem.MMFFOptimizeMolecule(self._mol, maxIters=int(max_iters))  # type: ignore[attr-defined]
        elif force_field == "UFF":
            AllChem.UFFOptimizeMolecule(self._mol, maxIters=int(max_iters))  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"Unknown force field: {force_field}. Use 'MMFF' or 'UFF'."
            )
        self.sync_from_mol()
        return self

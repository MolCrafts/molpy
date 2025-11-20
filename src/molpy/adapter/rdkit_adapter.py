"""RDKit adapter for MolPy.

This module provides bidirectional conversion between RDKit Chem.Mol objects
and MolPy Atomistic structures, as well as utilities for 3D coordinate generation,
molecular visualization, and structure optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import ChiralType

from molpy import Atomistic, Wrapper
from molpy.parser.smiles import SmilesIR

if TYPE_CHECKING:
    from molpy.core.wrappers.monomer import Monomer


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
    """Compute 2D coordinates in-place if not present."""
    if not mol.GetNumConformers():
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)


def _normalize_symbol(symbol: str | None) -> str:
    """Normalize an atomic symbol (SMILES-style) to RDKit-style symbol."""
    if not symbol:
        return "C"
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _build_mon_heavy_index(mon_atoms: list[Any]) -> dict[int, Any]:
    """Build index mapping entity IDs to heavy atom entities."""
    table: dict[int, Any] = {}
    for i, ent in enumerate(mon_atoms):
        if ent.get("atomic_num", 0) == 1:
            continue
        hid = ent.get("id", i)
        table[int(hid)] = ent
    return table


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
        symbol = atom_ir.symbol or "C"
        is_aromatic = symbol.islower()  # aromatic atoms encoded as lower-case in SMILES
        rd_symbol = _normalize_symbol(symbol)

        rdkit_atom = Chem.Atom(rd_symbol)
        if is_aromatic:
            rdkit_atom.SetIsAromatic(True)

        if atom_ir.charge is not None:
            rdkit_atom.SetFormalCharge(int(atom_ir.charge))

        isotope = getattr(atom_ir, "isotope", None)
        if isotope is not None:
            rdkit_atom.SetIsotope(int(isotope))

        h_count = getattr(atom_ir, "h_count", None)
        if h_count is not None:
            rdkit_atom.SetNumExplicitHs(int(h_count))

        chiral = getattr(atom_ir, "chiral", None)
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
        start_key = id(bond_ir.start)
        end_key = id(bond_ir.end)
        begin = atom_map.get(start_key)
        end = atom_map.get(end_key)
        if begin is None or end is None or begin == end:
            continue
        if mol.GetBondBetweenAtoms(begin, end) is not None:
            continue

        bond_symbol = bond_ir.bond_type or "-"
        # If both atoms are aromatic and no explicit aromatic bond is set, use aromatic bond
        start_is_aromatic = aromatic_flags.get(start_key, False)
        end_is_aromatic = aromatic_flags.get(end_key, False)
        if bond_symbol == "-" and start_is_aromatic and end_is_aromatic:
            bond_symbol = ":"

        bond_type = bond_type_map.get(bond_symbol, Chem.BondType.SINGLE)
        mol.AddBond(begin, end, bond_type)

    Chem.SanitizeMol(mol)
    return mol.GetMol()


def smilesir_to_atomistic(ir: SmilesIR) -> Atomistic:
    """Convert SmilesIR to Atomistic structure via RDKit."""
    wrapper = RDKitWrapper.from_mol(smilesir_to_mol(ir), sync_coords=True)
    return wrapper.core


def atomistic_to_mol(atomistic: Atomistic) -> Chem.Mol:
    """Convert Atomistic structure to RDKit Mol."""
    mol = Chem.RWMol()
    atom_map: dict[int, int] = {}  # id(atom) -> rdkit_idx

    # Add atoms
    for i, atom in enumerate(atomistic.atoms):
        symbol = atom.get("symbol", "C")
        rd_atom = Chem.Atom(symbol)

        if "charge" in atom:
            rd_atom.SetFormalCharge(int(atom["charge"]))

        atom_id = atom.get("id", i)
        rd_atom.SetIntProp(MP_ID, int(atom_id))

        idx = mol.AddAtom(rd_atom)
        atom_map[id(atom)] = idx

    # Add bonds
    for bond in atomistic.bonds:
        begin_idx = atom_map.get(id(bond.itom))
        end_idx = atom_map.get(id(bond.jtom))
        if begin_idx is None or end_idx is None:
            continue

        order = bond.get("type", 1.0)
        bt = _rdkit_bond_type(order)
        mol.AddBond(begin_idx, end_idx, bt)

    Chem.SanitizeMol(mol)
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
        atomistic.def_bond(a1, a2, type=order)

    # Transfer coordinates if conformer exists
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        for i, atom in enumerate(atomistic.atoms):
            if i >= conf.GetNumAtoms():
                break
            pos = conf.GetAtomPosition(i)
            atom["xyz"] = [float(pos.x), float(pos.y), float(pos.z)]

    return atomistic


# =============================================================================
#   RDKitWrapper: Bidirectional RDKit Mol <-> MolPy Atomistic Bridge
# =============================================================================


class RDKitWrapper(Wrapper[Atomistic]):
    """Bridge associating RDKit Mol with MolPy Atomistic."""

    __slots__ = ("_atom_map", "_atom_map_reverse", "_mol")

    def __init__(
        self,
        inner: Atomistic | Wrapper[Atomistic],
        mol: Chem.Mol | None = None,
        atom_map: dict[int, Any] | None = None,
        **props: Any,
    ) -> None:
        """Initialize wrapper with core object."""
        super().__init__(inner, **props)

        # Core Atomistic object used for RDKit conversion
        core = self.unwrap()

        # Store / build RDKit molecule
        if mol is None:
            mol = atomistic_to_mol(core)
        object.__setattr__(self, "_mol", mol)

        # Build atom map if not provided
        if atom_map is None:
            try:
                atom_map = self._build_atom_map()
            except Exception as e:  # pragma: no cover - defensive
                import warnings

                warnings.warn(
                    f"Failed to build atom map: {e}. Using empty map.", stacklevel=2
                )
                atom_map = {}

        object.__setattr__(self, "_atom_map", atom_map or {})
        object.__setattr__(
            self,
            "_atom_map_reverse",
            {v: k for k, v in (atom_map or {}).items()},
        )

        # Ensure MP_ID tags are set for mapping
        try:
            self._ensure_mapping_tags()
        except Exception as e:  # pragma: no cover - defensive
            import warnings

            warnings.warn(f"Failed to ensure mapping tags: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    #   Small internal helpers
    # ------------------------------------------------------------------

    def _core_atoms(self) -> list[Any]:
        """Return a list of Atomistic atoms (convenience wrapper)."""
        core = self.unwrap()
        return list(core.atoms)

    def _ensure_atom_map_ready(self) -> None:
        """Ensure atom mapping dictionaries are available."""
        if not self._atom_map:
            self._rebuild_atom_map()

    # ------------------------------------------------------------------
    #   Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_mol(cls, mol: Chem.Mol, sync_coords: bool = False) -> RDKitWrapper:
        """Create RDKitWrapper from a Chem.Mol."""
        atomistic = mol_to_atomistic(mol)
        wrapper = cls(atomistic, mol=mol)
        if sync_coords:
            wrapper.sync_coords_from_mol()
        return wrapper

    @classmethod
    def from_atomistic(cls, atomistic: Atomistic | Wrapper[Atomistic]) -> RDKitWrapper:
        """Create RDKitWrapper from an Atomistic structure."""
        return cls(atomistic)

    # ------------------------------------------------------------------
    #   Properties
    # ------------------------------------------------------------------

    @property
    def mol(self) -> Chem.Mol:
        """Get the RDKit molecule."""
        return self._mol

    @property
    def core(self) -> Atomistic:
        """Get the Atomistic structure (alias for inner)."""
        return self._inner  # type: ignore[return-value]

    # ------------------------------------------------------------------
    #   Coordinate synchronization
    # ------------------------------------------------------------------

    def sync(self, direction: str = "both") -> None:
        """Synchronize data between RDKit `Mol` and MolPy `Atomistic`.

        Args:
            direction: one of "from_mol", "to_mol", or "both".

        Behavior:
        - "from_mol": copy coordinates, element/ids and hydrogen atoms from RDKit into
          the MolPy Atomistic (uses a hydrogen-expanded RDKit mol when available).
        - "to_mol": copy per-atom `xyz` from Atomistic into the RDKit conformer
          (creates a conformer if necessary).
        - "both": perform "from_mol" followed by "to_mol" (useful after structural
          edits on either side).
        """
        direction = (direction or "both").lower()
        if direction not in ("from_mol", "to_mol", "both"):
            raise ValueError("direction must be one of 'from_mol', 'to_mol', 'both'")

        # Ensure maps are available when needed
        if direction in ("from_mol", "both"):
            # Transfer coordinates + hydrogens from RDKit into Atomistic.
            if self._mol.GetNumConformers() > 0:
                # Build a hydrogen-expanded copy to capture explicit H atoms and coords
                mol_h = Chem.Mol(self._mol)
                mol_h = Chem.AddHs(mol_h, addCoords=True)

                # If the hydrogenized copy lacks conformers but original has one, copy it
                if mol_h.GetNumConformers() == 0 and self._mol.GetNumConformers() > 0:
                    mol_h.AddConformer(self._mol.GetConformer(), assignId=True)

                # Reuse internal transfer helper (handles hydrogen addition & mapping)
                self._transfer_coords_and_hydrogens(mol_h)

        if direction in ("to_mol", "both"):
            # Copy coordinates from Atomistic into RDKit conformer
            mon_atoms = self._core_atoms()
            n_atoms = len(mon_atoms)
            if n_atoms == 0:
                return

            self._ensure_atom_map_ready()

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

                ent = self._atom_map.get(ridx)
                if ent is None and ridx < len(mon_atoms):
                    ent = mon_atoms[ridx]
                if ent is None:
                    continue

                xyz = ent.get("xyz", [0.0, 0.0, 0.0])
                if len(xyz) >= 3:
                    conf.SetAtomPosition(ridx, (float(xyz[0]), float(xyz[1]), float(xyz[2])))

    # ------------------------------------------------------------------
    #   Convenience factories / helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_smiles(cls, smi: str, sync_coords: bool = False) -> "RDKitWrapper":
        """Create RDKitWrapper directly from a SMILES string.

        Args:
            smi: SMILES string
            sync_coords: if True, attempt to sync coordinates into Atomistic
        """
        mol = Chem.MolFromSmiles(smi)
        return cls.from_mol(mol, sync_coords=sync_coords)

    def to_smiles(self, canonical: bool = True) -> str:
        """Return SMILES for the wrapped RDKit molecule."""
        return Chem.MolToSmiles(Chem.Mol(self._mol), canonical=canonical)

    def embed_and_optimize(self, *, optimize: bool = True, random_seed: int = 42, max_iters: int = 200, add_hydrogens: bool = True) -> "RDKitWrapper":
        """Convenience wrapper around `generate_3d()` that returns self for chaining."""
        return self.generate_3d(optimize=optimize, random_seed=random_seed, max_iters=max_iters, add_hydrogens=add_hydrogens)

    # ------------------------------------------------------------------
    #   Atom mapping utilities
    # ------------------------------------------------------------------

    def _build_atom_map(self) -> dict[int, Any]:
        """Build atom mapping dictionary from current state.

        Maps RDKit atom indices to Atomistic atom entities.
        Uses MP_ID tags if available, otherwise maps by position / heavy atoms.
        """
        atom_map: dict[int, Any] = {}
        mon_atoms = self._core_atoms()
        atom_by_id = _build_atom_id_index(mon_atoms)

        mol_atoms = list(self._mol.GetAtoms())
        n_mol = len(mol_atoms)
        n_atomistic = len(mon_atoms)

        # Simple case: same atom count, map one-to-one by index
        if n_mol == n_atomistic:
            for rd_idx in range(n_mol):
                atom_map[rd_idx] = mon_atoms[rd_idx]
            return atom_map

        # Fallback: heavy-atom-based mapping
        heavy_atoms = [atom for atom in mon_atoms if atom.get("atomic_num", 0) != 1]
        heavy_pos = 0

        for rd_idx, rd_atom in enumerate(mol_atoms):
            if rd_atom.GetAtomicNum() == 1:
                continue  # Skip H for map

            ent: Any | None = None

            if rd_atom.HasProp(MP_ID):
                hid = int(rd_atom.GetIntProp(MP_ID))
                ent = atom_by_id.get(hid)

            if ent is None and heavy_pos < len(heavy_atoms):
                ent = heavy_atoms[heavy_pos]
                heavy_pos += 1

            if ent is not None:
                atom_map[rd_idx] = ent

        return atom_map

    def _ensure_mapping_tags(self) -> None:
        """Ensure MP_ID tags are set on RDKit atoms for mapping."""
        mon_atoms = self._core_atoms()
        for i, a in enumerate(self._mol.GetAtoms()):
            if a.GetAtomicNum() == 1:
                continue  # Skip hydrogens
            if a.HasProp(MP_ID):
                continue
            if i < len(mon_atoms):
                ent = mon_atoms[i]
                ent_id = ent.get("id", i)
                a.SetIntProp(MP_ID, int(ent_id))

    def _rebuild_atom_map(self) -> None:
        """Rebuild atom mapping dictionaries."""
        atom_map = self._build_atom_map()
        object.__setattr__(self, "_atom_map", atom_map)
        object.__setattr__(
            self, "_atom_map_reverse", {v: k for k, v in atom_map.items()}
        )

    # ------------------------------------------------------------------
    #   Visualization
    # ------------------------------------------------------------------

    def draw(
        self,
        *,
        size: tuple[int, int] = (320, 260),
        show_indices: bool = True,
        show_explicit_H: bool = False,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        title: str | None = None,
        show: bool = True,  # kept for API compatibility; unused here
    ) -> str:
        """Generate 2D molecular structure drawing as SVG."""
        _ensure_2d(self._mol)
        dm = Chem.Mol(self._mol)
        if show_explicit_H:
            dm = Chem.AddHs(dm)
            _ensure_2d(dm)

        w, h = size
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        opts.padding = 0.12
        opts.additionalAtomLabelPadding = 0.06
        opts.fixedFontSize = 13
        opts.minFontSize = 9
        opts.bondLineWidth = 2
        opts.addAtomIndices = bool(show_indices)
        opts.addStereoAnnotation = False
        opts.explicitMethyl = True

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            dm,
            highlightAtoms=highlight_atoms or [],
            highlightBonds=highlight_bonds or [],
            legend=(title or ""),
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg

    # ------------------------------------------------------------------
    #   3D generation / optimization
    # ------------------------------------------------------------------

    def generate_3d(
        self,
        *,
        optimize: bool = True,
        random_seed: int = 42,
        max_iters: int = 200,
        add_hydrogens: bool = True,
    ) -> RDKitWrapper:
        """Generate 3D coordinates using RDKit ETKDG + optional MMFF optimization."""
        mol = self._mol
        if mol.GetNumAtoms() == 0:
            raise ValueError("Cannot generate 3D coordinates for empty molecule")

        mol_working = Chem.Mol(mol)

        if add_hydrogens:
            molH = Chem.AddHs(mol_working, addCoords=True)
        else:
            molH = Chem.Mol(mol_working)
            if molH.GetNumConformers() == 0:
                conf = Chem.Conformer(molH.GetNumAtoms())
                molH.AddConformer(conf, assignId=True)

        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(random_seed)
        params.useRandomCoords = True

        embed_result = AllChem.EmbedMolecule(molH, params)  # type: ignore[attr-defined]
        if embed_result == -1:
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(molH, params)  # type: ignore[attr-defined]
            if embed_result == -1:
                raise RuntimeError("ETKDG embedding failed")

        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(molH, maxIters=int(max_iters))  # type: ignore[attr-defined]
            except Exception as e:  # pragma: no cover - defensive
                import warnings

                warnings.warn(f"MMFF optimization failed: {e}", stacklevel=2)

        # Synchronize coordinates and hydrogens to Atomistic
        self._transfer_coords_and_hydrogens(molH)

        # Update mol reference to include hydrogens
        object.__setattr__(self, "_mol", molH)

        # Rebuild atom mapping (may have added hydrogens)
        self._rebuild_atom_map()

        return self

    def optimize_geometry(
        self,
        *,
        max_iters: int = 200,
        force_field: str = "MMFF",
    ) -> RDKitWrapper:
        """Optimize molecular geometry using force field minimization."""
        if self._mol.GetNumConformers() == 0:
            raise ValueError("No conformer found. Call generate_3d() first.")

        mol = self._mol
        try:
            if force_field == "MMFF":
                AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))  # type: ignore[attr-defined]
            elif force_field == "UFF":
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))  # type: ignore[attr-defined]
            else:
                raise ValueError(
                    f"Unknown force field: {force_field}. Use 'MMFF' or 'UFF'."
                )
        except Exception as e:  # pragma: no cover - defensive
            import warnings

            warnings.warn(f"Geometry optimization failed: {e}", stacklevel=2)
            return self

        self.sync_coords_from_mol()
        return self

    # ------------------------------------------------------------------
    #   Internal: coordinate + hydrogen transfer
    # ------------------------------------------------------------------

    def _transfer_coords_and_hydrogens(self, mol_with_h: Chem.Mol) -> None:
        """Transfer coordinates and hydrogens from RDKit mol to Atomistic."""
        if mol_with_h.GetNumConformers() == 0:
            raise ValueError("RDKit mol has no conformer")

        self._ensure_atom_map_ready()

        conf = mol_with_h.GetConformer()
        core = self.unwrap()
        mon_atoms = list(core.atoms)
        heavy_index = _build_mon_heavy_index(mon_atoms)
        atom_by_id = _build_atom_id_index(mon_atoms)

        # 1) Transfer heavy atom coordinates
        for a in mol_with_h.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue

            if a.HasProp(MP_ID):
                hid = int(a.GetIntProp(MP_ID))
            else:
                hid = a.GetIdx()

            ent = heavy_index.get(hid)
            if ent is None:
                ent = self._atom_map.get(a.GetIdx())
            if ent is None:
                ent = atom_by_id.get(hid)
            if ent is None:
                continue

            p = conf.GetAtomPosition(a.GetIdx())
            ent["xyz"] = [float(p.x), float(p.y), float(p.z)]
            ent["atomic_num"] = a.GetAtomicNum()

        # 2) Add / sync hydrogen atoms
        existing_ids = {int(a.get("id")) for a in mon_atoms if "id" in a}
        next_id = max(existing_ids) + 1 if existing_ids else len(mon_atoms)

        def _new_id() -> int:
            nonlocal next_id
            val = next_id
            next_id += 1
            return val

        for a in mol_with_h.GetAtoms():
            if a.GetAtomicNum() != 1:
                continue

            rd_idx = a.GetIdx()

            # If we already have mapping for this RDKit index, treat it as already present H
            mapped_ent = self._atom_map.get(rd_idx)
            if mapped_ent is not None:
                p = conf.GetAtomPosition(rd_idx)
                mapped_ent["xyz"] = [float(p.x), float(p.y), float(p.z)]
                mapped_ent["atomic_num"] = 1
                continue

            # New H: need to attach to a heavy neighbor
            neighbors = list(a.GetNeighbors())
            if not neighbors:
                continue
            heavy = neighbors[0]

            # Determine ID for this H
            if a.HasProp(MP_ID):
                hid = int(a.GetIntProp(MP_ID))
                if hid in existing_ids:
                    hid = _new_id()
            else:
                hid = _new_id()

            p = conf.GetAtomPosition(rd_idx)
            h_ent = core.def_atom(
                symbol="H",
                atomic_num=1,
                id=hid,
                xyz=[float(p.x), float(p.y), float(p.z)],
            )
            existing_ids.add(hid)

            # Attach bond heavy-H if possible
            heavy_ent: Any | None = None

            if heavy.HasProp(MP_ID):
                heavy_id = int(heavy.GetIntProp(MP_ID))
                heavy_ent = heavy_index.get(heavy_id) or atom_by_id.get(heavy_id)

            if heavy_ent is None:
                heavy_ent = self._atom_map.get(heavy.GetIdx())

            if heavy_ent is not None:
                core.def_bond(heavy_ent, h_ent, type=1.0)

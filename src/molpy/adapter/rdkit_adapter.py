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

from molpy import Atomistic
from molpy.core.wrappers.base import Wrapper
from molpy.parser.smiles import SmilesIR

if TYPE_CHECKING:
    from molpy.core.wrappers.monomer import Monomer


# Stable property tag for bidirectional atom mapping
# Stores entity ID for reliable round-trip conversion
MP_ID = "mp_id"

# Bond order mappings between MolPy and RDKit
_BOND_ORDER_TO_RDKIT: dict[float, Chem.BondType] | None = None
_RDKIT_TO_BOND_ORDER: dict[Chem.BondType, float] | None = None


def _get_bond_order_to_rdkit() -> dict[float, Chem.BondType]:
    """Get MolPy bond order to RDKit bond type mapping.

    Returns:
        Dictionary mapping float bond orders to RDKit BondType enums.
    """
    global _BOND_ORDER_TO_RDKIT
    if _BOND_ORDER_TO_RDKIT is None:
        _BOND_ORDER_TO_RDKIT = {
            1.0: Chem.BondType.SINGLE,
            2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE,
            1.5: Chem.BondType.AROMATIC,  # Approximate; true aromaticity determined by RDKit
        }
    return _BOND_ORDER_TO_RDKIT


def _get_rdkit_to_bond_order() -> dict[Chem.BondType, float]:
    """Get RDKit bond type to MolPy bond order mapping.

    Returns:
        Dictionary mapping RDKit BondType enums to float bond orders.
    """
    global _RDKIT_TO_BOND_ORDER
    if _RDKIT_TO_BOND_ORDER is None:
        _RDKIT_TO_BOND_ORDER = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5,
        }
    return _RDKIT_TO_BOND_ORDER


def _ensure_2d(mol: Chem.Mol) -> None:
    """Compute 2D coordinates in-place if not present.

    Args:
        mol: RDKit molecule to add 2D coordinates to.
    """
    if not mol.GetNumConformers():
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)


def _build_mon_heavy_index(mon_atoms: list[Any]) -> dict[int, Any]:
    """Build index mapping entity IDs to heavy atom entities.

    Args:
        mon_atoms: List of atom entities from Atomistic structure.

    Returns:
        Dictionary mapping entity IDs to atom entities.
    """
    table: dict[int, Any] = {}
    for i, ent in enumerate(mon_atoms):
        hid = ent.get("id", i)
        table[int(hid)] = ent
    return table


def _rdkit_bond_type(order: float) -> Chem.BondType:
    """Convert MolPy bond order to RDKit BondType.

    Args:
        order: Bond order as float (1.0, 2.0, 3.0, 1.5).

    Returns:
        Corresponding RDKit BondType enum.
    """
    return _get_bond_order_to_rdkit().get(float(order), Chem.BondType.SINGLE)


def _order_from_rdkit(bt: Chem.BondType) -> float:
    """Convert RDKit BondType to MolPy bond order.

    Args:
        bt: RDKit BondType enum.

    Returns:
        Corresponding bond order as float.
    """
    return _get_rdkit_to_bond_order().get(bt, 1.0)


# =============================================================================
#   Conversion Functions: SmilesIR <-> Chem.Mol <-> Atomistic
# =============================================================================


def smilesir_to_mol(ir: SmilesIR) -> Chem.Mol:
    """Convert SmilesIR to RDKit Mol.

    Converts MolPy's internal SMILES representation to an RDKit molecule,
    preserving aromaticity, charges, stereochemistry, and explicit hydrogens.

    Args:
        ir: SmilesIR object from MolPy parser.

    Returns:
        Sanitized RDKit Mol object.

    Raises:
        RuntimeError: If molecule sanitization fails.
    """
    mol = Chem.RWMol()
    atom_map: dict[int, int] = {}
    aromatic_flags: dict[int, bool] = {}

    for atom_ir in ir.atoms:
        symbol = atom_ir.symbol or "C"
        # Aromatic atoms are encoded as lower-case in SMILES
        is_aromatic = symbol.islower()
        if len(symbol) == 1:
            rd_symbol = symbol.upper()
        else:
            rd_symbol = symbol[0].upper() + symbol[1:].lower()

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
    """Convert SmilesIR to Atomistic structure.

    Convenience function that converts SmilesIR to RDKit Mol, then to Atomistic.

    Args:
        ir: SmilesIR object from MolPy parser.

    Returns:
        Atomistic structure with atoms and bonds.
    """
    wrapper = RDKitWrapper.from_mol(smilesir_to_mol(ir), sync_coords=True)
    return wrapper.inner


# =============================================================================
#   RDKitWrapper: Bidirectional RDKit Mol <-> MolPy Atomistic Bridge
# =============================================================================


class RDKitWrapper(Wrapper[Atomistic]):
    """Bidirectional wrapper associating RDKit Mol with MolPy Atomistic.

    This class provides a bridge between RDKit's molecular representation and
    MolPy's Atomistic structure, maintaining consistent atom mappings and
    enabling coordinate synchronization, 3D generation, and visualization.

    The wrapper extends Wrapper[Atomistic] and stores both representations
    internally, with methods to convert between them and synchronize data.

    Key Features:
        - Bidirectional conversion between Chem.Mol and Atomistic
        - Stable atom mapping using MP_ID property tags
        - Coordinate synchronization (RDKit ↔ Atomistic)
        - 3D structure generation with ETKDG + MMFF optimization
        - 2D molecular visualization (SVG)
        - Geometry optimization (MMFF/UFF)

    Attributes:
        mol: The RDKit Mol object.
        atomistic: The wrapped Atomistic structure (alias for `inner`).
        inner: The wrapped Atomistic structure (from Wrapper base class).

    Examples:
        Create from SMILES and generate 3D coordinates:

        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> wrapper = RDKitWrapper.from_mol(mol)
        >>> wrapper.generate_3d(optimize=True)
        >>> atomistic = wrapper.inner  # Access MolPy structure

        Create from existing Atomistic:

        >>> wrapper = RDKitWrapper.from_atomistic(atomistic)
        >>> svg = wrapper.draw(show=False)  # Generate 2D drawing
    """

    __slots__ = ("_atom_map", "_atom_map_reverse", "_mol")

    def __post_init__(
        self,
        mol: Chem.Mol | None = None,
        atom_map: dict[int, Any] | None = None,
        **props: Any,
    ) -> dict[str, Any]:
        """Post-initialization hook for RDKit-specific setup.

        Args:
            mol: Optional RDKit molecule object (will be created if None)
            atom_map: Optional atom mapping dictionary (rdkit_idx -> atom entity)
            **props: Additional properties

        Returns:
            Remaining props dict
        """
        # Store RDKit molecule (create if not provided)
        if mol is None:
            mol = atomistic_to_mol(self)

        # Store RDKit molecule using object.__setattr__ (required for __slots__)
        object.__setattr__(self, "_mol", mol)

        # Build atom map if not provided
        # Note: _build_atom_map needs self._mol and self._inner to be set
        if atom_map is None:
            try:
                atom_map = self._build_atom_map()
            except Exception as e:
                # If building atom map fails, create empty map
                # This can happen if mol and atomistic don't match
                import warnings

                warnings.warn(
                    f"Failed to build atom map: {e}. Using empty map.", stacklevel=2
                )
                atom_map = {}

        object.__setattr__(self, "_atom_map", atom_map)
        object.__setattr__(
            self,
            "_atom_map_reverse",
            {v: k for k, v in atom_map.items()} if atom_map else {},
        )

        # Ensure MP_ID tags are set for mapping
        try:
            self._ensure_mapping_tags()
        except Exception as e:
            # If ensuring tags fails, continue anyway
            import warnings

            warnings.warn(f"Failed to ensure mapping tags: {e}", stacklevel=2)

        # Remove RDKit-specific props before returning
        remaining_props = {
            k: v for k, v in props.items() if k not in ("mol", "atom_map")
        }
        return remaining_props

    @classmethod
    def from_mol(cls, mol: Chem.Mol, sync_coords: bool = False) -> RDKitWrapper:
        """
        Create RDKitWrapper from a Chem.Mol.

        Args:
            mol: RDKit molecule object
            sync_coords: If True, synchronize coordinates from RDKit conformer

        Returns:
            RDKitWrapper instance
        """
        # Create Atomistic from Mol
        atomistic = mol_to_atomistic(mol)

        # Create wrapper with mol parameter passed to __post_init__
        wrapper = cls(atomistic, mol=mol)

        # Sync coordinates if requested
        if sync_coords:
            wrapper.sync_coords_from_mol()

        return wrapper

    @classmethod
    def from_atomistic(cls, atomistic: Atomistic | Wrapper[Atomistic]) -> RDKitWrapper:
        """
        Create RDKitWrapper from an Atomistic structure or a Wrapper[Atomistic].

        This method creates RDKitWrapper that wraps the input directly.
        When accessing Atomistic methods/properties, the wrapper automatically
        forwards to the innermost Atomistic via __getattr__.

        **Type Preservation**:
        - `wrapper.inner` returns the direct input (Atomistic or Wrapper)
        - `wrapper.unwrap()` returns the innermost Atomistic
        - `wrapper.atoms`, `wrapper.bonds`, etc. auto-forward to innermost Atomistic

        **Process**:
        1. Unwrap to get Atomistic for creating RDKit Mol
        2. Pass the ORIGINAL input to RDKitWrapper.__init__()
        3. Wrapper.__init__() stores it directly in _inner
        4. Accessing `wrapper.atoms` auto-forwards to innermost Atomistic

        Args:
            atomistic: Atomistic structure or Wrapper[Atomistic] (e.g., Monomer[Atomistic])
                      The wrapper chain is preserved.

        Returns:
            RDKitWrapper wrapping the input

        Examples:
            >>> # From plain Atomistic
            >>> wrapper = RDKitWrapper.from_atomistic(atomistic)
            >>> wrapper.inner  # Atomistic (same object)
            >>> wrapper.atoms  # Auto-forwards to atomistic.atoms

            >>> # From Monomer[Atomistic]
            >>> monomer: Monomer[Atomistic] = Monomer(atomistic)
            >>> wrapper = RDKitWrapper.from_atomistic(monomer)
            >>> wrapper.inner  # Monomer[Atomistic] (preserved!)
            >>> wrapper.unwrap()  # Atomistic (innermost)
            >>> wrapper.atoms  # Auto-forwards to atomistic.atoms via __getattr__
        """
        # Get underlying Atomistic to create RDKit Mol
        underlying = atomistic.unwrap() if isinstance(atomistic, Wrapper) else atomistic

        # Create Mol from underlying Atomistic
        mol = atomistic_to_mol(underlying)

        # Pass ORIGINAL atomistic to constructor (Wrapper.__init__ stores it as-is)
        return cls(atomistic, mol=mol)

    @property
    def mol(self) -> Chem.Mol:
        """Get the RDKit molecule."""
        return self._mol

    @property
    def atomistic(self) -> Atomistic:
        """Get the Atomistic structure (alias for inner)."""
        return self._inner

    def sync_coords_from_mol(self) -> None:
        """
        Synchronize coordinates from RDKit conformer to Atomistic.

        Updates atom positions in Atomistic based on RDKit conformer.
        """
        if self._mol.GetNumConformers() == 0:
            return

        conf = self._mol.GetConformer()
        mon_atoms = list(self.atoms)

        # Build reverse mapping if needed
        if not self._atom_map_reverse:
            self._rebuild_atom_map()

        # Update coordinates
        for a in self._mol.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue  # Skip hydrogens for now

            ridx = a.GetIdx()
            if not a.HasProp(MP_ID):
                # Fallback to index
                if ridx < len(mon_atoms):
                    ent = mon_atoms[ridx]
                else:
                    continue
            else:
                hid = int(a.GetProp(MP_ID))
                ent = self._atom_map.get(ridx)
                if ent is None:
                    # Try to find by id
                    for atom in mon_atoms:
                        if atom.get("id") == hid:
                            ent = atom
                            break
                    if ent is None:
                        continue

            p = conf.GetAtomPosition(ridx)
            ent["xyz"] = [float(p.x), float(p.y), float(p.z)]

    def sync_coords_to_mol(self) -> None:
        """
        Synchronize coordinates from Atomistic to RDKit conformer.

        Updates RDKit conformer based on atom positions in Atomistic.
        """
        mon_atoms = list(self.atoms)
        n_atoms = len(mon_atoms)

        if n_atoms == 0:
            return

        # Get or create conformer
        if self._mol.GetNumConformers() == 0:
            conf = Chem.Conformer(n_atoms)
            self._mol.AddConformer(conf, assignId=True)
        else:
            conf = self._mol.GetConformer()

        # Update coordinates
        for a in self._mol.GetAtoms():
            ridx = a.GetIdx()
            if a.GetAtomicNum() == 1:
                continue  # Skip hydrogens for now

            ent = self._atom_map.get(ridx)
            if ent is None:
                # Fallback to index
                if ridx < len(mon_atoms):
                    ent = mon_atoms[ridx]
                else:
                    continue

            xyz = ent.get("xyz", [0.0, 0.0, 0.0])
            if len(xyz) >= 3:
                conf.SetAtomPosition(
                    ridx, (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                )

    def _build_atom_map(self) -> dict[int, Any]:
        """Build atom mapping dictionary from current state.

        Maps RDKit atom indices to Atomistic atom entities.
        Uses MP_ID tags if available, otherwise maps by position.
        """
        atom_map: dict[int, Any] = {}
        mon_atoms = list(self.atoms)

        # Build index of Atomistic atoms by id
        atom_by_id: dict[int, Any] = {}
        for atom in mon_atoms:
            atom_id = atom.get("id")
            if atom_id is not None:
                atom_by_id[int(atom_id)] = atom

        # Map RDKit atoms to Atomistic atoms
        # Simple approach: map by index if atom counts match, or by MP_ID if available
        mol_atoms = list(self._mol.GetAtoms())
        n_mol = len(mol_atoms)
        n_atomistic = len(mon_atoms)

        # If counts match, map by index (mol_to_atomistic preserves order)
        if n_mol == n_atomistic:
            for rd_idx in range(n_mol):
                if rd_idx < n_atomistic:
                    atom_map[rd_idx] = mon_atoms[rd_idx]
        else:
            # Counts don't match, try to map by MP_ID or by heavy atom position
            # This handles cases where mol has hydrogens but atomistic doesn't, or vice versa
            heavy_atoms = [atom for atom in mon_atoms if atom.get("atomic_num", 0) != 1]
            heavy_pos = 0

            for rd_idx, rd_atom in enumerate(mol_atoms):
                # Skip hydrogens in mapping
                if rd_atom.GetAtomicNum() == 1:
                    continue

                ent = None

                # Try to find by MP_ID first
                if rd_atom.HasProp(MP_ID):
                    hid = int(rd_atom.GetProp(MP_ID))
                    ent = atom_by_id.get(hid)

                # Fallback: map by heavy atom position
                if ent is None and heavy_pos < len(heavy_atoms):
                    ent = heavy_atoms[heavy_pos]
                    heavy_pos += 1

                if ent is not None:
                    atom_map[rd_idx] = ent

        return atom_map

    def _ensure_mapping_tags(self) -> None:
        """Ensure MP_ID tags are set on RDKit atoms for mapping."""
        mon_atoms = list(self.atoms)
        for i, a in enumerate(self._mol.GetAtoms()):
            if a.GetAtomicNum() == 1:
                continue  # Skip hydrogens
            if not a.HasProp(MP_ID):
                # Set MP_ID based on atomistic
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

    def draw(
        self,
        *,
        size: tuple[int, int] = (320, 260),
        show_indices: bool = True,
        show_explicit_H: bool = False,
        highlight_atoms: list[int] | None = None,
        highlight_bonds: list[int] | None = None,
        title: str | None = None,
        show: bool = True,
    ) -> str:
        """Generate 2D molecular structure drawing as SVG.

        Creates a 2D depiction of the molecule using RDKit's drawing tools.
        Automatically computes 2D coordinates if not present.

        Args:
            size: Image dimensions (width, height) in pixels.
            show_indices: Whether to display atom indices.
            show_explicit_H: Whether to show explicit hydrogen atoms.
            highlight_atoms: List of atom indices to highlight.
            highlight_bonds: List of bond indices to highlight.
            title: Title text to display on the image.
            show: Whether to display in Jupyter (requires IPython).

        Returns:
            SVG string representation of the molecule.
        """
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

    def generate_3d(
        self,
        *,
        optimize: bool = True,
        random_seed: int = 42,
        max_iters: int = 200,
        add_hydrogens: bool = True,
    ) -> RDKitWrapper:
        """Generate 3D coordinates using RDKit ETKDG + optional MMFF optimization.

        Uses RDKit's ETKDGv3 (Experimental Torsion Knowledge Distance Geometry)
        method to generate a 3D conformer, optionally followed by MMFF force field
        optimization. Coordinates and added hydrogens are synchronized back to the
        Atomistic structure.

        Args:
            optimize: Whether to optimize with MMFF force field after embedding.
            random_seed: Random seed for reproducible coordinate generation.
            max_iters: Maximum iterations for MMFF optimization.
            add_hydrogens: Whether to add explicit hydrogen atoms.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If molecule is empty.
            RuntimeError: If ETKDG embedding fails after retries.
        """
        # Ensure molecule has atoms
        mol = self._mol
        if mol.GetNumAtoms() == 0:
            raise ValueError("Cannot generate 3D coordinates for empty molecule")

        # Create working copy to avoid modifying original mol
        mol_working = Chem.Mol(mol)

        # Add hydrogens if requested
        if add_hydrogens:
            molH = Chem.AddHs(mol_working, addCoords=True)
        else:
            molH = Chem.Mol(mol_working)
            # Create empty conformer if none exists
            if molH.GetNumConformers() == 0:
                conf = Chem.Conformer(molH.GetNumAtoms())
                molH.AddConformer(conf, assignId=True)

        # MP_ID tags for heavy atoms should already exist
        # New hydrogens will be handled in _transfer_coords_and_hydrogens

        # Use ETKDGv3 for 3D embedding
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
        params.randomSeed = int(random_seed)
        params.useRandomCoords = True

        embed_result = AllChem.EmbedMolecule(molH, params)  # type: ignore[attr-defined]
        if embed_result == -1:
            # Retry with random coords
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(molH, params)  # type: ignore[attr-defined]
            if embed_result == -1:
                raise RuntimeError("ETKDG embedding failed")

        # MMFF optimization (if enabled)
        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(molH, maxIters=int(max_iters))  # type: ignore[attr-defined]
            except Exception as e:
                # Keep embedded structure even if optimization fails
                import warnings

                warnings.warn(f"MMFF optimization failed: {e}", stacklevel=2)

        # Synchronize coordinates and hydrogens to Atomistic
        # Note: self._atom_map still points to original mol (without H)
        # _transfer_coords_and_hydrogens handles hydrogen addition
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
        """Optimize molecular geometry using force field minimization.

        Performs energy minimization on the existing 3D conformer using either
        MMFF94 or UFF force field. Coordinates are synchronized back to Atomistic.

        Args:
            max_iters: Maximum optimization iterations.
            force_field: Force field to use ("MMFF" or "UFF").

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no conformer exists or invalid force field specified.
        """
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
        except Exception as e:
            import warnings

            warnings.warn(f"Geometry optimization failed: {e}", stacklevel=2)
            return self

        # Sync coordinates to Atomistic
        self.sync_coords_from_mol()

        return self

    def _transfer_coords_and_hydrogens(self, mol_with_h: Chem.Mol) -> None:
        """Transfer coordinates and hydrogens from RDKit mol to Atomistic.

        Internal method that synchronizes heavy atom coordinates and adds any
        new hydrogen atoms from the RDKit molecule to the Atomistic structure.

        Args:
            mol_with_h: RDKit molecule with hydrogen atoms and conformer.

        Raises:
            ValueError: If RDKit molecule has no conformer.
        """
        if mol_with_h.GetNumConformers() == 0:
            raise ValueError("RDKit mol has no conformer")

        conf = mol_with_h.GetConformer()
        mon_atoms = list(self.atoms)
        heavy_index = _build_mon_heavy_index(mon_atoms)

        # Transfer heavy atom coordinates
        for a in mol_with_h.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue

            if not a.HasProp(MP_ID):
                # Fallback to index if no MP_ID tag
                hid = a.GetIdx()
            else:
                hid = int(a.GetProp(MP_ID))

            ent = heavy_index.get(hid)
            if ent is None:
                # Try to find via atom map
                rd_idx = a.GetIdx()
                ent = self._atom_map.get(rd_idx)
                if ent is None:
                    # Skip if still not found
                    continue

            p = conf.GetAtomPosition(a.GetIdx())
            ent["xyz"] = [float(p.x), float(p.y), float(p.z)]

            # Ensure element information is set correctly
            ent["atomic_num"] = a.GetAtomicNum()
            ent["element"] = a.GetSymbol()
            if "symbol" not in ent or ent["symbol"] is None:
                ent["symbol"] = a.GetSymbol()

        # Add hydrogens and connect to their heavy atom neighbors
        for a in mol_with_h.GetAtoms():
            if a.GetAtomicNum() != 1:
                continue

            bonds = list(a.GetBonds())
            if not bonds:
                continue

            other = bonds[0].GetOtherAtom(a)
            if not other.HasProp(MP_ID):
                continue

            hid = int(other.GetProp(MP_ID))
            heavy_ent = heavy_index.get(hid)
            if heavy_ent is None:
                # Try to find via atom map
                rd_idx = other.GetIdx()
                heavy_ent = self._atom_map.get(rd_idx)
                if heavy_ent is None:
                    continue

            # Check if hydrogen already exists at this position
            existing_h = False
            for bond in self.bonds:
                if bond.itom == heavy_ent or bond.jtom == heavy_ent:
                    other_atom = bond.jtom if bond.itom == heavy_ent else bond.itom
                    if other_atom.get("atomic_num") == 1:
                        # Check if positions are close
                        p = conf.GetAtomPosition(a.GetIdx())
                        pos_h = other_atom.get("xyz", [0, 0, 0])
                        dist_sq = (
                            (p.x - pos_h[0]) ** 2
                            + (p.y - pos_h[1]) ** 2
                            + (p.z - pos_h[2]) ** 2
                        )
                        if dist_sq < 0.01:  # Threshold: 0.1 Angstrom squared
                            existing_h = True
                            # Update coordinates
                            other_atom["xyz"] = [float(p.x), float(p.y), float(p.z)]
                            break

            if not existing_h:
                p = conf.GetAtomPosition(a.GetIdx())
                h_ent = self.def_atom(
                    symbol="H", atomic_num=1, xyz=[float(p.x), float(p.y), float(p.z)]
                )
                self.def_bond(heavy_ent, h_ent, order=1.0)

    def __repr__(self) -> str:
        """String representation of wrapper."""
        n_atoms = self._mol.GetNumAtoms() if self._mol else 0
        n_inner_atoms = len(self.atoms)
        has_conformer = self._mol.GetNumConformers() > 0 if self._mol else False
        return f"<RDKitWrapper mol_atoms={n_atoms} inner_atoms={n_inner_atoms} has_conformer={has_conformer}>"


# =============================================================================
#   Low-Level Conversion Functions
# =============================================================================


def mol_to_atomistic(mol: Chem.Mol) -> Atomistic:
    """Convert RDKit Mol to MolPy Atomistic structure.

    Performs a straightforward conversion preserving:
    - Atomic symbols and atomic numbers
    - Formal charges
    - Coordinates (if conformer exists)
    - Bond orders (mapped to 1.0/2.0/3.0/1.5)

    Args:
        mol: RDKit Mol object to convert.

    Returns:
        Atomistic structure with atoms and bonds.
    """
    atomistic = Atomistic()
    atom_map: dict[int, Any] = {}

    conf = mol.GetConformer(0) if mol.GetNumConformers() else None

    # atoms
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        data: dict[str, Any] = {
            "symbol": a.GetSymbol(),
            "atomic_num": a.GetAtomicNum(),
            "formal_charge": a.GetFormalCharge(),
        }
        if conf is not None:
            p = conf.GetAtomPosition(idx)
            data["xyz"] = [float(p.x), float(p.y), float(p.z)]
        atom = atomistic.def_atom(**data)
        atom_map[idx] = atom

    # bonds
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        order = _order_from_rdkit(b.GetBondType())
        atomistic.def_bond(
            atom_map[i], atom_map[j], order=order, type=str(b.GetBondType())
        )

    return atomistic


def atomistic_to_mol(atomistic: Atomistic) -> Chem.Mol:
    """Convert MolPy Atomistic structure to RDKit Mol.

    Creates an RDKit molecule from Atomistic, setting MP_ID property tags
    on each atom for stable round-trip conversion. If atom positions exist,
    creates a conformer with 3D coordinates.

    Args:
        atomistic: Atomistic structure to convert.

    Returns:
        RDKit Mol object with MP_ID tags and optional conformer.
    """
    rwmol = Chem.RWMol()
    atom_map: dict[Any, int] = {}
    mon_atoms = list(atomistic.atoms)

    # atoms
    for i, ent in enumerate(mon_atoms):
        sym = ent.get("symbol")
        rd_atom = Chem.Atom(sym)
        if "atomic_num" in ent:
            rd_atom.SetAtomicNum(int(ent["atomic_num"]))
        if "formal_charge" in ent:
            rd_atom.SetFormalCharge(int(ent["formal_charge"]))

        ridx = rwmol.AddAtom(rd_atom)
        rwmol.GetAtomWithIdx(ridx).SetIntProp(MP_ID, int(ent.get("id", i)))
        atom_map[ent] = ridx

    # bonds
    for b in atomistic.bonds:
        i = atom_map[b.itom]
        j = atom_map[b.jtom]
        bt = _rdkit_bond_type(b.get("order", 1))
        rwmol.AddBond(i, j, bt)

    mol = rwmol.GetMol()

    # Sanitize to ensure valence calculations
    Chem.SanitizeMol(mol)

    # coordinates
    conf = Chem.Conformer(len(mon_atoms))
    for ent, ridx in atom_map.items():
        xyz = ent.get("xyz", None)
        if xyz is None:
            continue
        conf.SetAtomPosition(ridx, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    return mol


def monomer_to_mol(monomer: Monomer) -> Chem.Mol:
    """Convert MolPy Monomer to RDKit Mol.

    Convenience function that unwraps the Monomer and converts the underlying
    Atomistic structure.

    Args:
        monomer: Monomer wrapper to convert.

    Returns:
        RDKit Mol object.
    """
    return atomistic_to_mol(monomer.unwrap())

from collections import defaultdict
from typing import Any

from .entity import (
    ConnectivityMixin,
    Entities,
    Entity,
    Link,
    MembershipMixin,
    SpatialMixin,
    Struct,
)


class Atom(Entity):
    """Atom entity (expects optional keys like {"type": "C", "xyz": [...]})"""

    def __repr__(self) -> str:
        identifier: str
        if "symbol" in self.data:
            identifier = str(self.data["symbol"])
        elif "type" in self.data:
            identifier = str(self.data["type"])
        else:
            identifier = str(id(self))
        return f"<Atom: {identifier}>"


class Bond(Link):
    def __init__(self, a: Atom, b: Atom, /, **attrs: Any):
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<Bond: {self.itom} - {self.jtom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]


class Angle(Link):
    def __init__(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any):
        super().__init__([a, b, c], **attrs)

    def __repr__(self) -> str:
        return f"<Angle: {self.itom} - {self.jtom} - {self.ktom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]


class Dihedral(Link):
    """Dihedral (torsion) angle between four atoms"""

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any):
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Dihedral: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]


class Atomistic(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin):
    def __init__(self, **props) -> None:
        super().__init__(**props)
        # Call __post_init__ if it exists (for template pattern)
        if hasattr(self, "__post_init__"):
            # Get the method from the actual class, not from parent
            for klass in type(self).__mro__:
                if klass is Atomistic:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break
        self.entities.register_type(Atom)
        self.links.register_type(Bond)
        self.links.register_type(Angle)
        self.links.register_type(Dihedral)

    @property
    def atoms(self) -> Entities[Atom]:
        return self.entities[Atom]

    @property
    def bonds(self) -> Entities[Bond]:  # type: ignore[type-var]
        return self.links[Bond]  # type: ignore[return-value]

    @property
    def angles(self) -> Entities[Angle]:  # type: ignore[type-var]
        return self.links[Angle]  # type: ignore[return-value]

    @property
    def dihedrals(self) -> Entities[Dihedral]:  # type: ignore[type-var]
        return self.links[Dihedral]  # type: ignore[return-value]

    @property
    def symbols(self) -> list[str]:
        atoms = list(self.atoms)
        return [str(a.get("symbol", "")) for a in atoms]

    @property
    def xyz(self):
        """
        Get atomic positions as numpy array.

        Returns:
            Nx3 array of atomic coordinates, or list of lists if numpy not available
        """
        atoms = list(self.atoms)
        positions = []
        for atom in atoms:
            xyz = atom.get("xyz", atom.get("xyz", [0.0, 0.0, 0.0]))
            positions.append(xyz)

        try:
            import numpy as np

            return np.array(positions)
        except ImportError:
            return positions

    @property
    def positions(self):
        """Alias for xyz property."""
        return self.xyz

    def __repr__(self) -> str:
        """
        Return a concise representation of the atomistic structure.

        Shows:
        - Number of atoms (with element composition)
        - Number of bonds
        - Bounding box if positions available
        """
        atoms = self.atoms
        bonds = self.bonds

        # Count atoms by symbol
        from collections import Counter

        symbols = [a.get("symbol", "?") for a in atoms]
        symbol_counts = Counter(symbols)

        # Format composition
        if len(symbol_counts) <= 5:
            composition = " ".join(
                f"{sym}:{cnt}" for sym, cnt in sorted(symbol_counts.items())
            )
        else:
            composition = f"{len(symbol_counts)} types"

        # Check if we have positions
        has_coords = any("xyz" in a or "xyz" in a for a in atoms)

        parts = ["Atomistic"]
        parts.append(f"{len(atoms)} atoms ({composition})")
        parts.append(f"{len(bonds)} bonds")

        if has_coords:
            parts.append("with coords")

        return f"<{', '.join(parts)}>"

    # ========== Factory Methods (def_*: create and add) ==========

    def def_atom(self, **attrs: Any) -> Atom:
        """Create a new Atom and add it to the structure."""
        atom = Atom(**attrs)
        self.entities.add(atom)
        return atom

    def def_bond(self, a: Atom, b: Atom, /, **attrs: Any) -> Bond:
        """Create a new Bond between two atoms and add it to the structure."""
        bond = Bond(a, b, **attrs)
        self.links.add(bond)
        return bond

    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle:
        """Create a new Angle between three atoms and add it to the structure."""
        angle = Angle(a, b, c, **attrs)
        self.links.add(angle)
        return angle

    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral:
        """Create a new Dihedral between four atoms and add it to the structure."""
        dihedral = Dihedral(a, b, c, d, **attrs)
        self.links.add(dihedral)
        return dihedral

    # ========== Add Methods (add_*: add existing entities) ==========

    def add_atom(self, atom: Atom, /) -> Atom:
        """Add an existing Atom object to the structure."""
        self.entities.add(atom)
        return atom

    def add_bond(self, bond: Bond, /) -> Bond:
        """Add an existing Bond object to the structure."""
        self.links.add(bond)
        return bond

    def add_angle(self, angle: Angle, /) -> Angle:
        """Add an existing Angle object to the structure."""
        self.links.add(angle)
        return angle

    def add_dihedral(self, dihedral: Dihedral, /) -> Dihedral:
        """Add an existing Dihedral object to the structure."""
        self.links.add(dihedral)
        return dihedral

    # ========== Batch Factory Methods (def_*s: create and add multiple) ==========

    def def_atoms(self, atoms_data: list[dict[str, Any]], /) -> list[Atom]:
        """Create multiple Atoms from a list of attribute dictionaries."""
        atoms = []
        for attrs in atoms_data:
            atom = self.def_atom(**attrs)
            atoms.append(atom)
        return atoms

    def def_bonds(
        self, bonds_data: list[tuple[Atom, Atom] | tuple[Atom, Atom, dict[str, Any]]], /
    ) -> list[Bond]:
        """Create multiple Bonds from a list of (atom1, atom2) or (atom1, atom2, attrs) tuples."""
        bonds = []
        for bond_spec in bonds_data:
            if len(bond_spec) == 2:
                a, b = bond_spec
                attrs = {}
            else:
                a, b, attrs = bond_spec
            bond = self.def_bond(a, b, **attrs)
            bonds.append(bond)
        return bonds

    def def_angles(
        self,
        angles_data: list[
            tuple[Atom, Atom, Atom] | tuple[Atom, Atom, Atom, dict[str, Any]]
        ],
        /,
    ) -> list[Angle]:
        """Create multiple Angles from a list of (atom1, atom2, atom3) or (atom1, atom2, atom3, attrs) tuples."""
        angles = []
        for angle_spec in angles_data:
            if len(angle_spec) == 3:
                a, b, c = angle_spec
                attrs = {}
            else:
                a, b, c, attrs = angle_spec
            angle = self.def_angle(a, b, c, **attrs)
            angles.append(angle)
        return angles

    def def_dihedrals(
        self,
        dihedrals_data: list[
            tuple[Atom, Atom, Atom, Atom]
            | tuple[Atom, Atom, Atom, Atom, dict[str, Any]]
        ],
        /,
    ) -> list[Dihedral]:
        """Create multiple Dihedrals from a list of (atom1, atom2, atom3, atom4) or (atom1, atom2, atom3, atom4, attrs) tuples."""
        dihedrals = []
        for dihe_spec in dihedrals_data:
            if len(dihe_spec) == 4:
                a, b, c, d = dihe_spec
                attrs = {}
            else:
                a, b, c, d, attrs = dihe_spec
            dihedral = self.def_dihedral(a, b, c, d, **attrs)
            dihedrals.append(dihedral)
        return dihedrals

    # ========== Batch Add Methods (add_*s: add multiple existing entities) ==========

    def add_atoms(self, atoms: list[Atom], /) -> list[Atom]:
        """Add multiple existing Atom objects to the structure."""
        for atom in atoms:
            self.entities.add(atom)
        return atoms

    def add_bonds(self, bonds: list[Bond], /) -> list[Bond]:
        """Add multiple existing Bond objects to the structure."""
        for bond in bonds:
            self.links.add(bond)
        return bonds

    def add_angles(self, angles: list[Angle], /) -> list[Angle]:
        """Add multiple existing Angle objects to the structure."""
        for angle in angles:
            self.links.add(angle)
        return angles

    def add_dihedrals(self, dihedrals: list[Dihedral], /) -> list[Dihedral]:
        """Add multiple existing Dihedral objects to the structure."""
        for dihedral in dihedrals:
            self.links.add(dihedral)
        return dihedrals

    # ========== Spatial Operations (return self for chaining) ==========

    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Atom
    ) -> "Atomistic":
        """Move all entities by delta. Returns self for chaining."""
        super().move(delta, entity_type=entity_type)
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        """Rotate entities around axis. Returns self for chaining."""
        super().rotate(axis, angle, about=about, entity_type=entity_type)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        """Scale entities by factor. Returns self for chaining."""
        super().scale(factor, about=about, entity_type=entity_type)
        return self

    def align(
        self,
        a: Entity,
        b: Entity,
        *,
        a_dir: list[float] | None = None,
        b_dir: list[float] | None = None,
        flip: bool = False,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        """Align entities. Returns self for chaining."""
        super().align(
            a, b, a_dir=a_dir, b_dir=b_dir, flip=flip, entity_type=entity_type
        )
        return self

    # ========== System Composition ==========

    def __iadd__(self, other: "Atomistic") -> "Atomistic":
        """
        Merge another Atomistic into this one (in-place).

        Example:
            system = Water()
            system += Water().move([5, 0, 0])
            system += Methane().move([10, 0, 0])
        """
        self.merge(other)
        return self

    def __add__(self, other: "Atomistic") -> "Atomistic":
        """
        Create a new Atomistic by merging two systems.

        Example:
            combined = water1 + water2
        """
        result = self.copy()
        result.merge(other)
        return result

    def replicate(self, n: int, transform=None) -> "Atomistic":
        """
        Create n copies and merge them into a new system.

        Args:
            n: Number of copies to create
            transform: Optional callable(copy, index) -> None to transform each copy

        Example:
            # Create 10 waters in a line
            waters = Water().replicate(10, lambda mol, i: mol.move([i*5, 0, 0]))

            # Create 3x3 grid
            grid = Methane().replicate(9, lambda mol, i: mol.move([i%3*5, i//3*5, 0]))
        """
        result = type(self)()  # Empty system of same type

        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)

        return result

    def __len__(self) -> int:
        return len(self.atoms)

    def get_topo(self, gen_angle: bool = False, gen_dihe=False):
        vertrices = {}
        for i, atom in enumerate(self.entities[Atom]):
            vertrices[atom] = i
        edges = []
        for bond in self.links[Bond]:  # type: ignore[arg-type]
            edges.append((vertrices[bond.itom], vertrices[bond.jtom]))
        atoms = list(vertrices.keys())
        from .topology import Topology

        topo = Topology(len(vertrices), edges=edges)
        if gen_angle:
            for angle in topo.angles:
                angle = angle.tolist()
                self.links.add(Angle(atoms[angle[0]], atoms[angle[1]], atoms[angle[2]]))
        if gen_dihe:
            for dihe in topo.dihedrals:
                dihe = dihe.tolist()
                self.links.add(
                    Dihedral(
                        atoms[dihe[0]], atoms[dihe[1]], atoms[dihe[2]], atoms[dihe[3]]
                    )
                )

        return topo

    # ========== Conversion Methods ==========

    def to_frame(self, atom_fields: list[str] | None = None) -> "Frame":
        """Convert to LAMMPS data Frame format.

        Converts this Atomistic structure into a Frame suitable for writing
        as a LAMMPS data file.

        Args:
            atom_fields: List of atom fields to extract. If None, extracts all fields.
            bond_fields: List of bond fields to extract. If None, extracts all fields.
            angle_fields: List of angle fields to extract. If None, extracts all fields.
            dihedral_fields: List of dihedral fields to extract. If None, extracts all fields.

        Returns:
            Frame with atoms, bonds, angles, and dihedrals

        Example:
            >>> butane = CH3() + CH2() + CH3()
            >>> # Extract all fields
            >>> frame = butane.to_frame()
            >>> # Extract specific fields only
            >>> frame = butane.to_frame(
            ...     atom_fields=['xyz', 'charge', 'element', 'type'],
            ...     bond_fields=['itom', 'jtom', 'type'],
            ... )
            >>> writer = LammpsDataWriter("system.data")
            >>> writer.write(frame)
        """
        import numpy as np

        from .frame import Block, Frame

        frame = Frame()

        # Get all topology data
        atoms_data = list(self.atoms)
        bonds_data = list(self.bonds)
        angles_data = list(self.angles)
        dihedrals_data = list(self.dihedrals)

        # Build atoms Block - convert array of struct to struct of array
        # Determine which keys to extract
        if atom_fields is None:
            # Collect all keys from all atoms
            all_keys = set()
            for atom in atoms_data:
                all_keys.update(atom.keys())
        else:
            all_keys = set(atom_fields)

        # Initialize dict for all keys
        atom_dict = {key: [] for key in all_keys}

        # Create atom ID to index mapping
        atom_id_to_index = {}

        # Convert: just iterate and append each key's value
        for atom in atoms_data:
            atom_id_to_index[id(atom)] = len(atom_id_to_index)

            for key in all_keys:
                atom_dict[key].append(atom.get(key, None))

        # Convert to numpy arrays
        atom_dict_np = {k: np.array(v) for k, v in atom_dict.items()}
        frame["atoms"] = Block.from_dict(atom_dict_np)

        # Build bonds Block - convert array of struct to struct of array
        if bonds_data:
            # Always include atom references
            bond_dict = defaultdict(list)

            for bond in bonds_data:
                # Atom references from properties
                bond_dict["atom_i"].append(atom_id_to_index[id(bond.itom)])
                bond_dict["atom_j"].append(atom_id_to_index[id(bond.jtom)])
                # Data fields
                for key in bond:
                    bond_dict[key].append(bond.get(key, None))
            bond_dict_np = {k: np.array(v) for k, v in bond_dict.items()}
            frame["bonds"] = Block.from_dict(bond_dict_np)

        # Build angles Block - convert array of struct to struct of array
        if angles_data:
            angle_dict = defaultdict(list)

            for angle in angles_data:
                # Atom references from properties
                angle_dict["atom_i"].append(atom_id_to_index[id(angle.itom)])
                angle_dict["atom_j"].append(atom_id_to_index[id(angle.jtom)])
                angle_dict["atom_k"].append(atom_id_to_index[id(angle.ktom)])
                # Data fields
                for key in angle:
                    angle_dict[key].append(angle.get(key, None))

            angle_dict_np = {k: np.array(v) for k, v in angle_dict.items()}
            frame["angles"] = Block.from_dict(angle_dict_np)
        # Build dihedrals Block - convert array of struct to struct of array
        if dihedrals_data:
            dihedral_dict = defaultdict(list)

            for dihedral in dihedrals_data:
                # Atom references from properties
                dihedral_dict["atom_i"].append(atom_id_to_index[id(dihedral.itom)])
                dihedral_dict["atom_j"].append(atom_id_to_index[id(dihedral.jtom)])
                dihedral_dict["atom_k"].append(atom_id_to_index[id(dihedral.ktom)])
                dihedral_dict["atom_l"].append(atom_id_to_index[id(dihedral.ltom)])
                # Data fields
                for key in dihedral:
                    dihedral_dict[key].append(dihedral.get(key, None))

            dihedral_dict_np = {k: np.array(v) for k, v in dihedral_dict.items()}
            frame["dihedrals"] = Block.from_dict(dihedral_dict_np)

        return frame

    def to_molecule_frame(self) -> "Frame":
        """Convert to LAMMPS molecule template Frame format.

        Converts this Atomistic structure into a Frame suitable for writing
        as a LAMMPS molecule template file. The molecule template format
        uses local atom IDs (starting from 1) and doesn't include box information.

        Returns:
            Frame with atoms, bonds, angles, and dihedrals blocks

        Example:
            >>> ch2 = CH2()
            >>> frame = ch2.to_molecule_frame()
            >>> writer = LammpsMoleculeWriter("ch2.mol")
            >>> writer.write(frame)
        """
        import numpy as np

        from .frame import Block, Frame

        frame = Frame()

        atoms_data = list(self.atoms)
        bonds_data = list(self.bonds)
        angles_data = list(self.angles)
        dihedrals_data = list(self.dihedrals)

        # Build atoms Block (molecule template format)
        atom_dict = {
            "id": [],
            "type": [],
            "q": [],
            "x": [],
            "y": [],
            "z": [],
        }

        atom_id_to_index = {}

        for i, atom in enumerate(atoms_data, 1):
            atom_id_to_index[id(atom)] = i
            atom_dict["id"].append(i)
            atom_type = atom.get("type")
            atom_dict["type"].append(atom_type)
            charge = atom.get("charge", atom.get("q"))
            atom_dict["q"].append(charge)
            xyz = atom.get("xyz", atom.get("xyz", [0.0, 0.0, 0.0]))
            atom_dict["x"].append(float(xyz[0]))
            atom_dict["y"].append(float(xyz[1]))
            atom_dict["z"].append(float(xyz[2]))

        atom_dict_np = {k: np.array(v) for k, v in atom_dict.items()}
        frame["atoms"] = Block.from_dict(atom_dict_np)

        # Build connectivity blocks
        if bonds_data:
            bond_dict = {"id": [], "type": [], "atom1": [], "atom2": []}
            for i, bond in enumerate(bonds_data, 1):
                bond_dict["id"].append(i)
                bond_type = bond.get("type")
                bond_dict["type"].append(bond_type)
                bond_dict["atom1"].append(atom_id_to_index[id(bond.itom)])
                bond_dict["atom2"].append(atom_id_to_index[id(bond.jtom)])
            bond_dict_np = {k: np.array(v) for k, v in bond_dict.items()}
            frame["bonds"] = Block.from_dict(bond_dict_np)

        if angles_data:
            angle_dict = {"id": [], "type": [], "atom1": [], "atom2": [], "atom3": []}
            for i, angle in enumerate(angles_data, 1):
                angle_dict["id"].append(i)
                angle_type = angle.get("type")
                angle_dict["type"].append(angle_type)
                angle_dict["atom1"].append(atom_id_to_index[id(angle.itom)])
                angle_dict["atom2"].append(atom_id_to_index[id(angle.jtom)])
                angle_dict["atom3"].append(atom_id_to_index[id(angle.ktom)])
            angle_dict_np = {k: np.array(v) for k, v in angle_dict.items()}
            frame["angles"] = Block.from_dict(angle_dict_np)

        if dihedrals_data:
            dihedral_dict = {
                "id": [],
                "type": [],
                "atom1": [],
                "atom2": [],
                "atom3": [],
                "atom4": [],
            }
            for i, dihedral in enumerate(dihedrals_data, 1):
                dihedral_dict["id"].append(i)
                dihedral_type = dihedral.get("type")
                dihedral_dict["type"].append(dihedral_type)
                dihedral_dict["atom1"].append(atom_id_to_index[id(dihedral.itom)])
                dihedral_dict["atom2"].append(atom_id_to_index[id(dihedral.jtom)])
                dihedral_dict["atom3"].append(atom_id_to_index[id(dihedral.ktom)])
                dihedral_dict["atom4"].append(atom_id_to_index[id(dihedral.ltom)])
            dihedral_dict_np = {k: np.array(v) for k, v in dihedral_dict.items()}
            frame["dihedrals"] = Block.from_dict(dihedral_dict_np)

        return frame

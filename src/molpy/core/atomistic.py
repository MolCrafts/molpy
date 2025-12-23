from collections import defaultdict
from typing import Any, Iterable

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
            x = atom.get("x", 0.0)
            y = atom.get("y", 0.0)
            z = atom.get("z", 0.0)
            positions.append([x, y, z])

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
        """Create a new Atom and add it to the structure.

        If 'xyz' is provided, it will be converted to separate x, y, z fields.
        """
        # Convert xyz to x, y, z if provided
        if "xyz" in attrs:
            xyz = attrs.pop("xyz")
            attrs["x"] = float(xyz[0])
            attrs["y"] = float(xyz[1])
            attrs["z"] = float(xyz[2])

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

    def extract_subgraph(
        self,
        center_entities: Iterable[Atom],
        radius: int,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> tuple["Atomistic", list[Atom]]:
        """Extract subgraph preserving all topology (bonds, angles, dihedrals).

        Overrides ConnectivityMixin.extract_subgraph to ensure all topology
        types (bonds, angles, dihedrals) are preserved in the extracted subgraph.

        Args:
            center_entities: Center atoms for extraction
            radius: Topological radius
            entity_type: Entity type (should be Atom)
            link_type: Link type for topology calculation (should be Bond)

        Returns:
            Tuple of (subgraph Atomistic, edge atoms)
        """
        from copy import deepcopy

        # Call parent method to extract subgraph with bonds
        subgraph, edge_entities = super().extract_subgraph(
            center_entities=center_entities,
            radius=radius,
            entity_type=entity_type,
            link_type=link_type,
        )

        # Build mapping from original atoms to subgraph atoms (by react_id or id)
        original_to_subgraph = {}
        subgraph_atoms_set = set(subgraph.atoms)

        for subgraph_atom in subgraph_atoms_set:
            # Try to match by react_id first, then by id
            subgraph_rid = subgraph_atom.get("react_id")
            subgraph_id = subgraph_atom.get("id")

            for orig_atom in self.atoms:
                if subgraph_rid and orig_atom.get("react_id") == subgraph_rid:
                    original_to_subgraph[orig_atom] = subgraph_atom
                    break
                elif subgraph_id and orig_atom.get("id") == subgraph_id:
                    original_to_subgraph[orig_atom] = subgraph_atom
                    break

        # Copy angles from original to subgraph
        for angle in self.angles:
            endpoints = angle.endpoints
            if all(ep in original_to_subgraph for ep in endpoints):
                subgraph_eps = [original_to_subgraph[ep] for ep in endpoints]
                # Check if angle already exists
                exists = any(
                    set(a.endpoints) == set(subgraph_eps) for a in subgraph.angles
                )
                if not exists:
                    attrs = deepcopy(getattr(angle, "data", {}))
                    subgraph.def_angle(*subgraph_eps, **attrs)

        # Copy dihedrals from original to subgraph
        for dihedral in self.dihedrals:
            endpoints = dihedral.endpoints
            if all(ep in original_to_subgraph for ep in endpoints):
                subgraph_eps = [original_to_subgraph[ep] for ep in endpoints]
                # Check if dihedral already exists
                exists = any(
                    set(d.endpoints) == set(subgraph_eps) for d in subgraph.dihedrals
                )
                if not exists:
                    attrs = deepcopy(getattr(dihedral, "data", {}))
                    subgraph.def_dihedral(*subgraph_eps, **attrs)

        # Convert edge_entities to list of Atoms
        edge_atoms = [e for e in edge_entities if isinstance(e, Atom)]

        return subgraph, edge_atoms
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

    def get_topo(
        self,
        entity_type: type[Entity] = Atom,
        link_type: type[Link] = Bond,
        gen_angle: bool = False,
        gen_dihe: bool = False,
        clear_existing: bool = False,
    ):
        """Generate topology (angles and dihedrals) from bonds.

        Args:
            entity_type: Entity type to include in topology (default: Atom)
            link_type: Link type to use for connections (default: Bond)
            gen_angle: Whether to generate angles
            gen_dihe: Whether to generate dihedrals
            clear_existing: If True, clear existing angles/dihedrals before generating new ones.
                          If False, only add angles/dihedrals that don't already exist.

        Returns:
            Topology object
        """
        # Use the generic ConnectivityMixin.get_topo method
        topo = super().get_topo(entity_type=entity_type, link_type=link_type)

        # Get the entity mapping from Topology
        atoms = topo.idx_to_entity

        # gen_angle and gen_dihe only work with Atom entities
        if gen_angle and entity_type is not Atom:
            raise ValueError("gen_angle=True requires entity_type=Atom")
        if gen_dihe and entity_type is not Atom:
            raise ValueError("gen_dihe=True requires entity_type=Atom")

        if gen_angle:
            if clear_existing:
                # Remove all existing angles
                existing_angles = list(self.links.bucket(Angle))
                if existing_angles:
                    self.links.remove(*existing_angles)

            # Build set of existing angle endpoints for deduplication
            existing_angle_endpoints: set[tuple[Atom, Atom, Atom]] = set()
            if not clear_existing:
                for angle in self.links.bucket(Angle):
                    existing_angle_endpoints.add((angle.itom, angle.jtom, angle.ktom))

            # Add new angles, avoiding duplicates
            for angle in topo.angles:
                angle_indices = angle.tolist()
                atom_i = atoms[angle_indices[0]]
                atom_j = atoms[angle_indices[1]]
                atom_k = atoms[angle_indices[2]]

                # Check if this angle already exists
                if (atom_i, atom_j, atom_k) not in existing_angle_endpoints:
                    new_angle = Angle(atom_i, atom_j, atom_k)
                    self.links.add(new_angle)
                    existing_angle_endpoints.add((atom_i, atom_j, atom_k))

        if gen_dihe:
            if clear_existing:
                # Remove all existing dihedrals
                existing_dihedrals = list(self.links.bucket(Dihedral))
                if existing_dihedrals:
                    self.links.remove(*existing_dihedrals)

            # Build set of existing dihedral endpoints for deduplication
            existing_dihedral_endpoints: set[tuple[Atom, Atom, Atom, Atom]] = set()
            if not clear_existing:
                for dihedral in self.links.bucket(Dihedral):
                    existing_dihedral_endpoints.add(
                        (dihedral.itom, dihedral.jtom, dihedral.ktom, dihedral.ltom)
                    )

            # Add new dihedrals, avoiding duplicates
            for dihe in topo.dihedrals:
                dihe_indices = dihe.tolist()
                atom_i = atoms[dihe_indices[0]]
                atom_j = atoms[dihe_indices[1]]
                atom_k = atoms[dihe_indices[2]]
                atom_l = atoms[dihe_indices[3]]

                # Check if this dihedral already exists
                if (atom_i, atom_j, atom_k, atom_l) not in existing_dihedral_endpoints:
                    new_dihedral = Dihedral(atom_i, atom_j, atom_k, atom_l)
                    self.links.add(new_dihedral)
                    existing_dihedral_endpoints.add((atom_i, atom_j, atom_k, atom_l))

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

            # Collect all keys from all bonds first to ensure consistent fields
            all_bond_keys = set()
            for bond in bonds_data:
                all_bond_keys.update(bond.keys())

            for bond_idx, bond in enumerate(bonds_data):
                # Atom references from properties
                # Validate that atoms exist in atoms_data
                if id(bond.itom) not in atom_id_to_index:
                    raise ValueError(
                        f"Bond {bond_idx + 1}: atom_i (id={id(bond.itom)}) is not in atoms list. "
                        f"This bond references an atom that was removed or is invalid."
                    )
                if id(bond.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Bond {bond_idx + 1}: atom_j (id={id(bond.jtom)}) is not in atoms list. "
                        f"This bond references an atom that was removed or is invalid."
                    )
                bond_dict["atom_i"].append(atom_id_to_index[id(bond.itom)])
                bond_dict["atom_j"].append(atom_id_to_index[id(bond.jtom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_bond_keys:
                    if key not in [
                        "atom_i",
                        "atom_j",
                    ]:  # Skip atom indices, already added
                        value = bond.get(key, None)
                        bond_dict[key].append(value)

            # Ensure a 'type' column exists for compatibility with writers
            # If missing, raise error instead of using default
            n_bonds = len(bonds_data)
            if "type" not in bond_dict:
                raise ValueError(
                    f"Bonds are missing 'type' field. All {n_bonds} bonds must have a 'type' attribute. "
                    f"This may indicate that ring closure bonds were created without proper typing."
                )
            elif len(bond_dict["type"]) != n_bonds:
                # Some bonds are missing 'type' field
                missing_count = n_bonds - len(bond_dict["type"])
                raise ValueError(
                    f"Bonds 'type' field has {len(bond_dict['type'])} values, but expected {n_bonds} "
                    f"(based on atom index fields). {missing_count} bond(s) are missing 'type' field. "
                    f"This may indicate that ring closure bonds were created without proper typing."
                )

            bond_dict_np = {k: np.array(v) for k, v in bond_dict.items()}
            frame["bonds"] = Block.from_dict(bond_dict_np)

        # Build angles Block - convert array of struct to struct of array
        if angles_data:
            angle_dict = defaultdict(list)

            # Collect all keys from all angles first to ensure consistent fields
            all_angle_keys = set()
            for angle in angles_data:
                all_angle_keys.update(angle.keys())

            for angle_idx, angle in enumerate(angles_data):
                # Atom references from properties
                # Validate that atoms exist in atoms_data
                if id(angle.itom) not in atom_id_to_index:
                    raise ValueError(
                        f"Angle {angle_idx + 1}: atom_i (id={id(angle.itom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                if id(angle.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Angle {angle_idx + 1}: atom_j (id={id(angle.jtom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                if id(angle.ktom) not in atom_id_to_index:
                    raise ValueError(
                        f"Angle {angle_idx + 1}: atom_k (id={id(angle.ktom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                angle_dict["atom_i"].append(atom_id_to_index[id(angle.itom)])
                angle_dict["atom_j"].append(atom_id_to_index[id(angle.jtom)])
                angle_dict["atom_k"].append(atom_id_to_index[id(angle.ktom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_angle_keys:
                    if key not in [
                        "atom_i",
                        "atom_j",
                        "atom_k",
                    ]:  # Skip atom indices, already added
                        value = angle.get(key, None)
                        angle_dict[key].append(value)

            # Ensure a 'type' column exists for compatibility with writers
            # If missing, raise error instead of using default
            n_angles = len(angles_data)
            if "type" not in angle_dict:
                raise ValueError(
                    f"Angles are missing 'type' field. All {n_angles} angles must have a 'type' attribute."
                )
            elif len(angle_dict["type"]) != n_angles:
                missing_count = n_angles - len(angle_dict["type"])
                raise ValueError(
                    f"Angles 'type' field has {len(angle_dict['type'])} values, but expected {n_angles} "
                    f"(based on atom index fields). {missing_count} angle(s) are missing 'type' field."
                )

            angle_dict_np = {k: np.array(v) for k, v in angle_dict.items()}
            frame["angles"] = Block.from_dict(angle_dict_np)
        # Build dihedrals Block - convert array of struct to struct of array
        if dihedrals_data:
            dihedral_dict = defaultdict(list)

            # Collect all keys from all dihedrals first to ensure consistent fields
            all_dihedral_keys = set()
            for dihedral in dihedrals_data:
                all_dihedral_keys.update(dihedral.keys())

            for dihedral_idx, dihedral in enumerate(dihedrals_data):
                # Atom references from properties
                # Validate that atoms exist in atoms_data
                if id(dihedral.itom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atom_i (id={id(dihedral.itom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atom_j (id={id(dihedral.jtom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.ktom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atom_k (id={id(dihedral.ktom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.ltom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atom_l (id={id(dihedral.ltom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                dihedral_dict["atom_i"].append(atom_id_to_index[id(dihedral.itom)])
                dihedral_dict["atom_j"].append(atom_id_to_index[id(dihedral.jtom)])
                dihedral_dict["atom_k"].append(atom_id_to_index[id(dihedral.ktom)])
                dihedral_dict["atom_l"].append(atom_id_to_index[id(dihedral.ltom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_dihedral_keys:
                    if key not in [
                        "atom_i",
                        "atom_j",
                        "atom_k",
                        "atom_l",
                    ]:  # Skip atom indices, already added
                        value = dihedral.get(key, None)
                        dihedral_dict[key].append(value)

            # Ensure a 'type' column exists for compatibility with writers
            # If missing, raise error instead of using default
            n_dihedrals = len(dihedrals_data)
            if "type" not in dihedral_dict:
                raise ValueError(
                    f"Dihedrals are missing 'type' field. All {n_dihedrals} dihedrals must have a 'type' attribute."
                )
            elif len(dihedral_dict["type"]) != n_dihedrals:
                missing_count = n_dihedrals - len(dihedral_dict["type"])
                raise ValueError(
                    f"Dihedrals 'type' field has {len(dihedral_dict['type'])} values, but expected {n_dihedrals} "
                    f"(based on atom index fields). {missing_count} dihedral(s) are missing 'type' field."
                )

            dihedral_dict_np = {k: np.array(v) for k, v in dihedral_dict.items()}
            frame["dihedrals"] = Block.from_dict(dihedral_dict_np)

        return frame

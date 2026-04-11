from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, TYPE_CHECKING

from .entity import (
    ConnectivityMixin,
    Entities,
    Entity,
    Link,
    MembershipMixin,
    SpatialMixin,
    Struct,
)

if TYPE_CHECKING:
    from .frame import Frame


class Atom(Entity):
    """Atom entity (common keys include {"element": "C", "xyz": [...]})"""

    def __repr__(self) -> str:
        identifier: str
        if "element" in self.data:
            identifier = str(self.data["element"])
        elif "type" in self.data:
            identifier = str(self.data["type"])
        else:
            identifier = str(id(self))
        return f"<Atom: {identifier}>"


class Bond(Link):
    """Covalent bond connecting two atoms.

    A Bond holds ordered references to exactly two Atom endpoints and optional
    key-value attributes (e.g., bond order, force-field type).

    Related symbols:
        Atom, Angle, Dihedral, Atomistic.def_bond
    """

    def __init__(self, a: Atom, b: Atom, /, **attrs: Any):
        """Create a bond between two atoms.

        Args:
            a: First atom endpoint.
            b: Second atom endpoint.
            **attrs: Arbitrary bond attributes (e.g., ``type="C-C"``,
                ``order=2``).

        Raises:
            AssertionError: If either argument is not an Atom instance.
        """
        assert isinstance(a, Atom), f"atom a must be an Atom instance, got {type(a)}"
        assert isinstance(b, Atom), f"atom b must be an Atom instance, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<Bond: {self.itom} - {self.jtom}>"

    @property
    def itom(self) -> Atom:
        """First atom endpoint of the bond."""
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        """Second atom endpoint of the bond."""
        return self.endpoints[1]


class Angle(Link):
    """Valence angle formed by three atoms (i--j--k).

    The central atom ``jtom`` is the vertex of the angle. Angle values are
    measured in radians by convention throughout MolPy.

    Related symbols:
        Atom, Bond, Dihedral, Atomistic.def_angle
    """

    def __init__(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any):
        """Create an angle between three atoms.

        Args:
            a: First atom (one arm of the angle).
            b: Central/vertex atom.
            c: Third atom (other arm of the angle).
            **attrs: Arbitrary angle attributes (e.g., ``type="C-C-C"``).

        Raises:
            AssertionError: If any argument is not an Atom instance.
        """
        assert isinstance(a, Atom), f"atom a must be an Atom instance, got {type(a)}"
        assert isinstance(b, Atom), f"atom b must be an Atom instance, got {type(b)}"
        assert isinstance(c, Atom), f"atom c must be an Atom instance, got {type(c)}"
        super().__init__([a, b, c], **attrs)

    def __repr__(self) -> str:
        return f"<Angle: {self.itom} - {self.jtom} - {self.ktom}>"

    @property
    def itom(self) -> Atom:
        """First atom endpoint of the angle."""
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        """Central (vertex) atom of the angle."""
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        """Third atom endpoint of the angle."""
        return self.endpoints[2]


class Dihedral(Link):
    """Dihedral (torsion) angle between four atoms"""

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any):
        """Create a dihedral angle between four atoms.

        Args:
            a: First atom.
            b: Second atom (part of the central bond).
            c: Third atom (part of the central bond).
            d: Fourth atom.
            **attrs: Arbitrary dihedral attributes (e.g., ``type="C-C-C-C"``).

        Raises:
            AssertionError: If any argument is not an Atom instance.
        """
        assert isinstance(a, Atom), f"atom a must be an Atom instance, got {type(a)}"
        assert isinstance(b, Atom), f"atom b must be an Atom instance, got {type(b)}"
        assert isinstance(c, Atom), f"atom c must be an Atom instance, got {type(c)}"
        assert isinstance(d, Atom), f"atom d must be an Atom instance, got {type(d)}"
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Dihedral: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        """First atom endpoint of the dihedral."""
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        """Second atom (part of the central bond)."""
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        """Third atom (part of the central bond)."""
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        """Fourth atom endpoint of the dihedral."""
        return self.endpoints[3]


class Atomistic(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin):
    """All-atom molecular structure with full topological information.

    Atomistic is the primary container for molecular systems in MolPy. It
    manages collections of atoms, bonds, angles, and dihedrals through
    typed buckets, and provides factory methods for creating and adding
    topology elements.

    Supports spatial operations (move, rotate, scale, align), system
    composition via ``+`` / ``+=``, and conversion to tabular Frame format
    for I/O.

    Related symbols:
        Atom, Bond, Angle, Dihedral, Struct, Frame
    """

    def __init__(self, **props) -> None:
        """Initialize an empty atomistic structure.

        Registers buckets for Atom, Bond, Angle, and Dihedral types.
        If the concrete subclass defines a ``__post_init__`` method, it is
        called automatically with the same keyword arguments.

        Args:
            **props (Any): Arbitrary properties stored on the structure (e.g.,
                ``name="water"``, ``charge=0.0``).
        """
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
        """All atoms in this structure.

        Returns:
            Entities[Atom]: Column-accessible list of Atom objects.
        """
        return self.entities[Atom]

    @property
    def bonds(self) -> Entities[Bond]:  # type: ignore[type-var]
        """All bonds in this structure.

        Returns:
            Entities[Bond]: Column-accessible list of Bond objects.
        """
        return self.links[Bond]  # type: ignore[return-value]

    @property
    def angles(self) -> Entities[Angle]:  # type: ignore[type-var]
        """All angles in this structure.

        Returns:
            Entities[Angle]: Column-accessible list of Angle objects.
        """
        return self.links[Angle]  # type: ignore[return-value]

    @property
    def dihedrals(self) -> Entities[Dihedral]:  # type: ignore[type-var]
        """All dihedrals in this structure.

        Returns:
            Entities[Dihedral]: Column-accessible list of Dihedral objects.
        """
        return self.links[Dihedral]  # type: ignore[return-value]

    @property
    def symbols(self) -> list[str]:
        """Element symbols for every atom in insertion order.

        Returns:
            list[str]: List of element strings (e.g., ``["C", "H", "H"]``).
                Atoms without an ``"element"`` key produce an empty string.
        """
        atoms = list(self.atoms)
        return [str(a.get("element") or a.get("symbol") or "") for a in atoms]

    @property
    def xyz(self) -> np.ndarray:
        """
        Get atomic positions as numpy array.

        Returns:
            np.ndarray: Nx3 array of atomic coordinates, or list of lists if numpy not available.
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

        # Count atoms by element
        from collections import Counter

        symbols = [a.get("element", "?") for a in atoms]
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

        If an ``xyz`` key is provided, it is expanded into separate ``x``,
        ``y``, ``z`` float fields on the created atom.

        Args:
            **attrs: Atom attributes. Common keys include ``element`` (str),
                ``type`` (str), ``charge`` (float, elementary charge units),
                ``mass`` (float, g/mol), and ``xyz`` (sequence of 3 floats
                in angstroms).

        Returns:
            Atom: The newly created and registered atom.

        Preferred for:
            Building structures atom-by-atom. Use ``add_atom`` instead when
            the Atom object already exists.

        Related symbols:
            Atom, def_atoms, add_atom
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
        """Create a new Bond between two atoms and add it to the structure.

        Args:
            a: First atom endpoint.
            b: Second atom endpoint.
            **attrs: Bond attributes (e.g., ``type="C-C"``, ``order=1``).

        Returns:
            Bond: The newly created and registered bond.

        Related symbols:
            Bond, def_bonds, add_bond
        """
        bond = Bond(a, b, **attrs)
        self.links.add(bond)
        return bond

    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle:
        """Create a new Angle between three atoms and add it to the structure.

        Args:
            a: First atom (one arm of the angle).
            b: Central/vertex atom.
            c: Third atom (other arm of the angle).
            **attrs: Angle attributes (e.g., ``type="C-C-C"``).

        Returns:
            Angle: The newly created and registered angle.

        Related symbols:
            Angle, def_angles, add_angle
        """
        angle = Angle(a, b, c, **attrs)
        self.links.add(angle)
        return angle

    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral:
        """Create a new Dihedral between four atoms and add it to the structure.

        Args:
            a: First atom.
            b: Second atom (part of the central bond).
            c: Third atom (part of the central bond).
            d: Fourth atom.
            **attrs: Dihedral attributes (e.g., ``type="C-C-C-C"``).

        Returns:
            Dihedral: The newly created and registered dihedral.

        Related symbols:
            Dihedral, def_dihedrals, add_dihedral
        """
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
        """Add an existing Atom object to the structure.

        Args:
            atom: Atom instance to register.

        Returns:
            Atom: The same atom passed in (for chaining convenience).

        Preferred for:
            Re-using an Atom created elsewhere. Use ``def_atom`` to create
            and add in one step.

        Related symbols:
            Atom, def_atom, add_atoms
        """
        self.entities.add(atom)
        return atom

    def add_bond(self, bond: Bond, /) -> Bond:
        """Add an existing Bond object to the structure.

        Args:
            bond: Bond instance to register.

        Returns:
            Bond: The same bond passed in.

        Related symbols:
            Bond, def_bond, add_bonds
        """
        self.links.add(bond)
        return bond

    def add_angle(self, angle: Angle, /) -> Angle:
        """Add an existing Angle object to the structure.

        Args:
            angle: Angle instance to register.

        Returns:
            Angle: The same angle passed in.

        Related symbols:
            Angle, def_angle, add_angles
        """
        self.links.add(angle)
        return angle

    def add_dihedral(self, dihedral: Dihedral, /) -> Dihedral:
        """Add an existing Dihedral object to the structure.

        Args:
            dihedral: Dihedral instance to register.

        Returns:
            Dihedral: The same dihedral passed in.

        Related symbols:
            Dihedral, def_dihedral, add_dihedrals
        """
        self.links.add(dihedral)
        return dihedral

    # ========== Delete Methods (del_*: remove atoms / bonds) ==========

    def del_atom(self, *atoms: Atom) -> None:
        """Remove atoms and all their incident bonds, angles, and dihedrals.

        Args:
            *atoms: Atom instances to remove.

        Related symbols:
            def_atom, add_atom, remove_entity
        """
        self.remove_entity(*atoms)

    def del_bond(self, *bonds: Bond) -> None:
        """Remove bonds (and any dependent angles / dihedrals that reference them).

        Args:
            *bonds: Bond instances to remove.

        Related symbols:
            def_bond, add_bond, remove_link
        """
        self.remove_link(*bonds)

    # ========== Batch Factory Methods (def_*s: create and add multiple) ==========

    def def_atoms(self, atoms_data: list[dict[str, Any]], /) -> list[Atom]:
        """Create multiple Atoms from a list of attribute dictionaries.

        Args:
            atoms_data: Each dict is passed as ``**attrs`` to ``def_atom``.
                See ``def_atom`` for supported keys.

        Returns:
            list[Atom]: Newly created atoms in the same order as input.

        Related symbols:
            def_atom, add_atoms
        """
        atoms = []
        for attrs in atoms_data:
            atom = self.def_atom(**attrs)
            atoms.append(atom)
        return atoms

    def def_bonds(
        self, bonds_data: list[tuple[Atom, Atom] | tuple[Atom, Atom, dict[str, Any]]], /
    ) -> list[Bond]:
        """Create multiple Bonds from a list of atom-pair tuples.

        Args:
            bonds_data: Each element is ``(itom, jtom)`` or
                ``(itom, jtom, attrs_dict)``.

        Returns:
            list[Bond]: Newly created bonds in the same order as input.

        Related symbols:
            def_bond, add_bonds
        """
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
        """Create multiple Angles from a list of atom-triple tuples.

        Args:
            angles_data: Each element is ``(itom, jtom, ktom)`` or
                ``(itom, jtom, ktom, attrs_dict)``.

        Returns:
            list[Angle]: Newly created angles in the same order as input.

        Related symbols:
            def_angle, add_angles
        """
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
        """Create multiple Dihedrals from a list of atom-quadruple tuples.

        Args:
            dihedrals_data: Each element is ``(itom, jtom, ktom, ltom)`` or
                ``(itom, jtom, ktom, ltom, attrs_dict)``.

        Returns:
            list[Dihedral]: Newly created dihedrals in the same order as input.

        Related symbols:
            def_dihedral, add_dihedrals
        """
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
        """Add multiple existing Atom objects to the structure.

        Args:
            atoms: Atom instances to register.

        Returns:
            list[Atom]: The same list passed in.

        Related symbols:
            add_atom, def_atoms
        """
        for atom in atoms:
            self.entities.add(atom)
        return atoms

    def add_bonds(self, bonds: list[Bond], /) -> list[Bond]:
        """Add multiple existing Bond objects to the structure.

        Args:
            bonds: Bond instances to register.

        Returns:
            list[Bond]: The same list passed in.

        Related symbols:
            add_bond, def_bonds
        """
        for bond in bonds:
            self.links.add(bond)
        return bonds

    def add_angles(self, angles: list[Angle], /) -> list[Angle]:
        """Add multiple existing Angle objects to the structure.

        Args:
            angles: Angle instances to register.

        Returns:
            list[Angle]: The same list passed in.

        Related symbols:
            add_angle, def_angles
        """
        for angle in angles:
            self.links.add(angle)
        return angles

    def add_dihedrals(self, dihedrals: list[Dihedral], /) -> list[Dihedral]:
        """Add multiple existing Dihedral objects to the structure.

        Args:
            dihedrals: Dihedral instances to register.

        Returns:
            list[Dihedral]: The same list passed in.

        Related symbols:
            add_dihedral, def_dihedrals
        """
        for dihedral in dihedrals:
            self.links.add(dihedral)
        return dihedrals

    # ========== Spatial Operations (return self for chaining) ==========

    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Atom
    ) -> "Atomistic":
        """Translate all atoms by a displacement vector.

        This is an in-place operation that returns ``self`` for method
        chaining.

        Args:
            delta: Translation vector ``[dx, dy, dz]`` in angstroms.
            entity_type: Entity type to translate (default: Atom).

        Returns:
            Atomistic: ``self``, for chaining (e.g.,
                ``mol.move([1, 0, 0]).rotate(...)``).
        """
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
        """Rotate all atoms around an axis using the Rodrigues formula.

        This is an in-place operation that returns ``self`` for method
        chaining.

        Args:
            axis: Rotation axis ``[ax, ay, az]`` (will be normalised
                internally).
            angle: Rotation angle in radians.
            about: Point ``[x, y, z]`` in angstroms to rotate around.
                Defaults to the origin ``[0, 0, 0]``.
            entity_type: Entity type to rotate (default: Atom).

        Returns:
            Atomistic: ``self``, for chaining.
        """
        super().rotate(axis, angle, about=about, entity_type=entity_type)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        """Scale all atom positions by a uniform factor.

        This is an in-place operation that returns ``self`` for method
        chaining.

        Args:
            factor: Multiplicative scale factor (dimensionless).
            about: Center of scaling ``[x, y, z]`` in angstroms.
                Defaults to the origin ``[0, 0, 0]``.
            entity_type: Entity type to scale (default: Atom).

        Returns:
            Atomistic: ``self``, for chaining.
        """
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
        """Align this structure so that atom ``a`` coincides with atom ``b``.

        When direction vectors ``a_dir`` and ``b_dir`` are given, the
        structure is first rotated to align the two directions, then
        translated so that ``a`` lands on ``b``.

        This is an in-place operation that returns ``self`` for method
        chaining.

        Args:
            a: Source reference atom (in this structure).
            b: Target reference atom (position to align to) with
                coordinates in angstroms.
            a_dir: Direction vector at ``a`` (will be normalised).
            b_dir: Direction vector at ``b`` (will be normalised).
            flip: If True, reverse ``b_dir`` before alignment.
            entity_type: Entity type to transform (default: Atom).

        Returns:
            Atomistic: ``self``, for chaining.
        """
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
        """Create n copies and merge them into a new system.

        Each copy is independently deep-copied from ``self``, optionally
        transformed, then merged into a single new Atomistic structure.

        Args:
            n: Number of copies to create.
            transform (Callable | None): Optional callable ``(copy: Atomistic, index: int) -> None``
                applied to each copy before merging. The callable receives
                the deep-copied replica and its zero-based index.

        Returns:
            Atomistic: A new structure containing all replicas merged together.

        Example:
            >>> waters = Water().replicate(10, lambda mol, i: mol.move([i*5, 0, 0]))
            >>> grid = Methane().replicate(9, lambda mol, i: mol.move([i%3*5, i//3*5, 0]))
        """
        result = type(self)()  # Empty system of same type

        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)

        return result

    def __len__(self) -> int:
        """Return the number of atoms in the structure."""
        return len(self.atoms)

    def get_topo(
        self,
        entity_type: type[Entity] = Atom,
        link_type: type[Link] = Bond,
        gen_angle: bool = False,
        gen_dihe: bool = False,
        clear_existing: bool = False,
    ) -> "Atomistic | Topology":
        """Generate topology (angles and dihedrals) from bonds.

        When ``gen_angle`` or ``gen_dihe`` is True, returns a **new** Atomistic
        with the generated interactions added — the original is not mutated.
        When both are False, falls through to the base-class method and returns
        a :class:`Topology` graph (used internally for traversal).

        Args:
            entity_type: Entity type to include in topology (default: Atom)
            link_type: Link type to use for connections (default: Bond)
            gen_angle: Whether to generate angles
            gen_dihe: Whether to generate dihedrals
            clear_existing: If True, clear existing angles/dihedrals before
                generating new ones.

        Returns:
            New Atomistic with angles/dihedrals added when gen_angle or
            gen_dihe is True; Topology graph otherwise.
        """
        if not gen_angle and not gen_dihe:
            # Pure topology query used internally (get_topo_neighbors, etc.)
            return super().get_topo(entity_type=entity_type, link_type=link_type)

        # gen_angle and gen_dihe only work with Atom entities
        if gen_angle and entity_type is not Atom:
            raise ValueError("gen_angle=True requires entity_type=Atom")
        if gen_dihe and entity_type is not Atom:
            raise ValueError("gen_dihe=True requires entity_type=Atom")

        # Work on a copy so the original is not mutated
        new_struct = self.copy()

        # Build the topology graph from the copy's bonds
        from molpy.core.entity import ConnectivityMixin

        topo = ConnectivityMixin.get_topo(
            new_struct, entity_type=entity_type, link_type=link_type
        )
        atoms = topo.idx_to_entity

        if gen_angle:
            if clear_existing:
                existing_angles = list(new_struct.links.bucket(Angle))
                if existing_angles:
                    new_struct.links.remove(*existing_angles)

            existing_angle_endpoints: set[tuple[Atom, Atom, Atom]] = set()
            if not clear_existing:
                for angle in new_struct.links.bucket(Angle):
                    existing_angle_endpoints.add((angle.itom, angle.jtom, angle.ktom))

            for angle in topo.angles:
                angle_indices = angle.tolist()
                itom = atoms[angle_indices[0]]
                jtom = atoms[angle_indices[1]]
                ktom = atoms[angle_indices[2]]
                if (itom, jtom, ktom) not in existing_angle_endpoints:
                    new_struct.links.add(Angle(itom, jtom, ktom))
                    existing_angle_endpoints.add((itom, jtom, ktom))

        if gen_dihe:
            if clear_existing:
                existing_dihedrals = list(new_struct.links.bucket(Dihedral))
                if existing_dihedrals:
                    new_struct.links.remove(*existing_dihedrals)

            existing_dihedral_endpoints: set[tuple[Atom, Atom, Atom, Atom]] = set()
            if not clear_existing:
                for dihedral in new_struct.links.bucket(Dihedral):
                    existing_dihedral_endpoints.add(
                        (dihedral.itom, dihedral.jtom, dihedral.ktom, dihedral.ltom)
                    )

            for dihe in topo.dihedrals:
                dihe_indices = dihe.tolist()
                itom = atoms[dihe_indices[0]]
                jtom = atoms[dihe_indices[1]]
                ktom = atoms[dihe_indices[2]]
                ltom = atoms[dihe_indices[3]]
                if (itom, jtom, ktom, ltom) not in existing_dihedral_endpoints:
                    new_struct.links.add(Dihedral(itom, jtom, ktom, ltom))
                    existing_dihedral_endpoints.add((itom, jtom, ktom, ltom))

        return new_struct

    # ========== Conversion Methods ==========

    def to_frame(self, atom_fields: list[str] | None = None) -> "Frame":
        """Convert to LAMMPS data Frame format.

        Converts this Atomistic structure into a Frame suitable for writing
        as a LAMMPS data file.

        Args:
            atom_fields (list[str] | None): List of atom fields to extract. If None, extracts all fields.

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
                        f"Bond {bond_idx + 1}: atomi (id={id(bond.itom)}) is not in atoms list. "
                        f"This bond references an atom that was removed or is invalid."
                    )
                if id(bond.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Bond {bond_idx + 1}: atomj (id={id(bond.jtom)}) is not in atoms list. "
                        f"This bond references an atom that was removed or is invalid."
                    )
                bond_dict["atomi"].append(atom_id_to_index[id(bond.itom)])
                bond_dict["atomj"].append(atom_id_to_index[id(bond.jtom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_bond_keys:
                    if key not in [
                        "atomi",
                        "atomj",
                    ]:  # Skip atom indices, already added
                        value = bond.get(key, None)
                        bond_dict[key].append(value)

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
                        f"Angle {angle_idx + 1}: atomi (id={id(angle.itom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                if id(angle.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Angle {angle_idx + 1}: atomj (id={id(angle.jtom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                if id(angle.ktom) not in atom_id_to_index:
                    raise ValueError(
                        f"Angle {angle_idx + 1}: atomk (id={id(angle.ktom)}) is not in atoms list. "
                        f"This angle references an atom that was removed or is invalid."
                    )
                angle_dict["atomi"].append(atom_id_to_index[id(angle.itom)])
                angle_dict["atomj"].append(atom_id_to_index[id(angle.jtom)])
                angle_dict["atomk"].append(atom_id_to_index[id(angle.ktom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_angle_keys:
                    if key not in [
                        "atomi",
                        "atomj",
                        "atomk",
                    ]:  # Skip atom indices, already added
                        value = angle.get(key, None)
                        angle_dict[key].append(value)

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
                        f"Dihedral {dihedral_idx + 1}: atomi (id={id(dihedral.itom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.jtom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atomj (id={id(dihedral.jtom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.ktom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atomk (id={id(dihedral.ktom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                if id(dihedral.ltom) not in atom_id_to_index:
                    raise ValueError(
                        f"Dihedral {dihedral_idx + 1}: atoml (id={id(dihedral.ltom)}) is not in atoms list. "
                        f"This dihedral references an atom that was removed or is invalid."
                    )
                dihedral_dict["atomi"].append(atom_id_to_index[id(dihedral.itom)])
                dihedral_dict["atomj"].append(atom_id_to_index[id(dihedral.jtom)])
                dihedral_dict["atomk"].append(atom_id_to_index[id(dihedral.ktom)])
                dihedral_dict["atoml"].append(atom_id_to_index[id(dihedral.ltom)])
                # Data fields - iterate over all keys to ensure consistent length
                for key in all_dihedral_keys:
                    if key not in [
                        "atomi",
                        "atomj",
                        "atomk",
                        "atoml",
                    ]:  # Skip atom indices, already added
                        value = dihedral.get(key, None)
                        dihedral_dict[key].append(value)

            dihedral_dict_np = {k: np.array(v) for k, v in dihedral_dict.items()}
            frame["dihedrals"] = Block.from_dict(dihedral_dict_np)

        return frame

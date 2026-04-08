from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .atomistic import Atomistic

from .entity import (
    ConnectivityMixin,
    Entities,
    Entity,
    Link,
    MembershipMixin,
    SpatialMixin,
    Struct,
)


class Bead(Entity):
    """Coarse-grain bead entity.

    A bead can optionally map to an Atomistic structure via the atomistic member.
    Supports arbitrary attributes via dictionary interface.

    Attributes:
        atomistic: Optional reference to an Atomistic structure this bead represents
    """

    def __init__(
        self, data_dict=None, /, atomistic: "Atomistic | None" = None, **attrs: Any
    ):
        # Handle being called with just a dict (for copy compatibility)
        if data_dict is not None and isinstance(data_dict, dict):
            super().__init__(**data_dict)
            self.atomistic = None  # Will be set by __deepcopy__ if needed
        else:
            # Normal creation: data_dict is actually atomistic or None
            if data_dict is not None:
                atomistic = data_dict
            super().__init__(**attrs)
            self.atomistic = atomistic

    def __deepcopy__(self, memo):
        """Custom deep copy to handle atomistic member."""
        from copy import deepcopy

        # Create new bead with copied data and atomistic reference (not copy)
        new_bead = Bead(atomistic=self.atomistic, **deepcopy(self.data, memo))
        return new_bead

    def __repr__(self) -> str:
        identifier: str
        if "type" in self.data:
            identifier = str(self.data["type"])
        elif "name" in self.data:
            identifier = str(self.data["name"])
        else:
            identifier = str(id(self))
        return f"<Bead: {identifier}>"


class CGBond(Link):
    """Coarse-grained bond connecting two beads."""

    def __init__(self, a: Bead, b: Bead, /, **attrs: Any):
        assert isinstance(a, Bead), f"bead a must be a Bead instance, got {type(a)}"
        assert isinstance(b, Bead), f"bead b must be a Bead instance, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<CGBond: {self.ibead} - {self.jbead}>"

    @property
    def ibead(self) -> Bead:
        """First bead endpoint."""
        return self.endpoints[0]

    @property
    def jbead(self) -> Bead:
        """Second bead endpoint."""
        return self.endpoints[1]


class CoarseGrain(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin):
    """Coarse-grained molecular structure container.

    Similar to Atomistic but for coarse-grained representations using Beads and CGBonds.
    Supports bidirectional conversion with Atomistic structures.
    """

    def __init__(self, **props) -> None:
        super().__init__(**props)
        # Call __post_init__ if it exists (for template pattern)
        if hasattr(self, "__post_init__"):
            # Get the method from the actual class, not from parent
            for klass in type(self).__mro__:
                if klass is CoarseGrain:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break
        self.entities.register_type(Bead)
        self.links.register_type(CGBond)

    @property
    def beads(self) -> Entities[Bead]:
        """Collection of all beads in the structure."""
        return self.entities[Bead]

    @property
    def cgbonds(self) -> Entities[CGBond]:  # type: ignore[type-var]
        """Collection of all CG bonds in the structure."""
        return self.links[CGBond]  # type: ignore[return-value]

    def __repr__(self) -> str:
        """Return a concise representation of the coarse-grained structure."""
        beads = self.beads
        cgbonds = self.cgbonds

        # Count beads by type
        from collections import Counter

        types = [b.get("type", "?") for b in beads]
        type_counts = Counter(types)

        # Format composition
        if len(type_counts) <= 5:
            composition = " ".join(
                f"{typ}:{cnt}" for typ, cnt in sorted(type_counts.items())
            )
        else:
            composition = f"{len(type_counts)} types"

        parts = ["CoarseGrain"]
        parts.append(f"{len(beads)} beads ({composition})")
        parts.append(f"{len(cgbonds)} bonds")

        return f"<{', '.join(parts)}>"

    # ========== Factory Methods (def_*: create and add) ==========

    def def_bead(self, /, atomistic: "Atomistic | None" = None, **attrs: Any) -> Bead:
        """Create a new Bead and add it to the structure.

        Args:
            atomistic: Optional Atomistic structure this bead represents
            **attrs: Bead attributes (x, y, z, type, etc.)
        """
        bead = Bead(atomistic=atomistic, **attrs)
        self.entities.add(bead)
        return bead

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        """Create a new CGBond between two beads and add it to the structure."""
        bond = CGBond(a, b, **attrs)
        self.links.add(bond)
        return bond

    # ========== Add Methods (add_*: add existing entities) ==========

    def add_bead(self, bead: Bead, /) -> Bead:
        """Add an existing Bead object to the structure."""
        self.entities.add(bead)
        return bead

    def add_cgbond(self, bond: CGBond, /) -> CGBond:
        """Add an existing CGBond object to the structure."""
        self.links.add(bond)
        return bond

    # ========== Batch Factory Methods (def_*s: create and add multiple) ==========

    def def_beads(self, beads_data: list[dict[str, Any]], /) -> list[Bead]:
        """Create multiple Beads from a list of attribute dictionaries."""
        beads = []
        for attrs in beads_data:
            bead = self.def_bead(**attrs)
            beads.append(bead)
        return beads

    def def_cgbonds(
        self, bonds_data: list[tuple[Bead, Bead] | tuple[Bead, Bead, dict[str, Any]]], /
    ) -> list[CGBond]:
        """Create multiple CGBonds from a list of (ibead, jbead) or (ibead, jbead, attrs) tuples."""
        bonds = []
        for bond_spec in bonds_data:
            if len(bond_spec) == 2:
                a, b = bond_spec
                attrs = {}
            else:
                a, b, attrs = bond_spec
            bond = self.def_cgbond(a, b, **attrs)
            bonds.append(bond)
        return bonds

    # ========== Batch Add Methods (add_*s: add multiple existing entities) ==========

    def add_beads(self, beads: list[Bead], /) -> list[Bead]:
        """Add multiple existing Bead objects to the structure."""
        for bead in beads:
            self.entities.add(bead)
        return beads

    def add_cgbonds(self, bonds: list[CGBond], /) -> list[CGBond]:
        """Add multiple existing CGBond objects to the structure."""
        for bond in bonds:
            self.links.add(bond)
        return bonds

    # ========== Spatial Operations (return self for chaining) ==========

    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Bead
    ) -> "CoarseGrain":
        """Move all beads by delta. Returns self for chaining."""
        super().move(delta, entity_type=entity_type)
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Rotate beads around axis. Returns self for chaining."""
        super().rotate(axis, angle, about=about, entity_type=entity_type)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Scale beads by factor. Returns self for chaining."""
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
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Align beads. Returns self for chaining."""
        super().align(
            a, b, a_dir=a_dir, b_dir=b_dir, flip=flip, entity_type=entity_type
        )
        return self

    # ========== System Composition ==========

    def __iadd__(self, other: "CoarseGrain") -> "CoarseGrain":
        """Merge another CoarseGrain into this one (in-place).

        Example:
            cg1 += cg2  # Merges cg2 into cg1
        """
        self.merge(other)
        return self

    def __add__(self, other: "CoarseGrain") -> "CoarseGrain":
        """Create a new CoarseGrain by merging two structures.

        Example:
            cg3 = cg1 + cg2  # Creates new structure
        """
        result = self.copy()
        result.merge(other)
        return result

    def replicate(self, n: int, transform=None) -> "CoarseGrain":
        """Create n copies and merge them into a new system.

        Args:
            n: Number of copies to create.
            transform (Callable | None): Optional callable(copy, index) -> None to transform each copy.

        Example:
            # Create 10 copies in a line
            cg_line = cg.replicate(10, lambda mol, i: mol.move([i*5, 0, 0]))
        """
        result = type(self)()  # Empty system of same type

        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)

        return result

    def __len__(self) -> int:
        """Return number of beads in the structure."""
        return len(self.beads)

    # ========== Conversion Methods ==========

    def to_atomistic(self) -> "Atomistic":
        """Convert CoarseGrain structure to Atomistic representation.

        Beads with atomistic member are expanded to their full atomistic structure.
        Beads without atomistic mapping create a single atom at the bead position.
        CGBonds are converted to atomistic bonds between representative atoms.

        Returns:
            Atomistic structure
        """
        from .atomistic import Atom, Atomistic, Bond

        result = Atomistic()

        # Map beads to their representative atoms (for bond creation)
        bead_to_atom: dict[Bead, Atom] = {}

        # Process each bead
        for bead in self.beads:
            if bead.atomistic is not None:
                # Bead has atomistic mapping - use reference and merge
                # Get first atom as representative for bond creation
                atoms_list = list(bead.atomistic.atoms)
                if atoms_list:
                    bead_to_atom[bead] = atoms_list[0]

                # Merge the referenced atomistic structure
                result.merge(bead.atomistic)
            else:
                # No atomistic mapping - create single atom at bead position
                atom_attrs = {}

                # Copy position if available
                if "x" in bead.data:
                    atom_attrs["x"] = bead.data["x"]
                if "y" in bead.data:
                    atom_attrs["y"] = bead.data["y"]
                if "z" in bead.data:
                    atom_attrs["z"] = bead.data["z"]

                # Copy type as element placeholder if available
                if "type" in bead.data:
                    atom_attrs["element"] = bead.data["type"]

                # Create atom
                atom = result.def_atom(**atom_attrs)
                bead_to_atom[bead] = atom

        # Convert CGBonds to atomistic bonds
        for cgbond in self.cgbonds:
            ibead = cgbond.ibead
            jbead = cgbond.jbead

            # Get representative atoms
            if ibead in bead_to_atom and jbead in bead_to_atom:
                iatom = bead_to_atom[ibead]
                jatom = bead_to_atom[jbead]

                # Create bond with attributes from CGBond
                bond_attrs = dict(cgbond.data)
                result.def_bond(iatom, jatom, **bond_attrs)

        return result

    @classmethod
    def from_atomistic(
        cls, atomistic: "Atomistic", bead_mask: "np.ndarray"
    ) -> "CoarseGrain":
        """Create CoarseGrain structure from Atomistic using a bead mask.

        Args:
            atomistic: Atomistic structure to convert
            bead_mask: Numpy array where each element indicates which bead the atom belongs to.
                      Can be boolean (True = bead 0) or integer indices.

        Returns:
            CoarseGrain structure with beads mapped to atomistic substructures

        Example:
            >>> atomistic = Atomistic()
            >>> # ... create atoms and bonds ...
            >>> # Group atoms: [0,1,2] -> bead 0, [3,4] -> bead 1
            >>> mask = np.array([0, 0, 0, 1, 1])
            >>> cg = CoarseGrain.from_atomistic(atomistic, mask)
        """
        import numpy as np

        result = cls()

        # Get atoms list
        atoms_list = list(atomistic.atoms)

        if len(atoms_list) != len(bead_mask):
            raise ValueError(
                f"Bead mask length ({len(bead_mask)}) must match number of atoms ({len(atoms_list)})"
            )

        # Group atoms by bead index
        bead_indices = np.unique(bead_mask)
        bead_to_atoms: dict[int, list] = {int(idx): [] for idx in bead_indices}

        for atom, bead_idx in zip(atoms_list, bead_mask):
            bead_to_atoms[int(bead_idx)].append(atom)

        # Map bead index to Bead object
        idx_to_bead: dict[int, Bead] = {}

        # Create beads from atom groups
        for bead_idx, atom_group in bead_to_atoms.items():
            # Calculate center of mass for bead position
            positions = []
            for atom in atom_group:
                x = atom.get("x", 0.0)
                y = atom.get("y", 0.0)
                z = atom.get("z", 0.0)
                positions.append([x, y, z])

            positions_array = np.array(positions)
            center = positions_array.mean(axis=0)

            # Extract subgraph for this bead
            from .atomistic import Atomistic

            subgraph, _ = atomistic.extract_subgraph(
                center_entities=atom_group,
                radius=0,  # Only include atoms in the group
                entity_type=type(atom_group[0]),
                link_type=(
                    type(list(atomistic.bonds)[0]) if len(atomistic.bonds) > 0 else None
                ),
            )

            # Create bead with atomistic mapping (reference, not copy)
            bead = result.def_bead(
                atomistic=subgraph,
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2]),
            )

            idx_to_bead[bead_idx] = bead

        # Infer CGBonds from atomistic bonds crossing bead boundaries
        bead_pairs: set[tuple[int, int]] = set()

        for bond in atomistic.bonds:
            # Find which beads the bond endpoints belong to
            atom_i = bond.itom
            atom_j = bond.jtom

            # Find bead indices
            try:
                idx_i = atoms_list.index(atom_i)
                idx_j = atoms_list.index(atom_j)
            except ValueError:
                continue  # Skip if atoms not found

            bead_idx_i = int(bead_mask[idx_i])
            bead_idx_j = int(bead_mask[idx_j])

            # If atoms belong to different beads, create CGBond
            if bead_idx_i != bead_idx_j:
                # Ensure consistent ordering to avoid duplicates
                pair = tuple(sorted([bead_idx_i, bead_idx_j]))
                bead_pairs.add(pair)

        # Create CGBonds
        for bead_idx_i, bead_idx_j in bead_pairs:
            if bead_idx_i in idx_to_bead and bead_idx_j in idx_to_bead:
                bead_i = idx_to_bead[bead_idx_i]
                bead_j = idx_to_bead[bead_idx_j]
                result.def_cgbond(bead_i, bead_j)

        return result

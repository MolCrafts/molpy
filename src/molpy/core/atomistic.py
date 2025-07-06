"""
Core molecular structure classes for the molpy framework.

This module provides the fundamental building blocks for molecular modeling:
- Entity: Base class with dictionary-like behavior
- Spatial operations for atoms and structures
- Topological entities: bonds, angles, dihedrals
- Hierarchical structure management
- Collections and containers
"""

import copy
from typing import Iterable

import numpy as np

from .frame import Frame
from .protocol import Entities, Entity, Struct
from .topology import Topology
from .utils import to_dict_of_list, to_list_of_dict
from .wrapper import Wrapper


class Atom(Entity):
    """
    Class representing an atom with spatial coordinates.

    Combines Entity's dictionary behavior with spatial operations through wrappers.
    """

    def __init__(self, name: str = "", **kwargs):
        """
        Initialize an atom.

        Args:
            name: Atom name/symbol
            xyz: 3D coordinates
            **kwargs: Additional properties
        """
        super().__init__(name=name, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the atom."""
        return f"<Atom {self.name}>"

    @property
    def name(self) -> str:
        """Get the name of the atom."""
        return self.get("name", "")


class ManyBody(Entity):
    """
    Base class for entities involving multiple atoms.

    Handles bonds, angles, dihedrals, and other multi-atom entities.
    """

    def __init__(self, *atoms, **kwargs):
        """
        Initialize a ManyBody entity.

        Args:
            *atoms: Atoms involved in the entity
            **kwargs: Additional properties
        """
        super().__init__(**kwargs)
        if not all(isinstance(atom, Atom) for atom in atoms):
            raise TypeError("All arguments must be Atom instances")
        self._atoms = tuple(atoms)

    @property
    def atoms(self):
        """Get the atoms involved in the entity."""
        return self._atoms


class Bond(ManyBody):
    """Class representing a bond between two atoms."""

    def __init__(self, itom=None, jtom=None, **kwargs):
        """
        Initialize a bond between two atoms.

        Args:
            itom: First atom in the bond
            jtom: Second atom in the bond
            **kwargs: Additional properties (e.g., bond_type, length)
        """
        if itom is jtom:
            raise ValueError("Cannot create bond between same atom")
        sorted_atoms = sorted([itom, jtom], key=lambda a: id(a))
        super().__init__(*sorted_atoms, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the bond."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the bond."""
        return self._atoms[1]

    def __repr__(self) -> str:
        """Return a string representation of the bond."""
        return f"<Bond: {self.itom.name}-{self.jtom.name}>"

    def __eq__(self, other) -> bool:
        """Check equality based on the atoms in the bond."""
        if not isinstance(other, Bond):
            return False
        return (self.itom is other.itom and self.jtom is other.jtom) or (
            self.itom is other.jtom and self.jtom is other.itom
        )


class Angle(ManyBody):
    """Class representing an angle between three atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, **kwargs):
        """
        Initialize an angle.

        Args:
            itom: First atom in the angle
            jtom: jtom atom (center of angle)
            ktom: Third atom in the angle
            **kwargs: Additional properties
        """
        if len({id(itom), id(jtom), id(ktom)}) != 3:
            raise ValueError("All three atoms must be different")
        # Sort end atoms for consistent ordering
        end_atoms = sorted([itom, ktom], key=lambda a: id(a))
        super().__init__(end_atoms[0], jtom, end_atoms[1], **kwargs)

    @property
    def itom(self):
        """Get the first atom in the angle."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the jtom atom (center of angle)."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the angle."""
        return self._atoms[2]

    def __repr__(self) -> str:
        """Return a string representation of the angle."""
        return f"<Angle: {self.itom.name}-{self.jtom.name}-{self.ktom.name}>"

    @property
    def value(self) -> float:
        """Calculate the angle value in radians."""
        v1 = self.itom.xyz - self.jtom.xyz
        v2 = self.jtom.xyz - self.jtom.xyz
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))


class Dihedral(ManyBody):
    """Class representing a dihedral angle between four atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize a dihedral angle.

        Args:
            itom: First atom in the dihedral
            jtom: Second atom in the dihedral
            ktom: Third atom in the dihedral
            ltom: Fourth atom in the dihedral
            **kwargs: Additional properties
        """
        if len({id(itom), id(jtom), id(ktom), id(ltom)}) != 4:
            raise ValueError("All four atoms must be different")

        # Ensure consistent ordering based on central bond
        if id(jtom) > id(ktom):
            itom, jtom, ktom, ltom = ltom, ktom, jtom, itom

        super().__init__(itom, jtom, ktom, ltom, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the dihedral."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the dihedral."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the dihedral."""
        return self._atoms[2]

    @property
    def ltom(self):
        """Alias for atom4."""
        return self._atoms[3]

    def __repr__(self) -> str:
        """Return a string representation of the dihedral."""
        return f"<Dihedral: {self.itom.name}-{self.jtom.name}-{self.ktom.name}-{self.ltom.name}>"

    @property
    def value(self) -> float:
        """Calculate the dihedral angle value in radians."""
        # Vectors along the bonds
        b1 = self.jtom.xyz - self.itom.xyz
        b2 = self.ktom.xyz - self.jtom.xyz
        b3 = self.ltom.xyz - self.ktom.xyz

        # Normal vectors to the planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # Normalize normal vectors
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0  # Degenerate case

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Check sign using the middle bond vector
        if np.dot(np.cross(n1, n2), b2) < 0:
            angle = -angle

        return float(angle)


class Improper(Dihedral):
    """
    Class representing an improper dihedral angle.

    An improper dihedral is used to maintain planarity in molecular structures.
    """

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize an improper dihedral angle.

        Args:
            itom: Central atom
            itom: First bonded atom
            jtom: Second bonded atom
            ltom: Third bonded atom
            **kwargs: Additional properties
        """
        # Sort the three bonded atoms for consistent ordering
        bonded_atoms = sorted([jtom, ktom, ltom], key=lambda a: id(a))
        super().__init__(
            itom, bonded_atoms[0], bonded_atoms[1], bonded_atoms[2], **kwargs
        )

    def __repr__(self) -> str:
        """Return a string representation of the improper dihedral."""
        return f"<Improper: {self.itom.name}({self.jtom.name},{self.ktom.name},{self.ltom.name})>"


class Atomistic(Wrapper):
    """
    Structure containing atoms, bonds, angles, and dihedrals.

    Basic structure functionality that can be enhanced with wrappers
    for spatial operations and hierarchical management.
    """

    def __init__(self, **props):
        """
        Initialize an atomic structure.

        Args:
            name: Structure name
            **props: Additional properties
        """
        super().__init__(Struct(**props))
        self["atoms"] = Entities[Atom]()
        self["bonds"] = Entities[Bond]()
        self["angles"] = Entities[Angle]()
        self["dihedrals"] = Entities[Dihedral]()

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return f"<Atomistic: {len(self.atoms)} atoms>"

    @property
    def atoms(self) -> Entities[Atom]:
        """Get the atoms in the structure."""
        return self._wrapped["atoms"]

    @atoms.setter
    def atoms(self, value: Iterable[Atom]):
        self._wrapped["atoms"] = Entities[Atom](value)

    @property
    def bonds(self) -> Entities[Bond]:
        """Get the bonds in the structure."""
        return self._wrapped["bonds"]

    @bonds.setter
    def bonds(self, value: Iterable[Bond]):
        self._wrapped["bonds"] = Entities[Bond](value)

    @property
    def angles(self) -> Entities[Angle]:
        """Get the angles in the structure."""
        return self._wrapped["angles"]

    @angles.setter
    def angles(self, value: Iterable[Angle]):
        self._wrapped["angles"] = Entities[Angle](value)

    @property
    def dihedrals(self) -> Entities[Dihedral]:
        """Get the dihedrals in the structure."""
        return self._wrapped["dihedrals"]

    @dihedrals.setter
    def dihedrals(self, value: Iterable[Dihedral]):
        self._wrapped["dihedrals"] = Entities[Dihedral](value)

    @property
    def symbols(self) -> list[str]:
        """Get atomic symbols from all atoms."""
        symbols = []
        for atom in self.atoms:
            if "symbol" in atom:
                symbols.append(atom["symbol"])
            elif "name" in atom:
                symbols.append(atom["name"])
            else:
                symbols.append("X")  # default symbol
        return symbols

    @property
    def positions(self) -> np.ndarray:
        """Get atomic positions from all atoms."""
        positions = []
        for atom in self.atoms:
            if "xyz" in atom:
                positions.append(atom["xyz"])
            else:
                positions.append([0.0, 0.0, 0.0])  # default position
        return np.array(positions)

    def add_atom(self, atom):
        """
        Add an atom to the structure.

        Args:
            atom: Atom to add

        Returns:
            The added atom
        """
        return self.atoms.add(atom)

    def def_atom(self, **props):
        """
        Create and add an atom with given properties.

        Args:
            **props: Atom properties

        Returns:
            The created atom
        """
        atom = Atom(**props)
        return self.add_atom(atom)

    def add_atoms(self, atoms):
        """
        Add multiple atoms to the structure.

        Args:
            atoms: Sequence of atoms to add
        """
        self.atoms.extend(atoms)

    def add_bond(self, bond):
        """
        Add a bond to the structure.

        Args:
            bond: Bond to add

        Returns:
            The added bond
        """
        return self.bonds.add(bond)

    def def_bond(self, i: Atom | int, j: Atom | int, **kwargs):
        """
        Create and add a bond between two atoms.

        Args:
            i: First atom or its index
            j: Second atom or its index
            **kwargs: Bond properties

        Returns:
            The created bond
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j

        if not isinstance(itom, Atom) or not isinstance(jtom, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")

        bond = Bond(itom, jtom, **kwargs)
        return self.add_bond(bond)

    def del_bond(self, i: Atom | int, j: Atom | int):
        """
        Remove a bond from the structure.

        Args:
            i: First atom or its index
            j: Second atom or its index
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j

        if not isinstance(itom, Atom) or not isinstance(jtom, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")

        bond = Bond(itom, jtom)
        return self.bonds.remove(bond)

    def add_bonds(self, bonds):
        """
        Add multiple bonds to the structure.

        Args:
            bonds: Sequence of bonds to add
        """

        self.bonds.extend(bonds)

    def add_angle(self, angle):
        """
        Add an angle to the structure.

        Args:
            angle: Angle to add

        Returns:
            The added angle
        """
        return self.angles.add(angle)

    def def_angle(self, i: Atom | int, j: Atom | int, k: Atom | int, **kwargs):
        """
        Create and add an angle between three atoms.

        Args:
            i: First atom or its index
            j: Center atom or its index
            k: Third atom or its index
            **kwargs: Angle properties

        Returns:
            The created angle
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j
        ktom = self.atoms[k] if isinstance(k, int) else k

        if not all(isinstance(atom, Atom) for atom in [itom, jtom, ktom]):
            raise TypeError("Arguments must be Atom instances or valid indices")

        angle = Angle(itom, jtom, ktom, **kwargs)
        return self.add_angle(angle)

    def add_angles(self, angles):
        """
        Add multiple angles to the structure.

        Args:
            angles: Sequence of angles to add
        """
        self.angles.extend(angles)

    def add_dihedral(self, dihedral):
        """
        Add a dihedral to the structure.

        Args:
            dihedral: Dihedral to add

        Returns:
            The added dihedral
        """
        return self.dihedrals.add(dihedral)

    def add_dihedrals(self, dihedrals):
        """
        Add multiple dihedrals to the structure.

        Args:
            dihedrals: Sequence of dihedrals to add
        """
        self.dihedrals.extend(dihedrals)

    def remove_atom(self, atom):
        """
        Remove an atom from the structure.

        Args:
            atom: Atom instance, index, or name to remove
        """

    def get_topology(self, attrs=None):
        """
        Get the topology of the structure.

        Args:
            attrs: List of atom attributes to include

        Returns:
            A Topology object representing the structure's topology.
        """

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self.atoms)}
        atom_attrs = {}

        if attrs:
            for attr in attrs:
                atom_attrs[attr] = [atom[attr] for atom in atoms]
        topo.add_atoms(len(atoms), **atom_attrs)
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in self.bonds])
        return topo

    def gen_topo_items(
        self,
        topo: Topology | None = None,
        is_angle: bool = False,
        is_dihedral: bool = False,
    ):
        """
        Generate topology items for angles or dihedrals and add them to the structure.

        Args:
            topo: Topology object to use. If None, creates from current structure.
            is_angle: Whether to generate angle items
            is_dihedral: Whether to generate dihedral items

        Returns:
            List of generated topology items (Angle or Dihedral objects)
        """
        if topo is None:
            topo = self.get_topology()

        generated_items = []

        if is_angle:
            # Generate angles using the dedicated method
            angles = self.gen_angles(topo)
            for angle in angles:
                self.angles.add(angle)
            generated_items.extend(angles)

        if is_dihedral:
            # Generate dihedrals using the dedicated method
            dihedrals = self.gen_dihedrals(topo)
            for dihedral in dihedrals:
                self.dihedrals.add(dihedral)
            generated_items.extend(dihedrals)

        return generated_items

    def add_struct(self, struct):
        """
        Add another structure to the current structure.

        This merges the atoms and bonds from the other structure.

        Args:
            struct: The structure to add

        Returns:
            Self for method chaining
        """
        # Merge atoms and topological entities
        self.add_atoms(struct.atoms)
        self.add_bonds(struct.bonds)
        self.add_angles(struct.angles)
        self.add_dihedrals(struct.dihedrals)

        return self

    @classmethod
    def concat(cls, structs, **new_prop):
        """
        Concatenate multiple structures into a new structure.

        Args:
            name: Name for the new structure
            structs: Sequence of structures to concatenate
            reindex: Whether to reindex the atoms in the concatenated structure

        Returns:
            New structure containing all input structures
        """
        result = cls(**new_prop)
        for struct in structs:
            result.add_struct(struct)
        return result

    def to_dict(self):

        data = {
            "atoms": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {},
            "impropers": {},
        }

        if not self.atoms:
            return data

        atom_dicts = [a.to_dict() for a in self.atoms]
        atom_map = {a: i for i, a in enumerate(self.atoms)}
        data["atoms"] = to_dict_of_list(atom_dicts)

        def serialize_topo(items, ref_keys):
            return [
                dict(
                    item.to_dict(),
                    **{k: atom_map[getattr(item, k + "tom")] for k in ref_keys},
                )
                for item in items
            ]

        data["bonds"] = to_dict_of_list(serialize_topo(self.bonds, ["i", "j"]))
        data["angles"] = to_dict_of_list(serialize_topo(self.angles, ["i", "j", "k"]))
        data["dihedrals"] = to_dict_of_list(
            serialize_topo(self.dihedrals, ["i", "j", "k", "l"])
        )
        data["impropers"] = to_dict_of_list(
            serialize_topo(self.impropers, ["i", "j", "k", "l"])
        )

        return data

    @classmethod
    def from_dict(cls, data):
        struct = cls()

        atom_dicts = to_list_of_dict(data["atoms"])
        struct.atoms = [Atom.from_dict(d) for d in atom_dicts]
        atoms_by_index = struct.atoms  # index to Atom list

        bonds = []
        bond_dicts = to_list_of_dict(data["bonds"])
        for bd in bond_dicts:
            itom = atoms_by_index[bd.pop("i")]
            jtom = atoms_by_index[bd.pop("j")]
            bond = Bond(itom, jtom, **{k: v for k, v in bd.items()})
            bonds.append(bond)
        struct.bonds = bonds

        angles = []
        angle_dicts = to_list_of_dict(data["angles"])
        for ad in angle_dicts:
            itom = atoms_by_index[ad.pop("i")]
            jtom = atoms_by_index[ad.pop("j")]
            ktom = atoms_by_index[ad.pop("k")]
            angle = Angle(itom, jtom, ktom, **{k: v for k, v in ad.items()})
            angles.append(angle)
        struct.angles = angles

        dihedrals = []
        dihedral_dicts = to_list_of_dict(data["dihedrals"])
        for dd in dihedral_dicts:
            itom = atoms_by_index[dd.pop("i")]
            jtom = atoms_by_index[dd.pop("j")]
            ktom = atoms_by_index[dd.pop("k")]
            ltom = atoms_by_index[dd.pop("l")]
            dihedral = Dihedral(itom, jtom, ktom, ltom, **{k: v for k, v in dd.items()})
            dihedrals.append(dihedral)
        struct.dihedrals = dihedrals

        return struct

    def to_frame(self) -> "Frame":
        """
        Convert the structure to a Frame object using the new Frame/Block API.

        Returns:
            Frame: Frame object containing all structure data in Block objects.
        """
        from .frame import Frame, Block  # Local import to avoid circular deps
        import numpy as np

        frame = Frame()

        # Handle empty structure case
        if not self.atoms:
            frame["atoms"] = Block()
            return frame

        # --- Atoms ---
        atom_dicts = [atom.to_dict() for atom in self.atoms]
        atom_map = {atom: i for i, atom in enumerate(self.atoms)}
        if atom_dicts:
            all_keys = set().union(*(d.keys() for d in atom_dicts))
            atoms_block = Block(
                {k: np.asarray([d.get(k) for d in atom_dicts]) for k in all_keys}
            )
            # Add atom IDs if not present (1-based indexing for LAMMPS)
            if "id" not in atoms_block:
                atoms_block["id"] = np.arange(1, len(atom_dicts) + 1)
            frame["atoms"] = atoms_block
        else:
            frame["atoms"] = Block()
        frame.metadata["n_atoms"] = frame["atoms"].nrows

        # --- Bonds ---
        if self.bonds:
            bond_dicts = []
            for bond in self.bonds:
                d = bond.to_dict()
                d["i"] = atom_map[bond.itom]
                d["j"] = atom_map[bond.jtom]
                bond_dicts.append(d)

            all_keys = set().union(*(d.keys() for d in bond_dicts))
            bonds_block = Block(
                {k: np.asarray([d.get(k) for d in bond_dicts]) for k in all_keys}
            )
            # Add bond IDs if not present (1-based indexing for LAMMPS)
            if "id" not in bonds_block:
                bonds_block["id"] = np.arange(1, len(bond_dicts) + 1)
            frame["bonds"] = bonds_block
            frame.metadata["n_bonds"] = frame["bonds"].nrows

        # --- Angles ---
        if self.angles:
            angle_dicts = []
            for angle in self.angles:
                d = angle.to_dict()
                d["i"] = atom_map[angle.itom]
                d["j"] = atom_map[angle.jtom]
                d["k"] = atom_map[angle.ktom]
                angle_dicts.append(d)
            all_keys = set().union(*(d.keys() for d in angle_dicts))
            angles_block = Block(
                {k: np.asarray([d.get(k) for d in angle_dicts]) for k in all_keys}
            )
            frame["angles"] = angles_block
            frame.metadata["n_angles"] = frame["angles"].nrows

        # --- Dihedrals ---
        if self.dihedrals:
            dihedral_dicts = []
            for dih in self.dihedrals:
                d = dih.to_dict()
                d["i"] = atom_map[dih.itom]
                d["j"] = atom_map[dih.jtom]
                d["k"] = atom_map[dih.ktom]
                d["l"] = atom_map[dih.ltom]
                dihedral_dicts.append(d)
            all_keys = set().union(*(d.keys() for d in dihedral_dicts))
            dihedrals_block = Block(
                {k: np.asarray([d.get(k) for d in dihedral_dicts]) for k in all_keys}
            )
            frame["dihedrals"] = dihedrals_block
            frame.metadata["n_dihedrals"] = frame["dihedrals"].nrows

        # --- Impropers ---
        if hasattr(self, "impropers") and self.impropers:
            improper_dicts = []
            for imp in self.impropers:
                d = imp.to_dict()
                d["i"] = atom_map[imp.itom]
                d["j"] = atom_map[imp.jtom]
                d["k"] = atom_map[imp.ktom]
                d["l"] = atom_map[imp.ltom]
                improper_dicts.append(d)
            all_keys = set().union(*(d.keys() for d in improper_dicts))
            impropers_block = Block(
                {k: np.asarray([d.get(k) for d in improper_dicts]) for k in all_keys}
            )
            frame["impropers"] = impropers_block

        # --- Metadata ---
        frame.metadata["name"] = self.get("name", "")

        # Optionally add more metadata as needed
        return frame

    def __call__(self, **new_prop):

        new_instance = self.__class__(**new_prop)

        atom_mapping = {}
        for old_atom in self.atoms:
            new_atom = Atom(**copy.deepcopy(old_atom.to_dict()))
            new_instance.add_atom(new_atom)
            atom_mapping[old_atom] = new_atom

        def _deepcopy_topo(items, cls, atom_keys, add_method):
            for item in items:
                atoms = [atom_mapping.get(getattr(item, k + "tom")) for k in atom_keys]
                if all(atoms):
                    data = copy.deepcopy(item.to_dict())
                    # Remove atom references from data if present
                    for k in atom_keys:
                        data.pop(f"{k}tom", None)
                    new_item = cls(*atoms, **data)
                    add_method(new_item)

        # Deep copy bonds, angles, dihedrals
        _deepcopy_topo(self.bonds, Bond, ["i", "j"], new_instance.add_bond)
        _deepcopy_topo(self.angles, Angle, ["i", "j", "k"], new_instance.add_angle)
        _deepcopy_topo(
            self.dihedrals, Dihedral, ["i", "j", "k", "l"], new_instance.add_dihedral
        )
        # _deepcopy_topo(self.dihedrals, Improper, ["i", "j", "k", "l"], new_instance.add_improper)

        return new_instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_wrapped":
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @classmethod
    def from_frame(cls, frame, **props):
        """
        Create a new Atomistic from a Frame object.

        Args:
            frame: Frame object containing the structure data
            name: Name for the new structure (optional)

        Returns:
            New Atomistic instance populated from the frame
        """
        struct = cls(**props)

        # Handle empty frame case
        if "atoms" not in frame or not frame["atoms"]:
            return struct

        atoms = frame["atoms"]
        n_atoms = atoms.nrows
        # Create atoms with all their properties
        for i in range(n_atoms):
            atom = Atom(**atoms[i])
            struct.add_atom(atom)

        # Create bonds if they exist in the frame
        if "bonds" in frame and frame["bonds"]:
            bonds = frame["bonds"]
            n_bonds = bonds.nrows
            for i in range(n_bonds):
                bond_props = bonds[i]
                i = bond_props.pop("i")
                j = bond_props.pop("j")
                struct.add_bond(Bond(struct.atoms[i], struct.atoms[j], **bond_props))

        # Create angles if they exist in the frame
        if "angles" in frame and frame["angles"]:
            angles = frame["angles"]
            n_angles = angles.nrows
            for i in range(n_angles):
                angle_props = angles[i]
                i = angle_props.pop("i")
                j = angle_props.pop("j")
                k = angle_props.pop("k")
                struct.add_angle(
                    Angle(
                        struct.atoms[i], struct.atoms[j], struct.atoms[k], **angle_props
                    )
                )

        # Create dihedrals if they exist in the frame
        if "dihedrals" in frame and frame["dihedrals"]:
            dihedrals = frame["dihedrals"]
            n_dihedrals = dihedrals.nrows
            for i in range(n_dihedrals):
                dihedral_props = dihedrals[i]
                i = dihedral_props.pop("i")
                j = dihedral_props.pop("j")
                k = dihedral_props.pop("k")
                l = dihedral_props.pop("l")
                struct.add_dihedral(
                    Dihedral(
                        struct.atoms[i],
                        struct.atoms[j],
                        struct.atoms[k],
                        struct.atoms[l],
                        **dihedral_props,
                    )
                )
        return struct

    def gen_angles(self, topo=None):
        """
        Generate angle objects from bond connectivity.

        Args:
            topo: Optional topology object to use. If None, creates from current structure.

        Returns:
            List of Angle objects
        """
        if topo is None:
            topo = self.get_topology()

        # Check if we have enough atoms for angles
        if len(self.atoms) < 3:
            return []

        # The Topology class automatically finds angles using graph isomorphism
        # Convert topology angles to Angle objects
        angles = []
        try:
            angle_indices_array = topo.angles
            atom_list = list(self.atoms)  # Convert to list for proper indexing

            if angle_indices_array is not None and len(angle_indices_array) > 0:
                # Handle both 1D and 2D arrays
                if angle_indices_array.ndim == 1:
                    if len(angle_indices_array) >= 3:
                        angle_indices_array = [angle_indices_array]
                    else:
                        return []

                for angle_indices in angle_indices_array:
                    if len(angle_indices) >= 3:
                        i, j, k = angle_indices[:3]
                        if (
                            i < len(atom_list)
                            and j < len(atom_list)
                            and k < len(atom_list)
                        ):
                            itom = atom_list[i]
                            jtom = atom_list[j]
                            atom3 = atom_list[k]
                            # Check that all atoms are different
                            if len({id(itom), id(jtom), id(atom3)}) == 3:
                                angle = Angle(itom, jtom, atom3)
                                angles.append(angle)
        except (IndexError, AttributeError) as e:
            # Handle cases where topology properties fail
            pass

        return angles

    def gen_dihedrals(self, topo=None):
        """
        Generate dihedral objects from bond connectivity.

        Args:
            topo: Optional topology object to use. If None, creates from current structure.

        Returns:
            List of Dihedral objects
        """
        if topo is None:
            topo = self.get_topology()

        # Check if we have enough atoms for dihedrals
        if len(self.atoms) < 4:
            return []

        # The Topology class automatically finds dihedrals using graph isomorphism
        # Convert topology dihedrals to Dihedral objects
        dihedrals = []
        try:
            dihedral_indices_array = topo.dihedrals
            atom_list = list(self.atoms)  # Convert to list for proper indexing

            if dihedral_indices_array is not None and len(dihedral_indices_array) > 0:
                # Handle both 1D and 2D arrays
                if dihedral_indices_array.ndim == 1:
                    if len(dihedral_indices_array) >= 4:
                        dihedral_indices_array = [dihedral_indices_array]
                    else:
                        return []

                for dihedral_indices in dihedral_indices_array:
                    if len(dihedral_indices) >= 4:
                        i, j, k, l = dihedral_indices[:4]
                        if all(idx < len(atom_list) for idx in [i, j, k, l]):
                            itom = atom_list[i]
                            jtom = atom_list[j]
                            atom3 = atom_list[k]
                            atom4 = atom_list[l]
                            # Check that all atoms are different
                            if len({id(itom), id(jtom), id(atom3), id(atom4)}) == 4:
                                dihedral = Dihedral(itom, jtom, atom3, atom4)
                                dihedrals.append(dihedral)
        except (IndexError, AttributeError) as e:
            # Handle cases where topology properties fail
            pass

        return dihedrals

    def get_substruct(self, atom_indices: list[int]):
        """Extract substructure containing only specified atoms."""
        new_struct = type(self)()

        # Map old atom to new atom
        atom_map = {}

        # Add atoms
        for old_idx in atom_indices:
            atom = self.atoms[old_idx]
            new_atom = atom.copy()
            new_struct.atoms.add(new_atom)
            atom_map[atom] = new_atom

        # Add bonds between extracted atoms
        for bond in self.bonds:
            atom1 = bond.itom
            atom2 = bond.jtom
            if atom1 in atom_map and atom2 in atom_map:
                new_bond = Bond(
                    atom_map[atom1], atom_map[atom2], **{k: v for k, v in bond.items()}
                )
                new_struct.bonds.add(new_bond)

        for angle in self.angles:
            atom1 = angle.itom
            atom2 = angle.jtom
            atom3 = angle.ktom
            if atom1 in atom_map and atom2 in atom_map and atom3 in atom_map:
                new_angle = Angle(
                    atom_map[atom1],
                    atom_map[atom2],
                    atom_map[atom3],
                    **{k: v for k, v in angle.items()},
                )
                new_struct.angles.add(new_angle)

        for dihedral in self.dihedrals:
            atom1 = dihedral.itom
            atom2 = dihedral.jtom
            atom3 = dihedral.ktom
            atom4 = dihedral.ltom
            if (
                atom1 in atom_map
                and atom2 in atom_map
                and atom3 in atom_map
                and atom4 in atom_map
            ):
                new_dihedral = Dihedral(
                    atom_map[atom1],
                    atom_map[atom2],
                    atom_map[atom3],
                    atom_map[atom4],
                    **{k: v for k, v in dihedral.items()},
                )
                new_struct.dihedrals.add(new_dihedral)

        return new_struct

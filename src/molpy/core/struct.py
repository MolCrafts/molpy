from copy import deepcopy
from molpy.op import rotate_by_rodrigues
from typing import Callable, Generic, TypeVar
import numpy as np
import pandas as pd
from molpy import op

T = TypeVar("entity")

class Entity(dict):
    """Base class representing a general entity with dictionary-like behavior."""

    def __call__(self):
        """Return a copy of the entity."""
        return self.copy()

    def clone(self):
        """Create a deep copy of the entity."""
        return deepcopy(self)

    def to_dict(self):
        """Convert the entity to a dictionary."""
        return dict(self)

    def copy(self):
        """Alias for the clone method."""
        return self.clone()


class SpatialMixin:
    """Mixin class for spatial operations on entities."""

    @property
    def xyz(self):
        """Get the 3D coordinates of the entity."""
        return np.array([self["x"], self["y"], self["z"]], dtype=float)

    @xyz.setter
    def xyz(self, value):
        """Set the 3D coordinates of the entity."""
        if len(value) != 3:
            raise ValueError("R must be a 3D vector")
        self["x"], self["y"], self["z"] = value

    def distance_to(self, other):
        """Calculate the Euclidean distance to another entity."""
        return np.linalg.norm(self.xyz - other.xyz)

    def translate(self, vector):
        """Translate the entity by a given vector."""
        self.xyz += vector

    def rotate(self, axis, theta):
        """Rotate the entity around a given axis by a specified angle."""
        self.xyz = rotate_by_rodrigues(self.xyz.reshape(1, -1), axis, theta).flatten()


class Atom(Entity, SpatialMixin):
    """Class representing an atom."""

    def __init__(self, **props):
        """
        Initialize an atom with a name.

        Parameters:
        - name: Name of the atom.
        """
        super(Entity, self).__init__(**props)

    def __repr__(self):
        """Return a string representation of the atom."""
        return f"<Atom {self['name']}>"

    def __hash__(self):
        """Return a unique hash for the atom."""
        return id(self)

    def __eq__(self, other):
        """Check equality based on the atom's name."""
        return id(self) == id(other)

    def __lt__(self, other):
        """Compare atoms based on their names."""
        return id(self) < id(other)
    


class ManyBody(Entity):
    """Base class for entities involving multiple atoms."""

    def __init__(self, *_atoms, **kwargs):
        """
        Initialize a ManyBody entity.

        Parameters:
        - _atoms: Atoms involved in the entity.
        - kwargs: Additional properties.
        """
        super().__init__(**kwargs)
        self._atoms = _atoms


class Bond(ManyBody):
    """Class representing a bond between two atoms."""

    def __init__(self, itom: Atom, jtom: Atom, **kwargs):
        """
        Initialize a bond.

        Parameters:
        - itom: First atom in the bond.
        - jtom: Second atom in the bond.
        - kwargs: Additional properties.
        """
        itom, jtom = sorted([itom, jtom])
        super().__init__(itom, jtom, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the bond."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the bond."""
        return self._atoms[1]

    def __repr__(self):
        """Return a string representation of the bond."""
        return f"<Bond {self.itom} {self.jtom}>"

    def __eq__(self, other):
        """Check equality based on the atoms in the bond."""
        if isinstance(other, Bond):
            return (self.itom == other.itom and self.jtom == other.jtom) or (
                self.itom == other.jtom and self.jtom == other.itom
            )
        return False

    def __hash__(self):
        """Return a unique hash for the bond."""
        return hash((self.itom, self.jtom))


class Angle(ManyBody):
    """Class representing an angle between three atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, **kwargs):
        """
        Initialize an angle.

        Parameters:
        - itom: First atom in the angle.
        - jtom: Second atom in the angle (vertex).
        - ktom: Third atom in the angle.
        - kwargs: Additional properties.
        """
        itom, ktom = sorted([itom, ktom])
        super().__init__(itom, jtom, ktom, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the angle."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the angle (vertex)."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the angle."""
        return self._atoms[2]

    def to_dict(self):
        """Convert the angle to a dictionary."""
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"]
        )


class Dihedral(ManyBody):
    """Class representing a dihedral angle between four atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize a dihedral angle.

        Parameters:
        - itom: First atom in the dihedral.
        - jtom: Second atom in the dihedral.
        - ktom: Third atom in the dihedral.
        - ltom: Fourth atom in the dihedral.
        - kwargs: Additional properties.
        """
        if jtom > ktom:
            jtom, ktom = ktom, jtom
            itom, ltom = ltom, itom
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
        """Get the fourth atom in the dihedral."""
        return self._atoms[3]

    def to_dict(self):
        """Convert the dihedral to a dictionary."""
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"], l=self.ltom["id"]
        )


class Improper(ManyBody):
    """Class representing an improper dihedral angle between four atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize an improper dihedral angle.

        Parameters:
        - itom: First atom in the improper dihedral.
        - jtom: Second atom in the improper dihedral.
        - ktom: Third atom in the improper dihedral.
        - ltom: Fourth atom in the improper dihedral.
        - kwargs: Additional properties.
        """
        jtom, ktom, ltom = sorted([jtom, ktom, ltom])
        super().__init__(itom, jtom, ktom, ltom, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the improper dihedral."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the improper dihedral."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the improper dihedral."""
        return self._atoms[2]

    @property
    def ltom(self):
        """Get the fourth atom in the improper dihedral."""
        return self._atoms[3]

    def to_dict(self):
        """Convert the improper dihedral to a dictionary."""
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"], l=self.ltom["id"]
        )


class Entities(Generic[T]):
    """Class representing a collection of entities."""

    def __init__(self, entities: list[T] = []):
        self._data = set(entities)

    def add(self, entity):
        """Add an entity to the collection."""
        self._data.add(entity)
        return self

    def get_by(self, condition: Callable[[T], bool]) -> T:
        """
        Get an entity based on a condition.

        Parameters:
        - condition: A callable that takes an entity and returns a boolean.

        Returns:
        - The first entity that satisfies the condition, or None.
        """
        return next((entity for entity in self._data if condition(entity)), None)

    def __len__(self):
        """Return the number of entities in the collection."""
        return len(self._data)

    def extend(self, entities):
        """Extend the collection with multiple entities."""
        for entity in entities:
            self.add(entity)
        return self

    def __iter__(self):
        """Return an iterator over the entities."""
        return iter(self._data)

    def __getitem__(self, key):
        """Get an entity by its index."""
        return list(self._data)[key]

class HierarchicalMixin(Generic[T]):
    """Mixin class for hierarchical operations on entities."""

    def __new__(self, *args, **kwargs):
        ins = super().__new__(self)
        if not hasattr(ins, "_entities"):
            ins._entities = Entities()

        return ins
    
    def add_child(self, entity: T):
        """Add a sub-entity to the collection."""
        self._entities.add(entity)
        return self


class Struct(Entity, HierarchicalMixin["Struct"]):
    """Class representing a molecular structure."""

    def __init__(
        self,
        name: str | None = None,
        atoms: Entities | list = [],
        bonds: Entities | list = [],
        angles: Entities | list = [],
        dihedrals: Entities | list = [],
        impropers: Entities | list = [],
    ):
        """Initialize a molecular structure with atoms, bonds, angles, etc."""
        super().__init__(
            {
                "name": name,
                "atoms": Entities(atoms),
                "bonds": Entities([Bond(atoms[bond[0]], atoms[bond[1]]) if isinstance(bond, tuple) else bond for bond in bonds]),
                "angles": Entities(angles),
                "dihedrals": Entities(dihedrals),
                "impropers": Entities(impropers),
            }
        )

    @property
    def atoms(self):
        """Get the atoms in the structure."""
        return self["atoms"]
    
    @property
    def bonds(self):
        """Get the bonds in the structure."""
        return self["bonds"]

    def __repr__(self):
        """Return a string representation of the structure."""
        return f"<Struct: {len(self['atoms'])} atoms>"

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the structure.

        Parameters:
        - memo: Dictionary of objects already copied during the current copying pass.

        Returns:
        - A deep copy of the structure.
        """
        new_atoms = [atom.clone() for atom in self["atoms"]]
        atom_mapping = {
            atom: new_atom for atom, new_atom in zip(self["atoms"], new_atoms)
        }
        new = Struct()
        for key, value in self.items():
            new[key] = Entities()
        atoms = new["atoms"]
        for atom in new_atoms:
            atoms.add(atom)
        for key, value in self.items():
            if isinstance(value, Entities):
                for v in value:
                    if isinstance(v, ManyBody):
                        try:
                            new[key].add(
                                v.__class__(
                                    *[atom_mapping[atom] for atom in v._atoms], **v
                                )
                            )
                        except KeyError:
                            raise KeyError(f"Atoms {v._atoms} not found in atom_map")
        return new

    def add_atom(self, **props):
        """
        Add an atom to the structure.

        Parameters:
        - props: Properties of the atom.
        """
        self["atoms"].add(Atom(**props))
        return self

    def add_bond(self, itom, jtom, **kwargs):
        """
        Add a bond to the structure.

        Parameters:
        - itom: First atom in the bond.
        - jtom: Second atom in the bond.
        - kwargs: Additional properties.
        """
        if isinstance(itom, str):
            itom = self["atoms"][itom]
        if isinstance(jtom, str):
            jtom = self["atoms"][jtom]
        bond = Bond(itom, jtom, **kwargs)
        self["bonds"][(itom, jtom)] = bond
        return self

    def del_atom(self, atom):
        """
        Delete an atom from the structure.

        Parameters:
        - atom: Atom to delete (can be name, ID, or Atom object).
        """
        if isinstance(atom, str):
            atom = self.get_atom_by(lambda atom: atom["name"] == atom)
        if isinstance(atom, int):
            atom = self.get_atom_by_id(atom)
        if isinstance(atom, Atom):
            self["atoms"].remove(atom)
        else:
            raise ValueError(f"Cannot delete {atom}")
        return self

    def del_bond(self, itom, jtom):
        """
        Delete a bond from the structure.

        Parameters:
        - itom: First atom in the bond.
        - jtom: Second atom in the bond.
        """
        if isinstance(itom, str):
            itom = self.get_atom_by(lambda atom: atom["name"] == itom)
        if isinstance(jtom, str):
            jtom = self.get_atom_by(lambda atom: atom["name"] == jtom)
        to_be_deleted = Bond(itom, jtom)
        for bond in self["bonds"]:
            if bond == to_be_deleted:
                self["bonds"].remove(bond)
        return self

    def get_atom_by(self, condition: Callable[[Atom], bool]) -> Atom:
        """
        Get an atom based on a condition.

        Parameters:
        - condition: A callable that takes an atom and returns a boolean.

        Returns:
        - The first atom that satisfies the condition, or None.
        """
        return next((atom for atom in self["atoms"] if condition(atom)), None)

    def move(self, r):
        """
        Translate all atoms in the structure by a given vector.

        Parameters:
        - r: Translation vector.
        """
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.translate(xyz, r)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self

    def rotate(self, axis, theta):
        """
        Rotate all atoms in the structure around a given axis.

        Parameters:
        - axis: Rotation axis.
        - theta: Rotation angle in radians.
        """
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.rotate(xyz, axis, theta)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self

    def to_frame(self):
        """
        Convert the structure to a Frame object.

        Returns:
        - A Frame object containing the structure's data.
        """
        from .frame import Frame

        frame = Frame()

        atom_name_idx_mapping = {}
        for i, atom in enumerate(self["atoms"]):
            atom_name_idx_mapping[atom["name"]] = i
            atom["id"] = i

        frame["atoms"] = pd.DataFrame(
            [atom.to_dict() for atom in self["atoms"]]
        )
        if "bonds" in self and len(self["bonds"]) > 0:
            bdicts = []
            for bond in self["bonds"]:
                bdict = bond.to_dict()
                iname = bond.itom["name"]
                jname = bond.jtom["name"]
                bdict["i"] = atom_name_idx_mapping[iname]
                bdict["j"] = atom_name_idx_mapping[jname]
                bdicts.append(bdict)
            frame["bonds"] = pd.DataFrame(bdicts)
            frame["bonds"]["id"] = range(len(frame["bonds"]))

        # if "angles" in self and len(self["angles"]) > 0:
        #     angle_dict = [angle.to_dict() for angle in self["angles"]]
        #     frame["angles"] = pd.DataFrame(angle_dict)
        #     frame["angles"]["id"] = range(len(frame["angles"]))

        # if "dihedrals" in self and len(self["dihedrals"]) > 0:
        #     dihedral_dict = [dihedral.to_dict() for dihedral in self["dihedrals"]]
        #     frame["dihedrals"] = pd.DataFrame(dihedral_dict)
        #     frame["dihedrals"]["id"] = range(len(frame["dihedrals"]))

        # if "impropers" in self and len(self["impropers"]) > 0:
        #     improper_dict = [improper.to_dict() for improper in self["impropers"]]
        #     frame["impropers"] = pd.DataFrame(improper_dict)

        return frame

    def get_substruct(self, atom_names):
        """
        Get a substructure of the current structure by atom names.

        Parameters:
        - atom_names: List of atom names to include in the substructure.

        Returns:
        - A new Struct object containing the substructure.
        """
        substruct = Struct()
        atom_names = set(atom_names)
        for atom in self["atoms"]:
            if atom["name"] in atom_names:
                substruct.add_atom(**atom)
        for bond in self["bonds"]:
            if bond.itom["name"] in atom_names and bond.jtom["name"] in atom_names:
                itom = substruct["atoms"][bond.itom["name"]]
                jtom = substruct["atoms"][bond.jtom["name"]]
                substruct.add_bond(itom, jtom, **bond)
        return substruct

    def get_topology(self):
        """
        Get the topology of the structure.

        Returns:
        - A Topology object representing the structure's topology.
        """
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self["atoms"])}
        atom_attrs = {}
        if all("number" in atom for atom in self["atoms"]):
            atom_attrs["number"] = [atom["number"] for atom in self["atoms"]]
        if all("name" in atom for atom in self["atoms"]):
            atom_attrs["name"] = [atom["name"] for atom in self["atoms"]]

        # TODO: atom name if no number
        topo.add_atoms(len(atoms), **atom_attrs)
        bonds = self["bonds"]
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in bonds])
        return topo

    def add_struct(self, struct: "Struct"):
        """
        Add another structure to the current structure.

        Parameters:
        - struct: The structure to add.
        """
        self.add_child(struct)
        return self
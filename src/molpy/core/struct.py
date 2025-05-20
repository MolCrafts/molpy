from collections import UserDict, namedtuple
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from dataclasses import field
from typing import Callable, Generic, Protocol, Sequence, TypeVar

from attr import dataclass
import numpy as np
from nesteddict import ArrayDict
from numpy.typing import ArrayLike

from molpy.op import rotate_by_rodrigues

T = TypeVar("entity")


class Entity(UserDict):
    """Base class representing a general entity with dictionary-like behavior."""

    def __call__(self, **modify):
        """Return a copy of the entity."""
        return self.clone(**modify)

    def clone(self, **modify):
        """Create a deep copy of the entity."""
        ins = deepcopy(self)
        for k, v in modify.items():
            ins[k] = v
        return ins

    def __hash__(self):
        """Return a unique hash for the atom."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        """Compare entities based on their IDs."""
        return id(self) < id(other)

    def to_dict(self):
        return dict(self)
    
    def keys(self):
        """Return the keys of the entity."""
        return self.data.keys()
    

class Spatial(Protocol):
    """Mixin class for spatial operations on entities."""

    xyz: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def distance_to(self, other):
        """Calculate the Euclidean distance to another entity."""
        return np.linalg.norm(self.xyz - other.xyz)

    def move(self, vector: ArrayLike):
        """Move the entity by a given vector."""
        self.xyz = np.add(self.xyz, vector)
        return self

    def rotate(self, theta, axis):
        """Rotate the entity around a given axis by a specified angle."""
        self.xyz = rotate_by_rodrigues(self.xyz, axis, theta)
        return self

    def reflact(self, axis: ArrayLike):
        """Reflect the entity across a given axis."""
        self.xyz = np.dot(self.xyz, np.eye(3) - 2 * np.outer(axis, axis))
        return self


class Atom(Entity):
    """Class representing an atom."""

    def __repr__(self):
        """Return a string representation of the atom."""
        return f"<Atom {str(self['name'])}>"

    @property
    def xyz(self):
        return np.array(self["xyz"], dtype=float)

    @xyz.setter
    def xyz(self, value):
        assert len(value) == 3, "xyz must be a 3D vector"
        self["xyz"] = np.array(value)

    @property
    def name(self):
        """Get the name of the atom."""
        return self["name"]


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

    def __hash__(self):
        return sum([hash(atom) for atom in self._atoms]) + hash(self.__class__.__name__)
    
    @property
    def atoms(self):
        """Get the atoms involved in the entity."""
        return self._atoms


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
        return f"<Bond: {self.itom}-{self.jtom}>"

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

    def __repr__(self):
        return f"<Angle: {self.itom.name}-{self.jtom.name}-{self.ktom.name}>"

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

    def __repr__(self):
        """Return a string representation of the dihedral."""
        return f"<Dihedral: {self.itom.name}-{self.jtom.name}-{self.ktom.name}-{self.ltom.name}>"


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


class EntityContainer(Sequence[T]):

    def add(self, entity: T) -> T:
        """Add an entity to the container."""
        ...

    def get_by(self, condition: Callable[[T], bool]) -> T:
        """
        Get an entity based on a condition.

        Parameters:
        - condition: A callable that takes an entity and returns a boolean.

        Returns:
        - The first entity that satisfies the condition, or None.
        """
        ...

    def __len__(self) -> int:
        """Return the number of entities in the container."""
        ...

    def extend(self, entities: Sequence[T]) -> None:
        """Extend the container with multiple entities."""
        ...

    
class Entities(EntityContainer[T]):
    """Class representing a collection of entities."""

    def __init__(self, entities: list[T] = []):
        self._data = list(entities)

    def add(self, entity):
        """Add an entity to the collection."""
        self._data.append(entity)
        return entity

    def get_by(self, condition: Callable[[T], bool]) -> T | None:
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
        self._data.extend(entities)

    def __iter__(self):
        """Return an iterator over the entities."""
        return iter(self._data)

    def __getitem__(self, key: int|Sequence[int]):
        """Get an entity by its index."""
        if isinstance(key, (int, slice)):
            return self._data[key]
        elif isinstance(key, Iterable):
            return [self._data[i] for i in key]
        else:
            return self._data[key]
        

    def __repr__(self):
        """Return a string representation of the collection."""
        return f"<Entities: {len(self._data)}>"

    def remove(self, entity):
        e = self[entity]
        self._data.remove(e)


class Hierarchical(Generic[T]):
    """Mixin class for hierarchical operations on entities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.childern = Entities()

    def add_child(self, entity: T):
        """Add a sub-entity to the collection."""
        self.childern.add(entity)
        return self

    def reduce(self, func: Callable[[T], T] = lambda x: x) -> list[T]:
        """Reduce the collection of entities using a function."""
        return [func(entity) for entity in self.childern]


class Atomistic(MutableMapping):
    """Mixin class for structures containing all atoms."""

    @property
    def atoms(self) -> Entities[Atom]:
        return self["atoms"]

    @property
    def bonds(self) -> Entities[Bond]:
        return self["bonds"]
    
    @property
    def angles(self) -> Entities[Angle]:
        return self["angles"]
    
    @property
    def dihedrals(self) -> Entities[Dihedral]:
        return self["dihedrals"]
    
    def add_atom(self, atom: Atom):
        """
        Add an atom to the structure.

        Parameters:
        - props: Properties of the atom.
        """
        return self.atoms.add(atom)

    def def_atom(self, **props):
        """
        Define an atom with given properties.

        Parameters:
        - props: Properties of the atom.
        """
        atom = Atom(**props)
        return self.add_atom(atom)

    def add_atoms(self, atoms: Sequence[Atom]):
        """
        Add multiple atoms to the structure.

        Parameters:
        - atoms: List of atoms to add.
        """
        self.atoms.extend(atoms)

    def add_angles(self, angles: Sequence[Angle]):
        """
        Add multiple angles to the structure.

        Parameters:
        - angles: List of angles to add.
        """
        self.angles.extend(angles)

    def add_bond(self, bond: Bond):
        """
        Add a bond to the structure.

        Parameters:
        - bond: Bond to add.
        """
        return self.bonds.add(bond)

    def def_bond(self, itom, jtom, **kwargs):
        """
        Add a bond to the structure.

        Parameters:
        - itom: First atom in the bond.
        - jtom: Second atom in the bond.
        - kwargs: Additional properties.
        """
        if isinstance(itom, int):
            itom = self["atoms"][itom]
        if isinstance(jtom, int):
            jtom = self["atoms"][jtom]

        return self.add_bond(Bond(itom, jtom, **kwargs))

    def add_bonds(self, bonds: Sequence[Bond]):
        """
        Add multiple bonds to the structure.

        Parameters:
        - bonds: List of bonds to add.
        """
        self.bonds.extend(bonds)

    def del_atom(self, atom):
        """
        Delete an atom from the structure.

        Parameters:
        - atom: Atom to delete (can be name, ID, or Atom object).
        """
        self.atoms.remove(atom)

    def del_atoms(self, atoms: Sequence[Atom]):
        """
        Delete multiple atoms from the structure.

        Parameters:
        - atoms: List of atoms to delete.
        """
        for atom in atoms:
            self.del_atom(atom)
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

    def get_atom_by(self, condition: Callable[[Atom], bool]) -> Atom|None:
        """
        Get an atom based on a condition.

        Parameters:
        - condition: A callable that takes an atom and returns a boolean.

        Returns:
        - The first atom that satisfies the condition, or None.
        """
        return next((atom for atom in self["atoms"] if condition(atom)), None)

class Struct(Entity, Atomistic, Spatial):
    """Class representing a molecular structure."""

    def __init__(
        self,
        name: str = "",
        **props,
    ):
        """Initialize a molecular structure with atoms, bonds, angles, etc."""
        super().__init__(name=name, **props)
        self["atoms"] = Entities[Atom]()
        self["bonds"] = Entities[Bond]()
        self["angles"] = Entities[Angle]()
        self["dihedrals"] = Entities[Dihedral]()

    @classmethod
    def from_frame(cls, frame, name=""):

        struct = cls(name=name)
        atoms = frame["atoms"]
        for atom in atoms.iterrows():
            struct.def_atom(**atom)

        if "bonds" in frame:
            struct["bonds"] = Entities()
            bonds = frame["bonds"]
            for bond in bonds.iterrows():
                i, j = bond.pop("i"), bond.pop("j")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                struct["bonds"].add(
                    Bond(
                        itom,
                        jtom,
                        **{k: v for k, v in bond.items()},
                    )
                )

        if "angles" in frame:
            struct["angles"] = Entities()
            angles = frame["angles"]
            for _, angle in angles.iterrows():
                i, j, k = angle.pop("i"), angle.pop("j"), angle.pop("k")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                ktom = struct["atoms"].get_by(lambda atom: atom["id"] == k)
                struct["angles"].add(
                    Angle(
                        itom,
                        jtom,
                        ktom,
                        **{k: v for k, v in angle.items()},
                    )
                )

        if "dihedrals" in frame:
            struct["dihedrals"] = Entities()
            dihedrals = frame["dihedrals"]
            for _, dihedral in dihedrals.iterrows():
                i, j, k, l = (
                    dihedral.pop("i"),
                    dihedral.pop("j"),
                    dihedral.pop("k"),
                    dihedral.pop("l"),
                )
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                ktom = struct["atoms"].get_by(lambda atom: atom["id"] == k)
                ltom = struct["atoms"].get_by(lambda atom: atom["id"] == l)
                struct["dihedrals"].add(
                    Dihedral(
                        itom,
                        jtom,
                        ktom,
                        ltom,
                        **{k: v for k, v in dihedral.items()},
                    )
                )

        return struct


    @property
    def xyz(self):
        """Get the coordinates of the atoms in the structure."""
        return np.array([atom["xyz"] for atom in self["atoms"]], dtype=float)

    @xyz.setter
    def xyz(self, value):
        """Set the coordinates of the atoms in the structure."""
        assert len(value) == len(self["atoms"]), "xyz must match the number of atoms"
        for i, atom in enumerate(self.atoms):
            atom["xyz"] = value[i]

    def __repr__(self):
        """Return a string representation of the structure."""
        return f"<Struct: {len(self['atoms'])} atoms>"

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the structure, preserving atom references in bonds/angles/etc.
        """
        return self._deepcopy(memo)[0]

    def _deepcopy(self, memo):
        # Step 1: Deep-copy atoms
        new = self.__class__()
        new["atoms"] = Entities()
        atom_mapping = {}

        for atom in self["atoms"]:
            new_atom = atom.copy()
            new["atoms"].add(new_atom)
            atom_mapping[atom] = new_atom

        # Step 2: Copy all other entities
        for key, value in self.items():
            if key == "atoms":
                continue

            if isinstance(value, Entities):
                new[key] = Entities()

                for v in value:
                    if isinstance(v, ManyBody):
                        # Rebuild the new ManyBody object with remapped atoms
                        new_atoms = [atom_mapping[a] for a in v._atoms]
                        new_entity = v.__class__(*new_atoms, **v)
                        new[key].add(new_entity)
                    else:
                        # If not a ManyBody (unlikely here), use deepcopy
                        new[key].add(deepcopy(v, memo))
            else:
                # For non-Entities fields, shallow-copy or deepcopy as needed
                new[key] = deepcopy(value, memo)

        return new, atom_mapping

    def to_frame(self, atom_keys: list[str]|None = None, bond_keys: list[str]|None = None):
        """
        Convert the structure to a Frame object.

        Returns:
        - A Frame object containing the structure's data.
        """
        from .frame import Frame

        frame = Frame()
        frame["name"] = self["name"]

        atom_name_idx_mapping = {}
        for i, atom in enumerate(self["atoms"]):
            atom_name_idx_mapping[atom] = i

        frame["atoms"] = ArrayDict.from_dicts(
            [atom.to_dict() for atom in self["atoms"]], include=atom_keys
        )
        if "bonds" in self:
            bdicts = []
            for i, bond in enumerate(self["bonds"]):
                bdict = bond.to_dict()
                bdict["id"] = i
                bdict["i"] = atom_name_idx_mapping[bond.itom]
                bdict["j"] = atom_name_idx_mapping[bond.jtom]
                bdicts.append(bdict)
            frame["bonds"] = ArrayDict.from_dicts(bdicts, bond_keys)

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

    def get_substruct(self, indices: Sequence[int]):
        """
        Get a substructure of the current structure by atom names.

        Parameters:
        - indices: List of atom names to include in the substructure.

        Returns:
        - A new Struct object containing the substructure.
        """
        substruct = self.__class__()
        _indices = set(indices)
        atoms = self["atoms"][_indices]
        substruct["atoms"].extend(atoms)
        for bond in self["bonds"]:
            if bond.itom in atoms and bond.jtom in atoms:
                substruct.add_bond(bond)
        return substruct

    def get_topology(self, attrs: list[str] = []):
        """
        Get the topology of the structure.

        Returns:
        - A Topology object representing the structure's topology.
        """
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self.atoms)}
        atom_attrs = {}

        for attr in attrs:
            atom_attrs[attr] = [atom[attr] for atom in atoms]
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
        # self.add_child(struct)
        self.add_atoms(struct.atoms)
        self.add_bonds(struct.bonds)
        return self

    def gen_angles(self, topology):
        """
        Calculate angles in the structure.

        Returns:
        - list of Angle entities.
        """

        angle_idx = topology.angles
        angles = [
            Angle(
                self["atoms"][i],
                self["atoms"][j],
                self["atoms"][k],
            )
            for i, j, k in angle_idx
        ]
        return angles

    def gen_dihedrals(self, topology):
        """
        Calculate dihedrals in the structure.

        Returns:
        - list of Dihedral entities.
        """

        dihedral_idx = topology.dihedrals
        dihedrals = [
            Dihedral(
                self["atoms"][i],
                self["atoms"][j],
                self["atoms"][k],
                self["atoms"][l],
            )
            for i, j, k, l in dihedral_idx
        ]
        return dihedrals

    def get_topology(self):
        """
        Get the topology of the structure.

        Returns:
        - A Topology object representing the structure's topology.
        """
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self.atoms)}
        topo.add_atoms(len(atoms))
        bonds = self["bonds"]
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in bonds])
        return topo

    @classmethod
    def concat(cls, name, structs: Sequence["Struct"]):
        """
        Concatenate multiple structures into the current structure.

        Parameters:
        - structs: List of structures to concatenate.
        """
        _struct = cls(name)
        for struct in structs:
            _struct.add_struct(struct)
        return _struct


@dataclass
class LinkSite:
    """Class representing a link site in a monomer."""

    anchor: Atom
    deletes : list[Atom] = field(default_factory=list)
    label: str = ""
    direction: None|ArrayLike = None

class MonomerLike(MutableMapping):
    """Mixin class for monomer"""

    def def_link_site(self, this: Atom, deletes=[], label=""):
        """
        Define a link site for the monomer.

        Parameters:
        - this: Head atom of the link site.
        - that: Tail atom of the link site.
        - delete: Whether to delete the link site.
        """
        site = LinkSite(this, deletes, label)
        self["ports"].append(site)
        return site
    

class Monomer(Struct, MonomerLike):
    """Class representing a monomer."""

    def __init__(self, **props):
        super().__init__(**props)
        self["ports"] = []

    def __repr__(self):
        """Return a string representation of the monomer."""
        return f"<Monomer: {len(self['atoms'])} atoms>"
    
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the monomer, preserving atom references in bonds/angles/etc.
        """
        new, atom_mapping = self._deepcopy(memo)
        new["ports"] = []
        for port in self["ports"]:
            new_port = LinkSite(
                atom_mapping[port.anchor],
                [atom_mapping[atom] for atom in port.deletes],
                port.label,
                port.direction,
            )
            new["ports"].append(new_port)
        return new

class Polymer(Struct):
    """Class representing a polymer."""

    def __init__(self, name, **props):
        super().__init__(name, **props)

    # def polymerize(self, structs):
    #     """
    #     Polymerize the given structures.

    #     Parameters:
    #     - structs: List of structures to polymerize.
    #     """
    #     # Implement polymerization logic
    #     # add atom and bond one by one

    #     for struct in structs:

    #         ports = struct["ports"]
    #         deletes = [d for p in ports for d in p.delete]

    #         for atom in struct["atoms"]:
    #             if atom not in deletes:
    #                 self["atoms"].add(atom)

    #         for bond in struct["bonds"]:
    #             if bond.itom not in deletes and bond.jtom not in deletes:
    #                 self.def_bond(bond.itom, bond.jtom, **bond)

    #         for port in ports:
    #             self.def_bond(port.this, port.that)

    #     return self
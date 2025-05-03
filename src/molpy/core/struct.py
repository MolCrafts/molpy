from copy import deepcopy
from molpy.op import rotate_by_rodrigues
from typing import Callable, Generic, Sequence, TypeVar
import numpy as np
from nesteddict import ArrayDict

T = TypeVar("entity")

class Entity(dict):
    """Base class representing a general entity with dictionary-like behavior.""" 

    def __call__(self):
        """Return a copy of the entity."""
        return self.copy()

    def copy(self):
        """Create a deep copy of the entity."""
        return deepcopy(self)

    def __hash__(self):
        """Return a unique hash for the atom."""
        return id(self)
    
    def __eq__(self, other):
        return self is other
    
    def __ne__(self, other):
        """Check if two entities are not equal."""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Compare entities based on their IDs."""
        return id(self) < id(other)
    
    def to_dict(self):
        return dict(self)
    

class SpatialMixin:
    """Mixin class for spatial operations on entities."""

    xyz: np.ndarray

    def distance_to(self, other):
        """Calculate the Euclidean distance to another entity."""
        return np.linalg.norm(self.xyz - other.xyz)

    def translate(self, vector: np.ndarray):
        """Translate the entity by a given vector."""
        self.xyz = np.add(self.xyz, vector)

    def rotate(self, theta, axis):
        """Rotate the entity around a given axis by a specified angle."""
        self.xyz = rotate_by_rodrigues(self.xyz, axis, theta)

    def reflact(self, axis: np.ndarray):
        """Reflect the entity across a given axis."""
        self.xyz = np.dot(self.xyz, np.eye(3) - 2 * np.outer(axis, axis))


class Atom(Entity, SpatialMixin):
    """Class representing an atom."""

    def __init__(self, **props):
        """
        Initialize an atom with a name.

        Parameters:
        - name: Name of the atom.
        """
        super().__init__(**props)

    def __repr__(self):
        """Return a string representation of the atom."""
        return f"<Atom {self['name']}>"
    
    @property
    def xyz(self):
        return np.array(self["xyz"], dtype=float)
    
    @xyz.setter
    def xyz(self, value):
        assert len(value) == 3, "xyz must be a 3D vector"
        self["xyz"] = np.ndarray(value)

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
        return sum(
            [hash(atom) for atom in self._atoms]
        ) + hash(self.__class__.__name__)


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


class Entities(Generic[T]):
    """Class representing a collection of entities."""

    def __init__(self, entities: list[T] = []):
        self._data = list(entities)

    def add(self, entity):
        """Add an entity to the collection."""
        self._data.append(entity)
        return entity

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
        self._data.extend(entities)

    def __iter__(self):
        """Return an iterator over the entities."""
        return iter(self._data)

    def __getitem__(self, key: int):
        """Get an entity by its index."""
        return self._data[key]
    
    def __repr__(self):
        """Return a string representation of the collection."""
        return f"<Entities: {len(self._data)} entities>"


class HierarchicalMixin(Generic[T]):
    """Mixin class for hierarchical operations on entities."""


    def add_child(self, entity: T):
        """Add a sub-entity to the collection."""
        self.childern.add(entity)
        return self
    
    def reduce(self, func: Callable[[T], T] = lambda x: x) -> list[T]:
        """Reduce the collection of entities using a function."""
        return [func(entity) for entity in self.childern]

class AllAtomStructMixin:
    """Mixin class for structures containing all atoms."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["atoms"] = Entities()
        self["bonds"] = Entities()
        self["angles"] = Entities()
        self["dihedrals"] = Entities()
        self["impropers"] = Entities()



class Struct(Entity, SpatialMixin, HierarchicalMixin["Struct"]):
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

        # TODO: move atoms/bonds etc. to a mixin class
        super().__init__(
            **{
                "name": name,
                "atoms": Entities(atoms),
                "bonds": Entities(
                    [
                        (
                            Bond(atoms[bond[0]], atoms[bond[1]])
                            if isinstance(bond, tuple)
                            else bond
                        )
                        for bond in bonds
                    ]
                ),
                "angles": Entities(angles),
                "dihedrals": Entities(dihedrals),
                "impropers": Entities(impropers),
            }
        )
        self.childern = Entities()

    @property
    def atoms(self):
        """Get the atoms in the structure."""
        return self["atoms"]
    
    def get_atoms(self):
        """Get all atoms in the structure."""
        return self["atoms"]

    @property
    def bonds(self):
        """Get the bonds in the structure."""
        return self["bonds"]
    
    def get_bonds(self):
        """Get all bonds in the structure."""
        return self["bonds"]
    
    @property
    def angles(self):
        """Get the angles in the structure."""
        return self["angles"]
    
    def get_angles(self):
        """Get all angles in the structure."""
        return self["angles"]
    
    @property
    def dihedrals(self):
        """Get the dihedrals in the structure."""
        return self["dihedrals"]
    
    def get_dihedrals(self):
        """Get all dihedrals in the structure."""
        return self["dihedrals"]
    
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

        # Step 1: Deep-copy atoms
        new = Struct()
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

        return new

    def add_atom(self, **props):
        """
        Add an atom to the structure.

        Parameters:
        - props: Properties of the atom.
        """
        return self["atoms"].add(Atom(**props))
    
    def add_atoms(self, atoms: Sequence[Atom]):
        """
        Add multiple atoms to the structure.

        Parameters:
        - atoms: List of atoms to add.
        """
        self["atoms"].extend(atoms)
        return self

    def add_bond(self, itom, jtom, **kwargs):
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
        
        return self["bonds"].add(Bond(itom, jtom, **kwargs))
    
    def add_bonds(self, bonds: Sequence[Bond]):
        """
        Add multiple bonds to the structure.

        Parameters:
        - bonds: List of bonds to add.
        """
        self["bonds"].extend(bonds)
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
            atom_name_idx_mapping[atom] = i

        frame["atoms"] = ArrayDict.from_dicts(
            [atom.to_dict() for atom in self["atoms"]]
        )
        if "bonds" in self and len(self["bonds"]) > 0:
            bdicts = []
            for i, bond in enumerate(self["bonds"]):
                bdict = bond.to_dict()
                bdict["id"] = i
                bdict["i"] = atom_name_idx_mapping[bond.itom]
                bdict["j"] = atom_name_idx_mapping[bond.jtom]
                bdicts.append(bdict)
            frame["bonds"] = ArrayDict.from_dicts(bdicts)

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

    def get_topology(self, attrs:list[str] = []):
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
        self.add_child(struct)
        self.add_atoms(struct.atoms)
        self.add_bonds(struct.bonds)
        return self

    def calc_angles(self, topology):
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
    
    def calc_dihedrals(self, topology):
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

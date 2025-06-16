"""
Core molecular structure classes for the molpy framework.

This module provides the fundamental building blocks for molecular modeling:
- Entity: Base class with dictionary-like behavior
- Spatial operations for atoms and structures
- Topological entities: bonds, angles, dihedrals
- Hierarchical structure management
- Collections and containers
"""

from collections import UserDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Union, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from molpy.op import rotate_by_rodrigues


class Entity(UserDict):
    """
    Base class representing a general entity with dictionary-like behavior.
    
    Provides flexible storage of attributes using dictionary semantics,
    along with cloning and comparison functionality.
    """

    def __call__(self, **modify):
        """
        Return a copy of the entity with optional modifications.
        
        Args:
            **modify: Key-value pairs to modify in the copy
            
        Returns:
            A new Entity instance with modifications applied
        """
        return self.clone(**modify)

    def clone(self, **modify):
        """
        Create a deep copy of the entity with optional modifications.
        
        Args:
            **modify: Key-value pairs to modify in the copy
            
        Returns:
            A new Entity instance
        """
        ins = deepcopy(self)
        for k, v in modify.items():
            ins[k] = v
        return ins

    def __hash__(self) -> int:
        """Return a unique hash for the entity based on object identity."""
        return id(self)

    def __eq__(self, other) -> bool:
        """Check equality based on object identity."""
        return self is other

    def __lt__(self, other) -> bool:
        """Compare entities based on their memory addresses."""
        return id(self) < id(other)

    def to_dict(self) -> dict:
        """Convert entity to a standard dictionary."""
        return dict(self)

    def keys(self):
        """Return the keys of the entity."""
        return self.data.keys()


class SpatialMixin(ABC):
    """
    Abstract mixin class providing spatial operations for entities.
    
    Defines the interface for entities that have spatial coordinates
    and can perform geometric transformations.
    """
    
    @property
    @abstractmethod
    def xyz(self) -> np.ndarray:
        """Get the xyz coordinates as a numpy array."""
        pass
    
    @xyz.setter
    @abstractmethod
    def xyz(self, value: ArrayLike) -> None:
        """Set the xyz coordinates from an array-like object."""
        pass

    def distance_to(self, other: "SpatialMixin") -> float:
        """
        Calculate the Euclidean distance to another spatial entity.
        
        Args:
            other: Another spatial entity
            
        Returns:
            Distance as a float
        """
        return float(np.linalg.norm(self.xyz - other.xyz))

    def move(self, vector: ArrayLike):
        """
        Move the entity by a given vector.
        
        Args:
            vector: Translation vector
            
        Returns:
            Self for method chaining
        """
        self.xyz = np.add(self.xyz, vector)
        return self

    def rotate(self, theta: float, axis: ArrayLike):
        """
        Rotate the entity around a given axis by a specified angle.
        
        Args:
            theta: Rotation angle in radians
            axis: Rotation axis vector
            
        Returns:
            Self for method chaining
        """
        self.xyz = rotate_by_rodrigues(self.xyz, axis, theta)
        return self

    def reflect(self, axis: ArrayLike):
        """
        Reflect the entity across a given axis.
        
        Args:
            axis: Reflection axis vector (normalized)
            
        Returns:
            Self for method chaining
        """
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize axis
        self.xyz = np.dot(self.xyz, np.eye(3) - 2 * np.outer(axis, axis))
        return self


class Atom(Entity, SpatialMixin):
    """
    Class representing an atom with spatial coordinates.
    
    Combines Entity's dictionary behavior with spatial operations.
    """

    def __init__(self, name: str = "", xyz: Optional[ArrayLike] = None, **kwargs):
        """
        Initialize an atom.
        
        Args:
            name: Atom name/symbol
            xyz: 3D coordinates
            **kwargs: Additional properties
        """
        super().__init__(name=name, **kwargs)
        if xyz is not None:
            self.xyz = xyz

    def __repr__(self) -> str:
        """Return a string representation of the atom."""
        return f"<Atom {self.get('name', 'unnamed')}>"

    @property
    def xyz(self) -> np.ndarray:
        """Get the xyz coordinates as a numpy array."""
        return np.array(self.get("xyz", [0.0, 0.0, 0.0]), dtype=float)

    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """Set the xyz coordinates."""
        value = np.asarray(value, dtype=float)
        if value.shape != (3,):
            raise ValueError("xyz must be a 3D vector")
        self["xyz"] = value.tolist()  # Store as list for easier comparison

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

    def __hash__(self) -> int:
        """Return a hash based on constituent atoms and class."""
        return hash(self._atoms) + hash(self.__class__.__name__)

    @property
    def atoms(self):
        """Get the atoms involved in the entity."""
        return self._atoms


class Bond(ManyBody):
    """Class representing a bond between two atoms."""

    def __init__(self, atom1=None, atom2=None, itom=None, jtom=None, **kwargs):
        """
        Initialize a bond between two atoms.

        Args:
            atom1: First atom in the bond
            atom2: Second atom in the bond
            itom: Alternative name for first atom (for backward compatibility)
            jtom: Alternative name for second atom (for backward compatibility)
            **kwargs: Additional properties (e.g., bond_type, length)
        """
        # Handle backward compatibility with itom/jtom naming
        if itom is not None:
            atom1 = itom
        if jtom is not None:
            atom2 = jtom
            
        if atom1 is None or atom2 is None:
            raise ValueError("Must provide both atoms for the bond")
        if atom1 is atom2:
            raise ValueError("Cannot create bond between same atom")
        # Sort atoms for consistent ordering
        sorted_atoms = sorted([atom1, atom2], key=lambda a: id(a))
        super().__init__(*sorted_atoms, **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the bond."""
        return self._atoms[0]

    @property
    def atom2(self):
        """Get the second atom in the bond."""
        return self._atoms[1]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for atom2."""
        return self.atom2

    def __repr__(self) -> str:
        """Return a string representation of the bond."""
        return f"<Bond: {self.atom1.name}-{self.atom2.name}>"

    def __eq__(self, other) -> bool:
        """Check equality based on the atoms in the bond."""
        if not isinstance(other, Bond):
            return False
        return (self.atom1 is other.atom1 and self.atom2 is other.atom2) or \
               (self.atom1 is other.atom2 and self.atom2 is other.atom1)

    @property
    def length(self) -> float:
        """Calculate the bond length."""
        return float(np.linalg.norm(self.atom1.xyz - self.atom2.xyz))


class Angle(ManyBody):
    """Class representing an angle between three atoms."""

    def __init__(self, atom1: Atom, vertex: Atom, atom2: Atom, **kwargs):
        """
        Initialize an angle.

        Args:
            atom1: First atom in the angle
            vertex: Vertex atom (center of angle)
            atom2: Third atom in the angle
            **kwargs: Additional properties
        """
        if len({id(atom1), id(vertex), id(atom2)}) != 3:
            raise ValueError("All three atoms must be different")
        # Sort end atoms for consistent ordering
        end_atoms = sorted([atom1, atom2], key=lambda a: id(a))
        super().__init__(end_atoms[0], vertex, end_atoms[1], **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the angle."""
        return self._atoms[0]

    @property
    def vertex(self):
        """Get the vertex atom (center of angle)."""
        return self._atoms[1]

    @property
    def atom2(self):
        """Get the third atom in the angle."""
        return self._atoms[2]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for vertex."""
        return self.vertex

    @property
    def ktom(self):
        """Alias for atom2."""
        return self.atom2

    def __repr__(self) -> str:
        """Return a string representation of the angle."""
        return f"<Angle: {self.atom1.name}-{self.vertex.name}-{self.atom2.name}>"

    @property
    def value(self) -> float:
        """Calculate the angle value in radians."""
        v1 = self.atom1.xyz - self.vertex.xyz
        v2 = self.atom2.xyz - self.vertex.xyz
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def to_dict(self) -> dict:
        """Convert the angle to a dictionary."""
        result = super().to_dict()
        if hasattr(self.atom1, 'get') and 'id' in self.atom1:
            result.update({
                'i': self.atom1["id"],
                'j': self.vertex["id"], 
                'k': self.atom2["id"]
            })
        return result


class Dihedral(ManyBody):
    """Class representing a dihedral angle between four atoms."""

    def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom, **kwargs):
        """
        Initialize a dihedral angle.

        Args:
            atom1: First atom in the dihedral
            atom2: Second atom in the dihedral
            atom3: Third atom in the dihedral  
            atom4: Fourth atom in the dihedral
            **kwargs: Additional properties
        """
        if len({id(atom1), id(atom2), id(atom3), id(atom4)}) != 4:
            raise ValueError("All four atoms must be different")
        
        # Ensure consistent ordering based on central bond
        if id(atom2) > id(atom3):
            atom1, atom2, atom3, atom4 = atom4, atom3, atom2, atom1
            
        super().__init__(atom1, atom2, atom3, atom4, **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the dihedral."""
        return self._atoms[0]

    @property
    def atom2(self):
        """Get the second atom in the dihedral."""
        return self._atoms[1]

    @property
    def atom3(self):
        """Get the third atom in the dihedral."""
        return self._atoms[2]

    @property
    def atom4(self):
        """Get the fourth atom in the dihedral."""
        return self._atoms[3]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for atom2."""
        return self.atom2

    @property
    def ktom(self):
        """Alias for atom3."""
        return self.atom3

    @property
    def ltom(self):
        """Alias for atom4."""
        return self.atom4

    def __repr__(self) -> str:
        """Return a string representation of the dihedral."""
        return f"<Dihedral: {self.atom1.name}-{self.atom2.name}-{self.atom3.name}-{self.atom4.name}>"

    @property
    def value(self) -> float:
        """Calculate the dihedral angle value in radians."""
        # Vectors along the bonds
        b1 = self.atom2.xyz - self.atom1.xyz
        b2 = self.atom3.xyz - self.atom2.xyz
        b3 = self.atom4.xyz - self.atom3.xyz
        
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

    def to_dict(self) -> dict:
        """Convert the dihedral to a dictionary."""
        result = super().to_dict()
        if all(hasattr(atom, 'get') and 'id' in atom for atom in self._atoms):
            result.update({
                'i': self.atom1["id"],
                'j': self.atom2["id"],
                'k': self.atom3["id"],
                'l': self.atom4["id"]
            })
        return result


class Improper(Dihedral):
    """
    Class representing an improper dihedral angle.
    
    An improper dihedral is used to maintain planarity in molecular structures.
    """

    def __init__(self, center: Atom, atom1: Atom, atom2: Atom, atom3: Atom, **kwargs):
        """
        Initialize an improper dihedral angle.

        Args:
            center: Central atom
            atom1: First bonded atom
            atom2: Second bonded atom 
            atom3: Third bonded atom
            **kwargs: Additional properties
        """
        # Sort the three bonded atoms for consistent ordering
        bonded_atoms = sorted([atom1, atom2, atom3], key=lambda a: id(a))
        super().__init__(center, bonded_atoms[0], bonded_atoms[1], bonded_atoms[2], **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the improper dihedral."""
        return f"<Improper: {self.atom1.name}({self.atom2.name},{self.atom3.name},{self.atom4.name})>"


class Entities:
    """
    Container for storing and managing collections of molecular entities.
    
    Provides a list-like interface with additional functionality for
    finding entities by conditions and names.
    """

    def __init__(self, entities=None):
        """
        Initialize the container.
        
        Args:
            entities: Optional initial entities to add
        """
        self._data = list(entities) if entities is not None else []

    def add(self, entity):
        """
        Add an entity to the collection.
        
        Args:
            entity: Entity to add
            
        Returns:
            The added entity
        """
        self._data.append(entity)
        return entity

    def remove(self, entity):
        """
        Remove an entity from the collection.
        
        Args:
            entity: Entity instance, index, or name to remove
        """
        if isinstance(entity, int):
            self._data.pop(entity)
        elif isinstance(entity, str):
            # Find by name and remove
            for i, e in enumerate(self._data):
                if hasattr(e, 'get') and e.get("name") == entity:
                    self._data.pop(i)
                    break
        else:
            self._data.remove(entity)

    def get_by(self, condition: Callable):
        """
        Get an entity based on a condition.

        Args:
            condition: Function that takes an entity and returns a boolean

        Returns:
            The first entity that satisfies the condition, or None
        """
        return next((entity for entity in self._data if condition(entity)), None)

    def __len__(self) -> int:
        """Return the number of entities in the collection."""
        return len(self._data)

    def extend(self, entities) -> None:
        """
        Extend the collection with multiple entities.
        
        Args:
            entities: Sequence of entities to add
        """
        self._data.extend(entities)

    def __iter__(self):
        """Return an iterator over the entities."""
        return iter(self._data)

    def __getitem__(self, key):
        """
        Get an entity by its index, name, or multiple entities.
        
        Args:
            key: Index, slice, name, or sequence of indices/names
            
        Returns:
            Entity or list of entities
        """
        if isinstance(key, (int, slice)):
            return self._data[key]
        elif isinstance(key, str):
            # Find by name
            for entity in self._data:
                if hasattr(entity, 'get') and entity.get("name") == key:
                    return entity
            return None
        elif isinstance(key, (list, tuple)):
            result = []
            for k in key:
                if isinstance(k, int):
                    result.append(self._data[k])
                elif isinstance(k, str):
                    found = None
                    for entity in self._data:
                        if hasattr(entity, 'get') and entity.get("name") == k:
                            found = entity
                            break
                    result.append(found)
            return result
        return None

    def __repr__(self) -> str:
        """Return a string representation of the collection."""
        return f"<Entities: {len(self._data)} items>"


class HierarchyMixin:
    """
    Mixin class providing hierarchical structure management capabilities.
    
    Enables parent-child relationships and tree traversal operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize hierarchy attributes."""
        super().__init__(*args, **kwargs)
        self._parent = None
        self._children = []
    
    @property
    def parent(self):
        """Get the parent structure."""
        return self._parent
    
    @property 
    def children(self):
        """Get a copy of the child structures list."""
        return self._children.copy()
    
    @property
    def root(self):
        """Get the root structure of the hierarchy."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root structure."""
        return self._parent is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf structure (no children)."""
        return len(self._children) == 0
    
    @property
    def depth(self) -> int:
        """Get the depth of this structure in the hierarchy (root is 0)."""
        depth = 0
        current = self
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth
    
    def add_child(self, child):
        """
        Add a child structure.
        
        Args:
            child: The child structure to add
            
        Returns:
            The added child
        """
        if child in self._children:
            return child
            
        # Remove from previous parent if any
        if hasattr(child, '_parent') and child._parent is not None:
            child._parent.remove_child(child)
            
        # Add as child
        self._children.append(child)
        if hasattr(child, '_parent'):
            child._parent = self
        
        return child
    
    def remove_child(self, child):
        """
        Remove a child structure.
        
        Args:
            child: The child structure to remove
        """
        if child in self._children:
            self._children.remove(child)
            if hasattr(child, '_parent'):
                child._parent = None
    
    def get_descendants(self):
        """Get all descendant structures (children, grandchildren, etc.)."""
        descendants = []
        for child in self._children:
            descendants.append(child)
            if hasattr(child, 'get_descendants'):
                descendants.extend(child.get_descendants())
        return descendants
    
    def get_ancestors(self):
        """Get all ancestor structures (parent, grandparent, etc.)."""
        ancestors = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = getattr(current, '_parent', None)
        return ancestors
    
    def get_siblings(self):
        """Get all sibling structures."""
        if self._parent is None:
            return []
        return [child for child in getattr(self._parent, '_children', []) if child is not self]
    
    def traverse_dfs(self, visit_func):
        """
        Traverse the hierarchy using depth-first search.
        
        Args:
            visit_func: Function to call for each node
        """
        visit_func(self)
        for child in self._children:
            if hasattr(child, 'traverse_dfs'):
                child.traverse_dfs(visit_func)
    
    def traverse_bfs(self, visit_func):
        """
        Traverse the hierarchy using breadth-first search.
        
        Args:
            visit_func: Function to call for each node
        """
        from collections import deque
        queue = deque([self])
        
        while queue:
            current = queue.popleft()
            visit_func(current)
            queue.extend(current._children)
    
    def find_by_name(self, name: str):
        """
        Find a structure by name in the hierarchy.
        
        Args:
            name: Name to search for
            
        Returns:
            The first structure with the given name, or None
        """
        # Check if this object has a 'get' method and matches the name
        try:
            get_method = getattr(self, 'get', None)
            if get_method and get_method('name') == name:
                return self
        except (AttributeError, KeyError, TypeError):
            pass
            
        for child in self._children:
            if hasattr(child, 'find_by_name'):
                result = child.find_by_name(name)
                if result is not None:
                    return result
        return None
    
    def get_hierarchy_info(self) -> dict:
        """Get information about the hierarchical structure."""
        parent_name = None
        if self._parent and hasattr(self._parent, 'get'):
            parent_name = self._parent.get('name')
            
        children_names = []
        for child in self._children:
            if hasattr(child, 'get'):
                children_names.append(child.get('name'))
            
        return {
            'depth': self.depth,
            'is_root': self.is_root,
            'is_leaf': self.is_leaf,
            'num_children': len(self._children),
            'num_descendants': len(self.get_descendants()),
            'parent_name': parent_name,
            'children_names': children_names
        }


class Struct(Entity):
    """
    Base class for molecular structures.
    
    Provides fundamental structure functionality without requiring
    spatial or atomic properties.
    """

    def __init__(self, name: str = "", **props):
        """
        Initialize a molecular structure.
        
        Args:
            name: Structure name
            **props: Additional properties
        """
        super().__init__(name=name, **props)

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return f"<Struct: {self.get('name', 'unnamed')}>"


class AtomicStructure(Struct, SpatialMixin, HierarchyMixin):
    """
    Structure containing atoms, bonds, angles, and dihedrals.
    
    Combines basic structure functionality with spatial operations
    and hierarchical management.
    """

    def __init__(self, name: str = "", **props):
        """
        Initialize an atomic structure.
        
        Args:
            name: Structure name
            **props: Additional properties
        """
        # Initialize hierarchy first
        if not hasattr(self, '_parent'):
            self._parent = None
        if not hasattr(self, '_children'):
            self._children = []
            
        super().__init__(name=name, **props)
        self["atoms"] = Entities()
        self["bonds"] = Entities()
        self["angles"] = Entities()
        self["dihedrals"] = Entities()

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return f"<AtomicStructure: {len(self.atoms)} atoms>"

    @property
    def atoms(self):
        """Get the atoms in the structure."""
        return self["atoms"]

    @property
    def bonds(self):
        """Get the bonds in the structure."""
        return self["bonds"]

    @property
    def angles(self):
        """Get the angles in the structure."""
        return self["angles"]

    @property
    def dihedrals(self):
        """Get the dihedrals in the structure."""
        return self["dihedrals"]

    @property
    def xyz(self) -> np.ndarray:
        """Get the coordinates of all atoms in the structure."""
        if len(self.atoms) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([atom.xyz for atom in self.atoms], dtype=float)

    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """
        Set the coordinates of all atoms in the structure.
        
        Args:
            value: Array of shape (n_atoms, 3) with new coordinates
        """
        value = np.asarray(value)
        if value.shape != (len(self.atoms), 3):
            raise ValueError(f"xyz must have shape ({len(self.atoms)}, 3), got {value.shape}")
        
        for i, atom in enumerate(self.atoms):
            atom.xyz = value[i]

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

    def def_bond(self, atom1, atom2, **kwargs):
        """
        Create and add a bond between two atoms.
        
        Args:
            atom1: First atom or its index
            atom2: Second atom or its index
            **kwargs: Bond properties
            
        Returns:
            The created bond
        """
        if isinstance(atom1, int):
            atom1 = self.atoms[atom1]
        if isinstance(atom2, int):
            atom2 = self.atoms[atom2]
            
        if not isinstance(atom1, Atom) or not isinstance(atom2, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")
            
        bond = Bond(atom1, atom2, **kwargs)
        return self.add_bond(bond)

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
        self.atoms.remove(atom)

    def remove_bond(self, bond):
        """
        Remove a bond from the structure.
        
        Args:
            bond: Bond instance or tuple of atoms
        """
        if isinstance(bond, tuple):
            # Find and remove bond between these atoms
            atom1, atom2 = bond
            target_bond = self.bonds.get_by(
                lambda b: (b.atom1 is atom1 and b.atom2 is atom2) or
                         (b.atom1 is atom2 and b.atom2 is atom1)
            )
            if target_bond:
                self.bonds.remove(target_bond)
        else:
            self.bonds.remove(bond)

    def get_atom_by(self, condition: Callable):
        """
        Get an atom based on a condition.

        Args:
            condition: Function that takes an atom and returns a boolean

        Returns:
            The first atom that satisfies the condition, or None
        """
        return self.atoms.get_by(condition)

    def get_topology(self, attrs=None):
        """
        Get the topology of the structure.

        Args:
            attrs: List of atom attributes to include

        Returns:
            A Topology object representing the structure's topology.
        """
        # Create a simple topology-like object for testing
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self.atoms)}
        atom_attrs = {}

        if attrs:
            for attr in attrs:
                atom_attrs[attr] = [atom[attr] for atom in atoms]
        topo.add_atoms(len(atoms), **atom_attrs)
        bonds = self["bonds"]
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in bonds])
        return topo

    def add_struct(self, struct):
        """
        Add another structure to the current structure.
        
        This merges the atoms and bonds from the other structure.
        Also establishes a parent-child hierarchy relationship.

        Args:
            struct: The structure to add
            
        Returns:
            Self for method chaining
        """
        # Add hierarchical relationship
        self.add_child(struct)
        
        # Merge atoms and topological entities
        self.add_atoms(struct.atoms)
        self.add_bonds(struct.bonds)
        self.add_angles(struct.angles)
        self.add_dihedrals(struct.dihedrals)
        
        return self

    @classmethod
    def concat(cls, name: str, structs):
        """
        Concatenate multiple structures into a new structure.

        Args:
            name: Name for the new structure
            structs: Sequence of structures to concatenate
            
        Returns:
            New structure containing all input structures
        """
        result = cls(name)
        for struct in structs:
            result.add_struct(struct)
        return result


class MolecularStructure(AtomicStructure):
    """
    Complete molecular structure implementation.
    
    This is the main class for representing complete molecular systems
    with atoms, bonds, angles, and dihedrals.
    """

    def __repr__(self) -> str:
        """Return a string representation of the molecular structure."""
        return f"<MolecularStructure: {len(self.atoms)} atoms, {len(self.bonds)} bonds>"

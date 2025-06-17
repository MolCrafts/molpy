"""
Core protocols and mixins for the molpy framework.

This module provides the fundamental protocols and abstract base classes
used throughout the molecular modeling framework.
"""

from collections import UserDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Union, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ..op import rotate_by_rodrigues

class Entity(UserDict):
    """
    Base class representing a general entity with dictionary-like behavior.
    
    Provides flexible storage of attributes using dictionary semantics,
    along with cloning and comparison functionality.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize Entity with support for mixins.
        
        This method ensures proper initialization of all mixins in the MRO
        by calling super().__init__() to continue the initialization chain.
        
        Args:
            *args: Positional arguments for UserDict
            **kwargs: Keyword arguments for UserDict and mixins
        """
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_parent'):
            self._parent = None
        if not hasattr(self, '_children'):
            self._children = []

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

    def get_by_name(self, name):
        """Get an entity by its 'name' attribute."""
        return self[ name ]

    def filter_by(self, condition: Callable):
        """Return a list of all entities matching the condition."""
        return [entity for entity in self._data if condition(entity)]

    def to_list(self):
        """Return all entities as a list."""
        return list(self._data)

    def clear(self):
        """Remove all entities from the collection."""
        self._data.clear()

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
        Calculate the distance to another spatial entity.
        
        Args:
            other: Another spatial entity
            
        Returns:
            The Euclidean distance between the two entities
        """
        return float(np.linalg.norm(self.xyz - other.xyz))

    def move(self, vector: ArrayLike) -> None:
        """
        Translate the entity by a given vector.
        
        Args:
            vector: Translation vector
        """
        self.xyz = self.xyz + np.array(vector)

    def scale(self, factor: Union[float, ArrayLike], origin: Optional[ArrayLike] = None) -> None:
        """
        Scale the entity by a given factor around an origin.
        
        Args:
            factor: Scaling factor (scalar or per-axis)
            origin: Origin point for scaling (defaults to current position)
        """
        if origin is None:
            origin = self.xyz
        else:
            origin = np.array(origin)
        
        factor = np.array(factor)
        self.xyz = origin + factor * (self.xyz - origin)

    def rotate(self, axis: ArrayLike, angle: float, origin: Optional[ArrayLike] = None) -> None:
        """
        Rotate the entity around an axis by a given angle.
        
        Args:
            axis: Rotation axis vector
            angle: Rotation angle in radians
            origin: Origin point for rotation (defaults to [0,0,0])
        """
        if origin is None:
            origin = np.zeros(3)
        else:
            origin = np.array(origin)
        axis = np.array(axis, dtype=float)
        translated = self.xyz - origin
        rotated = rotate_by_rodrigues(translated, axis, angle)
        if isinstance(rotated, np.ndarray) and rotated.shape == (1, 3):
            rotated = rotated[0]
        rotated = np.where(np.abs(rotated) < 1e-12, 0.0, rotated)
        self.xyz = rotated + origin


class HierarchyMixin:
    """
    Mixin class providing hierarchical structure functionality.
    
    Allows entities to have parent-child relationships and provides
    methods for navigating and manipulating the hierarchy.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_parent'):
            self._parent = None
        if not hasattr(self, '_children'):
            self._children = []
    
    @property
    def parent(self):
        """Get the parent entity."""
        return self._parent
    
    @property
    def children(self):
        """Get the list of child entities."""
        return self._children.copy()  # Return a copy to prevent external modification
    
    @property
    def is_root(self) -> bool:
        """Check if this entity is a root (has no parent)."""
        return self._parent is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this entity is a leaf (has no children)."""
        return len(self._children) == 0
    
    @property
    def depth(self) -> int:
        """Get the depth of this entity in the hierarchy (root = 0)."""
        if self.is_root or self.parent is None:
            return 0
        return self.parent.depth + 1
    
    def add_child(self, child: "HierarchyMixin") -> None:
        """
        Add a child entity to this entity.
        
        Args:
            child: The child entity to add
        """
        if child not in self._children:
            self._children.append(child)
            child._parent = self
    
    def remove_child(self, child: "HierarchyMixin") -> None:
        """
        Remove a child entity from this entity.
        
        Args:
            child: The child entity to remove
        """
        if child in self._children:
            self._children.remove(child)
            child._parent = None
    
    def get_root(self) -> "HierarchyMixin":
        """
        Get the root entity of the hierarchy.
        
        Returns:
            The root entity
        """
        if self.is_root or self.parent is None:
            return self
        return self.parent.get_root()
    
    def get_ancestors(self) -> list["HierarchyMixin"]:
        """
        Get all ancestors of this entity (from parent to root).
        
        Returns:
            List of ancestor entities
        """
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self) -> list["HierarchyMixin"]:
        """
        Get all descendants of this entity.
        
        Returns:
            List of descendant entities
        """
        descendants = []
        for child in self._children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def find_by_condition(self, condition: Callable[["HierarchyMixin"], bool]):
        """
        Find the first descendant (including self) that satisfies a condition.
        
        Args:
            condition: A function that takes an entity and returns True if it matches
            
        Returns:
            The first matching entity, or None
        """
        if condition(self):
            return self
        for child in self._children:
            found = child.find_by_condition(condition)
            if found:
                return found
        return None


class IdentifierMixin:
    """
    Mixin class providing identifier functionality for entities.
    
    Provides consistent identifier management with optional auto-generation.
    """
    
    _id_counter = 0
    
    def __init__(self, id: Optional[Union[int, str]] = None, *args, **kwargs):
        """
        Initialize with an identifier.
        
        Args:
            id: Optional identifier (auto-generated if None)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        if id is None:
            IdentifierMixin._id_counter += 1
            self._id = IdentifierMixin._id_counter
        else:
            self._id = id
    
    @property
    def id(self) -> Union[int, str]:
        """Get the entity identifier."""
        return self._id
    
    @id.setter
    def id(self, value: Union[int, str]) -> None:
        """Set the entity identifier."""
        self._id = value
    
    def __str__(self) -> str:
        """Return string representation using identifier."""
        return f"{self.__class__.__name__}({self.id})"
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"{self.__class__.__name__}(id={self.id})"

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

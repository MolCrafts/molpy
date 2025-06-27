"""
Core protocols and mixins for the molpy framework.

This module provides the fundamental protocols and abstract base classes
used throughout the molecular modeling framework.
"""

from collections import UserDict
from copy import deepcopy
from typing import Callable, TypeVar, Generic, Iterable, Type

T = TypeVar("T", bound="Entity")

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
        """
        Convert the entity and all nested components to a dictionary,
        including class path for deserialization.
        """
        return {
                key: value
                for key, value in self.data.items()
            }

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstruct the current class from a dictionary.
        """
        obj = cls()
        for key, value in data.items():
            obj[key] = value
        return obj

    def keys(self):
        """Return the keys of the entity."""
        return self.data.keys()


T = TypeVar("T", bound=Entity)

class Entities(Generic[T]):
    """
    Container for storing and managing collections of molecular entities.
    
    Provides a list-like interface with additional functionality for
    finding entities by conditions and names.
    """

    def __init__(self, entities: Iterable[T] | None =None):
        """
        Initialize the container.
        
        Args:
            entities: Optional initial entities to add
        """
        self._data: list[T] = list(entities) if entities is not None else []

    def add(self, entity: T):
        """
        Add an entity to the collection.

        Args:
            entity: Entity to add
        """
        self._data.append(entity)
        return entity

    def remove(self, entity: T):
        """
        Remove an entity from the collection.

        Args:
            entity: Entity instance, index, or name to remove
        """
        self._data.remove(entity)

    def get_by(self, condition: Callable[[T], bool]) -> T | None:
        """
        Get an entity based on a condition.

        Args:
            condition: Function that takes an entity and returns a boolean

        Returns:
            The first entity that satisfies the condition, or None
        """
        return next((entity for entity in self._data if condition(entity)), None)

    def get_by_name(self, name: str) -> T | None:
        """Get an entity by its 'name' attribute."""
        return self.get_by(lambda e: e.get("name") == name)

    def filter_by(self, condition: Callable[[T], bool]) -> list[T]:
        """Return a list of all entities matching the condition."""
        return [entity for entity in self._data if condition(entity)]

    def to_list(self) -> list[T]:
        """Return all entities as a list."""
        return list(self._data)

    def clear(self):
        """Remove all entities from the collection."""
        self._data.clear()

    def __len__(self) -> int:
        """Return the number of entities in the collection."""
        return len(self._data)

    def extend(self, entities: Iterable[T]) -> None:
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
        return self._data[key]

    def __repr__(self) -> str:
        """Return a string representation of the collection."""
        return f"<Entities: {len(self._data)} items>"

class Struct(Entity):
    """
    Base class for molecular structures.
    
    Provides fundamental structure functionality without requiring
    spatial or atomic properties.
    """

    def to_frame(self):
        """
        Convert the structure to a frame representation.
        
        This method should be implemented by subclasses to provide
        a specific frame representation of the structure.
        
        Returns:
            Frame: A frame representation of the structure
        """
        raise NotImplementedError("Subclasses must implement to_frame()")

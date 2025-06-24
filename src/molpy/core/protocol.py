"""
Core protocols and mixins for the molpy framework.

This module provides the fundamental protocols and abstract base classes
used throughout the molecular modeling framework.
"""

from collections import UserDict
from copy import deepcopy
from typing import Callable

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
        Convert entity to a dictionary for serialization.
        
        Includes class information for proper reconstruction and recursively
        converts any nested components that support to_dict.
        
        Returns:
            dict: Dictionary representation including class info and all data
        """
        result = {
            "__class__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        }
        
        # Recursively convert data, calling to_dict on nested components if available
        for key, value in self.data.items():
            if hasattr(value, "to_dict") and callable(value.to_dict):
                result[key] = value.to_dict()
            else:
                result[key] = value
                
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """
        Create Entity from dictionary representation.
        
        Supports recursive reconstruction of nested components that have from_dict methods.
        
        Args:
            data: Dictionary containing entity data
            
        Returns:
            Entity: Reconstructed entity instance
        """
        # Extract class info if present for validation
        class_info = data.get("__class__")
        if class_info and not class_info.endswith(cls.__name__):
            # Allow subclasses but warn about potential mismatches
            pass
            
        # Create new instance
        instance = cls()
        
        # Recursively reconstruct data
        for key, value in data.items():
            if key == "__class__":
                continue
                
            # If value is a dict with class info, try to reconstruct it
            if isinstance(value, dict) and "__class__" in value:
                class_path = value["__class__"]
                try:
                    # Import and get the class
                    module_name, class_name = class_path.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[class_name])
                    target_class = getattr(module, class_name)
                    
                    # Call from_dict if available
                    if hasattr(target_class, "from_dict") and callable(target_class.from_dict):
                        instance[key] = target_class.from_dict(value)
                    else:
                        instance[key] = value
                except (ImportError, AttributeError, ValueError):
                    # Fallback to raw data if reconstruction fails
                    instance[key] = value
            else:
                instance[key] = value
                
        return instance

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

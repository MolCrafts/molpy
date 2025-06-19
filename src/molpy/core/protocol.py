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

    def __call__(self, **kwargs):
        """
        Create a new instance of this structure with optional modifications.
        
        This method enables structures to be used as factory functions,
        creating copies of themselves with potentially modified properties.
        
        For AtomicStructure, this creates a new instance and deep copies all
        atoms, bonds, angles, etc. with new parameters applied.
        
        Args:
            **kwargs: Properties to pass to the constructor and/or override in the instance
            
        Returns:
            A new instance of the same class with copied data
        """
        import copy
        import inspect
        
        # For AtomicStructure and subclasses, we need special handling
        # to avoid double-initialization
        if hasattr(self, 'atoms') and hasattr(self.__class__, '__init__'):
            # Get the __init__ signature of the class
            init_signature = inspect.signature(self.__class__.__init__)
            constructor_kwargs = {}
            modification_kwargs = {}
            
            # Separate constructor arguments from modification arguments
            for key, value in kwargs.items():
                if key in init_signature.parameters:
                    constructor_kwargs[key] = value
                else:
                    modification_kwargs[key] = value
            
            # Use existing name if not specified in kwargs  
            if 'name' not in constructor_kwargs:
                constructor_kwargs['name'] = self.get('name', '')
                
            # Create a new instance with constructor arguments
            new_instance = self.__class__(**constructor_kwargs)
            
            # Deep copy atoms with modifications
            if hasattr(self, 'atoms') and len(self.atoms) > 0:
                # Clear the auto-created atoms/bonds/angles from constructor
                new_instance['atoms'].clear()
                new_instance['bonds'].clear() 
                new_instance['angles'].clear()
                if hasattr(new_instance, 'dihedrals'):
                    new_instance['dihedrals'].clear()
                
                # Deep copy atoms with modifications applied
                for atom in self.atoms:
                    new_atom_data = copy.deepcopy(atom.to_dict())
                    # Apply modifications to atom data
                    for key, value in modification_kwargs.items():
                        new_atom_data[key] = value
                    new_atom = type(atom)(**new_atom_data)
                    new_instance.add_atom(new_atom)
                
                # Deep copy bonds
                if hasattr(self, 'bonds') and len(self.bonds) > 0:
                    atom_mapping = {}  # Map old atoms to new atoms
                    for old_atom, new_atom in zip(self.atoms, new_instance.atoms):
                        atom_mapping[old_atom] = new_atom
                    
                    for bond in self.bonds:
                        new_atom1 = atom_mapping.get(bond.atom1)
                        new_atom2 = atom_mapping.get(bond.atom2)
                        if new_atom1 and new_atom2:
                            bond_data = copy.deepcopy(bond.to_dict())
                            # Remove atom references from bond data
                            bond_data.pop('atoms', None)
                            new_bond = type(bond)(new_atom1, new_atom2, **bond_data)
                            new_instance.add_bond(new_bond)
                
                # Deep copy angles
                if hasattr(self, 'angles') and len(self.angles) > 0:
                    atom_mapping = {}
                    for old_atom, new_atom in zip(self.atoms, new_instance.atoms):
                        atom_mapping[old_atom] = new_atom
                    
                    for angle in self.angles:
                        new_atom1 = atom_mapping.get(angle.itom)
                        new_atom2 = atom_mapping.get(angle.jtom) 
                        new_atom3 = atom_mapping.get(angle.ktom)
                        if new_atom1 and new_atom2 and new_atom3:
                            angle_data = copy.deepcopy(angle.to_dict())
                            # Remove atom references from angle data
                            angle_data.pop('atoms', None)
                            new_angle = type(angle)(new_atom1, new_atom2, new_atom3, **angle_data)
                            new_instance.add_angle(new_angle)
            
            # Copy other non-structural properties
            for key, value in self.items():
                if key not in ['atoms', 'bonds', 'angles', 'dihedrals'] and key not in modification_kwargs:
                    if hasattr(value, 'copy'):
                        new_instance[key] = value.copy()
                    else:
                        new_instance[key] = value
            
            return new_instance
        
        # Original fallback implementation for other classes
        try:
            # Get the __init__ signature of the class
            init_signature = inspect.signature(self.__class__.__init__)
            constructor_kwargs = {}
            remaining_kwargs = {}
            
            # Separate constructor arguments from other properties
            for key, value in kwargs.items():
                if key in init_signature.parameters:
                    constructor_kwargs[key] = value
                else:
                    remaining_kwargs[key] = value
            
            # Create a new instance with constructor arguments
            # Use existing name if not specified in kwargs  
            if 'name' not in constructor_kwargs:
                constructor_kwargs['name'] = self.get('name', '')
                
            new_instance = self.__class__(**constructor_kwargs)
            
        except Exception:
            # Final fallback: create with just name if introspection fails
            new_instance = self.__class__(name=self.get('name', ''))
            remaining_kwargs = kwargs.copy()
        
        # Copy all data from the current instance (deep copy for collections)
        for key, value in self.items():
            if key not in remaining_kwargs:  # Don't override specified kwargs
                if hasattr(value, 'copy'):
                    # For collections/containers, create a shallow copy
                    new_instance[key] = value.copy()
                else:
                    # For simple values, just assign
                    new_instance[key] = value
        
        # Apply any remaining overrides from kwargs
        for key, value in remaining_kwargs.items():
            new_instance[key] = value
            
        return new_instance


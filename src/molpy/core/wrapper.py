"""
Wrapper classes for the molpy framework.

This module provides wrapper classes that can be composed to add functionality 
to molecular entities. Replaces the old mixin-based approach with a more 
flexible composition pattern.
"""

from typing import Callable, Union, Optional, Any
import numpy as np
from numpy.typing import ArrayLike

from ..op import rotate_by_rodrigues


class Wrapper:
    """
    Base class for all wrappers.
    
    Wrappers provide a composable way to add functionality to entities
    without using inheritance mixins.
    """
    
    def __init__(self, wrapped):
        """
        Initialize wrapper with an entity to wrap.
        
        Args:
            wrapped: The entity to wrap
        """
        self._wrapped = wrapped
    
    def unwrap(self):
        """Get the wrapped entity."""
        return self._wrapped
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped entity."""
        return getattr(self._wrapped, name)
    
    def __setattr__(self, name, value):
        """Set attributes on wrapper or delegate to wrapped entity."""
        if name.startswith('_') or name in self.__dict__ or hasattr(type(self), name):
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped, name, value)

    def __getitem__(self, key):
        """Delegate item access to the wrapped entity."""
        return self._wrapped[key]
    
    def __setitem__(self, key, value):
        """Set items on the wrapped entity."""
        self._wrapped[key] = value

    def __contains__(self, key):
        """Check if the wrapped entity contains a key."""
        return key in self._wrapped


class SpatialWrapper(Wrapper):
    """
    Wrapper class providing spatial operations for entities.
    
    Defines spatial functionality for entities that have xyz coordinates
    and can perform geometric transformations.
    """
    
    @property
    def xyz(self) -> np.ndarray:
        """Get the xyz coordinates as a numpy array."""
        # Handle Atomistic specifically
        if hasattr(self._wrapped, 'atoms') and hasattr(self._wrapped, '__getitem__'):
            # This is likely an Atomistic
            try:
                coords = self._wrapped["atoms", "xyz"]
                # Filter out None values and convert to numpy array
                valid_coords = [coord for coord in coords if coord is not None]
                if not valid_coords:
                    return np.array([]).reshape(0, 3)
                return np.array(valid_coords, dtype=float)
            except (KeyError, TypeError):
                # Fallback: try to get xyz from individual atoms
                if hasattr(self._wrapped, 'atoms') and len(self._wrapped.atoms) > 0:
                    coords = []
                    for atom in self._wrapped.atoms:
                        if "xyz" in atom:
                            coords.append(atom["xyz"])
                        else:
                            coords.append([0.0, 0.0, 0.0])
                    return np.array(coords, dtype=float) if coords else np.array([]).reshape(0, 3)
                return np.array([]).reshape(0, 3)
        
        # Handle single entity (Atom, etc.)
        if hasattr(self._wrapped, 'xyz'):
            return self._wrapped.xyz
        else:
            return np.array(self._wrapped.get("xyz", [0.0, 0.0, 0.0]), dtype=float)
    
    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """Set the xyz coordinates from an array-like object."""
        # Handle Atomistic specifically
        if hasattr(self._wrapped, 'atoms') and hasattr(self._wrapped, '__getitem__'):
            # This is likely an Atomistic
            value = np.asarray(value, dtype=float)
            
            # Check if we have the right number of atoms
            n_atoms = len(self._wrapped.atoms)
            if n_atoms == 0:
                return  # Nothing to set
                
            # Handle different input shapes
            if value.ndim == 1 and len(value) == 3:
                # Single coordinate - broadcast to all atoms
                coords = [value.tolist()] * n_atoms
            elif value.ndim == 2 and value.shape[0] == n_atoms and value.shape[1] == 3:
                # Coordinate array matching number of atoms
                coords = value.tolist()
            elif value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == 3:
                # Single coordinate in 2D array - broadcast to all atoms
                coords = [value[0].tolist()] * n_atoms
            else:
                raise ValueError(f"xyz must be shape (3,) or ({n_atoms}, 3), got {value.shape}")
            
            try:
                self._wrapped["atoms", "xyz"] = coords
            except (KeyError, TypeError):
                # Fallback: set individual atom coordinates
                for i, atom in enumerate(self._wrapped.atoms):
                    if i < len(coords):
                        atom["xyz"] = coords[i]
            return
        
        # Handle single entity (Atom, etc.)
        if hasattr(self._wrapped, 'xyz') and hasattr(type(self._wrapped).xyz, 'setter'):
            self._wrapped.xyz = value
        else:
            value = np.asarray(value, dtype=float)
            if value.shape != (3,):
                raise ValueError("xyz must be a 3D vector")
            self._wrapped["xyz"] = value.tolist()

    def distance_to(self, other) -> float:
        """
        Calculate the distance to another spatial entity.
        
        Args:
            other: Another spatial entity (can be wrapped or unwrapped)
            
        Returns:
            The Euclidean distance between the two entities
        """
        other_xyz = other.xyz if hasattr(other, 'xyz') else other
        return float(np.linalg.norm(self.xyz - other_xyz))

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
        self.xyz = origin + rotated

    def __call__(self, **kwargs):
        """
        Create a new instance of the wrapped entity with optional modifications.
        
        This method enables wrapped entities to be used as factory functions,
        creating copies of themselves with potentially modified properties.
        
        Args:
            **kwargs: Properties to pass to the wrapped entity's constructor
            
        Returns:
            A new instance of the wrapped entity with copied data
        """
        # If the wrapped entity has a __call__ method, use it
        if hasattr(self._wrapped, '__call__'):
            new_wrapped = self._wrapped(**kwargs)
            # Return a new SpatialWrapper wrapping the new instance
            return SpatialWrapper(new_wrapped)
        
        # Otherwise, create a new instance using the wrapped entity's class
        wrapped_class = type(self._wrapped)
        
        # Try to create a new instance with kwargs
        new_instance = wrapped_class(**kwargs)
        
        # Copy relevant data from the current wrapped entity
        if hasattr(self._wrapped, 'items'):
            for key, value in self._wrapped.items():
                if key not in kwargs:  # Don't override kwargs
                    if hasattr(value, 'copy'):
                        new_instance[key] = value.copy()
                    else:
                        new_instance[key] = value
        
        return new_instance


class HierarchyWrapper(Wrapper):
    """
    Wrapper class providing hierarchical structure functionality.
    
    Allows entities to have parent-child relationships and provides
    methods for navigating and manipulating the hierarchy.
    """
    
    def __init__(self, wrapped):
        """
        Initialize hierarchy wrapper.
        
        Args:
            wrapped: The entity to wrap
        """
        super().__init__(wrapped)
        if not hasattr(self._wrapped, '_parent'):
            self._wrapped._parent = None
        if not hasattr(self._wrapped, '_children'):
            self._wrapped._children = []
    
    @property
    def parent(self):
        """Get the parent entity."""
        return self._wrapped._parent
    
    @property
    def children(self):
        """Get the list of child entities."""
        return self._wrapped._children.copy()  # Return a copy to prevent external modification
    
    @property
    def is_root(self) -> bool:
        """Check if this entity is a root (has no parent)."""
        return self._wrapped._parent is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this entity is a leaf (has no children)."""
        return len(self._wrapped._children) == 0
    
    @property
    def depth(self) -> int:
        """Get the depth of this entity in the hierarchy (root = 0)."""
        if self.is_root or self.parent is None:
            return 0
        # Handle case where parent might also be wrapped
        parent_depth = self.parent.depth if hasattr(self.parent, 'depth') else 0
        return parent_depth + 1
    
    def add_child(self, child) -> None:
        """
        Add a child entity to this entity.
        
        Args:
            child: The child entity to add (can be wrapped or unwrapped)
        """
        # Get the underlying entity if child is wrapped
        child_entity = child.unwrap() if hasattr(child, 'unwrap') else child
        
        if child_entity not in self._wrapped._children:
            self._wrapped._children.append(child_entity)
            child_entity._parent = self._wrapped
    
    def remove_child(self, child) -> None:
        """
        Remove a child entity from this entity.
        
        Args:
            child: The child entity to remove (can be wrapped or unwrapped)
        """
        # Get the underlying entity if child is wrapped
        child_entity = child.unwrap() if hasattr(child, 'unwrap') else child
        
        if child_entity in self._wrapped._children:
            self._wrapped._children.remove(child_entity)
            child_entity._parent = None
    
    def get_root(self):
        """
        Get the root entity of the hierarchy.
        
        Returns:
            The root entity
        """
        if self.is_root or self.parent is None:
            return self._wrapped
        # Handle case where parent might also be wrapped
        parent_root = self.parent.get_root() if hasattr(self.parent, 'get_root') else self.parent
        return parent_root
    
    def get_ancestors(self) -> list:
        """
        Get all ancestors of this entity (from parent to root).
        
        Returns:
            List of ancestor entities
        """
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent if hasattr(current, 'parent') else None
        return ancestors
    
    def get_descendants(self) -> list:
        """
        Get all descendants of this entity.
        
        Returns:
            List of descendant entities
        """
        descendants = []
        for child in self._wrapped._children:
            descendants.append(child)
            # Handle case where child might be wrapped
            if hasattr(child, 'get_descendants'):
                descendants.extend(child.get_descendants())
            elif hasattr(child, '_children'):
                # Recursively get descendants for unwrapped children
                child_wrapper = HierarchyWrapper(child)
                descendants.extend(child_wrapper.get_descendants())
        return descendants
    
    def find_by_condition(self, condition: Callable[[Any], bool]):
        """
        Find the first descendant (including self) that satisfies a condition.
        
        Args:
            condition: A function that takes an entity and returns True if it matches
            
        Returns:
            The first matching entity, or None
        """
        if condition(self._wrapped):
            return self._wrapped
        for child in self._wrapped._children:
            # Check the child directly
            if condition(child):
                return child
            # Recursively search child's descendants
            if hasattr(child, '_children'):
                child_wrapper = HierarchyWrapper(child)
                found = child_wrapper.find_by_condition(condition)
                if found:
                    return found
        return None


class IdentifierWrapper(Wrapper):
    """
    Wrapper class providing identifier functionality for entities.
    
    Provides consistent identifier management with optional auto-generation.
    """
    
    _id_counter = 0
    
    def __init__(self, wrapped, id: Optional[Union[int, str]] = None):
        """
        Initialize with an identifier.
        
        Args:
            wrapped: The entity to wrap
            id: Optional identifier (auto-generated if None)
        """
        super().__init__(wrapped)
        if id is None:
            IdentifierWrapper._id_counter += 1
            self._wrapped._id = IdentifierWrapper._id_counter
        else:
            self._wrapped._id = id
    
    @property
    def id(self) -> Optional[Union[int, str]]:
        """Get the entity identifier."""
        return getattr(self._wrapped, '_id', None)
    
    @id.setter
    def id(self, value: Union[int, str]) -> None:
        """Set the entity identifier."""
        self._wrapped._id = value
    
    def __str__(self) -> str:
        """Return string representation using identifier."""
        return f"{self._wrapped.__class__.__name__}({self.id})"
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"{self._wrapped.__class__.__name__}(id={self.id})"


class VisualWrapper(Wrapper):
    """
    Wrapper class providing visualization functionality for entities.
    
    Manages visual properties like color, size, and rendering options.
    """
    
    def __init__(self, wrapped, color: Optional[str] = None, size: Optional[float] = None, **visual_props):
        """
        Initialize visual wrapper.
        
        Args:
            wrapped: The entity to wrap
            color: Color specification
            size: Size specification
            **visual_props: Additional visual properties
        """
        super().__init__(wrapped)
        if not hasattr(self._wrapped, '_visual_props'):
            self._wrapped._visual_props = {}
        
        if color is not None:
            self._wrapped._visual_props['color'] = color
        if size is not None:
            self._wrapped._visual_props['size'] = size
        
        self._wrapped._visual_props.update(visual_props)
    
    @property
    def color(self) -> Optional[str]:
        """Get the color property."""
        return self._wrapped._visual_props.get('color')
    
    @color.setter
    def color(self, value: str) -> None:
        """Set the color property."""
        if not hasattr(self._wrapped, '_visual_props'):
            self._wrapped._visual_props = {}
        self._wrapped._visual_props['color'] = value
    
    @property
    def size(self) -> Optional[float]:
        """Get the size property."""
        return self._wrapped._visual_props.get('size')
    
    @size.setter
    def size(self, value: float) -> None:
        """Set the size property."""
        if not hasattr(self._wrapped, '_visual_props'):
            self._wrapped._visual_props = {}
        self._wrapped._visual_props['size'] = value
    
    def get_visual_prop(self, key: str, default=None):
        """Get a visual property by key."""
        return self._wrapped._visual_props.get(key, default)
    
    def set_visual_prop(self, key: str, value) -> None:
        """Set a visual property by key."""
        if not hasattr(self._wrapped, '_visual_props'):
            self._wrapped._visual_props = {}
        self._wrapped._visual_props[key] = value
    
    @property
    def visual_props(self) -> dict:
        """Get all visual properties."""
        return getattr(self._wrapped, '_visual_props', {}).copy()


# Utility functions for working with wrappers
def wrap(entity, *wrapper_classes, **wrapper_kwargs):
    """
    Wrap an entity with multiple wrapper classes.
    
    Args:
        entity: The entity to wrap
        *wrapper_classes: Wrapper classes to apply
        **wrapper_kwargs: Keyword arguments for wrapper initialization
        
    Returns:
        The wrapped entity
    """
    wrapped = entity
    for wrapper_class in wrapper_classes:
        # Handle different wrapper initialization signatures
        if wrapper_class == VisualWrapper:
            # VisualWrapper accepts keyword arguments
            wrapped = wrapper_class(wrapped, **wrapper_kwargs)
        elif wrapper_class == IdentifierWrapper:
            # IdentifierWrapper accepts id parameter
            id_arg = wrapper_kwargs.get('id', None)
            wrapped = wrapper_class(wrapped, id=id_arg)
        else:
            # Other wrappers typically just take the wrapped entity
            wrapped = wrapper_class(wrapped)
    return wrapped


def unwrap_all(wrapped_entity):
    """
    Completely unwrap an entity, removing all wrapper layers.
    
    Args:
        wrapped_entity: The wrapped entity
        
    Returns:
        The original unwrapped entity
    """
    current = wrapped_entity
    while hasattr(current, 'unwrap'):
        current = current.unwrap()
    return current


def is_wrapped(entity) -> bool:
    """
    Check if an entity is wrapped.
    
    Args:
        entity: The entity to check
        
    Returns:
        True if the entity is wrapped, False otherwise
    """
    return hasattr(entity, 'unwrap') and callable(getattr(entity, 'unwrap'))

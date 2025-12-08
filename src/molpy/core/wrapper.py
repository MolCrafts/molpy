"""
Base wrapper class for Struct objects.

Provides a semi-transparent wrapper design:
- Internally: composition (holds `inner` object)
- Externally: explicit forwarding of selected APIs (no generic __getattr__)

Subclasses should explicitly forward the methods/properties they want to expose
from the inner object, rather than relying on automatic delegation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Self, TypeVar

from .entity import Struct

# Type variable for the inner type (bound to Struct)
TInner = TypeVar("TInner", bound=Struct)


class Wrapper[TInner: Struct]:
    """
    Base wrapper class for Struct objects.

    This is a **semi-transparent wrapper** that uses composition internally
    and requires explicit forwarding of inner APIs. It does NOT automatically
    forward all attributes via `__getattr__`.

    Design principles:
    - **Composition over inheritance**: The wrapper holds an `inner` object
    - **Explicit forwarding**: Subclasses should explicitly forward selected
      methods/properties from `inner` by writing wrapper methods that delegate
    - **Type safety**: Uses generics to preserve type information about the inner object

    Type parameter:
        TInner: The type of Struct being wrapped (bound to Struct)

    Example:
        >>> class Monomer(Wrapper[Atomistic]):
        ...     def __init__(self, inner: Atomistic, name: str):
        ...         super().__init__(inner)
        ...         self.name = name
        ...
        ...     # Explicitly forward selected properties
        ...     @property
        ...     def n_atoms(self) -> int:
        ...         return len(self.inner.atoms)
        ...
        ...     @property
        ...     def positions(self):
        ...         return self.inner.positions
        ...
        ...     def copy(self) -> "Monomer":
        ...         return Monomer(self.inner.copy(), self.name)
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: TInner | Wrapper[TInner], **props: Any) -> None:
        """Initialize wrapper with inner object.

        Args:
            inner: Either a concrete Struct instance or another Wrapper.
                  The wrapper stores exactly what is passed in.
            **props: Additional properties (passed to __post_init__)

        The wrapper stores the provided inner object as-is, preserving
        wrapper chains if inner is itself a Wrapper.
        """
        # Store the inner object directly (no unwrapping!)
        object.__setattr__(self, "_inner", inner)

        # Call post-init hook for subclass initialization
        remaining_props = self.__post_init__(**props)

        # Pass remaining props to inner if any
        if remaining_props:
            for key, value in remaining_props.items():
                self._inner[key] = value

    def __post_init__(self, **props) -> dict[str, Any] | None:
        """Post-initialization hook for subclass setup.

        Override this in subclasses to:
        - Initialize wrapper-specific state
        - Consume relevant kwargs from props
        - Return remaining props dict (or None)

        Args:
            **props: Properties passed during initialization

        Returns:
            dict or None: Remaining props to pass down, or None
        """
        return props

    @property
    def inner(self) -> TInner | Wrapper[TInner]:
        """Access the directly wrapped inner object.

        Returns:
            The directly wrapped object (may be Struct or Wrapper)
        """
        return self._inner

    def unwrap(self) -> TInner:
        """Recursively unwrap to get the innermost Struct.

        Returns:
            The innermost Struct instance (unwraps all wrapper layers)
        """
        current: TInner | Wrapper[TInner] = self._inner
        while isinstance(current, Wrapper):
            current = current._inner
        return current

    def with_inner(self, inner: TInner) -> Self:
        """Create a new wrapper instance with a different inner object.

        This is useful for creating modified wrappers while preserving
        wrapper-specific state.

        Args:
            inner: New inner object to wrap

        Returns:
            A new wrapper instance of the same type with the new inner object
        """
        # Create new instance without calling __init__ to avoid double unwrapping
        new_wrapper = object.__new__(type(self))
        object.__setattr__(new_wrapper, "_inner", inner)

        # Copy wrapper-specific attributes (excluding _inner)
        # Handle both __slots__ and __dict__ cases
        if hasattr(self, "__dict__"):
            for key, value in self.__dict__.items():
                if key != "_inner":
                    object.__setattr__(new_wrapper, key, deepcopy(value))

        # Also copy __slots__ attributes (excluding _inner)
        if hasattr(type(self), "__slots__"):
            for slot in type(self).__slots__:
                if slot != "_inner" and hasattr(self, slot):
                    try:
                        value = getattr(self, slot)
                        object.__setattr__(new_wrapper, slot, deepcopy(value))
                    except AttributeError:
                        pass

        return new_wrapper

    # Dict-like access: These are explicitly forwarded for convenience
    # since Struct supports dict-like access to props

    def __getitem__(self, key: str) -> object:
        """Delegate dict-style access to innermost Struct.

        Args:
            key: Key to access

        Returns:
            Value from innermost Struct's props

        Raises:
            KeyError: If key doesn't exist
        """
        return self.unwrap()[key]

    def __setitem__(self, key: str, value: object) -> None:
        """Delegate dict-style write to innermost Struct.

        Args:
            key: Key to set
            value: Value to assign
        """
        self.unwrap()[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in innermost Struct.

        Args:
            key: Key to check

        Returns:
            True if key exists in innermost Struct
        """
        return key in self.unwrap()

    def get(self, key: str, default: object | None = None) -> object:
        """Get value from innermost Struct with default.

        Args:
            key: Key to get
            default: Default value if key not found

        Returns:
            Value from innermost Struct or default
        """
        return self.unwrap().get(key, default)

    def copy(self) -> Self:
        """Create a deep copy of the wrapper and its wrapped entity.

        Returns:
            A new wrapper instance with deep-copied inner Struct.
            Wrapper-specific attributes are also deep-copied.
        """
        # Deep copy the inner
        new_inner = deepcopy(self._inner)

        # Create new wrapper with copied inner
        new_wrapper = type(self)(new_inner)  # type: ignore[arg-type]

        # Copy wrapper-specific attributes (excluding _inner which is already set)
        # Handle both __slots__ and __dict__ cases
        if hasattr(self, "__dict__"):
            for key, value in self.__dict__.items():
                if key != "_inner":
                    if hasattr(new_wrapper, "__dict__"):
                        new_wrapper.__dict__[key] = deepcopy(value)
                    else:
                        object.__setattr__(new_wrapper, key, deepcopy(value))

        # Also copy __slots__ attributes (excluding _inner)
        if hasattr(type(self), "__slots__"):
            for slot in type(self).__slots__:
                if slot != "_inner" and hasattr(self, slot):
                    try:
                        value = getattr(self, slot)
                        object.__setattr__(new_wrapper, slot, deepcopy(value))
                    except AttributeError:
                        pass

        return new_wrapper

    def __repr__(self) -> str:
        """Simple representation showing wrapper type and wrapped object."""
        return f"<{type(self).__name__} wrapping {self._inner!r}>"

    def __eq__(self, other: object) -> bool:
        """Equality comparison.

        Two wrappers are equal if they are of the same type and their
        innermost Structs are equal.

        Args:
            other: Object to compare with

        Returns:
            True if wrappers are equal, False otherwise
        """
        if not isinstance(other, type(self)):
            return False
        return self.unwrap() == other.unwrap()

    def __hash__(self) -> int:
        """Hash based on wrapper type and innermost Struct.

        Returns:
            Hash value
        """
        return hash((type(self), self.unwrap()))

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute access to innermost Struct.

        If an attribute is not found on the wrapper itself,
        delegate to the innermost Struct (via unwrap()).

        Args:
            name: Attribute name

        Returns:
            Attribute value from innermost Struct

        Raises:
            AttributeError: If attribute not found on innermost Struct
        """
        # Avoid infinite recursion for special attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Delegate to innermost Struct
        try:
            return getattr(self.unwrap(), name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

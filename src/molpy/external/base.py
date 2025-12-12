"""Base Adapter class for MolPy.

This module provides the abstract base class for adapters that maintain
bidirectional synchronization between MolPy's internal data structures
and external library representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# Type variables for adapter generics
InternalT = TypeVar("InternalT")
ExternalT = TypeVar("ExternalT")


class Adapter(ABC, Generic[InternalT, ExternalT]):
    """Abstract base class for representation adapters.

    Adapters maintain bidirectional synchronization between MolPy's internal
    data structures (e.g., Atomistic, Frame) and external library
    representations (e.g., RDKit Mol, OpenMM System).

    **Responsibilities:**
    - Maintain references to both internal and external representations
    - Provide explicit synchronization methods in both directions
    - Handle representation conversion

    **Limitations:**
    - Adapters do NOT execute external tools or spawn subprocesses
    - Adapters do NOT perform domain logic (e.g., geometry optimization)
    - Adapters do NOT assume specific index mappings or topology semantics
      (concrete adapters manage their own mapping strategies)

    **Usage Pattern:**
        >>> adapter = RDKitAdapter()
        >>> adapter.set_internal(atomistic_structure)
        >>> adapter.sync_to_external()  # Update external from internal
        >>> rdkit_mol = adapter.get_external()
        >>> # ... modify rdkit_mol ...
        >>> adapter.sync_to_internal()  # Update internal from external
        >>> updated_structure = adapter.get_internal()

    Type Parameters:
        InternalT: Type of the MolPy internal representation
        ExternalT: Type of the external library representation
    """

    def __init__(
        self,
        internal: InternalT | None = None,
        external: ExternalT | None = None,
    ) -> None:
        """Initialize adapter with optional internal and external objects.

        Args:
            internal: Optional internal MolPy object
            external: Optional external library object

        At least one of internal or external must be provided, or both can
        be None initially and set later via setter methods.
        """
        self._internal: InternalT | None = internal
        self._external: ExternalT | None = external

    # =====================================================================
    #   Getters
    # =====================================================================

    def get_internal(self) -> InternalT:
        """Get the internal MolPy representation.

        If internal is None, attempts to synchronize from external.
        If both are None, raises ValueError.

        Returns:
            The internal MolPy object

        Raises:
            ValueError: If both internal and external are None
            RuntimeError: If synchronization fails
        """
        if self._internal is not None:
            return self._internal

        if self._external is not None:
            # Try to sync from external
            self.sync_to_internal()
            if self._internal is not None:
                return self._internal

        raise ValueError(
            "Cannot get internal representation: both internal and external are None. "
            "Set at least one using set_internal() or set_external()."
        )

    def get_external(self) -> ExternalT:
        """Get the external library representation.

        If external is None, attempts to synchronize from internal.
        If both are None, raises ValueError.

        Returns:
            The external library object

        Raises:
            ValueError: If both internal and external are None
            RuntimeError: If synchronization fails
        """
        if self._external is not None:
            return self._external

        if self._internal is not None:
            # Try to sync from internal
            self.sync_to_external()
            if self._external is not None:
                return self._external

        raise ValueError(
            "Cannot get external representation: both internal and external are None. "
            "Set at least one using set_internal() or set_external()."
        )

    # =====================================================================
    #   Setters
    # =====================================================================

    def set_internal(self, internal: InternalT) -> None:
        """Set or update the internal MolPy representation.

        Args:
            internal: The internal MolPy object to set

        Note:
            This does NOT automatically synchronize to external.
            Call sync_to_external() explicitly if needed.
        """
        self._internal = internal

    def set_external(self, external: ExternalT) -> None:
        """Set or update the external library representation.

        Args:
            external: The external library object to set

        Note:
            This does NOT automatically synchronize to internal.
            Call sync_to_internal() explicitly if needed.
        """
        self._external = external

    # =====================================================================
    #   Synchronization Methods (Abstract)
    # =====================================================================

    @abstractmethod
    def sync_to_internal(self) -> None:
        """Synchronize internal representation from external.

        Updates the internal MolPy object based on the current state
        of the external library object. The external object is the
        source of truth for this operation.

        Raises:
            ValueError: If external is None
            RuntimeError: If synchronization fails
        """
        if self._external is None:
            raise ValueError(
                "Cannot sync to internal: external representation is None. "
                "Set external using set_external() first."
            )

    @abstractmethod
    def sync_to_external(self) -> None:
        """Synchronize external representation from internal.

        Updates the external library object based on the current state
        of the internal MolPy object. The internal object is the source
        of truth for this operation.

        Raises:
            ValueError: If internal is None
            RuntimeError: If synchronization fails
        """
        if self._internal is None:
            raise ValueError(
                "Cannot sync to external: internal representation is None. "
                "Set internal using set_internal() first."
            )

    # =====================================================================
    #   Convenience Methods
    # =====================================================================

    def has_internal(self) -> bool:
        """Check if internal representation is set.

        Returns:
            True if internal is not None
        """
        return self._internal is not None

    def has_external(self) -> bool:
        """Check if external representation is set.

        Returns:
            True if external is not None
        """
        return self._external is not None

    def __repr__(self) -> str:
        """String representation of adapter."""
        internal_str = "set" if self.has_internal() else "None"
        external_str = "set" if self.has_external() else "None"
        return (
            f"<{self.__class__.__name__}(internal={internal_str}, "
            f"external={external_str})>"
        )

"""Base Adapter class for MolPy.

This module provides the abstract base class for adapters that maintain
bidirectional synchronization between MolPy's internal data structures
and external representations.

Adapters do NOT execute external tools or spawn subprocesses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InternalT = TypeVar("InternalT")
ExternalT = TypeVar("ExternalT")


class Adapter(ABC, Generic[InternalT, ExternalT]):
    """Abstract base class for representation adapters.

    Adapters maintain bidirectional synchronization between MolPy's internal
    data structures (e.g., Atomistic, Frame) and external representations.

    Adapters MUST NOT execute external tools or spawn subprocesses.
    """

    def __init__(
        self,
        internal: InternalT | None = None,
        external: ExternalT | None = None,
    ) -> None:
        self._internal: InternalT | None = internal
        self._external: ExternalT | None = external

    def get_internal(self) -> InternalT:
        if self._internal is not None:
            return self._internal

        if self._external is not None:
            self.sync_to_internal()
            if self._internal is not None:
                return self._internal

        raise ValueError(
            "Cannot get internal representation: both internal and external are None. "
            "Set at least one using set_internal() or set_external()."
        )

    def get_external(self) -> ExternalT:
        if self._external is not None:
            return self._external

        if self._internal is not None:
            self.sync_to_external()
            if self._external is not None:
                return self._external

        raise ValueError(
            "Cannot get external representation: both internal and external are None. "
            "Set at least one using set_internal() or set_external()."
        )

    def set_internal(self, internal: InternalT) -> None:
        self._internal = internal

    def set_external(self, external: ExternalT) -> None:
        self._external = external

    @abstractmethod
    def sync_to_internal(self) -> None:
        if self._external is None:
            raise ValueError(
                "Cannot sync to internal: external representation is None. "
                "Set external using set_external() first."
            )

    @abstractmethod
    def sync_to_external(self) -> None:
        if self._internal is None:
            raise ValueError(
                "Cannot sync to external: internal representation is None. "
                "Set internal using set_internal() first."
            )

    def has_internal(self) -> bool:
        return self._internal is not None

    def has_external(self) -> bool:
        return self._external is not None

    def __repr__(self) -> str:
        internal_str = "set" if self.has_internal() else "None"
        external_str = "set" if self.has_external() else "None"
        return f"<{self.__class__.__name__}(internal={internal_str}, external={external_str})>"

    def check(self) -> None:
        """Validate the adapter has enough state to do useful work.

        This is intentionally lightweight and side-effect free.
        Concrete adapters may override with stronger validation.
        """

        if self._internal is None and self._external is None:
            raise ValueError(
                "Adapter has neither internal nor external representation set. "
                "Provide at least one of internal=... or external=..."
            )

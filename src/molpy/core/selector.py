from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .frame import Block

__all__ = [
    "AtomIndexSelector",
    "AtomTypeSelector",
    "CoordinateRangeSelector",
    "DistanceSelector",
    "ElementSelector",
    "MaskPredicate",
]


class MaskPredicate(ABC):
    """Boolean mask producer combinable with &, |, ~."""

    @abstractmethod
    def mask(self, block: "Block") -> np.ndarray: ...

    def __call__(self, block: "Block") -> "Block":
        return block[self.mask(block)]

    # Compositional logic
    def __and__(self, other: "MaskPredicate") -> "MaskPredicate":
        return _And(self, other)

    def __or__(self, other: "MaskPredicate") -> "MaskPredicate":
        return _Or(self, other)

    def __invert__(self) -> "MaskPredicate":
        return _Not(self)

    __rand__ = __and__
    __ror__ = __or__


Selector = MaskPredicate


class AtomTypeSelector(MaskPredicate):
    """Select atoms by their type (integer or string)."""

    def __init__(self, atom_type: int | str, field: str = "type") -> None:
        """
        Initialize atom type Selector.

        Args:
            atom_type: The atom type to select (integer or string)
            field: The field name containing atom types (default: "type")
        """
        self.atom_type = atom_type
        self.field = field

    def mask(self, block: "Block") -> np.ndarray:
        assert self.field in block, f"Field '{self.field}' not found in block"
        return block[self.field] == self.atom_type


class AtomIndexSelector(MaskPredicate):
    """Select atoms by their indices."""

    def __init__(self, indices: list[int] | np.ndarray, id_field: str = "id") -> None:
        """
        Initialize atom index Selector.

        Args:
            indices: List or array of atom indices to select
            id_field: The field name containing atom IDs (default: "id")
        """
        # Convert to numpy array and validate
        if isinstance(indices, list):
            self.indices = np.array(indices, dtype=int)
        elif isinstance(indices, np.ndarray):
            self.indices = indices.astype(int)
        else:
            raise TypeError("indices must be a list[int] or np.ndarray")

        self.id_field = id_field

    def mask(self, block: "Block") -> np.ndarray:
        if self.id_field not in block:
            return np.zeros(block.nrows, dtype=bool)
        return np.isin(block[self.id_field], self.indices)


class ElementSelector(MaskPredicate):
    """Select atoms by their element symbol."""

    def __init__(self, element: str, field: str = "element"):
        """
        Initialize element Selector.

        Args:
            element: The element symbol to select (e.g., "C", "H", "O")
            field: The field name containing element symbols (default: "element")
        """
        if not isinstance(element, str):
            raise TypeError("element must be a string")
        self.element = element
        self.field = field

    def mask(self, block: "Block") -> np.ndarray:
        if self.field not in block:
            return np.zeros(block.nrows, dtype=bool)
        return block[self.field] == self.element


class CoordinateRangeSelector(MaskPredicate):
    """Select atoms within a coordinate range."""

    def __init__(
        self,
        axis: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize coordinate range Selector.

        Args:
            axis: The coordinate axis ("x", "y", or "z")
            min_value: Minimum coordinate value (inclusive)
            max_value: Maximum coordinate value (inclusive)
        """
        if axis not in ["x", "y", "z"]:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")

        self.axis = axis
        self.min_value = min_value
        self.max_value = max_value

    def mask(self, block: "Block") -> np.ndarray:
        if self.axis not in block:
            return np.zeros(block.nrows, dtype=bool)

        values = block[self.axis]
        mask = np.ones(block.nrows, dtype=bool)

        if self.min_value is not None:
            mask &= values >= self.min_value

        if self.max_value is not None:
            mask &= values <= self.max_value

        return mask


class DistanceSelector(MaskPredicate):
    """Select atoms within a distance from a reference point."""

    def __init__(
        self,
        center: list[float] | np.ndarray,
        max_distance: float,
        min_distance: float | None = None,
    ) -> None:
        """
        Initialize distance-based Selector.

        Args:
            center: Reference point [x, y, z]
            max_distance: Maximum distance from center (inclusive)
            min_distance: Minimum distance from center (inclusive, optional)
        """
        # Convert center to numpy array and validate
        if isinstance(center, list):
            self.center = np.array(center, dtype=float)
        elif isinstance(center, np.ndarray):
            self.center = center.astype(float)
        else:
            raise TypeError("center must be a list[float] or np.ndarray")

        if len(self.center) != 3:
            raise ValueError("center must have exactly 3 coordinates")

        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")

        if min_distance is not None:
            if min_distance < 0:
                raise ValueError("min_distance must be non-negative")
            if min_distance > max_distance:
                raise ValueError("min_distance cannot be greater than max_distance")

        self.max_distance = max_distance
        self.min_distance = min_distance

    def mask(self, block: "Block") -> np.ndarray:
        required_fields = ["x", "y", "z"]
        if not all(field in block for field in required_fields):
            return np.zeros(block.nrows, dtype=bool)

        positions = np.column_stack([block["x"], block["y"], block["z"]])
        distances = np.linalg.norm(positions - self.center, axis=1)

        mask = distances <= self.max_distance

        if self.min_distance is not None:
            mask &= distances >= self.min_distance

        return mask


# ------------------------------------------------------------------ combinators
class _And(MaskPredicate):
    """Logical AND combination of two predicates."""

    def __init__(self, a: MaskPredicate, b: MaskPredicate):
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return self.a.mask(block) & self.b.mask(block)


class _Or(MaskPredicate):
    """Logical OR combination of two predicates."""

    def __init__(self, a: MaskPredicate, b: MaskPredicate):
        self.a = a
        self.b = b

    def mask(self, block: "Block") -> np.ndarray:
        return self.a.mask(block) | self.b.mask(block)


class _Not(MaskPredicate):
    """Logical NOT of a predicate."""

    def __init__(self, a: MaskPredicate):
        self.a = a

    def mask(self, block: "Block") -> np.ndarray:
        return ~self.a.mask(block)

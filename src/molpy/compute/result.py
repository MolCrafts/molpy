"""Result base class for compute operations.

This module provides the base Result class for storing computation outputs.
All compute operations return a Result or a subclass of Result.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    """Base class for computation results.

    Attributes:
        name: Optional name for the result.
        meta: Dictionary for storing arbitrary metadata.

    Examples:
        >>> result = Result(name="my_computation")
        >>> result.meta["timestamp"] = "2024-01-01"
        >>> print(result.name)
        my_computation
    """

    name: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

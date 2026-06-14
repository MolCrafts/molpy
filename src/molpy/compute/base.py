"""Base class for compute operations.

A :class:`Compute` is a configurable callable. Construction parameters go to
``__init__`` (stored for serialization via :meth:`Compute.dump`); data inputs
go to ``__call__``. Operators take one *or more* data inputs directly — there
is no single-input restriction:

    >>> rdf = RDF(n_bins=100, r_max=10.0)
    >>> result = rdf(frames, neighbors)      # two data inputs

The heavy numerics live in molrs; molpy operators are thin, typed shells that
forward to the Rust kernels and return the molrs native result unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Compute(ABC):
    """Abstract base class for compute operations.

    Subclasses implement :meth:`__call__` with a concrete, fully typed
    signature (one positional parameter per data input) and pass their
    construction parameters to ``super().__init__(**config)`` so that
    :meth:`dump` can round-trip the configuration.

    Examples:
        >>> class MyCompute(Compute):
        ...     def __init__(self, scale: float):
        ...         super().__init__(scale=scale)
        ...         self.scale = scale
        ...
        ...     def __call__(self, frames: Sequence[Frame]) -> MyResult:
        ...         return MyResult(value=42 * self.scale)
        >>>
        >>> compute = MyCompute(scale=2.0)
        >>> result = compute(frames)
        >>> compute.dump()
        {'scale': 2.0}
    """

    def __init__(self, **config: Any) -> None:
        """Store construction parameters for serialization.

        Args:
            **config: Configuration parameters, returned verbatim by
                :meth:`dump`.
        """
        self._config = config

    @abstractmethod
    def __call__(self, *inputs: Any) -> Any:
        """Run the computation on the given data inputs.

        Subclasses override this with a concrete, typed signature — one
        parameter per data input (e.g. ``(frames, neighbors)``).

        Args:
            *inputs: Data inputs for the computation.

        Returns:
            The computation result.
        """
        ...

    def dump(self) -> dict[str, Any]:
        """Serialize construction configuration to a dictionary.

        Returns:
            The configuration parameters passed to ``__init__``.
        """
        return dict(self._config)

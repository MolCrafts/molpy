"""Base classes for compute operations.

This module provides the core abstractions for defining computation units:
- Compute: Generic base class for all compute operations
- ComputeContext: Optional context for sharing expensive intermediates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from .result import Result

InT = TypeVar("InT")
OutT = TypeVar("OutT", bound=Result)


@dataclass
class ComputeContext:
    """Context for sharing expensive intermediate computations.

    This allows multiple compute operations to reuse expensive intermediates
    like neighbor lists, distance matrices, etc.

    Attributes:
        data: Dictionary storing intermediate computation results.

    Examples:
        >>> context = ComputeContext()
        >>> context.data["neighbor_list"] = compute_neighbors()
        >>> compute1 = SomeCompute(context=context)
        >>> compute2 = AnotherCompute(context=context)
        >>> # Both compute1 and compute2 can access the neighbor list
    """

    data: dict[str, object] = field(default_factory=dict)


class Compute[InT, OutT: Result](ABC):
    """Abstract base class for compute operations.

    A Compute is a callable object that takes an input of type InT and returns
    a result of type OutT (which must be a Result subclass).

    To implement a new compute operation:
    1. Subclass Compute[InT, OutT] with appropriate types
    2. Override the compute() method with your core logic
    3. Optionally override before() and after() for setup/cleanup

    The __call__ method orchestrates the computation:
    1. Calls before(input) for setup
    2. Calls compute(input) for the main computation
    3. Calls after(input, result) for cleanup
    4. Returns the result

    Attributes:
        context: Optional shared context for expensive intermediates.

    Examples:
        >>> class MyCompute(Compute[Frame, MyResult]):
        ...     def compute(self, frame: Frame) -> MyResult:
        ...         # Your computation logic here
        ...         return MyResult(value=42)
        >>>
        >>> compute = MyCompute()
        >>> result = compute(frame)  # Calls __call__, which calls compute()
    """

    def __init__(self, context: ComputeContext | None = None) -> None:
        """Initialize the compute operation.

        Args:
            context: Optional context for sharing expensive intermediates.
        """
        self.context = context

    def __call__(self, input: InT) -> OutT:
        """Execute the computation.

        This method orchestrates the computation by calling before(), compute(),
        and after() in sequence.

        Args:
            input: Input data for the computation.

        Returns:
            Computation result.
        """
        self.before(input)
        result = self.compute(input)
        self.after(input, result)
        return result

    @abstractmethod
    def compute(self, input: InT) -> OutT:
        """Perform the core computation.

        This method must be overridden by subclasses to implement the actual
        computation logic.

        Args:
            input: Input data for the computation.

        Returns:
            Computation result.
        """
        ...

    def before(self, input: InT) -> None:
        """Hook called before compute().

        Override this method to perform setup operations before the main
        computation. The default implementation does nothing.

        Args:
            input: Input data for the computation.
        """
        pass

    def after(self, input: InT, result: OutT) -> None:
        """Hook called after compute().

        Override this method to perform cleanup or post-processing operations
        after the main computation. The default implementation does nothing.

        Args:
            input: Input data for the computation.
            result: Result from the compute() method.
        """
        pass

"""Base classes for compute operations.

This module provides the core abstractions for defining computation units:
- Compute: Generic base class for all compute operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class Compute[InT, OutT](ABC):
    """Abstract base class for compute operations.

    A Compute is a callable object that takes an input of type InT and returns
    a result of type OutT.

    To implement a new compute operation:
    1. Subclass Compute[InT, OutT] with appropriate types
    2. Override the compute() method with your core logic
    3. Optionally override before() and after() for setup/cleanup

    The __call__ method orchestrates the computation:
    1. Calls before(input) for setup
    2. Calls compute(input) for the main computation
    3. Calls after(input, result) for cleanup
    4. Returns the result

    Examples:
        >>> class MyCompute(Compute[Frame, MyResult]):
        ...     def compute(self, frame: Frame) -> MyResult:
        ...         # Your computation logic here
        ...         return MyResult(value=42)
        >>>
        >>> compute = MyCompute()
        >>> result = compute(frame)  # Calls __call__, which calls compute()
    """

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

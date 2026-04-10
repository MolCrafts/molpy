"""Base classes for compute operations.

This module provides the core abstractions for defining computation units:
- Compute: Generic base class for all compute operations

The Compute class is compatible with molexp workflow tasks via structural typing (TaskProtocol).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class Compute[InT, OutT](ABC):
    """Abstract base class for compute operations.

    A Compute is a callable object that takes an input of type InT and returns
    a result of type OutT. It is compatible with molexp workflow tasks via
    structural typing (implements execute() and dump() methods).

    To implement a new compute operation:
    1. Subclass Compute[InT, OutT] with appropriate types
    2. Override the _compute() method with your core logic
    3. Optionally override before() and after() for setup/cleanup
    4. Pass **config_kwargs to super().__init__() to enable config serialization

    The __call__ method orchestrates the computation:
    1. Calls before(input) for setup
    2. Calls _compute(input) for the main computation
    3. Calls after(input, result) for cleanup
    4. Returns the result

    Examples:
        >>> class MyCompute(Compute[Frame, MyResult]):
        ...     def __init__(self, param1, param2, **config_kwargs):
        ...         super().__init__(param1=param1, param2=param2, **config_kwargs)
        ...         self.param1 = param1
        ...         self.param2 = param2
        ...
        ...     def _compute(self, frame: Frame) -> MyResult:
        ...         # Your computation logic here
        ...         return MyResult(value=42)
        >>>
        >>> compute = MyCompute(param1="value", param2=42)
        >>> result = compute(frame)  # Calls __call__, which calls _compute()
        >>> config = compute.dump()  # {"param1": "value", "param2": 42}
        >>>
        >>> # Use in molexp workflow
        >>> result_dict = compute.execute(input=frame)  # {"result": MyResult(...)}
    """

    # Configuration for workflow integration
    input_key: str = "input"
    output_key: str = "result"

    def __init__(self, **config_kwargs: Any):
        """Initialize compute with configuration parameters.

        Args:
            **config_kwargs: Configuration parameters stored for serialization
        """
        self._config = config_kwargs

    def __call__(self, input: InT) -> OutT:
        """Execute the computation.

        This method orchestrates the computation by calling before(), _compute(),
        and after() in sequence.

        Args:
            input: Input data for the computation.

        Returns:
            Computation result.
        """
        self.before(input)
        result = self._compute(input)
        self.after(input, result)
        return result

    def execute(self, ctx: Any | None = None, **inputs: Any) -> dict[str, Any]:
        """Execute as a workflow task (molexp-compatible API).

        This method adapts the Compute interface to work with molexp workflows.
        Subclasses can override to customize input/output mapping.

        Args:
            ctx: Optional runtime context
            **inputs: Named input values

        Returns:
            Dictionary with output_key -> result mapping

        Raises:
            ValueError: If required input is missing
        """
        input_value = inputs.get(self.input_key)
        if input_value is None:
            raise ValueError(f"Missing required input: {self.input_key}")

        result = self(input_value)  # Calls __call__ -> before/_compute/after
        return {self.output_key: result}

    # Alias for API consistency with molexp.Task
    compute = execute

    def dump(self) -> dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Configuration parameters as dict
        """
        return self._config.copy()

    @abstractmethod
    def _compute(self, input: InT) -> OutT:
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
        """Hook called before _compute().

        Override this method to perform setup operations before the main
        computation. The default implementation does nothing.

        Args:
            input: Input data for the computation.
        """
        pass

    def after(self, input: InT, result: OutT) -> None:
        """Hook called after _compute().

        Override this method to perform cleanup or post-processing operations
        after the main computation. The default implementation does nothing.

        Args:
            input: Input data for the computation.
            result: Result from the _compute() method.
        """
        pass

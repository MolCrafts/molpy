"""Compute abstraction for molecular computations.

This module provides a unified interface for defining and executing
computational operations on molecular structures. The core abstractions are:

- Result: Base class for computation outputs
- Compute: Base class for computation operations
- ComputeContext: Optional context for sharing expensive intermediates

Example usage:
    >>> from molpy.compute import CountAtomsCompute
    >>> compute = CountAtomsCompute()
    >>> result = compute(frame)
    >>> print(result.value)
"""

from .base import Compute, ComputeContext
from .result import Result

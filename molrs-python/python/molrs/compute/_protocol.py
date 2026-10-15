"""The one Compute contract: ``compute(...)``.

A :func:`typing.runtime_checkable` :class:`~typing.Protocol` mirroring the
core ``molrs::compute::Compute`` trait: an instance is configured by its
constructor and answers one call, ``compute``, whose keyword arguments are the
fields of the core input (the data). Every compute class in
:mod:`molrs.compute` satisfies it as it stands. ``isinstance`` / ``issubclass``
only check that the method is present (PEP 544); they are not a hot-path
dispatch tool.

``__call__`` and ``dump()`` are not part of this contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Compute(Protocol):
    """Analysis kernel: configured at construction, one method ``compute``
    taking the data as keyword arguments."""

    def compute(self, **inputs: Any) -> Any:
        raise NotImplementedError("Compute.compute must be implemented")

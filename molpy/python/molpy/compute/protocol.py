"""The two compute contracts: ``compute(...)`` and ``accumulator()``.

:class:`Compute` is a :func:`typing.runtime_checkable`
:class:`~typing.Protocol` mirroring the core ``molrs::compute::Compute``
trait: an instance is configured by its constructor and answers one call,
``compute``, whose keyword arguments are the fields of the core input (the
data). :class:`Stream` mirrors the core ``molrs::compute::Stream`` trait: an
instance hands out a fresh accumulator (``push`` one frame at a time,
``finish`` reads the result) — ``Msd``, ``Vacf`` and ``Rdf`` are streams
only. ``isinstance`` / ``issubclass`` only check that the method is present
(PEP 544); they are not a hot-path dispatch tool.

``__call__`` and ``dump()`` are not part of either contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Compute(Protocol):
    """Analysis kernel: configured at construction, one method ``compute``
    taking the data as keyword arguments."""

    def compute(self, **inputs: Any) -> Any:
        raise NotImplementedError("Compute.compute must be implemented")


@runtime_checkable
class Stream(Protocol):
    """Streaming analysis: configured at construction, one method
    ``accumulator`` handing out a fresh accumulator."""

    def accumulator(self) -> Any:
        raise NotImplementedError("Stream.accumulator must be implemented")

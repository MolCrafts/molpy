"""The one Potential contract: ``evaluate(pos) -> (energy, forces)``.

The Python face of the core ``molrs::ff::potential::Potential::evaluate``:
energy and forces (``-∇E``) at the coordinates, always together. This is a
:func:`typing.runtime_checkable` :class:`~typing.Protocol`. Any object that
defines ``evaluate`` satisfies it structurally — explicit subclassing is
optional. ``isinstance`` only checks that the method **exists**, not its
signature (PEP 544), and is not suitable for hot-path dispatch. Integrators
dispatch on concrete Rust types first and fall back to duck typing last.

The Python contract is coordinates-only. Neighbour / pair data never
crosses the FFI into a duck-typed implementation; custom forces (NN /
Torch) own their own neighbour correctness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from numpy.typing import NDArray


@runtime_checkable
class Potential(Protocol):
    """Structured force provider: one method, coordinates in, energy+forces out."""

    def evaluate(self, pos: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Energy and forces at ``pos`` (``(N, 3)``): ``(energy, forces)``,
        forces a float64 ``(N, 3)`` array."""
        raise NotImplementedError(
            "Potential.evaluate must return "
            "(energy: float, forces: float64 (N, 3) ndarray)"
        )

"""The one Python Potential contract: ``bind`` once, ``compute`` per evaluation.

A Python potential is anything that turns positions into energy and forces by
itself: a neural network on a GPU, an external program, a test double. It is
a :func:`typing.runtime_checkable` :class:`~typing.Protocol`. Any object with
the attributes and methods below satisfies it structurally; explicit
subclassing is optional. ``isinstance`` only checks that the members
**exist**, not their signatures (PEP 544).

Three consumers take one:

- ``MD.set_evaluator`` mounts it as the run's evaluator. The driver integrates
  on the CPU and makes **one** Python call per step. It owns the neighbour-list
  cadence: the potential searches at ``cutoff + skin`` on its own device, and
  ``compute(..., rebuild=True, ...)`` tells it the list is stale. Raise from
  ``compute`` to stop the run: the exception reaches the caller of
  ``MD.run`` unchanged, at the step it was raised.
- ``Potentials.push`` adds it to a collection whose ``evaluate`` sums it with
  native kernels.
- ``Lbfgs`` relaxes a structure on such a collection.

Outside MD there is no cadence: every call says ``rebuild=True``.

Exchange buffers are the potential's. ``bind`` returns them once. At every
call the caller writes the positions into the first buffer and reads the
forces back from the second, in place, so no buffer is reallocated per call.
For a device model the buffers are numpy views of pinned host tensors, so the
host-to-device copy can be asynchronous.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from molpy.core import Box, Frame


@runtime_checkable
class Potential(Protocol):
    """A Python potential: ``bind`` to a frame once, ``compute`` per call.

    Attributes:
        units: The LAMMPS units style (``"real"``, ``"metal"``, …) of the
            positions it reads and the energy, forces and virial it returns.
            MD converts nothing: it refuses a potential whose style is not
            the run's.
        cutoff: The interaction cutoff of the potential's own neighbour
            search, or ``None`` when it searches none.
    """

    units: str
    cutoff: float | None

    def bind(
        self, frame: Frame, skin: float
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ):
        """Prepare for evaluations over ``frame`` and return the exchange
        buffers.

        Called before the first evaluation and again after any change to the
        caller (MD: a setter; ``Potentials.evaluate`` / ``Lbfgs.minimize``:
        each call with a frame). ``frame`` is a snapshot to read species,
        charges and the box from; writes to it reach nothing. Evaluated from
        bare coordinates, a potential is bound to a frame of those
        coordinates alone.

        Args:
            frame: The frame the evaluations are over.
            skin: Build the neighbour list at ``cutoff + skin``.

        Returns:
            ``(positions, forces)``, or ``(positions, forces, virial)`` from a
            potential that computes a virial. Each is a writeable C-contiguous
            float64 array, ``(N, 3)``, ``(N, 3)`` and ``(6,)``, kept alive by
            the potential. They must not share memory. The virial is
            ``W = Σ_pairs f ⊗ d`` (``−∂E/∂ε`` at zero strain): positive under
            compression, in ``units``' energy unit, in
            ``[xx, yy, zz, xy, xz, yz]`` order. An ASE stress is ``−W/V`` in
            ``[xx, yy, zz, yz, xz, xy]`` order; negate and reorder it.

            The caller holds these exact arrays until the next ``bind``. Fill
            them in place (``forces[:] = …``). An array rebound to a new name
            is never read, and one resized in place is refused.
        """
        ...

    def compute(self, step: int, rebuild: bool, box: Box | None) -> float:
        """Evaluate at the positions just written into the positions buffer.

        Write every component of the forces buffer (and of the virial buffer,
        if there is one) before returning. Under MD a non-finite energy or
        force stops the run with a ``ValueError``. Do not read or write the
        frame being stepped: it is held by the run, and an access raises
        ``RuntimeError``.

        Args:
            step: MD's step index (the evaluation before the first step is
                step 0), or the call count outside MD.
            rebuild: The neighbour list built at ``cutoff + skin`` is stale:
                search again before evaluating. Always ``True`` on the first
                call after ``bind``, and on every call outside MD.
            box: The cell, when it differs from the one last passed, and on
                the first call after ``bind``. ``None`` otherwise.

        Returns:
            The potential energy, in ``units``.
        """
        ...

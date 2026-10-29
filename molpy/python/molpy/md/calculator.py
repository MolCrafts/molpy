"""The Calculator contract: a force provider of its own, mounted on :class:`MD`.

A calculator is what ASE calls one — a neural potential on a GPU, an external
program — something that turns positions into energy and forces by itself.
:meth:`MD.set_calculator <molpy.md.MD.set_calculator>` mounts it. The driver
integrates on the CPU, and each step it makes **one** Python call into the
calculator.

Exchange buffers are the calculator's. ``bind`` returns them once, and every
step the driver writes the positions into the first buffer and reads the
forces back from the second, in place, so no buffer is reallocated per step.
For a device model the buffers are numpy views of pinned host tensors, so the
host-to-device copy can be asynchronous.

The driver owns the neighbour-list cadence. The calculator searches at
``cutoff + skin`` on its own device, and ``compute(..., rebuild=True, ...)``
tells it the list is stale. Raise from ``compute`` to stop the run: the
exception reaches the caller of :meth:`MD.run <molpy.md.MD.run>` unchanged,
at the step it was raised.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from molpy.core import Box, Frame, UnitPreset


@runtime_checkable
class Calculator(Protocol):
    """A force provider of its own: ``bind`` once, ``compute`` every step.

    Attributes:
        units: The unit system of what ``compute`` returns: a preset name such
            as ``"metal"``, or a :class:`~molpy.core.UnitPreset`. The driver
            converts the energy, the forces and the virial into the run's units
            once per step. Lengths are not converted, so the length unit must
            be the run's.
        cutoff: The interaction cutoff of the calculator's own neighbour
            search, or ``None`` when it searches none.
    """

    units: str | UnitPreset
    cutoff: float | None

    def bind(
        self, frame: Frame, skin: float
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ):
        """Prepare for a run over ``frame`` and return the exchange buffers.

        Called before the first evaluation, and again after any change to the
        run. ``frame`` is a snapshot to read species, charges and the box from.
        Writes to it reach nothing.

        Args:
            frame: The frame the run will step.
            skin: Build the neighbour list at ``cutoff + skin``.

        Returns:
            ``(positions, forces)``, or ``(positions, forces, virial)`` from a
            calculator that computes a virial. Each is a writeable C-contiguous
            float64 array, ``(N, 3)``, ``(N, 3)`` and ``(6,)``, kept alive by
            the calculator. They must not share memory. The virial is
            ``W = Σ_pairs f ⊗ d`` (``−∂E/∂ε`` at zero strain) in
            ``[xx, yy, zz, xy, xz, yz]`` order.

            The driver holds these exact arrays until the next ``bind``. Fill
            them in place (``forces[:] = …``). An array rebound to a new name
            is never read, and one resized in place is refused.
        """
        ...

    def compute(self, step: int, rebuild: bool, box: Box | None) -> float:
        """Evaluate at the positions just written into the positions buffer.

        Write every component of the forces buffer (and of the virial buffer,
        if there is one) before returning. A non-finite energy or force stops
        the run with a ``ValueError``. Do not read or write the frame being
        stepped: it is held by the run, and an access raises ``RuntimeError``.

        Args:
            step: The run's step index. The evaluation before the first step
                is step 0.
            rebuild: The neighbour list built at ``cutoff + skin`` is stale:
                search again before evaluating. Always ``True`` on the first
                call after ``bind``.
            box: The cell, when it differs from the one last passed, and on
                the first call after ``bind``. ``None`` otherwise.

        Returns:
            The potential energy, in ``units``.
        """
        ...

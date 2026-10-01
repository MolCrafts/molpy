"""The simulation cell: a real subclass of the native box.

Everything a box computes — ``wrap``, ``unwrap``, ``images``, ``delta``,
``distances``, ``to_frac`` / ``to_cart``, ``lengths``, ``volume()`` — and the
factories ``Box.cube``, ``Box.ortho`` and ``Box.from_bounds`` are the native
box's own. molpy adds two things: a constructor that also takes no matrix (a
free box), a ``(3,)`` diagonal or an all-zero matrix, and the ``Box.Style``
vocabulary of the native ``style`` strings.
"""

from enum import Enum

import molrs
import numpy as np
from numpy.typing import ArrayLike


class Box(molrs.Box):
    """Simulation box, accepted by every native API without conversion.

    Args:
        matrix: A ``(3, 3)`` cell matrix (lattice vectors as columns) or a
            ``(3,)`` diagonal. ``None`` or an all-zero matrix makes a free
            box: an identity placeholder cell, non-periodic on every axis
            unless ``pbc`` says otherwise.
        pbc: Periodic-boundary flags per axis, shape ``(3,)``. Defaults to
            fully periodic for a cell and non-periodic for a free box.
        origin: Cartesian origin in Angstroms, shape ``(3,)``. Defaults to
            ``[0, 0, 0]``.

    Raises:
        ValueError: If ``matrix`` is neither ``(3, 3)`` nor ``(3,)``, or is
            singular without being all zero.
    """

    class Style(str, Enum):
        """The native ``style`` strings as an enumeration.

        Members compare equal to the strings the native ``style`` returns
        (``Box.Style.ORTHOGONAL == "orthogonal"``), so they work on any box,
        including ``frame.box``.
        """

        FREE = "free"
        ORTHOGONAL = "orthogonal"
        TRICLINIC = "triclinic"

    def __new__(
        cls,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ):
        h = None if matrix is None else np.asarray(matrix, dtype=float)
        if h is not None and h.shape == (3,):
            h = np.diag(h)
        if h is not None and h.shape != (3, 3):
            raise ValueError(f"matrix must be (3, 3) or (3,), got {h.shape}")
        is_free = h is None or np.allclose(h, 0.0)
        if pbc is None:
            pbc = np.full(3, not is_free)
        return super().__new__(
            cls,
            np.eye(3) if is_free else h,
            origin=np.zeros(3) if origin is None else np.asarray(origin, dtype=float),
            pbc=np.asarray(pbc, dtype=bool).reshape(3),
            cell_defined=not is_free,
        )

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        pbc: ArrayLike | None = None,
        origin: ArrayLike | None = None,
    ):
        # The native base is fully built in ``__new__``; this initializer only
        # accepts the constructor arguments.
        pass

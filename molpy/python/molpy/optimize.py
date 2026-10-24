"""Geometry optimization — ``molrs::optimize``.

An optimizer minimizes a :class:`~molpy.ff.potential.Potentials` aggregate over a
coordinate vector; it does not build the potentials and does not own a force
field. Construct the potentials with :mod:`molpy.ff.potential`, hand them here.
"""

from molpy._core import (
    Lbfgs,
    OptimizationReport,
)

__all__ = [
    "Lbfgs",
    "OptimizationReport",
]

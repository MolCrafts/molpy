"""Evaluable force terms — ``molrs::ff::potential``.

* :class:`Potentials` — kernels evaluated together (``evaluate``, the
  core ``Potential::evaluate``: energy and forces at once); ``push`` moves one
  more member in. A force field compiles into one through
  :class:`molpy.ff.PotentialCompiler`.
* :class:`WeightedTerms` — kernels each with its special-bonds weights, for a
  neighbour-driven integrator (``PotentialCompiler.compile_typed``).
* :class:`PairLjCut` — the one-type ``lj/cut`` kernel a neighbour loop feeds
  (the MD integrators' nonbond kernel): ``pair_energy_force`` for one pair,
  ``evaluate_pairs`` over a ``Neighbors`` table.
* :class:`Potential` — the protocol every Python-defined force provider
  satisfies (one method, ``evaluate(pos) -> (energy, forces)``).
"""

from molpy._core import (
    PairLjCut,
    Potentials,
    WeightedTerms,
)

from molpy.ff.potential.protocol import Potential

Potential.__module__ = __name__

__all__ = [
    "PairLjCut",
    "Potential",
    "Potentials",
    "WeightedTerms",
]

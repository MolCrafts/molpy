"""Geometry optimization: the native L-BFGS minimizer.

``LBFGS(potentials, *, fmax=0.05, max_steps=500, max_step=0.2, memory=8)``
takes the ``Potentials`` compiled from a force field for the frame under
study (``PotentialCompiler(forcefield).compile(frame)``); ``run(frame)``
returns ``(frame, OptReport)``. Composition — typify, compile, relax — is the
caller's, exactly as with any other native primitive.
"""

from molrs.optimize import LBFGS, OptReport

__all__ = ["LBFGS", "OptReport"]

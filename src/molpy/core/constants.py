"""Physical and force-field constants — :mod:`molrs.core.constants`, by identity.

``mp.core.constants.BOLTZMANN == molrs.core.constants.BOLTZMANN``: the CODATA
constants and the per-engine Coulomb and 1-4 scaling factors molrs defines.
"""

from molrs.core.constants import *  # noqa: F403
from molrs.core.constants import __all__ as _native

__all__ = [*_native]

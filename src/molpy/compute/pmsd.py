"""Einstein–Helfand ionic conductivity from the collective charge-dipole MSD.

``EinsteinConductivity`` returns the raw MSD of ``M(t) = Σ q r`` (unwrapped)
only. Fit the slope with ``LinearFit`` and apply the ``1/(6·V·k_B·T)`` SI
prefactor yourself.
"""

from molrs.compute.transport import EinsteinConductivity as EinsteinConductivity

__all__ = ["EinsteinConductivity"]

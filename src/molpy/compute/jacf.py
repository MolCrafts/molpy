"""Green–Kubo ionic conductivity from the charge-current ACF.

``GreenKuboConductivity`` returns the raw current autocorrelation
``⟨J(0)·J(t)⟩`` only. Integrate with ``CumulativeTrapezoid`` and apply the
``1/(3·V·k_B·T)`` SI prefactor yourself.
"""

from molrs.compute.transport import GreenKuboConductivity as GreenKuboConductivity

__all__ = ["GreenKuboConductivity"]

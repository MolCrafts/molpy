"""Signal processing — :mod:`molrs.signal`, mirrored by identity.

FFT autocorrelation (``acf_fft``) and cross-correlation (``xcorr_fft``),
window functions (``apply_window``) and frequency grids
(``frequency_grid``); ``mp.signal.acf_fft is molrs.signal.acf_fft``.
"""

from molrs.signal import *  # noqa: F403
from molrs.signal import __all__ as __all__

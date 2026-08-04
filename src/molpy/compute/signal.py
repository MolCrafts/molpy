"""Signal processing over sampled data — FFT autocorrelation, windows, grids.

Module-level functions: arrays in, arrays out. Import as
``molpy.compute.signal.acf_fft`` (not a top-level ``molpy.acf_fft``).
"""

from __future__ import annotations

from molrs.signal import acf_fft, apply_window, frequency_grid

__all__ = [
    "acf_fft",
    "apply_window",
    "frequency_grid",
]

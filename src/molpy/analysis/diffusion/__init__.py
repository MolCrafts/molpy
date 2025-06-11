
from molpy.analysis import BaseCompute
import numpy as np

import logging

from molpy.analysis.base import Result1D
from molpy.analysis.parallel import get_num_threads

logger = logging.getLogger(__name__)

# Use fastest available fft library
try:
    import pyfftw

    logger.info("Using PyFFTW for FFTs")
    
    pyfftw.config.NUM_THREADS = min(1, get_num_threads())
    logger.info(f"Setting number of threads to {get_num_threads()}")

    # Note that currently these functions are defined to match only the parts
    # of the numpy/scipy API that are actually used below. There is no promise
    # that other aspects of the API will be preserved.
    def fft(x, n, axis):
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.fft(a, n=n, axis=axis)
        return fft_object()

    def ifft(x, axis):
        a = pyfftw.empty_aligned(x.shape, "complex64")
        a[:] = x
        fft_object = pyfftw.builders.ifft(a, axis=axis)
        return fft_object()
except ImportError:
    try:
        from scipy.fftpack import fft, ifft

        logger.info("Using SciPy's fftpack for FFTs")
    except ImportError:
        from numpy.fft import fft, ifft

        logger.info("Using NumPy for FFTs")


def _autocorrelation(x):
    r"""Compute the autocorrelation of a sequence"""
    N = x.shape[0]
    F = fft(x, n=2 * N, axis=0)
    PSD = F * F.conjugate()
    res = ifft(PSD, axis=0)
    res = (res[:N]).real
    n = np.arange(1, N + 1)[::-1]  # N to 1
    return res / n[:, np.newaxis]



class DirectMSD(BaseCompute):
    """Direct Mean Square Displacement (MSD) calculation."""

    def __init__(self, box=None):

        if box is None:
            self._box = None
        else:
            self._box = box

    def compute(self, positions, images=None, reset=True):

        if reset:
            self._particle_msd = Result1D()

        self._called_compute = True

        positions = np.asarray(positions)
        assert len(positions.shape) == 3 and positions.shape[-1] == 3
        if images is not None:
            images = np.asarray(
                images, dtype=np.int32
            )

        # Make sure we aren't modifying the provided array
        if self._box is not None and images is not None:
            unwrapped_positions = positions.copy()
            positions = self._box.wrap(unwrapped_positions, images)

        self._particle_msd.append(
            np.linalg.norm(positions - positions[[0], :, :], axis=-1) ** 2
        )

        return self
    
class WindowedMSD(BaseCompute):
    """Windowed Mean Square Displacement (MSD) calculation."""

    def __init__(self, window_size=1, box=None):
        self.window_size = window_size
        if box is None:
            self._box = None
        else:
            self._box = box

    def compute(self, positions, images=None, reset=True):

        if reset:
            self._particle_msd = Result1D()

        self._called_compute = True

        positions = np.asarray(positions)
        assert len(positions.shape) == 3 and positions.shape[-1] == 3
        if images is not None:
            images = np.asarray(
                images, dtype=np.int32
            )

        # Make sure we aren't modifying the provided array
        if self._box is not None and images is not None:
            unwrapped_positions = positions.copy()
            positions = self._box.wrap(unwrapped_positions, images)


        # First compute the first term r^2(k+m) - r^2(k)
        N = positions.shape[0]
        D = np.square(positions).sum(axis=2)
        D = np.append(D, np.zeros(positions.shape[:2]), axis=0)
        Q = 2 * D.sum(axis=0)
        S1 = np.zeros(positions.shape[:2])
        for m in range(N):
            Q -= D[m - 1, :] + D[N - m, :]
            S1[m, :] = Q / (N - m)

        # The second term can be computed via autocorrelation
        corrs = []
        for i in range(positions.shape[2]):
            corrs.append(_autocorrelation(positions[:, :, i]))
        S2 = np.sum(corrs, axis=0)

        self._particle_msd.append(S1 - 2 * S2)

        return self
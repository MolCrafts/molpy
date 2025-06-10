
_num_threads = 0


def get_num_threads():
    r"""Get the number of threads for parallel computation.

    Returns:
        (int): Number of threads.
    """
    return _num_threads


def set_num_threads(nthreads=None):
    r"""Set the number of threads for parallel computation.

    Args:
        nthreads (int, optional):
            Number of threads to use. If :code:`None`, use all threads
            available. (Default value = :code:`None`).
    """
    global _num_threads  # noqa: PLW0603 - freud needs to maintain the state
    if nthreads is None or nthreads < 0:
        nthreads = 0

    _num_threads = nthreads


class NumThreads:
    r"""Context manager for managing the number of threads to use.

    Args:
        N (int, optional): Number of threads to use in this context. If
            :code:`None`, which will use all available threads.
            (Default value = :code:`None`).
    """

    def __init__(self, N=None):
        self.restore_N = _num_threads
        self.N = N

    def __enter__(self):
        set_num_threads(self.N)
        return self

    def __exit__(self, *args):
        set_num_threads(self.restore_N)
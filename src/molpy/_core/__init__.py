"""molpy's own root types: the column-value selectors and the trajectory splitters.

This package is private and exports nothing. Each public type has one path,
the ``molpy`` root (``mp.ElementSelector``, ``mp.TrajectorySplitter``); the
native types they work on (``mp.Frame``, ``mp.Box``, ``mp.Trajectory``, …) are
the molrs objects, re-exported there too. The package docstring of
:mod:`molpy` lists which layer owns each name.
"""

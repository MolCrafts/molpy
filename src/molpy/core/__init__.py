"""molpy's own core modules (boxes, regions, selectors, trajectories, units, …).

This package exports nothing. Every public core type has one path, the
``molpy`` root (``mp.Box``, ``mp.Region``, ``mp.Trajectory``); the native types
the core builds on (``mp.Atomistic``, ``mp.Frame``, ``mp.ForceField``, …) are
re-exported there too. The package docstring of :mod:`molpy` lists which layer
owns each name.
"""

"""molpy's own core types: the box, the configuration, selectors, trajectories.

This package exports nothing. Every public core type has one path, the
``molpy`` root (``mp.Box``, ``mp.Config``, ``mp.ElementSelector``,
``mp.Trajectory``); the native types the core builds on (``mp.Atomistic``,
``mp.Frame``, ``mp.Cuboid``, …) are re-exported there too. The package
docstring of :mod:`molpy` lists which layer owns each name.
"""

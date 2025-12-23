"""
Potential functions for molecular simulations.

This module provides potential functions for bonds, angles, dihedrals, and pair interactions.
Each potential class accepts specific parameters (e.g., r, bond_idx, bond_types) rather than
Frame objects, making them more flexible and easier to use in different contexts.

For convenience, utility functions are provided to extract data from Frame objects and
call potential functions with the extracted data.
"""

# Import all potential implementations to register them
from . import angle, bond, pair
from .angle import *
from .base import Potential, Potentials
from .bond import *
from .pair import *

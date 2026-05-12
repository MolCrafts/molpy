"""Experimental I/O — molrs-backed readers and writers.

This package mirrors :mod:`molpy.io` with implementations that delegate
to :mod:`molrs.io`.  Every public symbol has a direct counterpart in
:mod:`molpy.io` with an identical signature.

When the experimental layer matures, the files under this directory will
be promoted to replace their counterparts in :mod:`molpy.io`, and this
package will be deleted.
"""

from .readers import read_lammps_data, read_pdb, read_xyz
from .writers import write_lammps_data, write_pdb, write_xyz

__all__ = [
    "read_lammps_data",
    "read_pdb",
    "read_xyz",
    "write_lammps_data",
    "write_pdb",
    "write_xyz",
]

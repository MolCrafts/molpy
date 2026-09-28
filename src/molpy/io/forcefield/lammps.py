"""LAMMPS force-field include (``*.ff``) I/O.

Read/write of the AMBER/GAFF-style include is implemented in the native core
(the native ``read_lammps_forcefield``, the native ``write_lammps_forcefield``).
This module exposes the molpy entry points.
"""

from pathlib import Path
from typing import TextIO

from molpy.core.forcefield import ForceField


class LAMMPSForceFieldWriter:
    """Write a :class:`~molpy.ForceField` to a LAMMPS ``*.ff`` include."""

    def __init__(
        self,
        fpath: str | Path | TextIO,
        precision: int = 6,
        *,
        units: str = "real",
    ):
        """
        Args:
            fpath: Output path or file-like object.
            precision: Decimal places for floating-point coefficients.
            units: LAMMPS ``units`` style for the file (``real``, ``metal``, ``lj``).
        """
        self.precision = precision
        self.units = units
        self._fpath = fpath

    def write(
        self,
        forcefield: ForceField,
        atom_types: set[str] | None = None,
        bond_types: set[str] | None = None,
        angle_types: set[str] | None = None,
        dihedral_types: set[str] | None = None,
        improper_types: set[str] | None = None,
        skip_pair_style: bool = False,
        skip_units: bool = False,
        units: str | None = None,
    ) -> None:
        """Write ``forcefield`` (native store units) as a LAMMPS include.

        Args:
            forcefield: Force field to write.
            atom_types: Optional atom-type whitelist for pair coeffs.
            bond_types: Optional bond type-name whitelist.
            angle_types: Optional angle type-name whitelist.
            dihedral_types: Optional dihedral type-name whitelist.
            improper_types: Optional improper type-name whitelist.
            skip_pair_style: If True, omit ``pair_style`` and ``special_bonds``.
            skip_units: If True, omit the ``units`` line.
            units: Override constructor ``units`` for this write.
        """
        import molrs

        kwargs = dict(
            precision=self.precision,
            skip_pair_style=skip_pair_style,
            skip_units=skip_units,
            units=units if units is not None else self.units,
            atom_types=atom_types,
            bond_types=bond_types,
            angle_types=angle_types,
            dihedral_types=dihedral_types,
            improper_types=improper_types,
        )
        if isinstance(self._fpath, (str, Path)):
            molrs.ff.write_lammps_forcefield(str(self._fpath), forcefield, **kwargs)
        else:
            self._fpath.write(
                molrs.ff.write_lammps_forcefield_str(forcefield, **kwargs)
            )

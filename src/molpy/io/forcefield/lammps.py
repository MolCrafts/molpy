"""LAMMPS force-field include (``*.ff``) I/O.

Read/write of the AMBER/GAFF-style include is implemented in molrs
(:func:`molrs.ff.read_lammps_forcefield`, :func:`molrs.ff.write_lammps_forcefield`).
This module exposes the molpy entry points and parameter formatters for
specialized pair styles (CL&Pol Thole / Tang−Toennies).
"""

from pathlib import Path
from typing import TextIO

from molpy import ForceField
from molpy.core.fields import ForceFieldFormatter
from molpy.core.forcefield import PairCoulTTStyle, PairTholeStyle
from molpy.io.data.lammps import LammpsFieldFormatter


def _format_pair_thole(typ) -> list[float]:
    """Thole pair coefficients: alpha, a_thole (LAMMPS ``pair_style thole``)."""
    kwargs = typ.params.kwargs
    return [kwargs.get("alpha", 0.0), kwargs.get("a_thole", 2.6)]


def _format_pair_coul_tt(typ) -> list[float]:
    """Tang−Toennies pair coefficients: b, n, c (LAMMPS ``pair_style coul/tt``)."""
    kwargs = typ.params.kwargs
    return [kwargs.get("b", 4.5), kwargs.get("n", 4), kwargs.get("c", 1.0)]


class LammpsForceFieldFormatter(LammpsFieldFormatter, ForceFieldFormatter):
    """Parameter formatters for LAMMPS pair styles beyond the AMBER/GAFF set."""

    _param_formatters = {
        PairTholeStyle: _format_pair_thole,
        PairCoulTTStyle: _format_pair_coul_tt,
    }


class LAMMPSForceFieldWriter:
    """Write a :class:`~molpy.ForceField` to a LAMMPS ``*.ff`` include."""

    def __init__(self, fpath: str | Path | TextIO, precision: int = 6):
        """
        Args:
            fpath: Output path or file-like object.
            precision: Decimal places for floating-point coefficients.
        """
        self.precision = precision
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
    ) -> None:
        """Write ``forcefield`` (molrs units) as a LAMMPS include.

        Args:
            forcefield: Force field to write.
            atom_types: Optional atom-type whitelist for pair coeffs.
            bond_types: Optional bond type-name whitelist.
            angle_types: Optional angle type-name whitelist.
            dihedral_types: Optional dihedral type-name whitelist.
            improper_types: Optional improper type-name whitelist.
            skip_pair_style: If True, omit the ``pair_style`` line.
        """
        import molrs

        kwargs = dict(
            precision=self.precision,
            skip_pair_style=skip_pair_style,
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

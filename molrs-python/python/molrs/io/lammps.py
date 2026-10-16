"""LAMMPS's files — ``molrs::io::lammps``.

The doors are functions of :mod:`molrs.io`: the data file
(:func:`~molrs.io.read_lammps_data`, :func:`~molrs.io.write_lammps_data`),
molecule templates (:func:`~molrs.io.read_lammps_molecule`,
:func:`~molrs.io.read_lammps_molecule_json` and their writers), dumps
(:func:`~molrs.io.read_lammps_dump_trajectory`,
:func:`~molrs.io.write_lammps_dump_trajectory`,
:func:`~molrs.io.write_lammps_dump_local`), ``fix bond/react`` file sets
(:func:`~molrs.io.write_lammps_bond_react_map`,
:func:`~molrs.io.write_lammps_bond_react_system`), logs
(:func:`~molrs.io.read_lammps_log`, :func:`~molrs.io.read_lammps_log_str`)
and force-field files (:func:`~molrs.io.read_lammps_forcefield`, …).

This module holds the family's classes: :class:`LammpsDumpReader`, the lazy
dump reader; :class:`BondReactTemplate`, one reaction's pre/post template
pair; and the log records :func:`~molrs.io.read_lammps_log` hands out —
:class:`LammpsLog` and its runs, thermo tables, warnings, and performance and
timing summaries; :func:`is_lammps_log`, which tells a log by its first
run before it is read; and :func:`infer_lammps_atom_style`, the
``atom_style`` :func:`~molrs.io.write_lammps_data` writes a frame in when
none is named, for the input script's ``atom_style`` line.
"""

from .._native import (
    BondReactTemplate,
    LammpsCpuUse,
    LammpsDumpReader,
    LammpsLoadBalance,
    LammpsLog,
    LammpsLogHeader,
    LammpsLoopTime,
    LammpsMemoryUsage,
    LammpsNeighborStatistics,
    LammpsPerformance,
    LammpsRun,
    LammpsThermo,
    LammpsTimingBreakdown,
    LammpsTimingRow,
    LammpsWarning,
    infer_lammps_atom_style,
    is_lammps_log,
)

__all__ = [
    "BondReactTemplate",
    "LammpsCpuUse",
    "LammpsDumpReader",
    "LammpsLoadBalance",
    "LammpsLog",
    "LammpsLogHeader",
    "LammpsLoopTime",
    "LammpsMemoryUsage",
    "LammpsNeighborStatistics",
    "LammpsPerformance",
    "LammpsRun",
    "LammpsThermo",
    "LammpsTimingBreakdown",
    "LammpsTimingRow",
    "LammpsWarning",
    "infer_lammps_atom_style",
    "is_lammps_log",
]

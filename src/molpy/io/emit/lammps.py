"""LAMMPS emitter: data + in.settings + in.init + starter in-script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molrs import Atomistic
from molrs.ff import ForceField, write_lammps_forcefield_str

from molpy.io.data.lammps import write_lammps_data


class LammpsEmitter:
    """Emits a complete LAMMPS input set.

    Files written (given ``prefix="system"``):
      * ``system.data``        -- LAMMPS data file (coords + topology).
      * ``system.in.settings`` -- molrs's LAMMPS force-field include: the
        ``*_style`` line of every category the system uses (built-in or a
        registered force-field IR style; ``hybrid`` when a category spans
        several styles), its coefficients, ``special_bonds``.
      * ``system.in.init``     -- units/atom_style/boundary.
      * ``system.in``          -- starter run script.

    The styles and coefficients come from one place, molrs's LAMMPS writer
    (:func:`molpy.io.write_lammps_forcefield`); molpy does not name styles
    itself, so a style molrs can write is emitted and one it cannot is
    refused by molrs, by name.
    """

    name = "lammps"

    def emit(
        self,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        atom_style: str = "full",
        units: str = "real",
        **_opts: Any,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        data_path = out_dir / f"{prefix}.data"
        settings_path = out_dir / f"{prefix}.in.settings"
        init_path = out_dir / f"{prefix}.in.init"
        run_path = out_dir / f"{prefix}.in"

        init_lines = [
            f"# MolPy-generated LAMMPS init for {prefix}",
            f"units {units}",
            "atom_style " + atom_style,
            "boundary p p p",
        ]

        # 1) in.settings first, so a style molrs cannot write fails before any
        # file is written; 2) the data file, keyed by the same frame's labels.
        # The settings are included after read_data, where LAMMPS rejects a
        # ``units`` command; in.init owns it, the styles and coefficients
        # follow it.
        frame = atomistic.to_frame()
        settings = write_lammps_forcefield_str(ff, frame, skip_units=True, units=units)
        write_lammps_data(data_path, frame)
        settings_path.write_text(settings)

        # 3) in.init
        init_path.write_text("\n".join(init_lines) + "\n")

        # 4) starter run script
        run_path.write_text(
            _LAMMPS_RUN_TEMPLATE.format(
                prefix=prefix,
                init=init_path.name,
                data=data_path.name,
                settings=settings_path.name,
            )
        )
        return [data_path, settings_path, init_path, run_path]


_LAMMPS_RUN_TEMPLATE = """\
# MolPy-generated LAMMPS starter script for {prefix}
include {init}
read_data {data}
include {settings}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Minimise
minimize        1.0e-4 1.0e-6 1000 10000

# Basic NVT ensemble — edit temperature and timestep as needed
velocity        all create 300.0 12345 loop geom
fix             1 all nvt temp 300.0 300.0 100.0
timestep        1.0
thermo          100
thermo_style    custom step temp pe ke etotal press
run             1000
unfix           1
"""

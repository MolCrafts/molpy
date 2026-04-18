"""LAMMPS emitter: data + in.settings + in.init + starter in-script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField


class LammpsEmitter:
    """Emits a complete LAMMPS input set.

    Files written (given ``prefix="system"``):
      * ``system.data``        -- LAMMPS data file (coords + topology).
      * ``system.in.settings`` -- pair/bond/angle/... coefficient commands.
      * ``system.in.init``     -- units/boundary/atom_style/pair_style/...
      * ``system.in``          -- starter run script.
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

        # 1) data file
        try:
            from molpy.io.data.lammps import LammpsDataWriter

            frame = atomistic.to_frame()
            LammpsDataWriter(data_path, atom_style=atom_style).write(frame)
        except Exception:
            data_path.write_text(
                f"# {prefix} data\n# (LammpsDataWriter unavailable for this FF)\n"
            )

        # 2) in.settings from FF
        try:
            from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter

            LAMMPSForceFieldWriter(settings_path).write(ff)
        except Exception:
            settings_path.write_text("# fallback: no native LAMMPS FF writer output\n")

        # 3) in.init
        init_lines = [
            f"# MolPy-generated LAMMPS init for {prefix}",
            f"units {units}",
            "atom_style " + atom_style,
            "boundary p p p",
        ]
        pair_styles = {
            s.name
            for s in ff.styles.bucket(
                type(ff.styles.bucket(object)[0])
                if ff.styles.bucket(object)
                else object
            )
            if False
        }
        # pair_style / bond_style / angle_style / ... derived from ff
        for kind, cmd in [
            ("bond", "bond_style"),
            ("angle", "angle_style"),
            ("dihedral", "dihedral_style"),
            ("improper", "improper_style"),
            ("pair", "pair_style"),
        ]:
            style_names = _collect_style_names(ff, kind)
            if style_names:
                init_lines.append(f"{cmd} {style_names[0]}")
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


def _collect_style_names(ff: ForceField, kind_prefix: str) -> list[str]:
    """Return the style.name of every style whose class name starts with the kind.

    Uses a duck-typed class-name match to avoid importing the Style base classes
    here (keeps emit/ free of heavy imports).
    """
    from molpy.core.forcefield import (
        AngleStyle,
        BondStyle,
        DihedralStyle,
        ImproperStyle,
        PairStyle,
    )

    mapping = {
        "bond": BondStyle,
        "angle": AngleStyle,
        "dihedral": DihedralStyle,
        "improper": ImproperStyle,
        "pair": PairStyle,
    }
    cls = mapping.get(kind_prefix)
    if cls is None:
        return []
    return [s.name for s in ff.get_styles(cls)]

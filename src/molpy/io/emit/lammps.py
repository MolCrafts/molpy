"""LAMMPS emitter: data + in.settings + in.init + starter in-script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molrs import Atomistic
from molrs.ff import ForceField, write_lammps_forcefield

from molpy.io.data.lammps import write_lammps_data


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

        # in.init lines first: an unsupported (hybrid) style fails before any
        # file is written.
        init_lines = [
            f"# MolPy-generated LAMMPS init for {prefix}",
            f"units {units}",
            "atom_style " + atom_style,
            "boundary p p p",
        ]
        # pair_style / bond_style / angle_style / ... derived from ff
        for kind in ("bond", "angle", "dihedral", "improper", "pair"):
            style_name = _style_name(ff, kind)
            if style_name is not None:
                init_lines.append(f"{kind}_style {style_name}")

        # 1) data file and 2) in.settings, both keyed by the same frame's labels.
        # The settings are included after read_data, where LAMMPS rejects a
        # ``units`` command; in.init owns it, the coefficients follow it.
        frame = atomistic.to_frame()
        write_lammps_data(data_path, frame)
        write_lammps_forcefield(settings_path, ff, frame, skip_units=True, units=units)

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


def _style_name(ff: ForceField, category: str) -> str | None:
    """The one style ``ff`` defines in ``category`` (``"bond"``, ``"pair"``, …).

    Returns ``None`` when the category has no style. One ``*_style`` line is
    written per category; hybrid styles are not supported.

    Raises:
        ValueError: ``ff`` defines more than one style in ``category``.
    """
    names = [style.name for style in ff.get_styles(category)]
    if len(names) > 1:
        raise ValueError(
            f"one {category}_style line is written and hybrid styles are not "
            f"supported; the force field defines {category} styles {names}"
        )
    return names[0] if names else None

"""``AmberTools`` — one object for GAFF parameterisation via the Amber suite.

A single facade owning the conda-env / force-field / charge-method config, with
member functions for the three things every GAFF workflow needs:

* :meth:`parameterize` — a small molecule (antechamber → parmchk2 → tleap).
* :meth:`parameterize_ion` — a monatomic ion from literature Lennard-Jones
  parameters (no charge calc), with the ``addAtomTypes`` element mapping baked
  in so tleap never writes ``ATOMIC_NUMBER = -1``.
* :meth:`build_polymer` — a chain via :class:`AmberPolymerBuilder` (cached, so a
  whole config matrix reuses one monomer parameterisation).

Every result is charge-neutralised to its integer target, so an assembled
system is neutral by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.io import read_amber
from molpy.io.writers import write_pdb


@dataclass
class AmberResult:
    """A parameterised component: its :class:`Frame` and its ForceField."""

    frame: Any
    forcefield: Any

    @property
    def ff(self) -> Any:
        return self.forcefield


def _neutralize(frame: Any, target: float) -> None:
    q = np.asarray(frame["atoms"]["charge"], dtype=float)
    frame["atoms"]["charge"] = q - (q.sum() - target) / len(q)


class AmberTools:
    """GAFF parameterisation facade over antechamber / parmchk2 / tleap / prepgen."""

    def __init__(
        self,
        *,
        env: str | Path | None = None,
        env_manager: str | None = None,
        force_field: str = "gaff2",
        charge_method: str = "bcc",
        work_dir: str | Path = "amber_work",
    ) -> None:
        self.env = env
        self.env_manager = env_manager
        self.force_field = force_field
        self.charge_method = charge_method
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._polymer_builder: Any = None

    # -- small molecule ----------------------------------------------------
    def parameterize(
        self, struct: Atomistic, *, net_charge: int = 0, name: str = "MOL"
    ) -> AmberResult:
        """Parameterise a small molecule; returns a neutralised :class:`AmberResult`."""
        from molpy.wrapper.antechamber import AntechamberWrapper
        from molpy.wrapper.prepgen import Parmchk2Wrapper
        from molpy.wrapper.tleap import TLeapWrapper

        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for idx, atom in enumerate(struct.atoms, start=1):
            if atom.get("name") is None:
                atom["name"] = f"{atom.get('element', 'X')}{idx}"
        write_pdb(d / f"{name}.pdb", struct.to_frame())

        ac = AntechamberWrapper(
            name="antechamber", workdir=d, env=self.env, env_manager=self.env_manager
        )
        r = ac.atomtype_assign(
            input_file=(d / f"{name}.pdb").absolute(),
            output_file=(d / f"{name}.mol2").absolute(),
            input_format="pdb",
            output_format="mol2",
            charge_method=self.charge_method,
            atom_type=self.force_field,
            net_charge=net_charge,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"antechamber failed for {name}:\n{r.stderr or r.stdout}"
            )

        Parmchk2Wrapper(
            name="parmchk2", workdir=d, env=self.env, env_manager=self.env_manager
        ).generate_parameters(
            input_file=d / f"{name}.mol2",
            output_file=d / f"{name}.frcmod",
            force_field=self.force_field,
        )
        leap = (
            f"source leaprc.{self.force_field}\n"
            f"{name} = loadmol2 {d / f'{name}.mol2'}\n"
            f"loadamberparams {d / f'{name}.frcmod'}\n"
            f"saveamberparm {name} {d / f'{name}.prmtop'} {d / f'{name}.inpcrd'}\n"
            "quit\n"
        )
        TLeapWrapper(
            name="tleap", workdir=d, env=self.env, env_manager=self.env_manager
        ).run_from_script(leap)
        if not (d / f"{name}.prmtop").exists():
            raise RuntimeError(f"tleap failed to produce {name}.prmtop")
        frame, ff = read_amber(d / f"{name}.prmtop", d / f"{name}.inpcrd")
        _neutralize(frame, float(net_charge))
        return AmberResult(frame, ff)

    # -- monatomic ion -----------------------------------------------------
    def parameterize_ion(
        self,
        element: str,
        *,
        charge: float,
        rmin_half: float,
        epsilon: float,
        mass: float,
        name: str = "ION",
    ) -> AmberResult:
        """Parameterise a monatomic ion from literature LJ parameters (no charge calc)."""
        from molpy.wrapper.tleap import TLeapWrapper

        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        atype = element.upper()
        (d / f"{name}.frcmod").write_text(
            f"{element} ion parameters\nMASS\n{atype:<4}  {mass:.4f}  0.0\n\n"
            f"BOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\nNONBON\n"
            f"  {atype:<8} {rmin_half:.4f} {epsilon:.4f}\n\n"
        )
        (d / f"{name}.mol2").write_text(
            "@<TRIPOS>MOLECULE\n"
            f"{name}\n 1 0 0 0 0\nSMALL\nUSER_CHARGES\n\n@<TRIPOS>ATOM\n"
            f"      1 {atype:<4}      0.0000    0.0000    0.0000 {atype:<4} 1  {name}"
            f"      {charge:.6f}\n@<TRIPOS>BOND\n"
        )
        # addAtomTypes maps the bare atom type to its element so tleap writes a
        # real ATOMIC_NUMBER (else it emits -1 and read_amber fails).
        leap = (
            f"source leaprc.{self.force_field}\n"
            f'addAtomTypes {{ {{ "{atype}" "{element}" "sp3" }} }}\n'
            f"loadamberparams {d / f'{name}.frcmod'}\n"
            f"{name} = loadmol2 {d / f'{name}.mol2'}\n"
            f"saveamberparm {name} {d / f'{name}.prmtop'} {d / f'{name}.inpcrd'}\n"
            "quit\n"
        )
        TLeapWrapper(
            name="tleap", workdir=d, env=self.env, env_manager=self.env_manager
        ).run_from_script(leap)
        if not (d / f"{name}.prmtop").exists():
            raise RuntimeError(f"tleap failed to produce {name}.prmtop")
        frame, ff = read_amber(d / f"{name}.prmtop", d / f"{name}.inpcrd")
        _neutralize(frame, float(charge))
        return AmberResult(frame, ff)

    # -- polymer -----------------------------------------------------------
    def build_polymer(
        self,
        cgsmiles: str,
        *,
        library: Mapping[str, Atomistic],
        net_charges: Mapping[str, int] | None = None,
    ) -> AmberResult:
        """Assemble a chain from a CGSmiles sequence + monomer library (cached)."""
        from .polymer.ambertools import AmberPolymerBuilder

        if self._polymer_builder is None:
            self._polymer_builder = AmberPolymerBuilder(
                library=library,
                force_field=self.force_field,  # type: ignore[arg-type]
                charge_method=self.charge_method,
                work_dir=self.work_dir / "polymer",
                env=self.env,
                env_manager=self.env_manager,
                net_charges=net_charges,
            )
        result = self._polymer_builder.build(cgsmiles)
        _neutralize(result.frame, 0.0)
        return AmberResult(result.frame, result.forcefield)

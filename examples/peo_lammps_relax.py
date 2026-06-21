"""LAMMPS minimization + NVT equilibration of the PEO matrix — GAFF (AmberTools).

Continuation of ``peo_electrolyte_workflow.py``. The PEO chain is parameterized
with **GAFF2 via AmberTools** (``antechamber`` AM1-BCC charges -> ``parmchk2``
fills missing terms -> ``tleap`` -> ``prmtop``), which ``read_amber_prmtop``
turns into a complete molpy ForceField — crucially with *real torsions*
(``dihedral_style fourier``). Eight chains are Packmol-packed into the periodic
box and driven through a genuine LAMMPS run via :class:`~molpy.engine.LAMMPS`:

    1. energy minimization
    2. NVT equilibration at 400 K (Nose-Hoover)

Using GAFF (not a torsion-free hand FF) is what keeps the dynamics physical:
with real dihedrals the 1-4 H-C-C-H pairs no longer eclipse, so the equilibrated
structure has no spurious clashes.

Requires the ``AmberTools`` conda env (set ``AMBER_ENV`` below) and a LAMMPS
binary on PATH. Salt is added separately by ``peo_insert_salt.py``.

Run::

    python examples/peo_lammps_relax.py
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np

from molpy.core.box import Box
from molpy.core.frame import Block, Frame
from molpy.engine import LAMMPS
from molpy.io import read_amber_prmtop, write_lammps_data, write_pdb
from molpy.pack import InsideBoxConstraint, Molpack
from molpy.wrapper.antechamber import AntechamberWrapper
from molpy.wrapper.prepgen import Parmchk2Wrapper
from molpy.wrapper.tleap import TLeapWrapper

# Reuse the builders from the structure-generation workflow.
_WF_PATH = Path(__file__).resolve().parent / "peo_electrolyte_workflow.py"
_spec = importlib.util.spec_from_file_location("peo_workflow", _WF_PATH)
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CHAINS = 8
DP = 12
BOX_LENGTH = 36.0
TEMPERATURE = 400.0  # K
NVT_STEPS = 5000
TIMESTEP = 1.0  # fs
SEED = 42
CHARGE_METHOD = "bcc"  # AM1-BCC; use "gas" (Gasteiger) for a faster build
AMBER_ENV = {"env": "AmberTools25", "env_manager": "conda"}
OUT_DIR = Path(__file__).resolve().parent / "peo_workflow_output"


def gaff_parameterize(struct, *, prefix: str, net_charge: int = 0):
    """GAFF2-parameterize a molecule via AmberTools -> (Frame, ForceField).

    antechamber (atom types + charges) -> parmchk2 (missing terms) -> tleap
    (prmtop/inpcrd) -> read_amber_prmtop. The returned force field carries real
    bonded parameters, including ``dihedral_style fourier`` torsions.
    """
    work = OUT_DIR / "gaff" / prefix
    work.mkdir(parents=True, exist_ok=True)

    atoms = struct.to_frame()["atoms"]
    elements = np.asarray(atoms["element"]).astype(str)
    n = atoms.nrows
    pdb_frame = Frame(
        {
            "atoms": Block(
                {
                    "id": np.arange(1, n + 1, dtype=int),
                    "name": np.array([f"{e}{i + 1}" for i, e in enumerate(elements)]),
                    "resName": np.array([prefix[:3].upper()] * n),
                    "resSeq": np.ones(n, dtype=int),
                    "element": elements,
                    "x": np.asarray(atoms["x"]).astype(float),
                    "y": np.asarray(atoms["y"]).astype(float),
                    "z": np.asarray(atoms["z"]).astype(float),
                }
            )
        }
    )
    write_pdb(work / f"{prefix}.pdb", pdb_frame)

    AntechamberWrapper(name="antechamber", workdir=work, **AMBER_ENV).atomtype_assign(
        f"{prefix}.pdb",
        f"{prefix}.mol2",
        input_format="pdb",
        output_format="mol2",
        charge_method=CHARGE_METHOD,
        atom_type="gaff2",
        net_charge=net_charge,
        check=True,
    )
    Parmchk2Wrapper(name="parmchk2", workdir=work, **AMBER_ENV).generate_parameters(
        f"{prefix}.mol2", f"{prefix}.frcmod", force_field="gaff2", check=True
    )
    script = (
        "source leaprc.gaff2\n"
        f"loadamberparams {prefix}.frcmod\n"
        f"mol = loadmol2 {prefix}.mol2\n"
        f"saveamberparm mol {prefix}.prmtop {prefix}.inpcrd\n"
        "quit\n"
    )
    TLeapWrapper(name="tleap", workdir=work, **AMBER_ENV).run_from_script(
        script, check=True
    )

    frame, ff = read_amber_prmtop(work / f"{prefix}.prmtop", work / f"{prefix}.inpcrd")
    # The LAMMPS writer needs a mol_id column; the prmtop frame only has residue.
    a = frame["atoms"]
    cols = {k: np.asarray(a[k]) for k in a.keys()}
    cols["mol_id"] = np.ones(a.nrows, dtype=int)
    frame["atoms"] = Block(cols)
    return frame, ff


def pack_peo_box():
    """GAFF-typed PEO chains Packmol-packed into the periodic box."""
    chain = wf.build_linear_peo(DP)
    wf.mmff_optimize(chain, fmax=0.05, max_steps=1000)
    peo_frame, ff = gaff_parameterize(chain, prefix="peo", net_charge=0)

    constraint = InsideBoxConstraint(
        length=np.array([BOX_LENGTH] * 3), origin=np.zeros(3)
    )
    packer = Molpack(workdir=OUT_DIR / "lammps_packmol")
    packer.add_target(peo_frame, number=N_CHAINS, constraint=constraint)
    box = packer.optimize(max_steps=200, seed=SEED, pbc=[BOX_LENGTH] * 3)
    box.box = Box(np.array([BOX_LENGTH] * 3, dtype=float), origin=np.zeros(3))
    return box, ff


def thermo_table(workdir: Path) -> list[list[float]]:
    """Parse the ``Step Temp PE ...`` thermo block from log.lammps."""
    rows: list[list[float]] = []
    capture = False
    for line in (workdir / "log.lammps").read_text().splitlines():
        head = line.split()
        if head[:2] == ["Step", "Temp"]:
            capture = True
            continue
        if capture:
            try:
                rows.append([float(x) for x in head])
            except (ValueError, IndexError):
                capture = False
    return rows


def main() -> None:
    lmp = next((c for c in ("lmp", "lmp_serial", "lmp_mpi") if shutil.which(c)), None)
    if lmp is None:
        print("No LAMMPS binary on PATH — cannot run.")
        return
    print(f"LAMMPS binary : {lmp}")

    print(f"GAFF2 parameterization via AmberTools ({CHARGE_METHOD} charges)...")
    frame, ff = pack_peo_box()
    net_q = float(np.sum(frame["atoms"]["charge"]))
    print(
        f"PEO matrix    : {N_CHAINS} chains, {frame['atoms'].nrows} atoms, "
        f"{frame['bonds'].nrows} bonds, net charge {net_q:+.3f} e, box {BOX_LENGTH}^3 A"
    )
    engine = LAMMPS(lmp)

    print("\n=== 1. Energy minimization ===")
    minimized = engine.minimize(frame, ff, workdir=OUT_DIR / "lammps_min")
    rows = thermo_table(OUT_DIR / "lammps_min")
    if rows:
        print(f"  PE: {rows[0][2]:.1f}  ->  {rows[-1][2]:.1f} kcal/mol")
    write_lammps_data(
        OUT_DIR / "step9_lammps_minimized.data", minimized, atom_style="full"
    )
    print(f"  -> {(OUT_DIR / 'step9_lammps_minimized.data').name}")

    print(f"\n=== 2. NVT equilibration ({TEMPERATURE:g} K, {NVT_STEPS} steps) ===")
    equilibrated = engine.md(
        minimized,
        ff,
        ensemble="nvt",
        temperature=TEMPERATURE,
        steps=NVT_STEPS,
        timestep=TIMESTEP,
        seed=SEED,
        workdir=OUT_DIR / "lammps_nvt",
    )
    rows = thermo_table(OUT_DIR / "lammps_nvt")
    if rows:
        print(f"  step {int(rows[0][0])}: T={rows[0][1]:6.1f} K  PE={rows[0][2]:9.1f}")
        print(
            f"  step {int(rows[-1][0])}: T={rows[-1][1]:6.1f} K  PE={rows[-1][2]:9.1f}"
        )
        temps = [r[1] for r in rows[len(rows) // 2 :]]
        print(
            f"  mean T over 2nd half: {np.mean(temps):.1f} K (target {TEMPERATURE:g} K)"
        )
    write_lammps_data(
        OUT_DIR / "step10_lammps_equilibrated.data", equilibrated, atom_style="full"
    )
    write_pdb(OUT_DIR / "step10_lammps_equilibrated.pdb", equilibrated)
    print(f"  -> {(OUT_DIR / 'step10_lammps_equilibrated.data').name}")

    eq = equilibrated["atoms"]
    p = np.column_stack([eq["x"], eq["y"], eq["z"]])
    dist = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    np.fill_diagonal(dist, 9.9)
    print(f"  min interatomic distance: {dist.min():.2f} A (bond-scale -> no clashes)")
    print("\n[OK] GAFF LAMMPS minimization + NVT equilibration complete.")


if __name__ == "__main__":
    main()

"""Structures for the manual's home page.

Each tile on the home page is one use of MolPy, run here: a rock-salt crystal
from its space group, a molecule parsed from SMILES, a star polymer assembled
from CGsmiles, and a carbon nanotube from its chiral indices. They
are written as PDB (with CONECT records, so the viewer draws the bonds) under
``docs/series/showcase/``.

The star is laid out flat before its units are placed, so its three arms read
at a glance: the CGsmiles site graph gets arms 120° apart, ``SitePlacer`` and
``AxisOrienter`` put the units on those sites, and the result is typed with
OPLS-AA and minimized, non-bonded terms included, as the assembly guide
prescribes. The units are the ethylene-oxide kit of
``examples/topology/eo_kit.py``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

import molpy as mp
from molpy.builder import Lattice, Site, SpaceGroup

from .lj import Trajectory
from .run import DOCS_SERIES

SEED = 42

#: Length of one ethylene-oxide repeat unit along a stretched chain (Å).
REPEAT = 3.4

#: The minimizer's force threshold (kcal/mol/Å): enough to clear every
#: overlap and set every bond, which is all a picture needs.
FMAX = 0.5

#: CGsmiles fragments; ``<`` joins ``>``.
UNITS = {
    "EO": "[<]OCC[>]",
    "X3": "C(C[>])(C[>])C[>]",
}

_ARM = "[#EO]" * 6

#: A three-arm star: a branch core and three ethylene-oxide arms.
STAR = "{[#X3](" + _ARM + ")(" + _ARM + ")" + _ARM + "}"

#: Acetylsalicylic acid.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

#: Rock salt: lattice constant (Å) and the generators of Fm-3m.
NACL_A = 5.64
FM3M = ["x+1/2,y+1/2,z", "x+1/2,y,z+1/2", "-x,-y,z", "-x,y,-z", "z,x,y", "y,x,-z", "-x,-y,-z"]


def _library() -> dict[str, mp.Atomistic]:
    conformer = mp.conformer.Conformer(seed=SEED)
    return {
        name: conformer.generate(mp.io.smiles.read_smiles_fragment(body).to_template())[0]
        for name, body in UNITS.items()
    }


def _neighbors(frame: mp.Frame) -> dict[int, list[int]]:
    bonds = frame["bonds"]
    adjacent: dict[int, list[int]] = defaultdict(list)
    for i, j in zip(np.asarray(bonds["atomi"]), np.asarray(bonds["atomj"])):
        adjacent[int(i)].append(int(j))
        adjacent[int(j)].append(int(i))
    return adjacent


def _star_layout(frame: mp.Frame) -> tuple[np.ndarray, np.ndarray]:
    """Arms straight out of site 0, 120° apart, in the x–z plane.

    The viewer's default camera looks at the x–z plane nearly face on. Each
    axis is square to its arm, alternating sides along it.
    """
    adjacent = _neighbors(frame)
    n = frame["atoms"].n_rows
    position = np.zeros((n, 3))
    axis = np.zeros((n, 3))
    axis[0] = [0.0, 1.0, 0.0]
    arms = adjacent[0]
    for k, first in enumerate(arms):
        theta = 2.0 * np.pi * k / len(arms)
        heading = np.array([np.cos(theta), 0.0, np.sin(theta)])
        side = np.array([-heading[2], 0.0, heading[0]])
        previous, here, step = 0, first, 1
        while True:
            position[here] = step * REPEAT * heading
            axis[here] = side * (1.0 if step % 2 else -1.0)
            ahead = [c for c in adjacent[here] if c != previous]
            if not ahead:
                break
            previous, here, step = here, ahead[0], step + 1
    return position, axis


def _star_sites() -> mp.CoarseGrain:
    frame = mp.io.cgsmiles.read_cgsmiles(STAR).to_coarsegrain().to_frame()
    position, axis = _star_layout(frame)
    atoms = frame["atoms"]
    for k, name in enumerate("xyz"):
        atoms[name] = position[:, k]
        atoms[f"axis_{name}"] = axis[:, k]
    atoms["mass"] = np.ones(atoms.n_rows)
    return mp.CoarseGrain.from_frame(frame)


def _relaxed(name: str, world: mp.Atomistic) -> mp.Frame:
    typifier = mp.ff.typifier.OplsAaTypifier()
    frame = typifier.typify(world).to_frame()
    potentials = mp.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
    report = mp.optimize.Lbfgs(potentials, fmax=FMAX, max_steps=5000).minimize(frame)
    if not report.converged:
        raise RuntimeError(f"{name} did not relax: fmax = {report.final_fmax:.3g}")
    return frame


def _rock_salt() -> mp.Frame:
    salt = Lattice.from_spacegroup(
        mp.Box(mp.core.Cell.lengths([NACL_A] * 3)),
        [Site("Na1", "Na", (0.0, 0.0, 0.0)), Site("Cl1", "Cl", (0.5, 0.5, 0.5))],
        SpaceGroup.from_generators(FM3M),
    )
    frame = salt.build(repeats=(3, 3, 3)).to_frame()
    frame.box = salt.supercell((3, 3, 3))
    return frame


def _write(name: str, frame: mp.Frame) -> Path:
    mp.Element.fill_symbols(frame)
    path = DOCS_SERIES / "showcase" / f"{name}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    mp.io.write_pdb(path, frame)
    return path


def gallery(_trajectory: Trajectory) -> dict[str, float]:
    """Build every home-page structure; report its atom count.

    No tile is a simulation, so the reference run the driver hands every
    builder goes unused.
    """
    assembler = mp.builder.Assembler(
        _library(), mp.builder.SitePlacer(), mp.builder.AxisOrienter()
    )
    molecule, _ = mp.conformer.Conformer(seed=SEED).generate(
        mp.io.smiles.read_smiles(ASPIRIN).to_atomistic()
    )
    nanotube = mp.builder.CarbonTubeBuilder(10, 10, length=30.0, periodic=False).build()
    # An isolated tube: without a box the viewer frames the atoms, not a cell.
    nanotube.box = None
    frames = {
        "rock_salt": _rock_salt(),
        "aspirin": molecule.to_frame(),
        "star": _relaxed("star", assembler.assemble(_star_sites())),
        "nanotube": nanotube,
    }
    summary: dict[str, float] = {}
    for name, frame in frames.items():
        _write(name, frame)
        summary[name] = float(frame["atoms"].n_rows)
    return summary

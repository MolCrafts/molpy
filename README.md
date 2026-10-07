<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molpy
</h1>

<p><strong>A programmable toolkit for molecular simulation workflows</strong></p>

<p>
  <a href="https://github.com/MolCrafts/molpy/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molpy/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"></a>
  <a href="https://pypi.org/project/molcrafts-molpy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/molcrafts-molpy?style=flat-square&logo=pypi&logoColor=white&label=PyPI"></a>
  <a href="https://pypi.org/project/molcrafts-molpy/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/molcrafts-molpy?style=flat-square&logo=python&logoColor=white"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square"></a>
</p>

<p>
  <a href="https://docs.molcrafts.org/molpy/"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="https://docs.molcrafts.org/molpy/getting-started/examples/"><b>Examples</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

MolPy is a Python toolkit for the full molecular-system workflow — parsing,
building, editing, typing, analyzing, and reading/writing simulation formats.
Packing a box is the job of the companion package
[molpack](https://docs.molcrafts.org/molpack/).

> **Under active development.** Public APIs may change between minor releases.

## Vision

Molecular modeling is fragmented. Every simulation code has its own file formats
and conventions; every task — parsing, building, typing, analysis,
visualization — lives in a separate library; and moving a system between them
means writing throwaway glue.

molpy aims to be the **common foundation** beneath that workflow: one explicit,
programmable representation of a molecular system that every stage can share.
Parse a structure into it, build on it, type and analyze it — then hand the
*same object* onward, with no conversion step in between.

That representation is meant to be built on, not just used. It is the data model
the [MolCrafts ecosystem](#molcrafts-ecosystem) extends — visualization,
experiment management, agent access — and it reads the same whether a human
writes it or an agent calls it.

## Capabilities

Every name is reached through `import molpy as mp`, and molpy is a thin layer
over the native core (molrs): every native name is the molrs object. Every
molrs subsystem is mirrored by a molpy module (`mp.core` for `molrs.core` —
stores, graphs, box, neighbour search and units — then `mp.io`, `mp.ff`, `mp.perceive`,
`mp.compute`, `mp.builder`, …). The core data classes you handle directly
(`mp.Frame`, `mp.Block`, `mp.Trajectory`, `mp.Box`, `mp.Atomistic`, …) are
promoted to the root, as the same objects.

| Where | Capability |
|---|---|
| **`mp.*`, `core`, `perceive`, `conformer`, `optimize`** | Explicit data model — `Frame` / `Block` columnar arrays, `Trajectory`, editable `Atomistic` / `CoarseGrain` graphs and `Box` on the root; regions, neighbour search, units, molpy's selectors and trajectory splitters in `mp.core`; perception and SMARTS (`mp.perceive.SmartsPattern`), 3D conformers (`mp.conformer.Conformer`), the `mp.optimize.Lbfgs` minimizer |
| **`ff`** | Force fields, mirroring `molrs.ff` — `forcefield` (the `ForceField` data model), `potential` (kernels), `compile` (`PotentialCompiler`), `typifier` (OPLS-AA, MMFF94, GAFF natively; GAFF / GAFF2 via AmberTools), `charge`, `ir` (the IR's vocabulary), `style_registry` (your own styles from Python), `params`, `clpol_scaling` |
| **`builder`** | Site-graph assembly (`Assembler`), graphene and nanotubes, polymer planning and polydispersity, crystals, virtual sites, packing templates |
| **`compute`** · **`signal`** | Analysis — `Rdf`, `Msd`, conductivity, order parameters, Voronoi, dielectric and vibrational spectra, … (native kernels); FFT correlation and windows |
| **`io`** | Read/write — PDB, GRO, LAMMPS data, XYZ, AMBER, SMILES / CGsmiles, trajectories, `*.mrec` records, … |
| **`engine`** | MD input generation & run management — LAMMPS, GROMACS, OpenMM, CP2K |
| **`wrapper` · `adapter`** | External CLIs (Antechamber, tleap, …) and optional RDKit in-memory bridge |

## Install

```bash
pip install molcrafts-molpy

# Browser (Pyodide) — needs molcrafts-molrs Pyodide wheel on PyPI:
# await micropip.install("molcrafts-molrs")
# await micropip.install("molcrafts-molpy")
```

Core dependencies: NumPy and
[molrs](https://github.com/MolCrafts/molrs) (`molcrafts-molrs>=0.16.0,<0.17`)
plus the MolCrafts logging/config packages. Optional: RDKit (adapter example),
AmberTools (GAFF charges).

> **Nightly builds.** Bleeding-edge snapshots are published to the separate
> project `molcrafts-molpy-nightly` (versioned `X.Y.Z.devN`) on every push to
> the `nightly` branch. Install with `pip install --pre molcrafts-molpy-nightly`.
> It imports as `molpy`, so it cannot be installed alongside the stable
> `molcrafts-molpy` (same as `tensorflow` vs `tf-nightly`).

<details>
<summary>Install from source (development)</summary>

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
uv sync --extra dev
pre-commit install --hook-type pre-commit --hook-type pre-push
# the two gates (same as the hooks / CI):
uv run --no-project --with 'tox>=4.23' --with ruff==0.16.1 --with ty==0.0.65 tox -e lint
uv run --extra dev python -m pytest tests/ -n auto
```

`uv sync` installs `molcrafts-molrs` from PyPI within the pinned minor line.
To develop against a local molrs checkout instead, see
[Building molrs from source](https://docs.molcrafts.org/molpy/developer/development-setup/#building-molrs-from-source).

See [docs/developer/development-setup](https://docs.molcrafts.org/molpy/developer/development-setup/)
for the full workflow.

</details>

## Quick start

Parse a SMILES string, assign OPLS-AA types, and write LAMMPS input files:

```python
import molpy as mp

from pathlib import Path

mol       = mp.io.smiles.SmilesIr("CCO").to_atomistic()     # ethanol from SMILES
mol3d, _  = mp.conformer.Conformer(seed=42).generate(mol)   # hydrogens + 3D coordinates

typifier  = mp.ff.typifier.OplsAaTypifier()          # carries the OPLS-AA library
typed     = typifier.typify(mol3d)
ff        = typifier.forcefield()                 # parameters of the assigned types

system    = typed.to_frame()
system.box = mp.Box.cube(30.0)
system["atoms"]["mol_id"] = mp.Topology.from_frame(system).connected_components() + 1

# the pair cutoff is a run setting: you declare it, molpy never invents one
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0

out = Path("output"); out.mkdir(exist_ok=True)
mp.io.write_lammps_data(out / "system.data", system)
mp.io.write_lammps_forcefield(out / "system.ff", ff, system)
```

More workflows — packed solvent boxes, virtual-site models, polymer
architectures (the stress test for MolPy's editing machinery), AmberTools
parameterization — are in the
**[Example Gallery](https://docs.molcrafts.org/molpy/getting-started/examples/)**
and the task-oriented [Guides](https://docs.molcrafts.org/molpy/user-guide/).

## Documentation

Full documentation, including executable notebooks:
**[docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)**

- [Getting Started](https://docs.molcrafts.org/molpy/getting-started/) — install and first example
- [Example Gallery](https://docs.molcrafts.org/molpy/getting-started/examples/) — short copy-paste workflows
- [Guides](https://docs.molcrafts.org/molpy/user-guide/) — task-oriented notebooks
- [Concepts](https://docs.molcrafts.org/molpy/tutorials/) — data model deep dives
- [API Reference](https://docs.molcrafts.org/molpy/api/) — full API

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| **molpy** | Python toolkit — the shared molecular data model & workflow layer — this repo |
| [molrs](https://github.com/MolCrafts/molrs)     | Rust core — molecular data structures & compute kernels (native + WASM) |
| [molpack](https://github.com/MolCrafts/molpack) | Packmol-grade molecular packing (Rust + Python) |
| [molvis](https://github.com/MolCrafts/molvis)   | WebGL molecular visualization & editing |
| [molexp](https://github.com/MolCrafts/molexp)   | Workflow & experiment-management platform |
| [molnex](https://github.com/MolCrafts/molnex)   | Molecular machine-learning framework |
| [molq](https://github.com/MolCrafts/molq)       | Unified job queue — local / SLURM / PBS / LSF |
| [molcfg](https://github.com/MolCrafts/molcfg)   | Layered configuration library |
| [mollog](https://github.com/MolCrafts/mollog)   | Structured logging, stdlib-compatible |
| [molhub](https://github.com/MolCrafts/molhub)   | Molecular dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp)   | MCP server for the ecosystem |
| [molrec](https://github.com/MolCrafts/molrec)   | Atomistic record specification |

## Contributing

Issues and pull requests are welcome — see
[Contributing](https://docs.molcrafts.org/molpy/developer/contributing/).

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>

<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molpy
</h1>

<p><strong>A programmable toolkit for molecular simulation workflows</strong></p>

<p>
  <a href="https://github.com/MolCrafts/molrs/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molrs/test.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"></a>
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

MolPy is a toolkit for the full molecular-system workflow — parsing,
building, editing, typing, analyzing, and reading/writing simulation formats.
One Rust core, [molrs](crates/molrs), carries the data model and the
kernels; the same core is bound to Python (`molpy`), WebAssembly (`molwasm`),
C (`molc`) and C++ (`molcxx`). Packing a box is the job of the companion
package [molpack](https://docs.molcrafts.org/molpack/).

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

In Python every name is reached through `import molpy as mp`, one module per
subsystem of the Rust core, so the Python path and the Rust path are the same
words (`mp.core` is `molrs::core` — stores, graphs, box, neighbour search and
units — then `mp.io`, `mp.ff`, `mp.perceive`, `mp.compute`, `mp.builder`, …).
The core data classes you handle directly (`mp.Frame`, `mp.Block`,
`mp.Trajectory`, `mp.Box`, `mp.Atomistic`, …) are promoted to the root, as
the same objects.

| Where | Capability |
|---|---|
| **`mp.*`, `core`, `perceive`, `conformer`, `optimize`** | Explicit data model — `Frame` / `Block` columnar arrays, `Trajectory`, editable `Atomistic` / `CoarseGrain` graphs and `Box` on the root; regions, neighbour search, units, molpy's selectors and trajectory splitters in `mp.core`; perception and SMARTS (`mp.perceive.SmartsPattern`), 3D conformers (`mp.conformer.Conformer`), the `mp.optimize.Lbfgs` minimizer |
| **`ff`** | Force fields — `forcefield` (the `ForceField` data model), `potential` (kernels), `compile` (`PotentialCompiler`), `typifier` (OPLS-AA, MMFF94, GAFF / GAFF2 natively; AM1-BCC charges via AmberTools), `charge`, `ir` (the IR's vocabulary), `style_registry` (your own styles from Python), `params`, `clpol_scaling` |
| **`builder`** | Site-graph assembly (`Assembler`), graphene and nanotubes, polymer planning and polydispersity, crystals, virtual sites, packing templates |
| **`compute`** | Analysis — `Rdf`, `Msd`, conductivity, order parameters, Voronoi, dielectric and vibrational spectra, time-correlation functions, … (native kernels) |
| **`io`** | Read/write — PDB, GRO, LAMMPS data, XYZ, AMBER, SMILES / CGsmiles, trajectories, `*.mrec` records, … |
| **`engine`** | MD input generation & run management — LAMMPS, GROMACS, OpenMM, CP2K |
| **`tool`** | The external programs molpy runs (engines, the AmberTools pipeline): settings, environment, `ToolError` |

## Install

| Binding | Install | Use |
|---------|---------|-----|
| **Python** | `pip install molcrafts-molpy` | `import molpy as mp` |
| **Python (browser / Pyodide)** | `await micropip.install("molcrafts-molpy")` | `import molpy as mp` |
| **JavaScript (browser)** | `npm install @molcrafts/molwasm` | the wasm-bindgen API ([guide](https://docs.molcrafts.org/molpy/bindings/wasm/)) |
| **C** | `molc-*.tar.gz` from [GitHub Releases](https://github.com/MolCrafts/molrs/releases) | link `libmolc`, `#include "molc.h"` ([interop](https://docs.molcrafts.org/molpy/bindings/interop/)) |
| **C++** | built from source (`molcxx/`) | the CXX bridge |
| **Rust** | `cargo add molcrafts-molrs` | `use molrs::…` ([guide](https://docs.molcrafts.org/molpy/bindings/rust/)) |

The Python wheel carries the native core as `molpy._core`; its other
dependencies are NumPy and the MolCrafts logging/config packages. Optional:
AmberTools (AM1-BCC charges and AMBER polymer chains; GAFF typing itself is
native).

> **Nightly builds** are paused while the project is restructured; the last
> snapshots stay on `molcrafts-molpy-nightly` (versioned `X.Y.Z.devN`).

<details>
<summary>Build from source (development)</summary>

Building needs the Rust toolchain; `rust-toolchain.toml` pins it, so
[`rustup`](https://rustup.rs/) selects it on the first build.

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
uvx pre-commit install
uv --directory molpy sync --extra dev   # builds molpy (maturin) into molpy/.venv
scripts/check.sh verify                 # the pre-push gate: every check, once
```

Each binding is a standalone Cargo workspace (`molpy/`, `molwasm/`, `molc/`,
`molcxx/`), built with `cargo build --manifest-path <dir>/Cargo.toml`; the
Rust core has `cargo mrs-build` / `mrs-test` aliases (`.cargo/config.toml`).
See [Development Setup](https://docs.molcrafts.org/molpy/developer/development-setup/).

</details>

## Quick start

Parse a SMILES string, assign OPLS-AA types, and write LAMMPS input files:

```python
import molpy as mp

from pathlib import Path

mol       = mp.io.smiles.read_smiles("CCO").to_atomistic()  # ethanol from SMILES
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
mp.io.lammps.LammpsForcefieldWriter(system).write(out / "system.ff", ff)
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
| **molpy** | This project: the shared molecular data model & workflow layer, its Rust core (molrs) and its Python, WebAssembly, C and C++ bindings |
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

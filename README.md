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
building, editing, typing, analyzing, packing, and reading/writing simulation
formats.

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

Each row is one `src/molpy/` module — parse or build a structure, edit and type
it, analyze or minimize it, then read and write it across formats.

| Module | Capability |
|---|---|
| **`core`** | Explicit data model — editable `Atomistic` topology graph, `Frame`/`Block` columnar arrays, `ForceField`, `Box` |
| **`parser`** | SMILES / SMARTS (via molrs types: `SmilesIR`, `SmartsPattern`); moltemplate `.lt` reader |
| **`builder`** | System assembly — polymers, crosslinking, polydispersity, virtual sites, AmberTools facade |
| **`conformer`** | 3D coordinate generation (molrs ETKDG + MMFF cleanup) |
| **`typifier`** | Atom typing — OPLS-AA, CL&P, MMFF, GAFF via AmberTools |
| **`potential` · `optimize`** | Energy & force potentials with L-BFGS minimization |
| **`compute`** | Analysis modules under `molpy.compute` — `rdf`/`msd`/`dielectric`/`spectra`/`order`/`voronoi`/… (molrs kernels) |
| **`pack`** | Packmol-based packing with density targets |
| **`io`** | Read/write — PDB, GRO, LAMMPS data, XYZ, force fields, trajectories, … |
| **`engine`** | MD input generation & run management — LAMMPS, CP2K, OpenMM |
| **`wrapper` · `adapter`** | External CLIs (Antechamber, tleap, …) and optional RDKit in-memory bridge |

## Install

```bash
pip install molcrafts-molpy

# Browser (Pyodide) — needs molcrafts-molrs Pyodide wheel on PyPI:
# await micropip.install("molcrafts-molrs")
# await micropip.install("molcrafts-molpy")
```

Core dependencies: NumPy and
[molrs](https://github.com/MolCrafts/molrs) (`molcrafts-molrs>=0.12.0,<0.13`)
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
pip install -e ".[dev]"   # includes tox (gate driver)
prek install
# optional manual gates (same as prek/CI):
#   uv run --extra dev tox -e lint
#   uv run --extra dev tox -e py
pytest tests/
```

`pip install -e ".[dev]"` pulls the published `molcrafts-molrs` wheel from
PyPI. To develop molpy against a **local molrs checkout** (e.g. when changing
the Rust core), build molrs editable first — molrs ships its Python bindings as
a [maturin](https://www.maturin.rs/) project that needs the Rust toolchain via
[`rustup`](https://rustup.rs/):

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release   # installs `molrs` editable
cd ../molpy
pip install -e ".[dev]"                                # resolves molrs from the local build
```

See [docs/developer/development-setup](https://docs.molcrafts.org/molpy/developer/development-setup/)
for the full workflow.

</details>

## Quick start

Parse a SMILES string, assign OPLS-AA types, and write LAMMPS input files:

```python
import molpy as mp

mol   = mp.Atomistic.from_smiles("CCO")          # ethanol from SMILES
ff    = mp.io.read_xml_forcefield("oplsaa.xml")  # bundled OPLS-AA
typed = mp.typifier.OplsAtomisticTypifier(ff).typify(mol)

mp.io.write_lammps_system("output/", typed.to_frame(), ff)
# → output/system.data  output/system.in
```

More workflows — packed solvent boxes, virtual-site models, polymer chains and
networks (the stress test for MolPy's editing machinery), AmberTools
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

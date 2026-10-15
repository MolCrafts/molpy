<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molrs
</h1>

<p><strong>Rust core for molecular modeling — data structures, I/O, and compute kernels, native and in the browser.</strong></p>

<p>
  <a href="https://img.shields.io/github/actions/workflow/status/MolCrafts/molrs/test.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"><img src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molrs/test.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://crates.io/crates/molcrafts-molrs"><img src="https://img.shields.io/crates/v/molcrafts-molrs?style=flat-square&logo=rust&logoColor=white" alt="crates.io"></a>
  <a href="https://docs.rs/molcrafts-molrs"><img src="https://img.shields.io/docsrs/molcrafts-molrs?style=flat-square&logo=docsdotrs&logoColor=white" alt="docs.rs"></a>
  <a href="https://pypi.org/project/molcrafts-molrs/"><img src="https://img.shields.io/pypi/v/molcrafts-molrs?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@molcrafts/molrs"><img src="https://img.shields.io/npm/v/@molcrafts/molrs?style=flat-square&logo=npm&logoColor=white" alt="npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square" alt="License"></a>
</p>

<p>
  <a href="https://docs.molcrafts.org/molrs/"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

molrs is a Rust library for molecular modeling: a column-oriented data model, format readers and writers, trajectory analysis, force fields, and 3D structure generation. The same code runs natively, from Python (PyO3), and in the browser (WASM).

> **Under active development.** Public APIs may change between minor releases.

## Vision

Molecular modeling tools have long forced a choice between fast and reusable. The performance-critical kernels live in aging C and Fortran, while the science is written in Python wrappers that cannot run anywhere a Python interpreter does not. molrs exists to dissolve that split: one correct, well-tested implementation of the molecular data model and its compute kernels, written once in Rust.

That single implementation is meant to be portable everywhere molecules are studied — a native library, a Python package, and a WebAssembly module that runs in a browser tab with no install. The aspiration is that a researcher, a pipeline, and an interactive web tool can all reach for the exact same neighbor search, the same RDF, the same force-field evaluation, and get identical numbers.

By becoming the dependable core the rest of the MolCrafts ecosystem builds on, molrs aims to make high-performance, reference-grade molecular computation something you take for granted rather than something you reimplement.

## Capabilities

One crate, `molcrafts-molrs`, whose sub-systems are feature-gated modules
(`core` and `perceive` are always on):

| Module (feature) | Capability |
|------------------|------------|
| `core`, `perceive` *(always on)* | Frame / Block column store, MolGraph topology, elements, `SimBox` + minimum-image convention, spatial regions, neighbor search, chemical perception, SMARTS matching |
| `builder` | Structure builders: graphene, nanotubes, lattices, self-avoiding walks, site-graph assembly (`Assembler`, placers, orienters) |
| `io` | Readers / writers for PDB, XYZ, mol2, SDF, CIF, GRO, POSCAR, CHGCAR, Cube, LAMMPS data/dump, DCD, TRR, XTC; `*.mrec` record files following the [molrec](https://docs.molcrafts.org/molrec/) contract — frames, topologies, trajectories and force fields (`zarr` / `filesystem`); SMILES / CGsmiles parser under the `smiles` feature |
| `compute` | Trajectory analysis on one `Compute` contract (`new` = configuration, `compute` = data): one `Correlator` behind `Msd` / `Vacf` / … and the spectra (FFT, windows, frequency grids), RDF, clustering, gyration / inertia tensors, PCA, k-means, density, diffraction, PMFT, order parameters, dielectric, environment matching |
| `ff` | Force fields and potentials — `def_style` / `def_type` model, MMFF94 / OPLS-AA / GAFF / UFF typing through `Typing`, `PotentialCompiler`, LJ, PME; LAMMPS / GROMACS / OpenMM XML read and write; L-BFGS geometry optimization over a `Potential` |
| `conformer` | 3D conformer generation: ETKDGv3 distance geometry, experimental-torsion refinement, MMFF94 cleanup, stereo guards |
| `md` | In-process molecular dynamics: velocity-Verlet and Langevin integration over force-field potentials |
| `stream` | MessagePack/JSON frame transport and native WebSocket streaming |

A separate `molcrafts-molrs-cxxapi` crate (built from source, not published)
provides a CXX bridge for zero-copy integration with Atomiverse C++.

## Install

```bash
cargo add molcrafts-molrs
```

The default build is core only (`core`, `perceive`, the `optimize` contract)
plus `rayon`. Every sub-system is opt-in: name the modules you need, or `full`
for all of them (`builder`, `io`, `smiles`, `compute`, `voronoi`,
`ff`, `conformer`, `md`). `full` does not enable `stream` or `filesystem`
(path-backed Zarr); add them explicitly. `default-features = false` also
drops `rayon` (wasm, Pyodide).

```toml
molcrafts-molrs = { version = "0.17", features = ["io", "smiles", "conformer"] }
```

| Environment | Install | Import / use |
|-------------|---------|----------------|
| **Rust** | `cargo add molcrafts-molrs` | `use molrs::…` |
| **Python (desktop)** | `pip install molcrafts-molrs` | `import molrs` |
| **Python (browser / Pyodide)** | `await micropip.install("molcrafts-molrs")` | `import molrs` |
| **JS (browser)** | `npm install @molcrafts/molrs` | wasm-bindgen API (not `import molrs`) |
| **C / C++** | `molrs-capi-*.tar.gz` from [GitHub Releases](https://github.com/MolCrafts/molrs/releases) | link `libmolrs_capi` + `#include "molrs.h"` — see `docs/interop.md`, Path C |

PyPI ships **desktop** wheels (manylinux / macOS / Windows) and a **Pyodide
(Emscripten)** wheel for micropip. The npm package is a separate **wasm-bindgen**
build used by MolVis; it is not the Python extension.

```python
# In Pyodide (browser or Node)
import micropip
await micropip.install("molcrafts-molrs")
import molrs
```

> **Python nightly.** Bleeding-edge wheels go to `molcrafts-molrs-nightly`
> (`X.Y.Z.devN`) on the `nightly` branch. `pip install --pre molcrafts-molrs-nightly`
> imports as `molrs` and cannot sit beside stable `molcrafts-molrs`.

## Build from source

Building from source needs the Rust toolchain. `rust-toolchain.toml` pins one
exact rustc / rustfmt / clippy release plus the `wasm32-unknown-unknown`
target, so [`rustup`](https://rustup.rs/) selects them automatically on the
first build, and local builds, the prek hooks and CI all use the same compiler.

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
# `mrs-*` are cargo aliases committed in .cargo/config.toml. They pin one
# feature set, so every command reuses the same build instead of compiling
# the crate again under a slightly different spelling.
cargo mrs-build   # compile the Rust library
cargo mrs-test    # unit tests
cargo mrs-doctest # rustdoc examples
scripts/check.sh all  # every CI gate (prek runs the same script)
```

Binding crates are standalone workspaces. Build each with
`cargo build --manifest-path <crate>/Cargo.toml`.

**Python bindings** are built from the `molrs-python` crate with
[maturin](https://www.maturin.rs/). `maturin develop` compiles the PyO3
extension and installs it editable into the active virtualenv under the import
name `molrs`:

```bash
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release
python -c "import molrs; print(molrs.io.smiles.SmilesIr('O').n_components)"
```

**WASM / npm** is built with [wasm-pack](https://rustwasm.github.io/wasm-pack/),
using the same flags as release publishing:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

See the [installation guide](https://docs.molcrafts.org/molrs/getting-started/installation/)
for environment-verification snippets and the
[contributing guide](https://docs.molcrafts.org/molrs/contributing/) for the
documentation loop.

## Quick start

```rust
use molrs::conformer::{Conformer, ConformerOptions};
use molrs::io::read_smiles_str;

let mol = read_smiles_str("c1ccccc1").unwrap();      // benzene
let (mol3d, _report) = Conformer::new(ConformerOptions::default()).generate(&mol).unwrap();
```

Python and JavaScript/TypeScript quickstarts live in the documentation.

## Documentation

- [Documentation site](https://docs.molcrafts.org/molrs/) — guides and references
- [Getting started](https://docs.molcrafts.org/molrs/getting-started/installation/) — Rust, Python, and WASM quickstarts
- [Python reference](https://docs.molcrafts.org/molrs/reference/python/) — the binding surface, rendered from the installed package
- [Task-oriented guides](https://docs.molcrafts.org/molpy/) — data model, SMILES, neighbor search, 3D embedding, force fields, I/O, trajectory analysis (molpy, the Python library built on molrs)
- [Rust API reference](https://docs.rs/molcrafts-molrs) — full rustdoc on docs.rs
- [Record files](https://docs.molcrafts.org/molrs/guides/records/) — saving frames, trajectories and force fields as `*.mrec`

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| [molpy](https://github.com/MolCrafts/molpy)     | Python toolkit — the shared molecular data model & workflow layer |
| **molrs** | Rust core — molecular data structures & compute kernels (native + WASM) — this repo |
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

See [CONTRIBUTING](https://docs.molcrafts.org/molrs/contributing/) for development setup and guidelines.
The [release checklist](docs/releasing.md) covers local verification, package
inspection, and the tag-triggered publishing workflow.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>

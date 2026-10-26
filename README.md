<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molpy
</h1>

<p><strong>A multi-language toolkit for molecular systems</strong></p>

<p>
  <a href="https://github.com/MolCrafts/molpy/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molpy/test.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"></a>
  <a href="https://pypi.org/project/molcrafts-molpy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/molcrafts-molpy?style=flat-square&logo=pypi&logoColor=white&label=PyPI"></a>
  <a href="https://crates.io/crates/molcrafts-molrs"><img alt="crates.io" src="https://img.shields.io/crates/v/molcrafts-molrs?style=flat-square&logo=rust&logoColor=white&label=crates.io"></a>
  <a href="https://pypi.org/project/molcrafts-molpy/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/molcrafts-molpy?style=flat-square&logo=python&logoColor=white"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square"></a>
</p>

<p>
  <a href="https://docs.molcrafts.org/molpy/"><b>molpy</b></a> &nbsp;&middot;&nbsp;
  <a href="https://docs.molcrafts.org/molpy/molrs/"><b>molrs</b></a> &nbsp;&middot;&nbsp;
  <a href="https://docs.rs/molcrafts-molrs"><b>rustdoc</b></a> &nbsp;&middot;&nbsp;
  <a href="https://docs.molcrafts.org/molpy/api/"><b>Python API</b></a>
</p>

</div>

This repository is **molpy**. The core is Rust, published as the crate
`molcrafts-molrs` and imported as `molrs`. Python, WebAssembly, C and C++
each bind that core, and none of them reimplements it. NumPy is written in C
and is still NumPy. molpy is written in Rust and is still molpy.

The manual opens on [molpy](https://docs.molcrafts.org/molpy/). The next page
is [molrs](https://docs.molcrafts.org/molpy/molrs/), the core. Rust's
reference is [rustdoc](https://docs.rs/molcrafts-molrs). Python's reference
is generated from the docstrings. WebAssembly, C and C++ are documented by
wasm-bindgen, cbindgen and the CXX bridge.

> **Under active development.** Public APIs may change between minor releases.

## Layout

| Path | What it is | Published as |
| --- | --- | --- |
| `crates/molrs/` | the core, crate `molcrafts-molrs`, library `molrs` | crates.io `molcrafts-molrs` |
| `crates/molrs-ffi/` | handles and capsules, the only ABI | — |
| `molpy/` | Python binding, `import molpy as mp`, native module `molpy._core` | PyPI `molcrafts-molpy` |
| `molwasm/` | WebAssembly binding | npm `@molcrafts/molwasm` |
| `molc/` | C API, header `molc.h`, symbols `molc_*` | GitHub release archive |
| `molcxx/` | C++ bridge (`FrameView`, zero-copy) | built from source |
| `docs/` | the manual. The first page is molpy; it continues in molrs | [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/) |

There is no Python package named `molrs`. Lengths in a `Frame` are ångströms.
The version of every binding is `[workspace.package].version` in the root
`Cargo.toml`.

Packing a box is [molpack](https://github.com/MolCrafts/molpack), a separate
crate that depends on `molrs`.

## Install

| Language | Install | Use |
| --- | --- | --- |
| Python | `pip install molcrafts-molpy` | `import molpy as mp` |
| Python in the browser | `await micropip.install("molcrafts-molpy")` | `import molpy as mp` |
| Rust | `cargo add molcrafts-molrs` | `use molrs::…` |
| JavaScript | `npm install @molcrafts/molwasm` | the wasm-bindgen API |
| C | `molc-*.tar.gz` from [releases](https://github.com/MolCrafts/molpy/releases) | `#include "molc.h"` |
| C++ | build `molcxx/` | the CXX bridge |

`core`, `perceive`, `op` and `optimize` are always compiled. Name the Cargo
features you need, or `full` while exploring. `full` does not include
`stream`, and `*.mrec` paths need `filesystem`.

```toml
molrs = { package = "molcrafts-molrs", version = "0.20", features = ["io", "smiles"] }
```

The Python wheel carries the native core as `molpy._core`. AmberTools is
optional, and only for the programs molpy still shells out to.

<details>
<summary>Build from source</summary>

`rust-toolchain.toml` pins the toolchain.

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
uvx pre-commit install
uv --directory molpy sync --extra dev
scripts/check.sh verify
```

Each binding is its own Cargo workspace. Build it with
`cargo build --manifest-path <dir>/Cargo.toml`. The core uses the `cargo mrs-*`
aliases in `.cargo/config.toml`. All of them share `<repo>/target`.

</details>

## The same molecule

```python
import molpy as mp

mol = mp.io.smiles.read_smiles("CCO").to_atomistic()
```

```rust
use molrs::io::smiles::SmilesIr;

let mol = SmilesIr::parse("CCO")?.to_atomistic()?;
```

The Rust door needs the `smiles` feature. The chapters after molrs in the
manual walk the same model from Python: the
[tutorials](https://docs.molcrafts.org/molpy/tutorials/), the
[guides](https://docs.molcrafts.org/molpy/user-guide/) and the
[compute](https://docs.molcrafts.org/molpy/compute/) section.

## MolCrafts

| Project | Role |
| --- | --- |
| **molpy** | this repository |
| [molpack](https://github.com/MolCrafts/molpack) | molecular packing |
| [molvis](https://github.com/MolCrafts/molvis) | visualization |
| [molexp](https://github.com/MolCrafts/molexp) | experiment management |
| [molnex](https://github.com/MolCrafts/molnex) | molecular machine learning |
| [molq](https://github.com/MolCrafts/molq) | job queue |
| [molcfg](https://github.com/MolCrafts/molcfg) | configuration |
| [mollog](https://github.com/MolCrafts/mollog) | logging |
| [molhub](https://github.com/MolCrafts/molhub) | dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp) | MCP server |
| [molrec](https://github.com/MolCrafts/molrec) | record specification |

## Contributing

See [Contributing](https://docs.molcrafts.org/molpy/developer/contributing/).

## License

BSD-3-Clause. See [LICENSE](LICENSE).

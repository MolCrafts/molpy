# molrs

molpy is one project. This chapter is its core: the crate `molcrafts-molrs`, imported as `molrs`. The data model and the kernels live here. Python, WebAssembly, C and C++ bind the same types and do not reimplement them. Their references are produced by the tools that build those bindings. This manual does not copy those references.

`op`, `core`, `perceive` and `optimize` are always compiled. `optimize`'s force-field minimizer needs the `ff` feature. Everything else is a Cargo feature, so a program compiles only the subsystems it names. `full` turns on `builder`, `io`, `smiles`, `compute`, `ff`, `conformer`, `md` and `voronoi`. It does not turn on `stream`, and record files need `zarr` plus `filesystem` for their path doors. The default feature is `rayon` alone, which is the core with no I/O. `default-features = false` also drops `rayon`.

```toml
molrs = { package = "molcrafts-molrs", version = "0.19", features = ["io", "smiles"] }
```

A public item has one path, `molrs::<subsystem>::Item`. Nothing is re-exported at the crate root.

| Module | Feature | What it is |
| --- | --- | --- |
| `molrs::op` | always | Vectors, linear algebra, superposition |
| `molrs::core` | always | `Frame`, `Block`, graphs, boxes, regions, neighbour search, units |
| `molrs::perceive` | always (`perceive::smarts` needs `smiles`) | Rings, aromaticity, hydrogens, stereo, SMARTS |
| `molrs::optimize` | always (force-field optimizers need `ff`) | L-BFGS over a potential |
| `molrs::builder` | `builder` | Graphene, nanotubes, lattices, site-graph assembly |
| `molrs::io` | `io` | One reader and one writer per format |
| `molrs::io::smiles`, `molrs::io::cgsmiles` | `smiles` | SMILES and CGsmiles |
| `molrs::io::mrec` | `zarr` (`filesystem` for paths) | `*.mrec` records |
| `molrs::conformer` | `conformer` | 3D coordinates |
| `molrs::compute` | `compute` | RDF, correlations, spectra, clusters |
| `molrs::ff` | `ff` | Force fields, typing, potentials |
| `molrs::md` | `md` | In-process dynamics |
| `molrs::stream` | `stream` | MessagePack and JSON frames, WebSocket transport |

The [rustdoc on docs.rs](https://docs.rs/molcrafts-molrs) is the reference. That build enables `full`, `filesystem` and `stream`, so every module is present. A name that fails to resolve locally is a feature left off: `smiles` implies `io`, `conformer` implies `ff`, and `write_mrec_frame` needs `filesystem`.

```rust
use molrs::io::smiles::SmilesIr;

let molecule = SmilesIr::parse("CCO")?.to_atomistic()?;
assert_eq!(molecule.n_atoms(), 3);
```

`SmilesIr::parse` checks the text. `to_atomistic` builds the graph the rest of the core consumes. Packing a box is not part of this crate; it lives in [molpack](https://github.com/MolCrafts/molpack).

The other languages document themselves from the source. Python's reference is generated from the docstrings into the [API reference](../api/index.md) of this manual. WebAssembly's reference is the wasm-bindgen API shipped as `@molcrafts/molwasm`. C's reference is the cbindgen header `molc.h` (`molc` is both a static library and a cdylib). C++'s reference is the CXX bridge in `molcxx`.

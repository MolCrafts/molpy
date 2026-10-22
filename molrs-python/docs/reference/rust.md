# Rust Reference

The Rust API reference is built and hosted by docs.rs. This site links to it
instead of copying signatures or rustdoc text.

| Crate | Reference |
| --- | --- |
| Library (single crate) | [`molcrafts-molrs`](https://docs.rs/molcrafts-molrs) |
| C API (`libmolc` + `molrs.h`) | archives on [GitHub Releases](https://github.com/MolCrafts/molrs/releases); see [Path C](https://github.com/MolCrafts/molrs/blob/master/docs/interop.md#path-c--c-abi-libmolc) |
| CXX bridge (Atomiverse) | `molcrafts-molcxx`, built from source; not published to crates.io |

Always compiled inside `molcrafts-molrs`: `core`, `perceive`, `op` and the
`optimize` engine. Feature-gated modules: `builder`, `io` (with `smiles`
inside it), `ff`, `conformer`, `compute`, `voronoi`, `md` and
`stream`. `full` bundles every one of those except `stream`. Record files
(`io::mrec`) need `zarr`, and their path-based doors need `filesystem`.
The crate's default features are `rayon` only, so name what you use:

```toml
molrs = { package = "molcrafts-molrs", version = "0.19", features = ["full", "filesystem"] }
```

The docs.rs build enables `full`, `filesystem` and `stream`, so every module
appears there.

The Packmol-aligned packing workflow lives in the separate
[`molpack`](https://github.com/MolCrafts/molpack) repository.

# Rust Quickstart

molrs is one crate, `molcrafts-molrs`, imported as `molrs`. The core data
model is always compiled; every other subsystem is a Cargo feature, so an
application compiles only what it names, or `full` while exploring.

## 1. Create a Project

```toml
[dependencies]
molrs = { package = "molcrafts-molrs", version = "0.17", features = ["full", "filesystem"] }
```

The crate's default features are `rayon` only — core, no I/O. `full` enables
the builder, I/O, SMILES, compute, force-field, conformer, MD and Voronoi
subsystems. `filesystem` adds the path-based record-file doors
(`molrs::io::write_mrec_frame`, …) and the native zstd codec; it is not
part of `full`, and neither is `stream`. Once you know which layers your
application uses, replace `full` with a narrower list.

## 2. Parse Topology and Generate Coordinates

```rust
use molrs::conformer::{Conformer, ConformerOptions};
use molrs::io::smiles::SmilesIr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = SmilesIr::parse("c1ccccc1")?;
    let mol = ir.to_atomistic()?;

    let (mol3d, report) = Conformer::new(ConformerOptions::default()).generate(&mol)?;
    let frame = mol3d.to_frame()?;

    let atoms = frame.get("atoms").expect("an atoms block");
    let x = atoms.get("x").and_then(|c| c.as_float()).expect("an f64 x column");
    println!("atoms: {}", atoms.n_rows().unwrap_or(0));
    println!("first x: {:.3}", x[0]);
    println!("final energy: {:?}", report.final_energy);

    molrs::io::write_mrec_frame("benzene.mrec", &frame, None, None)?;
    Ok(())
}
```

The two-step parse is intentional. `SmilesIr::parse` validates the text and
produces an intermediate representation. `to_atomistic` turns that intermediate
form into the molecular graph consumed by embedding and force-field code
(`molrs::io::read_smiles_str` does both for one molecule).
`to_frame` converts the graph into the column store; it fails when a node
property contradicts the frame schema.

A column is read with one accessor: `Block::get(key)` returns the stored
`Column`, and a `Column::as_*` projection (`as_float`, `as_int`, `as_uint`,
`as_bool`, `as_string`, …) returns the array when the column is stored at
that dtype. Floats are always `f64`. `write_mrec_frame` saves the frame as a
[record file](../guides/records.md).

## 3. Understand the Module Layout

| Module | Feature | Purpose |
| --- | --- | --- |
| `molrs::core` | always | Core `Frame`, `Block`, topology, boxes, regions, neighbor search, units |
| `molrs::perceive` | always (`perceive::smarts` needs `smiles`) | Rings, aromaticity, hydrogens, stereo, SMARTS |
| `molrs::op` | always | Vector, linear-algebra and superposition kernels |
| `molrs::optimize` | always (force-field optimizers need `ff`) | L-BFGS geometry optimization over a `Potential` |
| `molrs::builder` | `builder` | Graphene, nanotubes, lattices, walks, site-graph assembly |
| `molrs::io` | `io` | File readers and writers, one module per format (`io::pdb`, `io::lammps`, …), every door `io::read_<fmt>` / `io::write_<fmt>` |
| `molrs::io::smiles`, `molrs::io::cgsmiles` | `smiles` | SMILES and CGsmiles parsing and graph conversion |
| `molrs::io::mrec` | `zarr` (+ `filesystem` for paths) | `*.mrec` record files |
| `molrs::conformer` | `conformer` | 3D coordinate generation |
| `molrs::compute` | `compute` | RDF, correlations (MSD, VACF, …), spectra, clusters, descriptors |
| `molrs::ff` | `ff` | Force fields, typing, potentials |
| `molrs::md` | `md` | In-process velocity-Verlet / Langevin dynamics |
| `molrs::stream` | `stream` | MessagePack/JSON frames, WebSocket transport |

The [docs.rs reference](https://docs.rs/molcrafts-molrs) documents every
module with all of these features enabled.

## 4. Common Compile Errors

If `molrs::io::smiles` or `molrs::conformer` cannot be found, the Cargo
feature is not enabled (`smiles` implies `io`, and `conformer` implies `ff`).
If `write_mrec_frame` cannot be found, enable `filesystem`. If code compiles
but embedding fails at runtime, inspect the topology: embedding expects
chemically meaningful atoms and bonds, not just a coordinate table.

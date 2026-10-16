# molrs

molrs is a molecular modeling toolkit with a Rust core and Python and
WebAssembly bindings. The project is organized around a shared data model:
`Frame` holds named `Block`s of columnar molecular data, optional simulation
box metadata, and enough topology to move between file I/O, geometry
generation, force-field evaluation, and trajectory analysis.

This site is the narrative layer for that system. The Rust API reference is on
docs.rs, the Python reference is rendered from the installed binding module,
and the npm package ships TypeScript declarations generated from the same Rust
doc comments.

## The same workflow runs in Python, Rust, and TypeScript

=== "Python"

    Parse ethanol from SMILES, generate a three-dimensional structure, convert
    it to a frame, and inspect coordinate columns.

    ```python
    import molrs

    ir = molrs.io.smiles.SmilesIr.parse("CCO")
    mol = ir.to_atomistic()

    mol3d, _report = molrs.conformer.Conformer(speed="fast", seed=42).generate(mol)
    frame = mol3d.to_frame()

    atoms = frame["atoms"]
    print("atoms:", atoms.n_rows)
    print("columns:", atoms.keys())
    print("x:", atoms["x"][:3])
    ```

    Expected shape of the result: the input graph has three heavy atoms, while
    the embedded molecule usually includes explicit hydrogens because
    `Conformer(add_hydrogens=True)` is the default.

=== "Rust"

    Use the crate with the `full` feature while learning, then narrow
    features when an application has a stable dependency boundary.

    ```rust
    use molrs::conformer::{Conformer, ConformerOptions};
    use molrs::io::smiles::SmilesIr;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mol = SmilesIr::parse("c1ccccc1")?.to_atomistic()?;
        let (mol3d, report) = Conformer::new(ConformerOptions::default()).generate(&mol)?;

        println!("atoms: {}", mol3d.n_atoms());
        println!("final energy: {:?}", report.final_energy);
        Ok(())
    }
    ```

=== "TypeScript"

    The npm package is a bundler build: importing it loads the WebAssembly
    module, and the generated classes and functions are regular exports.

    ```ts
    import { SmilesIr, Conformer, writeXyz } from "@molcrafts/molrs";

    const ir = SmilesIr.parse("CCO");
    const frame2d = ir.toFrame();
    const frame3d = new Conformer("fast", true, 42).generate(frame2d);

    console.log(writeXyz(frame3d));
    ```

## Highlights

molrs holds every force field in one force-field IR (LAMMPS standard):

- **One set of styles** with LAMMPS's energy expressions, factors and
  units; every angle-valued parameter is in degrees. Urey–Bradley, CMAP,
  CHARMM 1-4 interactions and per-pair overrides are supported. See
  [Force-field IR](guides/forcefield-ir.md).
- **Engines read and written whole.** LAMMPS, GROMACS (whole topologies),
  OpenMM XML and AMBER prmtop (chamber too) convert to the IR exactly or
  refuse by name, checked term by term against the engines.
- **The IR is a protocol.** A style or a category registers from Rust,
  Python or molpy with nothing rebuilt; see
  [Extending the force-field IR](guides/extending-forcefield-ir.md).
- **Records** (`*.mrec`) carry frames, trajectories and force fields; see
  [Record files](guides/records.md).

## What lives here

These docs cover the molrs **binding surface**: the per-language quickstarts,
the [record-file guide](guides/records.md), and the API reference.
Task-oriented Python guides (the data model, in-process MD,
SMILES and topology, neighbor search, 3D embedding, force fields, I/O, and
trajectory analysis) live in the
[molpy documentation](https://docs.molcrafts.org/molpy/), the Python library
built on molrs.

## Find your starting point

Start with [Installation](getting-started/installation.md), then choose the
quickstart for your host language:

- [Python Quickstart](getting-started/quickstart-python.md) is the most complete
  end-to-end tutorial and mirrors the style of a notebook.
- [Rust Quickstart](getting-started/quickstart-rust.md) explains crate features
  and the module layout.
- [WASM Quickstart](getting-started/quickstart-wasm.md) explains loading the
  module, typed-array columns, and browser bundling.

[Record files](guides/records.md) shows how to save frames, trajectories and
force fields as `*.mrec` records that every surface reads.

Use [Python Reference](reference/python.md), [Rust Reference](reference/rust.md),
and [WASM Reference](reference/wasm.md) when you need exact API details.

The data model is the same in all three: frames, topology, simulation boxes,
neighbor lists, force fields, and trajectories mean one thing across Rust,
Python, and WASM, so the mental model carries between them. The narrative
explanation of that model lives in the
[molpy documentation](https://docs.molcrafts.org/molpy/).

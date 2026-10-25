# Record files (`*.mrec`)

A molrs **record** is one self-describing package on disk: a `meta` document
plus one or more sections — a snapshot (`frame`), a topology (`system`), a
frame sequence (`trajectory`), a force field (`forcefield`), observables, or a
run `status`. It is a directory whose name ends in `.mrec`, with every array
stored as Zarr V3, and a closed directory can be packed into one
`*.mrec.zip` file.

molrs writes and reads records by the [molrec](https://docs.molcrafts.org/molrec/)
contract. The [layout chapter](https://docs.molcrafts.org/molrec/layout/)
names every group, array and attribute; this page shows how molrs maps onto
it. The examples are Python, but the Rust `molrs::io::write_mrec` and
`molrs::io::mrec` classes are the same doors with the same rules, and the WASM
package reads the same files.

| What you have | Write | Read |
| --- | --- | --- |
| A `Frame` (snapshot) | `mp.io.write_mrec(sink, frame=…)` | `mp.io.mrec.MrecFile(source).frame()` |
| A `Frame` (topology) | `mp.io.write_mrec(sink, system=…)` | `MrecFile(source).system()` |
| A `Trajectory` in memory | `mp.io.write_mrec(sink, trajectory=…)` | `mp.io.mrec.MrecReader(source).to_trajectory()` |
| A run too large for memory | `mp.io.mrec.MrecWriter` | `mp.io.mrec.MrecReader` |
| A `ForceField` | `mp.io.write_mrec(sink, forcefield=…)` | `MrecFile(source).forcefield()` |
| Any record | `meta=` on `mp.io.write_mrec` | `MrecFile(source).sections()`, `MrecFile(source).meta()` |

`write_mrec` takes every section as a keyword, so one call writes a snapshot
with its topology and force field. A sink is a path (`pathlib.Path`, written
as a directory store) or a `bytearray` (which receives the packed record); a
source is a path or the `bytes` of a packed record.

## Write and read a frame

Build a small frame. Canonical columns (`x`, `element`, `res_id`, `chain`, …)
adopt their schema dtype on insert, so `res_id` below is stored as `uint64`
however the array was spelled.

```python
from pathlib import Path

import numpy as np
import molpy as mp
atoms = mp.core.Block({
    "element": ["O", "H", "H"],
    "x": np.array([0.000, 0.757, -0.757]),
    "y": np.array([0.000, 0.586, 0.586]),
    "z": np.zeros(3),
    "res_id": np.array([1, 1, 1]),
    "res_name": ["HOH", "HOH", "HOH"],
    "chain": ["A", "A", "A"],
})
bonds = mp.core.Block({
    "atomi": np.array([0, 0], dtype=np.uint64),
    "atomj": np.array([1, 2], dtype=np.uint64),
})
frame = mp.core.Frame(
    {"atoms": atoms, "bonds": bonds},
    box=mp.core.Box.cube(20.0),
)
frame.meta["title"] = "water"
frame.meta["temperature"] = 300.0

print(frame["atoms"].dtype("res_id"))
```

`mp.io.write_mrec(path, frame=frame)` writes the `frame` section and a `meta` document. The
`meta` document is what you supplied; a record carries no version key, and a
reader keeps any `meta` key it does not recognise.

```python
mp.io.write_mrec(Path("water.mrec"), frame=frame, meta={"producer": "quickstart"})

record = mp.io.mrec.MrecFile(Path("water.mrec"))
print(record.sections())
print(record.meta())

back = record.frame()
print(back["atoms"]["chain"], back["atoms"]["res_id"])
print(back.box.lengths)
```

Frame meta is **typed on disk**: the writer adds a `_meta_types` attribute,
so every value reads back at the dtype it was stored with, `float("nan")`
included. A JSON object comes back as a frozen `MetaDocument`.

```python
frame.meta["n_steps"] = mp.core.MetaValue("i32", 5000)
frame.meta["run"] = {"ensemble": "NVT", "thermostat": "langevin"}
mp.io.write_mrec(Path("water.mrec"), frame=frame)

back = mp.io.mrec.MrecFile(Path("water.mrec")).frame()
print(back.meta.dtype("n_steps"), back.meta["n_steps"])
print(back.meta["run"]["ensemble"])
```

A topology goes in the `system` section, next to or instead of a snapshot:
`mp.io.write_mrec(path, frame=frame, system=topology)` writes both, and
`write_mrec(path, system=topology)` / `MrecFile(path).system()` handle a
topology on its own. Each `MrecFile` read decodes its own section only, so a
damaged trajectory in the same store cannot fail `MrecFile(path).frame()`; a
section the record does not hold reads as `None`.

## Topology conventions

The molrec conventions give fixed names to the residue and chain facts the
format readers produce. `chain`, `res_id`, `res_name`, `icode`, `altloc`,
`occupancy` and `b_factor` are canonical `atoms` columns; the PDB, mmCIF, GRO
and extxyz readers fill them under those names. Extra blocks with fixed
meanings are `constraints`, `virtual_sites`, `drudes` and `members` (bead →
atom membership in a coarse-grained frame). `mp.core.schema` prints the whole
vocabulary.

A `uint64` column can declare which block its values index. Writers store
that as the block's `targets` attribute and refuse a reference that does not
resolve, so a dangling index fails at write time instead of at analysis time:

```python
contacts = mp.core.Block({
    "site": np.array([0, 2], dtype=np.uint64),
    "distance": np.array([2.8, 3.1]),
})
contacts.set_target("site", "atoms")
frame["contacts"] = contacts
mp.io.write_mrec(Path("water.mrec"), frame=frame)

print(mp.io.mrec.MrecFile(Path("water.mrec")).frame()["contacts"].targets())
```

## Declared precision

Coordinates rarely carry 16 significant digits of information. An `f64`
column can declare an absolute precision `p` in its own units. The writer then
stores each value rounded to a binary grid — the largest power of two `q ≤ p`,
so the error is at most `p / 2` — and compresses the column with byte shuffle
plus zstd. The declaration is stored with the column and reads back.

```python
for key in ("x", "y", "z"):
    frame["atoms"].set_precision(key, 1e-3)
mp.io.write_mrec(Path("water.mrec"), frame=frame)

back = mp.io.mrec.MrecFile(Path("water.mrec")).frame()
print(back["atoms"].precision("x"))
print(np.abs(back["atoms"]["x"] - frame["atoms"]["x"]).max() <= 0.5e-3)
```

On a 3000-atom trajectory moving 0.05 Å per frame, lossless coordinates
take 24 B/atom/frame; `p = 1e-3` Å takes about 7.6 and `p = 1e-2` Å about
5.8. Without a declaration nothing is rounded. Memory is never touched: only
the stored copy is rounded. A store with a declared precision needs a reader
that decodes zstd and shuffle, as every molrs build does (the WASM package
included).

## Trajectories

A `Trajectory` that fits in memory is written by one call and read back
through `MrecReader.to_trajectory()`:

```python
frames = []
for i in range(4):
    f = frame.copy()
    f["atoms"]["x"] = frame["atoms"]["x"] + 0.01 * i
    frames.append(f)

traj = mp.core.Trajectory(
    frames,
    step=np.arange(4, dtype=np.int64) * 100,
    time=np.arange(4, dtype=np.float64) * 0.2,
)
mp.io.write_mrec(Path("run.mrec"), trajectory=traj, meta={"producer": "quickstart"})

loaded = mp.io.mrec.MrecReader(Path("run.mrec")).to_trajectory()
print(len(loaded), loaded.step)
```

Only what changes between frames is written: a block identical to the previous
frame's is not stored again.

### Streaming with a pinned schema

A long run is written frame by frame. Pin a `SequenceSchema` first — the
blocks, columns and dtypes every frame will carry — then append. Derive the
schema from representative frames (`SequenceSchema.from_frames`;
`from_frames([])` is the empty schema) and add declarations:

- `declare_precision(block, column, p)` — round that column on every frame,
  as above. A change smaller than half a grid step is not an update, so a
  rattling-but-static block costs nothing.
- `declare_target(block, column, target)` — a `uint64` column indexes rows of
  `target`.
- `declare_aligned(block, target)` — `block` has one row per row of `target`
  at every frame (per-atom forces next to `atoms`). The writer refuses a frame
  that breaks it.

```python
from molpy.io.mrec import SequenceSchema, MrecReader, MrecWriter

schema = SequenceSchema.from_frames(frames[:1])
for key in ("x", "y", "z"):
    schema.declare_precision("atoms", key, 1e-3)

with MrecWriter(Path("stream.mrec"), schema, meta={"producer": "quickstart"}) as writer:
    for i, f in enumerate(frames):
        writer.append(f, step=100 * i, time=0.2 * i)

with MrecReader(Path("stream.mrec")) as reader:
    print(len(reader), reader.step)
    last = reader[len(reader) - 1]
    print(last["atoms"]["x"])
```

`MrecReader` decodes one frame per access, so a store larger than
memory can be walked with a plain `for frame in reader:` loop. A writer
flushes every `flush_every` frames; if the process dies, the frames already
committed are readable and `MrecWriter.open(path)` resumes appending.

### Packing

A closed directory store packs into a single `*.mrec.zip` — one file to copy,
upload or serve. `pack_mrec_zip` replaces the directory with the archive and
returns its path; `MrecReader` and the WASM readers open the archive directly.

```python
from molpy.io.mrec import pack_mrec_zip

zipped = pack_mrec_zip(Path("stream.mrec"))
print(zipped)

with MrecReader(Path(zipped)) as reader:
    print(len(reader))
```

## Force fields

A `forcefield` section stores a force field as data: a document (styles,
units, mixing rule, special-bond weights) and one table per style. Units are
recorded as declared and read back as recorded. `ForceFieldSection.from_forcefield` and
`ForceFieldSection.to_forcefield` map between the two, and every record writer takes
a force field next to the structure it parameterizes:

```python
ff = mp.ff.forcefield.ForceField("water", units="real")
atom_style = ff.def_style("atom", "full")
o = atom_style.def_type("OW", mass=15.9994, charge=-0.8476)
h = atom_style.def_type("HW", mass=1.008, charge=0.4238)
ff.def_style("bond", "harmonic").def_type("OW-HW", o, h, k=450.0, r0=1.0)
ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
    "OW", o, epsilon=0.1553, sigma=3.166
)

mp.io.write_mrec(Path("water.mrec"), frame=frame, forcefield=ff)
record = mp.io.mrec.MrecFile(Path("water.mrec"))
print(record.sections())

section = record.forcefield()
print(section.name, sorted(section.tables))
restored = section.to_forcefield()
print([(s.category, s.name) for s in restored.get_styles(mp.ff.forcefield.Style)])
```

A pair style's table holds each atom type's own row (`itom == jtom`) and any
explicit cross row (`itom != jtom`, e.g. a CHARMM NBFIX), which prices that
type pair in place of the style's mixing rule. Both kinds round-trip. A
pair is found by its two types in either order, so a table holds one row per
pair: a section restating a pair with other parameters is refused on read.

`MrecFile.forcefield()` returns `None` for a record without a force field.
`mp.io.write_mrec(path, forcefield=ff)` writes a force-field package with
no structure at all.

## From Rust and the browser

The Rust doors have the Python names (features `zarr` and `filesystem`):
`molrs::io::write_mrec(sink, &record)` writes a whole `MolRec` (a path, or a
`Vec<u8>` that receives the packed record), and `molrs::io::mrec::MrecFile::open(src)`
reads it per section (`sections()`, `meta()`, `frame()`, `system()`,
`forcefield()`, `record()`); `molrs::io::mrec` also holds `pack_mrec_zip`,
and for streaming `SequenceSchema`, `MrecWriter` and `MrecReader`
(`MrecReader::open(src)`). Their rustdoc on [docs.rs](https://docs.rs/molcrafts-molrs) carries
compiled examples.

```rust
use std::path::Path;

use molrs::io::mrec::{MolRec, MrecFile};
use molrs::io::write_mrec;

fn main() -> std::io::Result<()> {
    let mut record = MolRec::new();
    record.frame = Some(molrs::core::Frame::new());
    write_mrec(Path::new("water.mrec"), &record)?;

    let file = MrecFile::open(Path::new("water.mrec"))?;
    let frame = file.frame()?.expect("a frame section");
    println!("{:?} {}", file.sections()?, frame.len());
    Ok(())
}
```

In the browser, `@molcrafts/molwasm` reads records from bytes:
`new MrecFile(source)` reads one section at a time (`sections()`, `frame()`,
`system()`, `meta()`) and `new MrecReader(source)` decodes trajectory frames,
each over a file map, a packed `*.mrec.zip`'s bytes, or a lazy
range-request store. See the
[package README](https://github.com/MolCrafts/molpy/tree/master/molwasm#trajectory-stores-mrec).

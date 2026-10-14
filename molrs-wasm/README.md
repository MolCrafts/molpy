# @molcrafts/molrs

[![npm](https://img.shields.io/npm/v/@molcrafts/molrs.svg)](https://www.npmjs.com/package/@molcrafts/molrs)

WebAssembly bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

Full documentation lives at <https://docs.molcrafts.org/molrs/>. The package
ships its TypeScript declarations (`molrs.d.ts`); the
[WASM reference page](https://docs.molcrafts.org/molrs/reference/wasm/) maps
the main exports.

## Install

```bash
npm install @molcrafts/molrs
```

## Quick start

```js
import { readSmilesStr, Conformer, writeXyzStr } from "@molcrafts/molrs";

// Parse SMILES → 3D coordinates → XYZ string
const frame = readSmilesStr("CCO");
const mol3d = new Conformer("fast").generate(frame);
console.log(writeXyzStr(mol3d));
```

The published package uses wasm-pack's `bundler` target: configure your bundler
to load WebAssembly modules. A custom `--target web` build instead exports an
async `init()` function that must be awaited before calling the API.

## API

The exports are grouped as molrs groups them: one section per molrs module
(`core`, `io`, `perceive`, `conformer`, `ff` / `optimize`, `compute`); the JS
namespace itself is flat.

### Core data model (`molrs::core`)

- **`Frame`** — container mapping string keys (`"atoms"`, `"bonds"`) to
  `Block`s, plus metadata (`getMeta` / `setMeta` / `metaKeys`, and
  `getMetaScalar` / `setMetaScalar` for numbers)
- **`Block`** — column store with typed arrays. Float columns are `Float64Array` (dtype `float`, stored as f64).
- **`Box`** — simulation box with periodic boundary conditions (`h()` the
  cell matrix, `toFrac` / `toCart`, `wrap`, `corners()`, …)
- **`NDArray`** — owned `Float64Array` plus a shape (what `Box.origin()`, `Box.h()` and `Box.lengths()` return)
- **`Topology`** — the bond graph (`Topology.fromFrame(frame)` reads
  `bonds.atomi` / `atomj`): angles, dihedrals, impropers, connected components
- **`TriMesh`** — a triangle surface (what `readStlBytes` returns)
- **`Sphere`**, **`Cuboid`**, **`Parallelepiped`**, **`HalfSpace`**,
  **`Cylinder`**, **`Ellipsoid`**, **`Polyhedron`**, **`SphereUnion`** — regions
  (signed `distance`, `contains`, `bounds`; `and` / `or` / `not` compose into a
  `Region`)
- **`NeighborList`**, **`Neighbors`**, **`NeighborQuery`** — neighbor search
  (see [Analysis](#analysis))
- `covalentRadius(symbol)` — the element table's covalent radius (Å)

Columns read like numpy: the column's dtype picks the array type, and no
method names a dtype. Dtype names are molrs core's (`DType::name()`), the
same strings the Frame schema and every binding use.

```js
const atoms = frame.get("atoms");                 // Block; throws if absent (frame.has)
atoms.set("x", new Float64Array([0, 1, 2]));      // dtype from the constructor: float
atoms.set("element", ["C", "C", "O"]);            // string[] → string
atoms.set("pos", new Float64Array(9), [3, 3]);    // optional row-major shape
const v = atoms.view("x");                        // zero-copy Float64Array; invalid after WASM memory grows
const x = atoms.copy("x");                        // owned Float64Array to keep
const q = atoms.copy("charge", new Float64Array(atoms.nRows)); // or this fallback if absent
atoms.dtype("x"); atoms.shape("pos"); atoms.has("x"); atoms.keys();
```

| dtype | `view` | `copy` | `set` accepts |
|-------|--------|--------|---------------|
| `float` | `Float64Array` | `Float64Array` | `Float64Array` (`Float32Array` is refused) |
| `i8` `i16` `int` `i64` | `Int8Array` … `BigInt64Array` (`int` is `Int32Array`) | same typed array | same |
| `u8` `u16` `u32` `uint` | `Uint8Array` … `BigUint64Array` (`uint` is `BigUint64Array`) | same typed array | same |
| `bool` | throws; use `copy` | `boolean[]` | `boolean[]` |
| `string` | throws; use `copy` | `string[]` | `string[]` (and `[]`) |
| `c64` / `c128` | throws; use `copy` | `{ real, imag, shape, dtype }` | never |

### Perception (`molrs::perceive`)

```js
const rings = assignRings(frame);     // atoms/bonds gain is_in_ring, n_rings
const arom  = assignAromaticity(frame);
const withH = addHydrogens(frame);
```

### I/O (`molrs::io`)

The binding's `io` sources mirror `molrs::io`: one module per format, holding
that format's reader (or stream) and writer.

- `readSmilesStr(smiles)` → `Frame` (one molecule); `SmilesIr.parse(smiles)` →
  `SmilesIr` → `.toFrame()` (any SMILES, a `.`-separated set included);
  `writeSmilesStr(frame)` → SMILES (molrs's default emit options);
  `readCgsmilesStr(text)` → `Frame`
- `readCsvBlockStr(text, delimiter?, header?)` → `Block`,
  `writeCsvBlockStr(block, delimiter?, header?)` → CSV text
- `XyzStream`, `PdbStream`, `SdfStream`, `LammpsDataStream`, `LammpsDumpStream`,
  `DcdStream`, `XtcStream`, `TrrStream` — chunk-fed readers, the one reader of
  their format (`allocInputBuffer` → `feedIndexChunk` / `finishIndex` →
  `parseRangeInInput` per frame)
- `CifReader`, `GroReader`, `Mol2Reader`, `VaspPoscarReader` — whole-content
  readers, the molrs reader classes of those formats; `readFrame(index)` reads
  frame `index` (0-based), as every reader in every binding does
- `readStlBytes(bytes)` → `TriMesh`
- `readLammpsLogStr(text, style?)` → the `LammpsLog` record (runs, thermo
  tables, timings — Rust and Python's field names); `isLammpsLog(text)` tells a
  log by its first run
- the in-memory doors, named as molrs's `read_<fmt>_str` / `_bytes` and
  `write_<fmt>_str` / `_bytes` (camelCased; Python's `molrs.io` has the same
  set): `readXyzStr` / `readXyzBytes` / `writeXyzStr`, `readPdbStr` /
  `readPdbBytes` / `writePdbStr`, `readSdfStr` / `readSdfBytes`,
  `readGroStr` / `writeGroStr`, `readMol2Str` / `writeMol2Str`, `readCifStr` /
  `writeCifStr`, `readXsfStr` / `writeXsfStr`, `readCubeStr` / `writeCubeStr`,
  `readVaspPoscarStr` / `writeVaspPoscarStr`, `readVaspChgcarStr`,
  `readAmberInpcrdStr`, `readAmberAcStr`, `readAmberPrmtopStr`,
  `readLammpsDataStr` / `readLammpsDataBytes` / `writeLammpsDataStr`,
  `readLammpsDumpStr` / `readLammpsDumpBytes` / `writeLammpsDumpStr(frame,
  columns?)`; `readDcdBytes(bytes, decoderState?)` / `writeDcdBytes`,
  `readTrrBytes` / `writeTrrBytes`, `readXtcBytes` / `writeXtcBytes` (binary,
  one frame)
- `readMsgpackFrameBytes` / `writeMsgpackFrameBytes`, `readJsonFrameStr` /
  `writeJsonFrameStr` — the `molrs::stream` wire encodings

### 3D generation (`molrs::conformer`)

- `new Conformer(speed?, addHydrogens?, seed?).generate(frame)` — distance
  geometry + MMFF94 coordinate generation (`"fast"` | `"medium"` | `"better"`)

### Force fields + geometry optimization (`molrs::ff`, `molrs::optimize`)

```js
const typifier = new UffTypifier();                 // or Mmff94Typifier / Mmff94sTypifier
const typed    = typifier.typify(frame);
const pots     = new PotentialCompiler(typifier.forcefield()).compile(typed);
const nl       = new NeighborList(12.5);            // or NeighborList.bruteForce(12.5)
nl.build(typed);
const report   = new Lbfgs(pots, nl.neighbors()).minimize(typed);  // pairs come from the NeighborList
```

- **UFF** — full RDKit default table (entire periodic table + oxidation states)
- **MMFF94 / MMFF94s** — Merck force fields
- **no GFN-FF**
- `typifier.forcefield()` → the `ForceField` typing assigned;
  `new PotentialCompiler(ff).compile(typedFrame)` → `Potentials` — the native
  composition, no typifier-to-potentials shortcut
- `Potentials.calcEnergyForces(coords)` → `{ energy, forces }`

### Analysis (`molrs::compute`)

```js
import { NeighborList, Rdf, Msd, LinearFit, Cluster, CenterOfMass, ComputeSpec } from "@molcrafts/molrs";

const nl = new NeighborList(5.0);         // cutoff = 5.0 A, O(N) cell list
nl.build(frame);                          // index only — no pair table
const pairs = nl.neighbors();             // materialize: distSq + disp

// Constructor = configuration; compute({ ... }) = data, named as the core Input.
const gr = new Rdf(100, 5.0, 0.0).compute([{ pairs }]);   // one object per frame
console.log(gr.toJSON().rdf.data);

const curve = new Msd(200).compute({ positions: traj, dt: 0.5 });  // TimeCorrelation
const fit = new LinearFit(0.2, 0.8).compute({ correlation: curve, channel: 0 });

const clusters = new Cluster(5).compute({ pairs });
const com = new CenterOfMass().compute({ positions, partition: clusters, masses });

ComputeSpec.all();                        // every compute: params, defaults, inputs
```

A self search is **half-shell**: each unordered pair appears once, with
`i < j`. `neighbors()` keeps both physical columns by default — that names the
*columns*, not the pair direction. Pass `{ distSq: false }` or `{ disp: false }`
to drop one; a column that was not stored reads back as `undefined`, never as a
fabricated zero array. `disp` is the unnormalized minimum-image displacement
`r_j - r_i` (Å), flattened three values per pair.

- **`NeighborList`** — neighbor-search engine (`build` / `update` index,
  `neighbors` materializes); `NeighborList.bruteForce(cutoff)` selects the
  O(N²) reference backend
- **`Neighbors`** — the materialized pair table (`nPairs`, `queryPointIndices()`,
  `pointIndices()`, `distSq()`, `disp()`)
- **`NeighborQuery`** — the cross search: `new NeighborQuery(refFrame, cutoff)`
  indexes a reference frame, `query(otherFrame)` returns the directed pairs
- **One class per entry of `ComputeSpec.all()`**, under its molrs name (`Rdf`,
  `Msd`, `Correlator`, `Steinhardt`, `HBonds`, `PmftXy`, `RadicalVoronoi`, …).
  The constructor takes the configuration only (optional parameters
  `undefined` keep the molrs default; choices are their canonical names);
  `compute(data)` takes one object whose fields are the core `Input`'s, with
  the snake_case names `spec().inputs` lists. Shaped data is an `NDArray`, a
  `{ dim, data }` object, or (for `N × 3` / `N × 4` rows) a flat array.
- Every `compute` returns a **handle** of the molrs output type
  (`TimeCorrelation`, `RdfResult`, `ClusterResult`, …): `toJSON()` gives its
  data with the molrs field names, and the handle itself is what a downstream
  compute takes (`TimeCorrelation` → `LinearFit`, `ClusterResult` →
  `CenterOfMass`, `PcaResult` → `Kmeans`, `SteinhardtResult.qlm(6)` →
  `SolidLiquid`).
- `ComputeSpec.all()` / `ComputeSpec.find(name)` and each class's static
  `spec()` serialise the molrs descriptors; `tunables()` reads an instance's
  parameter values. Labels, units and ranges are the UI's.
- Errors are the molrs `ComputeError`, thrown as an `Error`.

### Block column conventions

Names and dtypes are the Frame schema. `schemaDocument()` is that vocabulary
(`JSON.stringify` it for text). `keysDocument()` is the constant
names (`X`, `BOND_TYPE`, `ATOMS`, `UNITS`, …) projected from the same tables.
The Rust and Python bindings print the vocabulary with `schema.to_markdown()`.

`F` is the molrs core float type — always `f64`.

## Trajectory stores (`*.mrec`)

Records written by molrs in Python or Rust (see
[Record files](https://docs.molcrafts.org/molrs/guides/records/)) read here
from bytes. `readMrecFrameBytes(bytes)` returns the `frame` section (or
`undefined`) of a packed `*.mrec.zip` and `sectionNamesBytes(bytes)` lists its
sections; `readMrecFrameFiles(files)` and `sectionNamesFiles(files)` do the
same for a `Map<path, Uint8Array>` of the record's files. Every reader decodes `zstd` and `shuffle`, so columns written with
a declared precision read as they do natively.

`MrecReader` opens a MolRec trajectory (Zarr V3) and decodes one frame
per call; consecutive frames of the same chunk are slices, not decodes.

```js
import { MrecReader } from "@molcrafts/molrs";

// (a) every file in memory
const reader = new MrecReader(files);            // Map<path, Uint8Array>
// (b) a packed store
const zipped = MrecReader.fromZip(bytes);        // Uint8Array of *.mrec.zip
// (c) served on demand — only touched chunks cross into wasm
const lazy = MrecReader.fromStorage({
  get: (key) => ...,                                   // Uint8Array | null
  getRange: (key, offset, length) => ...,              // length -1 = to end
  size: (key) => ...,                                  // number | null
  list: (prefix) => [...],                             // keys under prefix
});

reader.nFrames();
const frame = reader.readFrame(t);                     // Frame | undefined
const xyz = reader.readColumns(t, ["atoms/x", "atoms/y", "atoms/z"]);
reader.blockUpdateAt("bonds", t);                      // same value ⇒ same rows
reader.boxAt(t);
```

The host callbacks of `fromStorage` are synchronous: in a Worker that is
`FileReaderSync` over `File` handles or a synchronous range request; on the
main thread hand the reader a `Map` instead.

## Build from source

```bash
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

This writes `pkg/` — the npm package. `pkg/package.json` is auto-generated by
wasm-pack with name `@molcrafts/molrs`. Consumers link it directly:

```jsonc
// consumer's package.json
"dependencies": {
  "@molcrafts/molrs": "link:../path/to/molrs-wasm/pkg"
}
```

Then `npm install` creates a symlink — rebuilding `pkg/` (via `wasm-pack build`)
is picked up immediately by the consumer's dev server. No `npm link` dance
needed.

### Variants (optional)

Default features are `smiles`, `io`, `compute`, `conformer`, `voronoi`, `stream`, and `builder`. To
build a smaller wasm containing only a subset, use Cargo features:

```bash
wasm-pack build --release --target bundler --scope molcrafts \
  --out-name molrs \
  --no-default-features --features io,smiles
```

Note: **variants are mutually exclusive at runtime** — you can't mix a
`compute`-only build with an `io`-only build in the same app, because each
produces a separate wasm module with its own `Frame` class identity.

## License

BSD-3-Clause

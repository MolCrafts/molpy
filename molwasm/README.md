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
import { SmilesIr, Conformer, fillSymbols, writeXyz } from "@molcrafts/molrs";

// Parse SMILES → 3D coordinates → XYZ string
const frame = SmilesIr.parse("CCO").toFrame();      // atoms.atomic_number
const mol3d = new Conformer("fast").generate(frame);
fillSymbols(mol3d);                                 // XYZ spells symbols: atoms.element
console.log(writeXyz(mol3d));
```

Species is `atoms.atomic_number` everywhere molrs reads it; `atoms.element`
is raw symbol data. Readers keep what the file spells (`readXyz` gives
`element`, `readCube` gives `atomic_number`) and nothing converts
implicitly — `fillAtomicNumbers(frame)` / `fillSymbols(frame)` are the two
conversions.

The published package uses wasm-pack's `bundler` target: configure your bundler
to load WebAssembly modules. A custom `--target web` build instead exports an
async `init()` function that must be awaited before calling the API.

## API

The exports are grouped as molrs groups them: one section per molrs module
(`core`, `io`, `perceive`, `conformer`, `ff` / `optimize`, `compute`); the JS
namespace itself is flat. Every operation has exactly one door: no `Str` /
`Bytes` twins, no constructor families, and removal is `del*` (`delBlock`,
`delAtom` / `delAtoms` — a single and a batch removal are distinct).

### Core data model (`molrs::core`)

- **`Frame`** — container mapping string keys (`"atoms"`, `"bonds"`) to
  `Block`s (`createBlock`, `get`, `set`, `has`, `delBlock`, `renameBlock`,
  `keys`, `clear`), the `box`, and typed metadata: `getMeta(name)` returns the
  value as its own JS type (`number` for `f64` / `i32` / `u32`, `bigint` for
  `i64` / `u64`, `string`, `boolean`, a typed array for the vector dtypes, a
  JSON value for `json`), `setMeta(name, value)` stores a JS value as the
  dtype it is (`number` → `f64`, `bigint` → `i64`, `Float64Array(3)` →
  `f64x3`, …), `metaKeys()
- **`Block`** — column store with typed arrays. Float columns are `Float64Array` (dtype `float`, stored as f64).
- **`Box`** — simulation box with periodic boundary conditions, one
  constructor as `SimBox::new`: `new Box(cell, origin, pbc)` with a **`Cell`**
  (`Cell.matrix(h)` — 9 values, row-major, lattice vectors as columns —
  `Cell.lengths([a, b, c])`, `Cell.lengthsAngles(lengths, angles)`,
  `Cell.lengthsTilts(lengths, tilts)`, `Cell.undefined()`), and
  `Box.bounding(points, padding, pbc)` for the box around a point cloud.
  One door per transform, each returning its result: `toFrac(xyz)`,
  `toCart(frac)`, `delta(a, b, minimumImage)` → `NDArray`, and
  `wrap(xyz)` → `{ xyz, shifts }` (the image shift applied to each point).
  Also `volume()`, `origin()`, `lengths()`, `tilts()`, `pbc()`, `corners()`,
  `h()` (the cell matrix, flat **column-major**)

```js
const box = new Box(Cell.lengths([10, 10, 10]), [0, 0, 0], [true, true, true]);
const { xyz, shifts } = box.wrap(NDArray.from(positions, [n, 3]));
```
- **`NDArray`** — owned `Float64Array` plus a shape (what `Box.origin()`, `Box.h()` and `Box.lengths()` return)
- **`Topology`** — the bond graph (`Topology.fromFrame(frame)` reads
  `bonds.atomi` / `atomj`): angles, dihedrals, impropers, connected
  components; `addAtoms` / `addBonds`, and `delAtom(i)` / `delAtoms(idxs)`,
  `delBond(i)` / `delBonds(idxs)` (swap-remove; an index out of range throws)
- **`TriMesh`** — a triangle surface (what `readStl` returns)
- **`Sphere`**, **`Cuboid`**, **`Parallelepiped`**, **`HalfSpace`**,
  **`Cylinder`**, **`Ellipsoid`**, **`Polyhedron`**, **`SphereUnion`** — regions
  (signed `distance`, `contains`, `bounds`; `and` / `or` / `not` compose into a
  `Region`); `Parallelepiped.create(cell, origin)` takes a `Cell`, as a `Box`
  does
- **`NeighborList`**, **`Neighbors`**, **`NeighborQuery`** — neighbor search
  (see [Analysis](#analysis))
- `covalentRadius(atomicNumber)` — the element table's covalent radius (Å)
- `fillAtomicNumbers(frame)` / `fillSymbols(frame)` — the explicit species
  conversions, in place: `atoms.element` symbols → `atoms.atomic_number`
  (`uint`, `"*"` is 0) and back. molrs chemistry reads `atomic_number` only,
  so a frame from a symbol format (XYZ, PDB, SDF, …) goes through
  `fillAtomicNumbers` first; nothing converts implicitly

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

Perception, `Conformer.generate`, the typifiers and `SmilesIr.fromFrame` read
`atoms.atomic_number`; a frame that has only `element` symbols throws until
`fillAtomicNumbers(frame)` converts it.

The semantics are molrs's: `assign*` **mutate** the frame they are given (as
`assign_*(&mut Atomistic)` does) — the perceived columns are added to its
`atoms` / `bonds` blocks, everything else is kept, and block handles taken
before the call must be taken again. `addHydrogens` / `removeHydrogens` change
the graph itself and **return a new frame**, leaving their input alone.

```js
assignRings(frame);                   // frame's atoms/bonds gain is_in_ring, n_rings
assignAromaticity(frame);             // bond_type 4 on aromatic bonds, bond_number everywhere
assignKekuleBondOrders(frame);        // → whether any bond was aromatic
const withH = addHydrogens(frame);    // a new frame
```

### I/O (`molrs::io`)

The binding's `io` sources mirror `molrs::io`: one module per format, and one
door per format and direction.

- **`read<Fmt>(source)`** takes the whole file's content as a `Source`: a
  `string` is always the file's **text** (WASM has no filesystem, so a string
  is never a path), a `Uint8Array` its bytes. A binary format (DCD, TRR, XTC)
  takes the `Uint8Array` alone. It returns the file's first (or only) frame.
- **`write<Fmt>(frame)`** returns the whole one-frame file: a `string` for a
  text format, a `Uint8Array` for a binary one.

| Format | Read | Write |
|--------|------|-------|
| XYZ / extended XYZ | `readXyz` | `writeXyz` |
| PDB | `readPdb` | `writePdb` |
| GRO | `readGro` | `writeGro` |
| SDF / MOL | `readSdf` | — |
| MOL2 | `readMol2` | `writeMol2` |
| CIF | `readCif` | `writeCif` |
| XSF | `readXsf` | `writeXsf` |
| Gaussian cube | `readCube` | `writeCube` |
| POSCAR / CONTCAR | `readPoscar` | `writePoscar` |
| CHGCAR | `readChgcar` | — |
| AMBER inpcrd, `.ac`, prmtop | `readInpcrd`, `readAc`, `readPrmtop` | — |
| LAMMPS data | `readLammpsData` | `writeLammpsData` |
| LAMMPS dump | `readLammpsDump` | `writeLammpsDump(frame, columns?)` |
| DCD, TRR, XTC (bytes) | `readDcd`, `readTrr`, `readXtc` | `writeDcd`, `writeTrr`, `writeXtc` |
| STL (→ `TriMesh`) | `readStl` | — |
| CSV (↔ `Block`) | `readCsvBlock(source, delimiter?, header?)` | `writeCsvBlock(block, delimiter?, header?)` |
| LAMMPS log | `readLammpsLog(source, style?)` → the `LammpsLog` record; `isLammpsLog(text)` | — |
| `molrs::stream` wire | `readWireFrame(source, encoding)` | `writeWireFrame(frame, encoding)` → `Uint8Array` |

`encoding` is a `FrameEncoding`: `"msgpack"` or `"json"`.

```js
const frame = readXyz(await file.text());                         // text
const same  = readXyz(new Uint8Array(await file.arrayBuffer()));  // bytes
const dcd   = writeDcd(frame);                                    // Uint8Array
```

- SMILES and CGsmiles are their classes, as in molrs:
  `SmilesIr.parse(text).toFrame()` reads (connectivity only — a `.`-separated
  set becomes one frame of every component), `SmilesIr.fromFrame(frame).toString()`
  writes (molrs's default emit options: canonical); `SmilesIr.parse(text).nComponents`;
  `CgSmilesIr.parse(text).toFrame()` reads the molecule a CGsmiles string states
- `XyzStream`, `PdbStream`, `SdfStream`, `LammpsDataStream`, `LammpsDumpStream`,
  `DcdStream`, `XtcStream`, `TrrStream` — chunk-fed readers, the one reader of
  their format (`allocInputBuffer` → `feedIndexChunk` / `finishIndex` →
  `parseRangeInInput` per frame)
- `CifReader`, `GroReader`, `Mol2Reader`, `VaspPoscarReader` — random-access
  readers over a `Source` (`new GroReader(source)`): `readFrame(index)`
  (0-based, `undefined` past the end), `len()`, `isEmpty()`

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
- `Potentials.evaluate(coords)` → `{ energy, forces }` (flat `3N` coordinates)
- `new Lbfgs(pots, neighbors, fmax?, maxStep?, memory?)`; `minimize(frame,
  maxSteps?, fixed?)` minimizes in place. A given knob that is not positive
  (or a `fixed` index outside the frame) throws; nothing is clamped

### Analysis (`molrs::compute`)

```js
import { NeighborList, Rdf, Msd, LinearFit, Cluster, CenterOfMass, ComputeSpec } from "@molcrafts/molrs";

const nl = new NeighborList(5.0);         // cutoff = 5.0 A, O(N) cell list
nl.build(frame);                          // index only — no pair table
const pairs = nl.neighbors();             // materialize: distSq + disp

// Constructor = configuration; compute({ ... }) = data, named as the core Input.
const gr = new Rdf(100, 5.0, 0.0).compute([{ pairs }]);   // one object per frame
console.log(gr.toJSON().rdf.data);

// Msd / Vacf are accumulator-only in molrs: one record per frame, folded in order.
const curve = new Msd(200).compute(traj.map((xyz) => ({ a: xyz, dt: 0.5 })));  // TimeCorrelation
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
  the snake_case names `spec().inputs` lists. A frame-form compute (`Rdf`,
  `Msd`, `Vacf`, the distributions — the molrs accumulators) takes an array of
  such objects, one per frame, and folds them in order (`Msd`: `{ a, dt,
  simbox? }`, `Vacf`: `{ a, dt }` with `a` that frame's flat velocities).
  Shaped data is an `NDArray`, a `{ dim, data }` object, or (for `N × 3` /
  `N × 4` rows) a flat array.
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
from an `MrecSource` — however JS holds the record:

- a `Uint8Array`: the bytes of a packed `*.mrec.zip`;
- a `Map<path, Uint8Array>` of every file of a `*.mrec` directory;
- an `MrecStorageHost`, serving keys on demand — only the chunks a frame
  touches cross into wasm.

Every reader decodes `zstd` and `shuffle`, so columns written with a declared
precision read as they do natively.

`MrecFile` reads a record one section at a time (as molrs `MrecFile`), and
`MrecReader` opens its `trajectory` section and decodes one frame per call;
consecutive frames of the same chunk are slices, not decodes.

```js
import { MrecFile, MrecReader } from "@molcrafts/molrs";

const file = new MrecFile(bytes);                // Uint8Array of *.mrec.zip
file.sections();                                 // ["frame", "meta", …]
const snapshot = file.frame();                   // Frame | undefined; also system(), meta()

const reader = new MrecReader(files);            // Map<path, Uint8Array>
const lazy = new MrecReader({                    // served on demand
  get: (key) => ...,                                   // Uint8Array | null
  getRange: (key, offset, length) => ...,              // optional; length -1 = to end
  size: (key) => ...,                                  // number | null
  list: (prefix) => [...],                             // keys under prefix
});

reader.nFrames();
const frame = reader.readFrame(t);                     // Frame | undefined
const xyz = reader.readColumns(t, ["atoms/x", "atoms/y", "atoms/z"]);
reader.blockUpdateAt("bonds", t);                      // same value ⇒ same rows
reader.boxAt(t);
```

The host callbacks are synchronous: in a Worker that is `FileReaderSync` over
`File` handles or a synchronous range request; on the main thread hand the
reader a `Map` instead.

## Build from source

```bash
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

This writes `pkg/` — the npm package. `pkg/package.json` is auto-generated by
wasm-pack with name `@molcrafts/molrs`. Consumers link it directly:

```jsonc
// consumer's package.json
"dependencies": {
  "@molcrafts/molrs": "link:../path/to/molwasm/pkg"
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

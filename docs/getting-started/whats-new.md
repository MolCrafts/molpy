# What's New

## 0.16 (unreleased)

MolPy 0.16 pairs with **molrs 0.16** (`molcrafts-molrs>=0.16.0,<0.17`). Two
things change at once: molrs 0.16 makes the force-field IR a *protocol*, and
molpy becomes a thin layer over molrs, with one module per job and one public
path per name.

### One module, one job

Every native name in molpy is the molrs object itself (`mp.Atomistic is
molrs.system.Atomistic`); molpy keeps no parallel IR, I/O, geometry, units or
regions. molrs 0.16 puts every symbol under the subsystem that owns it
(`molrs.store`, `molrs.system`, `molrs.spatial`, `molrs.ff`, `molrs.io`, …),
and molpy maps those subsystems in one of two ways:

- **Mirrored subpackages.** `mp.ff` mirrors `molrs.ff` submodule by submodule
  (`forcefield`, `potential`, `typifier`, `charge`, `ir`, `params`,
  `scale_lj`); `mp.io`, `mp.compute`, `mp.signal`, `mp.md`, `mp.op` and
  `mp.builder` mirror `molrs.io`, `molrs.compute`, `molrs.signal`,
  `molrs.md`, `molrs.op` and `molrs.builder`. molpy's own additions sit next
  to the native names: the AmberTools typifiers in `mp.ff.typifier`,
  `read_smiles` / `read_amber` in `mp.io`, crystals, polymers and virtual
  sites in `mp.builder`.
- **The flat root.** The data model and the operations on it stay on `mp`:
  everything in `molrs.store` (`Frame`, `Block`, `keys`, `schema`, …),
  `molrs.system`, `molrs.spatial` (regions, neighbour search),
  `molrs.units`, `molrs.perceive`, `molrs.optimize` and `molrs.conformer`.

So a force field is `mp.ff.forcefield.ForceField`, its compiler
`mp.ff.potential.PotentialCompiler`, a SMILES parser `mp.io.SmilesIR`, and an
assembler `mp.builder.Assembler`. The full list of moves is under
[Upgrading from 0.15](#upgrading-from-015).

### Your own styles: `mp.ff.ir`

A style or a whole category of the right form extends the force-field IR from
Python, with nothing rebuilt, and is typed, priced, saved and read back like a
built-in. `mp.ff.ir` is `molrs.ff.ir`: `StyleSpec`, `Param`,
`register_style`, `register_category`, `styles`, `categories`, `evaluate`,
`unregister`, `IrError` and its subclasses.

```python
class Fene(mp.ff.ir.StyleSpec):  # LAMMPS bond_style fene
    category, name = "bond", "fene"
    params = {"k": "E/L^2", "r0": "L", "epsilon": "E", "sigma": "L"}
    expression = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
                  "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")
```

A typifier's `Match` types terms with it (`links={mp.Bond: rows}`),
`mp.ff.potential.PotentialCompiler` prices it, and `mp.io.write_mrec` stores
its expression, so a process that never registered it reads the record and
prices it the same. [Extending the Force Field](../developer/extending-forcefield.md)
is the whole recipe in under 30 lines.

`mp.ff.potential.kernel(category, style, atoms, **params)` builds the kernel
of any registered style over explicit instances (one parameter row per term,
angle values in degrees) as a `Potentials`, which `Potentials.push` moves into
a larger one.

### File formats are molrs's

The LAMMPS data, XYZ, AMBER, PDB and `fix bond/react` code molpy carried is in
molrs now, and `mp.io` re-exports it:

- `mp.io.read_lammps_data(path, atom_style=None)` returns the `Frame`. Typed
  blocks carry `type_id` and the string `type` (the Type Labels, or the id);
  `atom_style` fixes the `Atoms` layout as LAMMPS's does (a layout the style
  does not have is an error, not a column drop). The `* Coeffs` sections stay
  text in `frame.meta["lammps_coeffs_text"]`, and
  `mp.ff.forcefield.read_lammps_data_coeffs(text, units=..., atom_labels=...)`
  makes them a force field; the header counts and Type Labels are
  `frame.meta["lammps_counts"]` and `frame.meta["<kind>_type_labels"]`.
- `mp.io.write_lammps_data(path, frame, type_labels={"atoms": [...]})`
  declares labels no row uses; a Drude system gets a `fix drude` flags
  header comment.
- `mp.io.read_xyz` reads an n-wide extended-XYZ property as one `(N, n)`
  column and `species` as `element`; `mp.io.read_amber_inpcrd(path, frame)`
  reads coordinates into an existing frame; `mp.io.read_ac` emits `charge`.
- `mp.io.write_pdb` writes `X` for an atom without an element.
- `mp.io.BondReactTemplate`, `write_bond_react_map` and
  `write_lammps_bond_react_system` write a `fix bond/react` system.
- `mp.io.read_frame` / `write_frame` pick the format from the file name.

Force-field file formats are `mp.ff.forcefield`'s (`read_forcefield_xml`,
`read_gromacs_system`, `write_lammps_forcefield`, …); `mp.io` holds
structure and trajectory formats only.

### One input writer per engine

`mp.io.emit` is gone. Each engine has one `generate_inputs`:
`LAMMPSEngine.generate_inputs(frame, ff, out)` writes the data file, the
force-field settings, the init and the input script — the same deck its
`minimize` / `md` run — and the new `GROMACSEngine.generate_inputs(frame, ff,
out)` writes the `.gro`, `.top` and `.mdp` templates (its `run` grompp's and
mdrun's an `.mdp`). `mp.Script` is `mp.engine.Script`.

Every LAMMPS style line and coefficient is molrs's LAMMPS include
(`mp.ff.forcefield.write_lammps_forcefield_str`), read after `read_data`, so
whatever molrs writes reaches LAMMPS: a category spanning two styles as
`hybrid`, and `angle charmm` with its Urey-Bradley term. The relaxation keeps
the force field's `special_bonds` and `pair_modify mix`. A frame without a
box (`Atomistic.to_frame()`) gets a non-periodic deck: `boundary s s s` with
`neighbor 2.0 nsq`, and a data-file box enclosing the atoms with a margin of 1
length unit.

### Regions, units and CL&Pol data are molrs's

- Regions are the native solids: `mp.Cuboid(origin, lengths)`,
  `mp.Cuboid.cube(edge)`, `mp.Sphere(center, radius)`, `mp.HalfSpace`, … with
  `mask(block)`, `region(block)` and `&` / `|` / `~` composition into a
  `mp.Region`. A selector composes with a region (`selector & region`).
- Units are `mp.UnitRegistry` and `mp.UnitPreset`, which now hold the
  `openmm` preset, `UnitPreset.register` / `UnitPreset.names()`, `k_B` and
  `define_lj_units`.
- The CL&Pol Drude table is `mp.ff.params.clpol_polarizability()`;
  `DrudeBuilder` reads it.

### Builders

`mp.builder.GrapheneBuilder` and `CarbonTubeBuilder` are molrs's and build a
`Frame` (`mp.Atomistic.from_frame` makes it a graph, and
`generate_topology(gen_angle=True, gen_dihedral=True)` adds its angles and
dihedrals). `mp.builder.Lattice` keeps its unit cell as a `Box`
(`lattice.box.to_cart(frac)`), and its `build` clips to any native region.
The builder's modules are private; every name is on `mp.builder`, including
`AmberPolymerBuilder` and its result types.

### `GaffTypifier`

`mp.ff.typifier.GaffTypifier(parameter_set=...)` assigns GAFF / GAFF2 bonded
terms and parameters natively to a molecule `AtdTypifier` has typed, without
AmberTools; `ForceField.materialize_params` writes a typed frame's parameters
as columns.

### Force fields price as molrs 0.16 prices them

Everything molpy compiles, types and writes goes through molrs, so molrs
0.16's force-field changes reach molpy unchanged
([molrs migration guide](https://docs.molcrafts.org/molrs/migration/)):

- **Force-field IR, LAMMPS standard.** A harmonic `k` is LAMMPS's `K` (no
  ½) and every equilibrium angle and phase is in degrees. A force field read
  from a file or typed by a typifier prices as before; one built by hand
  with `def_type` needs `k / 2` and degrees. A 0.15 `.mrec` record is
  converted on read.
- **Pair cutoffs.** `PotentialCompiler(ff).compile(frame)` prices a `pairs`
  row only inside its style's `cutoff`, as LAMMPS does (0.15 priced every
  listed pair). A style that states no `cutoff` is untruncated, as before.
- **`coul/long/pme` reads the frame's box**, as LAMMPS's kspace does; a frame
  without a periodic box is refused by name.
- **GAFF atom types follow antechamber's bond orders.** `AtdTypifier`
  perceives the bond orders from the connectivity as `antechamber` does
  (`bond_orders="perceive"`, the default), so a Kekulé structure types as
  antechamber types it; `bond_orders="input"` keeps the graph's orders.
  Types that depend on the Kekulé structure, ring classes or colouring can
  change, and with them GAFF parameters and AM1-BCC / Gasteiger charges.
- **Typed refusals.** Every refusal of the force-field IR is a subclass of
  `mp.ff.ir.IrError` (a `ValueError`): `MissingParam`, `BadValue`,
  `NoMixing`, `NoEngineForm`, … `except ValueError` still catches them.

### Upgrading from 0.15

Every moved or removed public path, old → new. A name not listed keeps its
path.

**Force fields — `mp.ff` mirrors `molrs.ff`**

| 0.15 | 0.16 |
|------|------|
| `mp.ForceField`, `mp.Style`, `mp.Type`, `mp.AtomStyle` / `AtomType`, `mp.BondStyle` / `BondType`, `mp.AngleStyle` / `AngleType`, `mp.DihedralStyle` / `DihedralType`, `mp.ImproperStyle` / `ImproperType`, `mp.PairStyle` / `PairType` | `mp.ff.forcefield.<same name>` |
| `mp.PotentialCompiler`, `mp.Potentials` | `mp.ff.potential.PotentialCompiler`, `mp.ff.potential.Potentials` |
| `mp.BccModel`, `mp.GasteigerModel`, `mp.MullikenModel` | `mp.ff.charge.<same name>` |
| `mp.FragmentScaling` | `mp.ff.scale_lj.FragmentScaling` |
| `mp.potential.StyleSpec`, `Param`, `register_style`, `register_category`, `styles`, `categories`, `evaluate`, `unregister`, `IrError` (0.16 pre-release) | `mp.ff.ir.<same name>` |
| `mp.potential.LJCut`, `Potential`, `kernel` (0.16 pre-release); `molpy.md.LJCut`, `molpy.md.Potential` | `mp.ff.potential.<same name>` |
| `mp.typifier.<X>` (every typifier, `Typifier`, `Match`), `molpy.typifier.ambertools` | `mp.ff.typifier.<X>` |
| `molpy.core.ops.scale_lj(ff, {label: atoms}, data)` | `mp.ff.scale_lj.scale_lj(ff, {label: (types, positions, masses)}, data)` |
| `molpy.core.ops.compute_k_ij`, `fragment_scaling_data` | `mp.ff.scale_lj.compute_k_ij`, `fragment_scaling_data` |
| `molpy.core.ops.intramolecular_pairs` | `mp.ff.potential.intramolecular_pairs` |
| `mp.builder.load_polarizability(path)`; `molpy/data/forcefield/alpha.ff` | `mp.ff.params.clpol_polarizability(path)` (molrs ships `alpha.ff`) |

**I/O — `mp.io` mirrors `molrs.io`; force-field files are `mp.ff.forcefield`'s**

| 0.15 | 0.16 |
|------|------|
| `mp.SmilesIR`, `mp.CGSmilesIR`, `mp.SmilesError` | `mp.io.<same name>` |
| `mp.io.read_xml_forcefield`, `mp.io.write_xml_forcefield` | `mp.ff.forcefield.read_forcefield_xml`, `write_forcefield_xml` |
| `mp.io.read_gromacs_forcefield`, `mp.io.write_gromacs_forcefield` | `mp.ff.forcefield.read_gromacs_top_ff`, `write_gromacs_top_ff` |
| `mp.io.read_lammps_forcefield`, `write_lammps_forcefield`, `write_lammps_forcefield_str`, `read_lammps_data_coeffs`, `write_lammps_data_coeffs` | `mp.ff.forcefield.<same name>` |
| `mp.io.read_top(path)`, `mp.io.write_top(path, frame)` | `mp.ff.forcefield.read_gromacs_system(path)` → `(ForceField, Frame)`, `write_gromacs_system(path, ff, frame)` (0-based) |
| `mp.io.read_amber_ac` | `mp.io.read_ac` (emits `charge`) |
| `mp.io.read_lammps_data(path, atom_style="full")` → `LammpsDataResult` (`.frame`, `.forcefield`, `.counts`, `.type_labels`) | `mp.io.read_lammps_data(path, atom_style=None)` → `Frame`; force field: `mp.ff.forcefield.read_lammps_data_coeffs(frame.meta["lammps_coeffs_text"], units=..., atom_labels=...)`; counts / labels: `frame.meta["lammps_counts"]`, `frame.meta["<kind>_type_labels"]` |
| `mp.io.LammpsDataResult` | removed |
| a missing box axis raised `ValueError`; `frame.meta["format" / "atom_style" / "source_file"]` | recorded in `frame.meta["lammps_box_axes"]`; not set |
| `mp.io.write_lammps_data(..., type_labels={"atom_types": [...]})` | `type_labels={"atoms": [...]}` (block names) |
| Drude header `#   fix DRUDE all drude C D N …` | `# fix drude flags (atom-type order): C D N …` |
| `mp.io.read_xyz` filled `atomic_number` | it does not (`mp.Element` maps symbols) |
| `mp.io.read_amber_inpcrd` mismatch: `ValueError` | `OSError` |
| `mp.io.write_pdb` element from `frame.meta["elements"]`; missing `x`: `ValueError` | `X`; `OSError` |
| `mp.io.BondReactTemplate`, `write_bond_react_map`, `write_lammps_bond_react_system` (molpy's) | the same names, molrs's |
| `molpy.io.mrec` (a molpy module) | `mp.io.mrec` is `molrs.io.mrec` |
| `molpy.io.readers`, `molpy.io.writers`, `molpy.io.data.*` | private / removed |

**Engines — one `generate_inputs` per engine**

| 0.15 | 0.16 |
|------|------|
| `mp.io.emit.emitters.emit("lammps", mol, ff, out, prefix=p)`, `LammpsEmitter().emit(...)` | `mp.engine.LAMMPSEngine(check_executable=False).generate_inputs(mol.to_frame(), ff, out, prefix=p)` → `{"data", "settings", "init", "input"}` |
| `mp.io.emit.emitters.emit("gromacs", mol, ff, out, temperature_K=T)`, `GromacsEmitter().emit(...)` | `mp.engine.GROMACSEngine(check_executable=False, prefix=p).generate_inputs(mol.to_frame(), ff, out, temperature=T)` → `{"gro", "top", "em", "nvt"}` |
| `mp.io.emit.Emitter`, `EmitterRegistry`, `emitters` | removed |
| `mp.Script`, `mp.ScriptLanguage`, `molpy.core.script` | `mp.engine.Script`, `mp.engine.ScriptLanguage` |
| `EnvSpec` / engine `env_manager="pip"` or `"virtualenv"` | `env_manager="venv"` |

**Geometry, units, selection**

| 0.15 | 0.16 |
|------|------|
| `mp.BoxRegion(lengths, origin)` | `mp.Cuboid(origin, lengths)` |
| `mp.Cube(edge, origin)` | `mp.Cuboid.cube(edge, origin)` |
| `mp.SphereRegion(radius, center)` | `mp.Sphere(center, radius)` |
| `mp.AndRegion(a, b)`, `mp.OrRegion(a, b)`, `mp.NotRegion(a)` | `a & b`, `a \| b`, `~a` (a `mp.Region`) |
| `region.isin(xyz)`, `region.bounds` (`(2, 3)`) | `region.contains(xyz)`, `region.bounds()` (`(3, 2)`) |
| `mp.DistanceSelector(center, r_max, min_distance=r_min)` | `mp.Sphere(center, r_max) & ~mp.Sphere(center, r_min)` |
| `mp.CoordinateRangeSelector("x", lo, hi)` | `mp.HalfSpace([-1, 0, 0], [lo, 0, 0]) & mp.HalfSpace([1, 0, 0], [hi, 0, 0])` |
| `mp.UnitSystem()` | `mp.UnitRegistry()` (`k_B` included) |
| `mp.UnitSystem.preset(name)`, `.preset_names()`, `.register_preset(...)` | `mp.UnitPreset(name)`, `mp.UnitPreset.names()`, `mp.UnitPreset.register(name, units, boltzmann=, coulomb=)` |
| `mp.UnitSystem.lj(mass=, sigma=, epsilon=)` | `reg = mp.UnitRegistry(); reg.define_lj_units(mass, sigma, epsilon)` |
| `units.convert(q, "nm")`, `units.factor(a, b)` | `q.to(units.parse("nm"))`, `(1.0 * units.parse(a)).to(units.parse(b)).magnitude` |
| `mp.fields.X` (a `str`), `molpy.core.fields.FieldFormatter` | `mp.keys.X` (a `Key`; `.key` is the `str`); readers emit canonical names |
| `mp.compute.signal` | `mp.signal` |

**Builders**

| 0.15 | 0.16 |
|------|------|
| `mp.Assembler`, `mp.SitePlacer`, `mp.AxisOrienter`, `mp.GrowthPlacer`, `mp.Coarsener` | `mp.builder.<same name>` |
| `mp.builder.GrapheneBuilder(...).build()` / `CarbonTubeBuilder(...).build()` → `Atomistic` | `mp.Atomistic.from_frame(mp.builder.GrapheneBuilder(...).build())` |
| `.build(finalize="topology")`; `molpy.builder._finalize.Finalization`, `StructureFinalizer` | `mol.generate_topology(gen_angle=True, gen_dihedral=True)`; removed |
| `molpy.builder.polymer.<X>`, `molpy.builder.nanostructure.<X>`, `molpy.builder.crystal` / `symmetry` / `virtualsite` / `packing` `.<X>` | `mp.builder.<X>` |
| `molpy.builder.polymer.AmberPolymerBuilder`, `AmberBuildResult`, `AmberCut`, `AmberPieces` | `mp.builder.<same name>` |
| `mp.builder.DistributionIR` | removed (unused) |
| `Lattice.frac_to_cart(f)`, `Lattice.cart_to_frac(c)` | `lattice.box.to_cart(f)`, `lattice.box.to_frac(c)` (`(N, 3)`) |
| `molpy.builder.virtualsite.K_DRUDE` | removed (the table holds each type's `k_D`) |

**Removed outright**

| 0.15 | 0.16 |
|------|------|
| `mp.Config`, `molpy.core.config` | removed: its one setting, `log_level`, configured the logger below |
| `molpy.core.logger.get_logger` | removed (no caller); molpy no longer depends on `molcrafts-mollog` / `molcrafts-molcfg` |
| `mp.Conformer` (a subclass adding an empty-molecule guard) | `mp.Conformer` is `molrs.conformer.Conformer`, which has the guard |

**Behaviour**

| 0.15 | 0.16 |
|------|------|
| emitted `.in.init`: `boundary p p p`; `.in`: `neighbor 2.0 bin` | a box-free frame: `boundary s s s`, `neighbor 2.0 nsq` |
| harmonic `k` with a ½ (`def_type(..., k=2K)`), angles in radians | LAMMPS's `K`, degrees |
| `AtdTypifier` types the graph's bond orders | antechamber's perceived orders; `bond_orders="input"` for the old behaviour |
| a new style: a molrs kernel, writer arms, a rebuilt wheel | `class MyStyle(mp.ff.ir.StyleSpec)` |

New on the root, flattened from molrs: `Cylinder`, `Ellipsoid`,
`Polyhedron`, `SphereUnion`, `TriMesh`, `RelationBuckets`, `AMBER_COULOMB`.

## 0.15

MolPy 0.15 pairs with **molrs 0.15** (`molcrafts-molrs>=0.15.0,<0.16`). Most of
what changed for a molpy user comes from that release: the graph, table and
force-field types, the notation parsers, the native typifiers and the file
readers and writers are molrs's own, re-exported on `mp` by identity. molpy
0.15 also finishes moving its own copies of those layers out, so every name now
has exactly one home.

This page lists what you will notice. If you are upgrading a 0.14 script, read
[Upgrading from 0.14](#upgrading-from-014) at the end.

## 0.15.1

Still on molrs 0.15 (`molcrafts-molrs>=0.15.0,<0.16`).

### GAFF polymers: `AmberPolymerBuilder`

A GAFF chain is now built the way AMBER residues are made, instead of being
joined from typed monomers. `mp.Assembler` folds each leaving group's charge
onto its anchor, which moves a PEO ether oxygen from about −0.42 e to about
−0.20 e; the 0.15.0 route of typing monomers with `AntechamberTypifier` and
finishing the assembled chain with `TLeapTypifier` inherits that.

- `mp.builder.polymer.AmberPolymerBuilder(library, cuts).assemble(sites)` runs
  antechamber and parmchk2 once on an oligomer whose head, chain and tail
  monomers are already bonded, prepgen once per residue it cuts from it, and
  tleap `sequence` over a linear site graph (`{[#PEO]|10}`). It returns an
  `AmberBuildResult`: the typed `chain`, its `forcefield` (merges with the
  typifiers' force fields) and the prmtop / inpcrd paths. Reruns reuse
  `work_dir` and redo only the steps whose inputs changed.
- `AmberCut` is one prepgen control file: the atoms a residue omits (their
  charge is spread over the rest), its connection atoms, and the atom (or
  GAFF type) across each junction.
- `AmberPieces(head, repeat, tail).oligomer()` writes the oligomer and its
  three cuts from three SMILES.
- `TLeapTypifier` now refuses a graph that still has ports.
- antechamber reads the oligomer as mol2 (bonds given, not perceived).
- `molpy.wrapper.run_step` runs one tool and requires the file it must write;
  a failure message carries the tool's stdout when its stderr is empty (tleap
  and prepgen report there).

See [AmberTools Integration](../user-guide/13_ambertools_integration.md).

## Highlights

### Record files: `*.mrec`

A `*.mrec` store is a Zarr-v3 directory that holds a scientific record — one
snapshot, a topology, or a whole trajectory, with ragged frames and `step`
labels. 0.15 brings it to the molrec contract:

- One door per section: `mp.io.write_mrec` / `read_mrec` (a frame),
  `write_mrec_system` / `read_mrec_system` (a topology),
  `write_mrec_trajectory` / `read_mrec_trajectory` (a whole trajectory), and
  `mp.io.mrec.TrajectoryWriter` / `TrajectoryReader` to append or read one frame
  at a time. `mp.io.mrec_sections(path)` says what a store holds.
- **A force field travels in the record.** `write_mrec(path, frame,
  forcefield=ff)` (or `write_mrec_forcefield`) stores it as its own `forcefield`
  section with its units declared and never converted;
  `read_mrec_forcefield(path)` returns that section, or `None`, and
  `mp.ForceField.from_section(section)` turns it back into a force field.
- **Frame meta is stored typed**: an `i32` stays `i32`, a vector stays a
  vector, NaN stays NaN.
- **Declared precision.** A float column can declare an absolute tolerance;
  the writer rounds to it and compresses the column, which takes coordinates
  from 24 to about 8 bytes per atom per frame at `1e-3` Å. Without a
  declaration, floats are stored exactly.
- `meta.molrec_version` is checked only when it is present, so a store written
  by another molrec producer without it opens.

```python
import molpy as mp

frame = mp.Frame(
    blocks={
        "atoms": {
            "type": ["OW", "HW", "HW"],
            "x": [0.0, 0.9572, -0.24],
            "y": [0.0, 0.0, 0.927],
            "z": [0.0, 0.0, 0.0],
        }
    }
)
frame["atoms"].set_precision("x", 1e-3)  # keep x to a thousandth of an Å

ff = mp.ForceField(name="water", units="real")
atom_style = ff.def_style("atom", "full")
ow = atom_style.def_type("OW", mass=15.999, charge=-0.834)
hw = atom_style.def_type("HW", mass=1.008, charge=0.417)

mp.io.write_mrec("water.mrec", frame, forcefield=ff)
print(sorted(mp.io.mrec_sections("water.mrec")))  # ['forcefield', 'frame', 'meta']
ff_back = mp.ForceField.from_section(mp.io.read_mrec_forcefield("water.mrec"))
print(ff_back.units)  # real
```

See [File I/O → mrec stores](../user-guide/11_io.md#mrec-stores-scientific-records).

### Typed frame meta

`frame.meta` keeps insertion order and knows each value's dtype:
`frame.meta.dtype(key)` reports it, `frame.meta.typed()` hands every value out
as an `mp.MetaValue`. What comes back is frozen — a JSON object is a read-only
`mp.MetaDocument` and an array a `tuple` — so a nested edit is copy, edit,
store:

```python
frame.meta["run"] = {"step": 0, "ensemble": "nvt"}
run = frame.meta["run"].copy()  # a plain dict
run["step"] = 3
frame.meta["run"] = run
print(frame.meta.dtype("run"), frame.meta["run"]["step"])  # json 3
```

See [Block and Frame](../tutorials/02_block_and_frame.md).

### One vocabulary for columns and blocks

The frame schema follows the molrec conventions
([Naming Conventions](../tutorials/naming-conventions.md)):

- New canonical atom columns: forces `fx` `fy` `fz`, `formal_charge` (integer),
  `name`, `chain`, `altloc`, `icode`, `occupancy` and `b_factor`. The PDB and
  GRO readers and writers use them (`chain_id` → `chain`, `resname` →
  `res_name`, `atom_name` → `name`).
- New relation blocks: `constraints`, `drudes`, `virtual_sites` and, on a
  coarse-grained frame, `members`. A coarse-grained frame is now `atoms` +
  `bonds` (+ `members`), not `beads` + `cgbonds`.
- Floats are `float64` only (a `float32` array is widened on insert),
  identifier and index columns are `uint64`, and image flags are `int32`.

### Force fields: built by name, compiled by `PotentialCompiler`

- A style is defined with `ff.def_style(category, name, params)` and a type with
  `style.def_type(name, *endpoints, **params)`, where the endpoints are the
  atom-type handles it connects. There are no per-style classes to import.
- `mp.PotentialCompiler(ff).compile(frame)` binds a force field to a typed
  frame and returns the `Potentials` that evaluate energies and forces, and
  that `mp.LBFGS` minimizes.
- A typifier's `typify(mol)` returns a typed **copy**; `forcefield()` holds only
  the types it assigned, `library()` the whole table.
- The LAMMPS writers take the frame: `write_lammps_forcefield(path, ff, frame)`
  writes exactly the coefficients the frame's type labels use. A pair cutoff is
  a run setting you declare on the pair style; molpy never invents one.
- `ForceField.merge` carries special bonds, style parameters and units, and
  raises on a conflicting definition instead of keeping the first.

```python
ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)

typed = mp.Frame(
    blocks={
        "atoms": {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "type": ["OW", "HW"]},
        "bonds": {"atomi": [0], "atomj": [1], "type": ["OW-HW"]},
    }
)
pots = mp.PotentialCompiler(ff).compile(typed)
print(round(pots.calc_energy(typed), 4))  # ½·k·(1.0 − 0.9572)² = 0.4122
```

See [Force Field](../tutorials/04_force_field.md).

### Harmonic impropers: `K = k` on LAMMPS I/O

The harmonic improper is `E = k(χ − χ₀)²`, LAMMPS's own form, so its constant
is LAMMPS's `K` as written. molrs 0.14 halved it when writing a LAMMPS file and
doubled it when reading one. In 0.15 an improper read from a LAMMPS file is
evaluated at **half** the 0.14 energy (the energy LAMMPS gives it), and a force
field whose impropers you built with the kernel's `k` writes a `K` **twice** the
0.14 value. Bond and angle constants keep the `½k` convention, and the GROMACS
reader and writer are unchanged.

### Energies that change

- The OPLS-AA tables are regenerated from GROMACS `oplsaa.ff` (2026.3): classes
  are the GROMACS bond types, and mixing is **geometric** (it was
  arithmetic). The LAMMPS include says so with `pair_modify mix geometric`.
- Force fields read with `read_xml_forcefield` / `read_lammps_forcefield`
  honour the mixing rule they declare.
- GROMACS function types 2 and 3 were swapped; 3 is now Ryckaert–Bellemans.

### Assembly from CGsmiles

A polymer is written as two CGsmiles strings — the units with their bonding
descriptors as ports, and the topology as a site graph — and built by
`mp.Assembler` with `mp.GrowthPlacer`. The same assembler backmaps a
coarse-grained model onto all-atom units with `mp.SubgraphMatcher`,
`mp.Coarsener`, `mp.SitePlacer` and `mp.AxisOrienter`. See
[Assembly](../user-guide/02_assembly.md) and
[Polymer Topologies](../user-guide/topology/index.md).

### Typing

- `mp.typifier.AtdTypifier(parameter_set="gaff2")` evaluates antechamber's
  atom-type tables natively (atom types only).
- `AntechamberTypifier` types one complete molecule through antechamber,
  parmchk2 and tleap; `TLeapTypifier` runs tleap alone over a molecule that
  already carries AMBER types and charges. GAFF polymers: see
  [0.15.1](#0151).

### Smaller things

- Rigid-body transforms are chainable methods on the graph:
  `mol.translate(d)`, `mol.rotate(axis, angle, about=None)` and
  `mol.scale([sx, sy, sz], about=None)` move it in place and return it.
- Every analysis is on `mp.compute`, including the diffusion routes
  (`EinsteinDiffusion`, `GreenKuboDiffusion`, `VACF`, `Plateau`) and their
  result types.
- `mp.Topology.from_frame(frame).connected_components()` numbers the molecules
  of a frame, for example to fill the `mol_id` column LAMMPS needs.

## Upgrading from 0.14

`pip install -U molcrafts-molpy` pulls molrs 0.15. A molrs on another minor line
fails at `import molpy`, naming both versions.

Everything the [molrs migration guide](https://docs.molcrafts.org/molrs/) lists
for 0.14 → 0.15 applies to the molrs names molpy re-exports (`mp.Frame`,
`mp.Block`, `mp.ForceField`, the `mp.io` readers and writers, …). The changes
most scripts hit are:

| 0.14 | 0.15 |
|------|------|
| `ff.def_bondstyle("harmonic").def_type("c", "h", k=..., r0=...)` | `ff.def_style("bond", "harmonic").def_type("c-h", c, h, k=..., r0=...)`, with `c`, `h` the `AtomType` handles |
| `ff.to_potentials(frame)` | `mp.PotentialCompiler(ff).compile(frame)` |
| `write_lammps_forcefield(path, ff, precision, …, frame=None)` | `write_lammps_forcefield(path, ff, frame, *, precision=…)` — the frame is required and selects the coefficients |
| `frame.meta["run"]["step"] = 3` | copy the document, edit it, store it back |
| `block.view(key)`, `block.to_dict()`, `Block.from_dict(d)` | `block[key]`, `{k: block[k] for k in block}`, `mp.Block(d)` |
| `angle.itom` … `angle.ltom` | `angle.endpoints` (a bond keeps `itom` / `jtom`) |
| `box.matrix` | `box.h` |
| `read_trr`, `read_xtc` (lists), `write_trr`, `write_xtc` | `read_trr_trajectory`, `read_xtc_trajectory` (lazy), `write_trr_trajectory`, `write_xtc_trajectory` |
| `mol.move(d)`, `mol.align(...)` | `mol.translate(d)`; compose `rotate` / `translate` |

### molpy-side breaking changes

There are no deprecation shims: an old spelling raises `AttributeError`,
`ImportError` or `TypeError`, so a script that still runs is not silently on an
old path.

**One public path per name.**

- Names are reached from the molpy root: `mp.Box`, `mp.Trajectory`,
  `mp.UnitSystem`, `mp.Script`, `mp.ElementSelector`, … `molpy.core` no longer
  re-exports them.
- The root no longer carries the analyses, the typifiers or per-style classes:
  use `mp.compute.RDF`, `mp.typifier.OPLSAATypifier`, and
  `ff.def_style(category, name)` instead of `BondHarmonicStyle`,
  `PairLJCutCoulLongStyle` and the rest.
- `Entity`, `Link`, `Entities`, `GraphViews`, `Parameters`,
  `AtomisticForcefield` and `FRAME_SCHEMA_VERSION` are gone.
- `ConformerReport` and `ConformerStageReport` are on the root (`mp.ConformerReport`).
- `Region`, `BoxRegion`, `SphereRegion` and `Cube` moved from `mp.builder` to the root.

**Removed subpackages.**

| 0.14 | 0.15 |
|------|------|
| `molpy.parser` | `mp.io.read_smiles`, `mp.SmilesIR`, `mp.SmartsPattern`, `mp.CGSmilesIR` |
| `molpy.potential` | `mp.PotentialCompiler` → `mp.Potentials` |
| `molpy.optimize` | `mp.LBFGS`, `mp.OptReport` |
| `molpy.io.forcefield`, `molpy.io.trajectory`, `molpy.io.log` and the reader / writer classes (`PDBReader`, `LammpsDataWriter`, …) | the `mp.io.read_*` / `mp.io.write_*` functions |
| `mp.io.write_lammps_system(dir, frame, ff)` | `mp.io.write_lammps_data` + `mp.io.write_lammps_forcefield` |
| `molpy.parser.moltemplate`, `molpy.cli` and the `molpy` command | removed, no replacement |
| `molpy.io.emit`: `EMITTERS`, `emit`, `register`, `OpenMMEmitter`, `XMLEmitter` | `molpy.io.emit.emitters` (`.emit`, `.names`, `.register`) |

**Building.** The 0.14 polymer stack is replaced by site-graph assembly:
`PolymerBuilder`, `GraphAssembler`, `MonomerLibrary`, `ResiduePlacer`, `Placer`,
`SiteMap`, `Replicas`, the reaction-selector family (`TopologySelector`,
`ProximitySelector`, `RandomSelector`, …), `AssemblyFinalizer` and the 0.14
`AmberPolymerBuilder` are gone. Write the units and the topology in CGsmiles and
build with `mp.Assembler`; for GAFF chains, use the new `AmberPolymerBuilder`
([0.15.1](#0151)), which takes an oligomer and prepgen cuts. Crosslinked
networks and gels (joining sites by proximity) are not available in 0.15.

**Typing.**

- `AmberToolsTypifier` → `AntechamberTypifier` + `TLeapTypifier`;
  `MMFFTypifier` → `MMFF94Typifier`.
- `ClpTypifier`, `SmartsTypifier`, `LocalTypifier`, `ForceFieldParams`,
  `TypeScope` and `UnboundedPatternSet` are removed.
- `typify` returns a typed copy instead of editing its input.
- The bundled `oplsaa.xml` is gone: `mp.typifier.OPLSAATypifier()` carries the
  table, and `.library()` returns all of it.

**Analysis.**

- `IonicConductivity`, `DielectricSusceptibility`, `DebyeSpectrumFit` and their
  result classes are removed: compose a raw compute with a fit and your own SI
  prefactor ([PMSD](../compute/pmsd.md), [Dielectric](../compute/dielectric.md)).
  The composed route takes the frame spacing in **femtoseconds**, where
  `IonicConductivity` took picoseconds.
- `mp.compute.NeighborList` → `mp.NeighborList`; `Pca` → `Pca2`.

**Other.** `molpy.core.ops.extract_coords(frame)` → `frame.coords.ravel()`.

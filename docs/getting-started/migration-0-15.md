# Migrating to 0.15

molpy 0.15 pairs with its native backend package on the 0.15 line
(`>=0.15.0,<0.16`). As before, the two release together, the backend ships
first, and a mismatched **minor** version fails at `import molpy`.

Every change below is a removal or a rename with one replacement. There are no
deprecation shims: the old spelling raises `AttributeError`, `ImportError` or
`TypeError` straight away, so a script that still runs after you upgrade is not
silently using an old path.

## Installing 0.15 before the backend is on PyPI

molpy 0.15 needs the 0.15 line of the backend, and that line is not published
on PyPI yet, so `pip install molcrafts-molpy` cannot resolve it. Until it is
published, install from source with `uv`: clone the backend repository next to
your molpy checkout and run `uv sync` in molpy, as described in
[Development setup → building the backend from source](../developer/development-setup.md#building-molrs-from-source).
`uv` builds the backend from that sibling checkout because `pyproject.toml`
names it as a path source; `pip` ignores path sources, so it cannot do this
step.

## Rigid-body transforms: `translate`, `rotate`, `scale`

A *rigid-body transform* moves a whole structure without changing any bond
length or angle inside it (a translation or a rotation), or stretches it
uniformly along each axis (a scale). In 0.15 these are three methods defined by
the native core on `Atomistic` and `CoarseGrain`. Each one changes the
coordinates in place and returns the structure itself, so calls chain.

| 0.14 | 0.15 |
|------|------|
| `mol.move([dx, dy, dz], entity_type=Atom)` | `mol.translate([dx, dy, dz])` |
| `mol.move(delta=[dx, dy, dz])` | `mol.translate([dx, dy, dz])` |
| `mol.scale(s)` (one number) | `mol.scale([s, s, s])` — the factor is now per axis |
| `mol.align(...)` | removed; see below |
| `mol.rotate(...)` | `mol.rotate(axis, angle, about=None)` — `angle` in radians, pivot `about` (default: the origin) |

```python
water = water_template.copy().translate([5.0, 0.0, 0.0]).rotate(
    [0.0, 0.0, 1.0], 1.5708, about=[5.0, 0.0, 0.0]
)
```

`move` took an `entity_type` argument that chose which kind of node to move.
`translate` has none: it moves every node that has coordinates. `scale` takes
three factors so that a box can be stretched along one axis only; pass the same
number three times for the old uniform scale. `scale` also accepts `about=`, the
point that stays fixed.

`align` is gone. To turn a structure so that a direction inside it points along
a target direction, either compute the rotation yourself and call `rotate`, or
use an *orienter* — the object a placer uses to decide which way a fragment
faces. `LineOrienter().orient(mol, anchor, body_axis, to_dir)` rotates `mol`
about the point `anchor` so that the vector `body_axis` points along `to_dir`;
`TangOrienter` does the same for a direction perpendicular to `body_axis`. Both
import from `molpy.builder`.

## Polymer assembly

### Placement is opt-in, and `ResiduePlacer` is `TracePlacer`

A *placer* gives freshly pasted monomer copies a geometry before the bonds
between them form. In 0.14 `PolymerBuilder` placed residues by default. In 0.15
the default is `placer=None`, which means **no placement**: every residue keeps
its template coordinates, so the copies sit on top of each other. Ask for
placement explicitly:

```python
# 0.14
builder = PolymerBuilder(library, reaction)                        # placed implicitly
builder = PolymerBuilder(library, reaction, placer=ResiduePlacer())

# 0.15
from molpy.builder import PolymerBuilder, TracePlacer

builder = PolymerBuilder(library, reaction, placer=TracePlacer())
```

`GraphAssembler` works the same way; it never placed by default, and still
does not.

`TracePlacer` is the old `ResiduePlacer` algorithm, now in the native core. It
walks the residue graph breadth-first from the lowest-numbered residue, which
stays where it is, and moves every other residue rigidly relative to its own
parent so that the new bond starts at *bonding range* — the two atoms' summed
covalent radii plus a small buffer. Linear chains, stars and combs place
completely. Two behaviours are worth knowing:

- **A ring's closing bond is formed but not placed.** The bond that closes a
  cycle joins two residues the walk already placed, so it spans whatever
  distance the placement left. Shorten it with a geometry optimization after
  `build_ring`, or give the placer an explicit ring-shaped trace.
- **A `Trace` steers the chain.** A `Trace` is a list of 3D points;
  `TracePlacer().with_trace(Trace(points))` grows each residue along the
  trace's tangents instead of its parent's outward direction. With a trace the
  residues must form one path or one ring; a truly branched topology (a residue
  joined to three or more others) raises `ValueError`.

`Trace`, `Placer`, `Orienter`, `LineOrienter` and `TangOrienter` import from
`molpy.builder` and `molpy.builder.assembly`. To write your own placement rule,
subclass `Placer` and implement `place(self, mol, bonds)`.

### `build` takes a residue topology, not a string

`PolymerBuilder.build` no longer parses notation. It takes a
`ResidueTopology` — a graph whose nodes are residues (one monomer copy each)
and whose edges say which residues are bonded — and passing a string raises
`TypeError`. Build the topology with a constructor, or call the shortcut that
builds it for you:

| 0.14 | 0.15 |
|------|------|
| `builder.build("{[#EO]|20}")` | `builder.build_linear("EO", 20)` or `builder.build(linear_topology(["EO"] * 20))` |
| `builder.build("{[#A][#B][#A]}")` | `builder.build_sequence(["A", "B", "A"])` |
| a ring written in notation | `builder.build_ring("EO", 6)` or `builder.build(ring_topology("EO", 6))` |
| a star written in notation | `builder.build_star("X3", "EO", n_arms=3, arm_length=4)` or `star_topology(...)` |

The topology types were renamed from their notation-derived names to what they
describe: the graph, node and edge IR types with a `CGSmiles` prefix
(`CGSmilesGraphIR`, `CGSmilesNodeIR`, `CGSmilesBondIR`) are now
`ResidueTopology`, `ResidueNode` and `ResidueBond`. Construct a node as
`ResidueNode(label)` — `id` is keyword-only and auto-assigned — and a bond as
`ResidueBond(node_i, node_j)`. All of them, and `linear_topology(labels)`,
`ring_topology(label, n)` and `star_topology(core, arm, *, n_arms,
arm_length, cap=None)`, import from `molpy.builder.assembly`.

`AmberPolymerBuilder.build` still accepts a notation string for a linear chain
(`"{[#EO]|10}"`), or a linear `ResidueTopology`.

### `MonomerLibrary.expand` returns an `Expansion`

Expanding a topology used to return the pasted world, and you built the pairing
rule from the same topology a second time. Now the topology goes in once and
both come back together:

```python
# 0.14
world = library.expand(topology)
pairing = TopologySelector(topology)

# 0.15
expansion = library.expand(topology)
world, pairing = expansion.world, expansion.pairing
```

Most code never calls `expand` directly; `PolymerBuilder.build` does.

### `SiteMap` is the native class

`SiteMap` (in `molpy.builder.assembly`) is now the native implementation,
re-exported. Every method that takes atoms accepts atom views (`Atom`) or
integer handles, and methods that return atoms return **integer handles**.
`fields.SITE` and `fields.Q0` are unchanged strings; they now come from the
native field table like every other canonical name.

## Compute: the dielectric recipe classes are removed

`IonicConductivity`, `DielectricSusceptibility` and their result classes
`ConductivityResult` and `DielectricSusceptibilityResult` are removed, and so is
the `molpy.compute.dielectric` module. Each recipe hid a fit window and a unit
conversion behind one call. Compose the steps yourself: a raw compute returns
the curve, a fit extracts the slope or integral, and you apply the SI prefactor.

```python
from molpy.compute import EinsteinConductivity, LinearFit

raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=500)  # dt in fs
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
sigma = 3.0988e9 * fit["slope"] / (volume_A3 * temperature_K)                 # S/m
```

!!! warning "The time unit changed"
    `IonicConductivity` took the frame spacing `dt` in **picoseconds**. The
    composed route works in **femtoseconds**, and the $3.0988\times10^{9}$
    prefactor above assumes a slope in $e^2\,\text{Å}^2\,\text{fs}^{-1}$
    (charges in elementary charges, positions in Å). Passing an old picosecond
    `dt` unchanged makes the lag axis 1000× too short and the conductivity
    1000× too large. The derivation of the prefactor is on the
    [Einstein conductivity](../compute/pmsd.md) page.

The dielectric primitives (`Dielectric`, `DebyeRelaxation`, `DebyeFit`,
`EinsteinHelfandSpectrum`, `GreenKuboSpectrum`, `DielectricResult`) and the
vibrational-spectrum classes (`PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`,
…) import from `molpy.compute` itself; there is no `molpy.compute.dielectric`
or `molpy.compute.spectra` module. See [Dielectric response](../compute/dielectric.md).

## I/O: engine emitters live on one registry

An *emitter* writes the complete input set for one MD engine (data file,
force-field file, starter run script). The module-level `EMITTERS` dict and the
free functions `emit` and `register` are replaced by one registry object,
`emitters`:

| 0.14 | 0.15 |
|------|------|
| `from molpy.io.emit import EMITTERS, emit, register` | `from molpy.io.emit import emitters` |
| `emit("lammps", atomistic, ff, out_dir, prefix="w")` | `emitters.emit("lammps", atomistic, ff, out_dir, prefix="w")` |
| `for name in EMITTERS: ...` | `for name in emitters.names(): ...` |
| `register("mine", MyEmitter())` | `emitters.register("mine", MyEmitter())` |

An unknown name still raises `KeyError` listing the registered ones.

## Moltemplate: two classes instead of free functions

| 0.14 | 0.15 |
|------|------|
| `emit_python(doc, "system.py")` | `PythonScriptEmitter(base_dir=".").emit(doc, "system.py")` |
| module-level `build_forcefield(...)` / `build_system(...)` | `builder = MolTemplateBuilder(doc, base_dir=".")`, then `builder.build_forcefield()` and `builder.build_system(ff=None, *, auto_topology=True)` |

Both classes import from `molpy.parser.moltemplate`. `base_dir` is the
directory the document's `import` statements resolve against; it is fixed when
the object is constructed and defaults to the current working directory at that
moment. `MolTemplateBuilder` resolves the imports once and builds the force
field once: `build_system()` types its system against that same `ForceField`
unless you pass another, and returns `(system, ff)`.
`read_moltemplate_system("water.lt")` and the `molpy moltemplate` command line
are unchanged. See the [Moltemplate CLI](../user-guide/14_moltemplate_cli.md)
guide.

## Removed without replacement

`Atomistic.move`, `Atomistic.align`, `CoarseGrain.move`, `CoarseGrain.align`,
`ResiduePlacer`, the `CGSmiles`-prefixed residue IR types, string input to
`PolymerBuilder.build`, `IonicConductivity`, `DielectricSusceptibility`,
`ConductivityResult`, `DielectricSusceptibilityResult`,
`molpy.compute.dielectric`, `molpy.io.emit.EMITTERS` / `emit` / `register`,
`molpy.parser.moltemplate.emit_python` and the module-level moltemplate
`build_forcefield` / `build_system`. Every one has the replacement named in its
section above.

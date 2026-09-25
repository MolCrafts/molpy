# Builder

System assembly: select every reaction, execute the graph edits as one batch,
retype the region each edit disturbed, then optionally finalize topology. Growing
a chain and crosslinking a melt are the same algorithm with a different pairing
rule, so there is one kernel and one variation point.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `GraphAssembler` | The kernel: `apply(world, selector)` | Crosslinking an existing graph |
| `PolymerBuilder` | Library + reaction; `.build(topology)` is the sole expand + apply path; `.build_*` only build that topology. No placement unless you pass `placer=` | Ruled polymer topologies |
| `ResidueTopology` | The residue graph `build` takes: `nodes` (residues) + `bonds` (which residues are joined) | Any architecture the shortcuts do not cover |
| `ResidueNode` | One residue: `ResidueNode(label, *, id=...)`; `label` is a `MonomerLibrary` key, `id` is auto-assigned | Hand-built topologies |
| `ResidueBond` | Undirected edge `ResidueBond(node_i, node_j)` between two nodes | Hand-built topologies |
| `linear_topology` / `ring_topology` / `star_topology` | `linear_topology(labels)`, `ring_topology(label, n)`, `star_topology(core, arm, *, n_arms, arm_length, cap=None)` → `ResidueTopology` | Common architectures |
| `Finalization` | `ATOMS`, `TOPOLOGY` (default), or `BONDED` | Choosing when topology is materialized |
| `StructureFinalizer` | Run the shared topology/bonded tail later | Deferred MD export for large systems |
| `AssemblyFinalizer` | Assembly finalizer with aromaticity perception | Molecular reaction products |
| `SiteMap` | Mark `fields.SITE` (and optional leaving H + charge fold) | Naming reaction sites |
| `Replicas` | Grid / linear copies of a strand with `mol_id` | Melt precursor before crosslinking |
| `MonomerLibrary` | Validated repeat-unit templates; `.expand(topology)` | Naming your monomers |
| `Selector` | The one variation point: which matched sites pair up; implement `select(context)` | Writing your own pairing rule |
| `MatchContext` | What a selector receives: `world`, `occurrences` (one list of `{map_number: atom handle}` per reactant), `map_a` / `map_b`, `comp_a` / `comp_b`, and `sites(component, map_number)` | Reading matches inside `select` |
| `TopologySelector` | Pairs adjacent residues (used by `PolymerBuilder`) | Residue-edge pairing |
| `ExhaustiveSelector` / `SpacingSelector` / `ExplicitPairSelector` | Deterministic crosslink rules | Reproducible networks |
| `RandomSelector` | Random pairing to a target `conversion`, seeded | Flory–Stockmayer networks |
| `Placer` | Base class: subclass it and implement `place(mol, bonds)`, which moves whole fragments in place so every bond joining a fragment to its parent starts at bonding range, or raises without moving anything | Writing your own placement rule |
| `TracePlacer` | Moves each residue rigidly next to its parent, one bonding range (summed covalent radii plus a buffer) away. Ring-closing bonds are formed but not placed | Giving freshly pasted templates a geometry |
| `Trace` | A list of 3D points whose tangents give growth directions: `TracePlacer().with_trace(Trace(points))` lays a path or ring along it | Steering a chain along a curve |
| `Orienter` / `LineOrienter` / `TangOrienter` | Facing rule for each placed residue: its site axis along the growth direction (`LineOrienter`) or perpendicular to it (`TangOrienter`); set with `TracePlacer().with_orienter(...)` | Choosing how residues face |
| `SystemPlanner` / `PolydisperseChainGenerator` | Sample a polydisperse chain plan | Bulk / MW-distributed systems |
| `AmberPolymerBuilder` | GAFF-parameterised build via AmberTools | AMBER/LAMMPS-bound workflows |
| `CarbonTubeBuilder` | `CarbonTubeBuilder(n, m, ...)` → `.build()` graph + `.cell()` box | Zigzag, armchair, and chiral nanotubes |
| `GrapheneBuilder` | `GrapheneBuilder(nx, ny, ...)` → `.build()` graph + `.cell()` box | Rectangular graphene honeycomb sheet |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` | Virtual-site augmentation | Polarizable / 4-site models |

## Canonical example

A repeat unit is an ordinary capped molecule with a few of its atoms named.
There is no port system: the reaction SMARTS is the only place the chemistry
lives, and `%a` / `%b` bind it to the atoms you marked. `placer=TracePlacer()`
gives the pasted copies a geometry; without it the copies stay stacked where
the template put them. `Placer`, `TracePlacer`, `Trace` and the three
orienters import from `molpy.builder` as well as `molpy.builder.assembly`; the
residue-topology types, the topology constructors and `MatchContext` import
from `molpy.builder.assembly`.

```python
import molpy as mp
from molpy.builder.assembly import (
    MonomerLibrary,
    PolymerBuilder,
    TracePlacer,
    SiteMap,
)
from molpy.conformer import Conformer
from molpy.core import fields

eo, _ = Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.read_smiles("OCCO")
)
SiteMap(eo).label_elements("O", "a", "b")

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, placer=TracePlacer())
chain = builder.build_linear("EO", 5)

assert chain.__class__.__name__ == "Atomistic"
assert len({int(a[fields.RES_ID]) for a in chain.atoms}) == 5
```

Each repeat unit is a residue, and that identity survives into a PDB or a
prmtop — it is output, not a build-time marker to scrub afterwards.

## React first; finalize when needed

`typifier=` never receives the growing polymer. The builder executes all selected
reactions in one batch against the intact world, then cuts an
`AffectedRegion` around each edit and types that. The order matters: a region cut
*before* the edit would have to model the edit itself — including every
neighbouring reaction close enough to change a type, and the unmapped atoms each
one consumes — while a region cut afterwards inherits a graph on which those
edits already succeeded.

Structurally identical junctions share one typing pass via `RetypeCache`, and
only the region's interior is written back. Bond/angle/dihedral annotations are
never copied out of a region.

The default finalization generates complete angle/dihedral topology once. For a
large system that will be written to an MD format later, select
`Finalization.ATOMS` while building and apply
`StructureFinalizer(Finalization.TOPOLOGY)` at export time. This keeps the public
examples focused on polymer architectures while still documenting the deferred
topology option.

Use `Finalization.BONDED` together with
`bonded=ForceFieldParams(forcefield)` when those generated terms also need type
and parameter annotations.

A full walk of architectures from one ethylene-glycol template — linear, ring,
star, comb, gels, dual network — is the user-guide section
[Polymer Topologies](../user-guide/topology/index.md) (paired with
`examples/topology/`).

## Nanostructure topology

Nanostructure builders keep their lattice planning private and expose two
products: the molecular graph, and the simulation cell it was laid out in.

```python
from molpy.builder import CarbonTubeBuilder

tube_builder = CarbonTubeBuilder(6, 6, cells=2, periodic=True)
tube = tube_builder.build()
assert len(tube.atoms) == 48
assert len(tube.bonds) == 72
assert tube_builder.cell().pbc.tolist() == [False, False, True]
```

The same builder accepts zigzag `(n, 0)`, armchair `(n, n)`, and general chiral
`(n, m)` tubes. See [Nanostructures](../user-guide/04_nanostructures.md) for
open ends, length selection, and deferred topology.

## Crosslinking is the same machine

Strip the library and the residue topology away and you have the kernel itself, which is
all crosslinking needs: a graph you already have, plus a rule for which sites
pair up.

```python
from molpy.builder.assembly import GraphAssembler, RandomSelector

melt = mp.Atomistic()
for i in range(4):
    melt.def_atom(element="N", x=float(i), y=0.0, z=0.0)
    melt.def_atom(element="O", x=float(i), y=1.0, z=0.0)

gel = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).apply(
    melt, RandomSelector(conversion=1.0, seed=1, cutoff=2.0)
)
assert len(list(gel.bonds)) == 4
```

No `placer` is passed here: the melt's coordinates are already meaningful and
must not be disturbed. That is a decision about your *input*, not about which
class you reached for, which is why it is an argument.

## Polydisperse systems

Sample a chain plan, then loop `build`:

```python
import numpy as np
from molpy.builder.polymer import (
    PolydisperseChainGenerator,
    SchulzZimmPolydisperse,
    SystemPlanner,
    WeightedSequenceGenerator,
)

planner = SystemPlanner(
    PolydisperseChainGenerator(
        WeightedSequenceGenerator({"EO": 1.0}),
        {"EO": 44.05},
        distribution=SchulzZimmPolydisperse(1500, 3000),
    ),
    target_total_mass=5e3,
)
plan = planner.plan_system(np.random.default_rng(42))
chains = [builder.build_linear("EO", len(c.monomers)) for c in plan.chains[:2]]
assert len(chains) == 2
```

## Your own pairing rule

You never subclass the assembler. You write a `Selector`, which answers the one
question that varies. The matching has already happened — the kernel does it
once, in linear time — so a selector never scans the system, it only decides.

```python
from molpy.builder.assembly import Selector


class FirstPairSelector(Selector):
    """React exactly one pairing: the first site of each reactant."""

    def select(self, context):
        a_sites = context.occurrences[context.comp_a]
        b_sites = context.occurrences[context.comp_b]
        yield {**a_sites[0], **b_sites[0]}


one = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).apply(
    melt, FirstPairSelector()
)
assert len(list(one.bonds)) == 1
```

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)

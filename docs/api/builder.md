# Builder

System assembly: select every reaction, compile and cache its local product,
execute the graph edits as one batch, then optionally finalize topology. Growing
a chain and crosslinking a melt are the same algorithm with a different pairing
rule, so there is one kernel and one variation point.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `GraphAssembler` | The kernel: `assemble(world, selector)` | Crosslinking an existing graph |
| `PolymerBuilder` | Library + reaction; `.build(cgsmiles)` is the sole expand + assemble path; `.build_*` only format CGSmiles | Ruled polymer topologies |
| `Finalization` | `ATOMS`, `TOPOLOGY` (default), or `BONDED` | Choosing when topology is materialized |
| `StructureFinalizer` | Run the shared topology/bonded tail later | Deferred MD export for large systems |
| `AssemblyFinalizer` | Assembly finalizer with aromaticity perception | Molecular reaction products |
| `SiteMap` | Mark `fields.SITE` (and optional leaving H + charge fold) | Naming reaction sites |
| `Replicas` | Grid / linear copies of a strand with `mol_id` | Melt precursor before crosslinking |
| `MonomerLibrary` | Validated repeat-unit templates; `.expand(topology)` | Naming your monomers |
| `Selector` | The one variation point: which matched sites pair up | Writing your own pairing rule |
| `TopologySelector` | Pairs adjacent residues (used by `PolymerBuilder`) | Residue-edge pairing |
| `ExhaustiveSelector` / `SpacingSelector` / `ExplicitPairSelector` | Deterministic crosslink rules | Reproducible networks |
| `RandomSelector` | Random pairing to a target `conversion`, seeded | Flory–Stockmayer networks |
| `ResiduePlacer` | Lays fresh template copies out in space | Building from templates |
| `SystemPlanner` / `PolydisperseChainGenerator` | Sample a polydisperse chain plan | Bulk / MW-distributed systems |
| `AmberPolymerBuilder` | GAFF-parameterised build via AmberTools | AMBER/LAMMPS-bound workflows |
| `CarbonTubeBuilder` | `.build(n, m, ...)` rolled-graphene topology | Zigzag, armchair, and chiral nanotubes |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` | Virtual-site augmentation | Polarizable / 4-site models |

## Canonical example

A repeat unit is an ordinary capped molecule with a few of its atoms named.
There is no port system and no direction: the reaction SMARTS is the only place
the chemistry lives, and `%a` / `%b` bind it to the atoms you marked.

```python
import molpy as mp
from molpy.builder.assembly import MonomerLibrary, PolymerBuilder, ResiduePlacer, SiteMap
from molpy.conformer import Conformer
from molpy.core import fields
from molpy.parser import parse_molecule

eo, _ = Conformer(add_hydrogens=True, seed=42).generate(parse_molecule("OCCO"))
SiteMap(eo).label_elements("O", "a", "b")

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
builder = PolymerBuilder(
    MonomerLibrary({"EO": eo}), ether, placer=ResiduePlacer()
)
chain = builder.build_linear("EO", 5)

assert chain.__class__.__name__ == "Atomistic"
assert len({int(a[fields.RES_ID]) for a in chain.atoms}) == 5
```

Each repeat unit is a residue, and that identity survives into a PDB or a
prmtop — it is output, not a build-time marker to scrub afterwards.

## Compile first; finalize when needed

`typifier=` never receives the growing polymer. The builder first compiles all
selected junction motifs against the intact monomer templates, caches distinct
rooted environments, and records scalar per-atom changes only. It then executes
all reactions in one batch. Local bond/angle/dihedral annotations are not copied
from a motif into the world.

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

Nanostructure builders keep their lattice planning private and expose one verb:

```python
from molpy.builder import CarbonTubeBuilder

tube = CarbonTubeBuilder().build(6, 6, cells=2, periodic=True)
assert len(tube.atoms) == 48
assert len(tube.bonds) == 72
assert tube["box"].pbc.tolist() == [False, False, True]
```

The same call accepts zigzag `(n, 0)`, armchair `(n, n)`, and general chiral
`(n, m)` tubes. See [Nanostructures](../user-guide/04_nanostructures.md) for
open ends, length selection, and deferred topology.

## Crosslinking is the same machine

Strip the library and the notation away and you have the kernel itself, which is
all crosslinking needs: a graph you already have, plus a rule for which sites
pair up.

```python
from molpy.builder.assembly import GraphAssembler, RandomSelector

melt = mp.Atomistic()
for i in range(4):
    melt.def_atom(element="N", x=float(i), y=0.0, z=0.0)
    melt.def_atom(element="O", x=float(i), y=1.0, z=0.0)

gel = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).assemble(
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
chains = [builder.build("{[#EO]|%d}" % len(c.monomers)) for c in plan.chains[:2]]
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

one = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).assemble(
    melt, FirstPairSelector()
)
assert len(list(one.bonds)) == 1
```

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)

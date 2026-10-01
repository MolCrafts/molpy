# Builder

System construction. Assembly joins copies of ported units along a site graph
through `mp.Assembler`, re-exported on the molpy root with its placers and
orienter; `molpy.builder` keeps the polymer planning layer, the nanostructure
and crystal builders, virtual sites and the topology finalizer. GAFF parameters
for an assembled chain come from the AmberTools typifiers — see
[Typifier](typifier.md).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.Assembler` | `Assembler(library, placer, orienter=None).assemble(sites, cls=None)`: one copy of `library[bead_type]` per site of an `mp.CoarseGrain`, each site bond joining one accepting port of each end; returns the world as `cls` (`mp.Graph` by default) | Every site-graph build: chains, blocks, rings, stars, combs, backmapping |
| `mp.GrowthPlacer` | Grows each molecule breadth-first onto its parent's ports; needs no site positions | Topologies from CGsmiles notation |
| `mp.SitePlacer` | Puts each copy's centre of mass on its site | Sites with positions (a coarsened CG model) |
| `mp.AxisOrienter` | Turns each copy onto its site: backbone-to-centre along the site axis and joining atoms along the bonds for chain units; port directions fitted to bond directions for branch units | Backmapping with `SitePlacer` |
| `mp.CGSmilesIR` | `.templates()` → name → ported `mp.Atomistic`; `.to_coarsegrain()` → site graph | Writing units and topologies |
| `mp.SmilesIR.from_fragment` | `SmilesIR.from_fragment(body).to_template()` → one ported `mp.Atomistic` from a fragment body | Writing one unit |
| `mp.SubgraphMatcher` / `mp.Coarsener` | Find bead groups in a CG model; turn them into sites with a position and an axis | Site graphs from a CG model |
| `Finalization` | `ATOMS` or `TOPOLOGY` (default) | Choosing when angles and dihedrals are generated |
| `StructureFinalizer` | `StructureFinalizer(stage).apply(graph)`: drop, or regenerate, angles and dihedrals once | Topology before MD export |
| `SystemPlanner` / `PolydisperseChainGenerator` / `Chain` / `SystemPlan` | Sample a polydisperse chain plan to a target mass | Bulk / MW-distributed systems |
| `SchulzZimmPolydisperse` / `FlorySchulzPolydisperse` / `PoissonPolydisperse` / `UniformPolydisperse` | Chain-length distributions (`MassDistribution` / `DPDistribution` protocols) | Choosing a molecular-weight distribution |
| `WeightedSequenceGenerator` / `BlockSequenceGenerator` / `AlternatingSequenceGenerator` | Monomer sequences (`SequenceGenerator` protocol) | Copolymer composition |
| `CarbonTubeBuilder` | `CarbonTubeBuilder(n, m, ...)` → `.build()` graph + `.cell()` box | Zigzag, armchair, and chiral nanotubes |
| `GrapheneBuilder` | `GrapheneBuilder(nx, ny, ...)` → `.build()` graph + `.cell()` box | Rectangular graphene honeycomb sheet |
| `Lattice` / `Site` / `SpaceGroup` | Bravais lattice with basis sites (fractional coordinates) and symmetry operators | Crystals |
| `Region` / `BoxRegion` / `SphereRegion` / `Cube` | Geometric regions with `mask(Block)` | Selecting a volume |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` / `load_polarizability` | Virtual-site augmentation | Polarizable / 4-site models |

The crystal `Site` is a lattice basis site; it is unrelated to the sites of an
assembly site graph.

## Canonical example

A unit is a CGsmiles fragment whose bonding descriptors are its ports (`<`
joins `>`, `$` joins `$`, `!` joins `!`; labels and bond orders must match),
embedded by `mp.Conformer`. The topology is a CGsmiles string turned into a
site graph. `GrowthPlacer` needs no coordinates.

```python
import molpy as mp

conformer = mp.Conformer(seed=42)
eo = conformer.generate(
    mp.SmilesIR.from_fragment("[<]OCC[>]").to_template()
)[0]
assert eo.n_ports == 2

sites = mp.CGSmilesIR("{[#EO]|5}").to_coarsegrain()
assembler = mp.Assembler({"EO": eo}, mp.GrowthPlacer())
chain = assembler.assemble(sites, mp.Atomistic)

atoms = chain.to_frame()["atoms"]
assert sorted(set(atoms["frag_id"].tolist())) == [0, 1, 2, 3, 4]  # one per site
assert set(atoms["mol_id"].tolist()) == {1}  # one connected component
assert chain.n_ports == 2  # the chain ends keep their ports and hydrogens
```

Each bond joins one accepting port of each end, chosen by the assembler; the
leaving handles are removed and their charge folds onto the anchors. A build
that cannot give every bond a port raises `ValueError` naming the site. For a
site graph coarsened from a CG model, use
`mp.Assembler(library, mp.SitePlacer(), mp.AxisOrienter())` — see
[Guide: Assembly](../user-guide/02_assembly.md#backmapping-a-cg-model).

## Finalize when needed

The assembled world has atoms and bonds only. Generate the angle and dihedral
topology once, when an MD writer needs it:

```python
from molpy.builder import Finalization, StructureFinalizer

assert not list(chain.angles)
chain = StructureFinalizer(Finalization.TOPOLOGY).apply(chain)
assert list(chain.angles)
```

`Finalization.ATOMS` removes any angles and dihedrals instead.

A walk of architectures from one ethylene-oxide kit — linear, block, ring,
star, comb, telechelic — is the user-guide section
[Polymer Topologies](../user-guide/topology/index.md) (paired with
`examples/topology/`). Statistical crosslinking of melts is not available.

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

## Polydisperse systems

Sample a chain plan, then build each chain's monomer sequence as a CGsmiles
path with the same assembler:

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
chains = [
    assembler.assemble(
        mp.CGSmilesIR(
            "{" + "".join(f"[#{m}]" for m in chain.monomers) + "}"
        ).to_coarsegrain(),
        mp.Atomistic,
    )
    for chain in plan.chains[:2]
]
assert len(chains) == 2
```

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)
- [Polymer Topologies](../user-guide/topology/index.md)

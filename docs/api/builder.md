# Builder

System construction. `mp.builder` mirrors `molrs.builder` by identity —
site-graph assembly (`Assembler` with its placers and orienter), the
`Coarsener`, and the graphene / nanotube builders — and adds molpy's own: the
polymer planning layer, crystals, virtual sites, `PackingTemplate`, and the
AmberTools polymer builder, which makes a GAFF chain without the assembler:
prepgen cuts one antechamber-typed oligomer and tleap `sequence` joins the
residues. Every name has one path, `mp.builder.<Name>`; the modules behind it
are private.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.builder.Assembler` | `Assembler(library, placer, orienter=None).assemble(sites, cls=None)`: one copy of `library[bead_type]` per site of an `mp.CoarseGrain`, each site bond joining one accepting port of each end; returns the world as `cls` (`mp.Graph` by default) | Every site-graph build: chains, blocks, rings, stars, combs, backmapping |
| `mp.builder.GrowthPlacer` | Grows each molecule breadth-first onto its parent's ports; needs no site positions | Topologies from CGsmiles notation |
| `mp.builder.SitePlacer` | Puts each copy's centre of mass on its site | Sites with positions (a coarsened CG model) |
| `mp.builder.AxisOrienter` | Turns each copy onto its site: backbone-to-centre along the site axis and joining atoms along the bonds for chain units; port directions fitted to bond directions for branch units | Backmapping with `SitePlacer` |
| `mp.io.smiles.CGSmilesIR` | `.templates()` → name → ported `mp.Atomistic`; `.to_coarsegrain()` → site graph | Writing units and topologies |
| `mp.io.smiles.SmilesIR.from_fragment` | `SmilesIR.from_fragment(body).to_template()` → one ported `mp.Atomistic` from a fragment body | Writing one unit |
| `mp.perceive.SubgraphMatcher` / `mp.builder.Coarsener` | Find bead groups in a CG model; turn them into sites with a position and an axis | Site graphs from a CG model |
| `SystemPlanner` / `PolydisperseChainGenerator` / `Chain` / `SystemPlan` | Sample a polydisperse chain plan to a target mass | Bulk / MW-distributed systems |
| `SchulzZimmPolydisperse` / `FlorySchulzPolydisperse` / `PoissonPolydisperse` / `UniformPolydisperse` | Chain-length distributions (`MassDistribution` / `DPDistribution` protocols) | Choosing a molecular-weight distribution |
| `WeightedSequenceGenerator` / `BlockSequenceGenerator` / `AlternatingSequenceGenerator` | Monomer sequences (`SequenceGenerator` protocol) | Copolymer composition |
| `CarbonTubeBuilder` | `CarbonTubeBuilder(n, m, ...)` → `.build()` `Frame` + `.cell()` box | Zigzag, armchair, and chiral nanotubes |
| `GrapheneBuilder` | `GrapheneBuilder(nx, ny, ...)` → `.build()` `Frame` + `.cell()` box | Rectangular graphene honeycomb sheet |
| `Lattice` / `Site` / `SpaceGroup` | Bravais lattice with basis sites (fractional coordinates; the cell is `lattice.box`) and symmetry operators | Crystals |
| `mp.core.Cuboid` / `mp.core.Sphere` / `mp.core.HalfSpace` / `mp.core.Region` | Native geometric regions with `mask(block)`, on the molpy root | Selecting a volume, clipping a crystal |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` | Virtual-site augmentation (Drude parameters from `mp.ff.params.clpol_polarizability`) | Polarizable / 4-site models |
| `AmberPolymerBuilder` | `AmberPolymerBuilder(library, cuts, force_field="gaff").assemble(sites)` → `AmberBuildResult` (`chain`, `forcefield`, prmtop / inpcrd paths): antechamber + parmchk2 on each oligomer, prepgen per residue, tleap `sequence` over a linear site graph; needs AmberTools | GAFF / GAFF2 polymer chains |
| `AmberCut` | One prepgen residue: `omit`, `head` / `tail` connection atoms, `pre_head` / `post_tail` (or `*_type`), `charge` | Cutting an oligomer you built |
| `AmberPieces` | `AmberPieces(head, repeat, tail).oligomer()` → the embedded oligomer and its head / chain / tail cuts, from three SMILES | Writing the oligomer and cuts from SMILES |

The crystal `Site` is a lattice basis site; it is unrelated to the sites of an
assembly site graph.

## Canonical example

A unit is a CGsmiles fragment whose bonding descriptors are its ports (`<`
joins `>`, `$` joins `$`, `!` joins `!`; labels and bond orders must match),
embedded by `mp.conformer.Conformer`. The topology is a CGsmiles string turned into a
site graph. `GrowthPlacer` needs no coordinates.

```python
import molpy as mp

conformer = mp.conformer.Conformer(seed=42)
eo = conformer.generate(
    mp.io.smiles.SmilesIR.from_fragment("[<]OCC[>]").to_template()
)[0]
assert eo.n_ports == 2

sites = mp.io.smiles.CGSmilesIR("{[#EO]|5}").to_coarsegrain()
assembler = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer())
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
`mp.builder.Assembler(library, mp.builder.SitePlacer(), mp.builder.AxisOrienter())` — see
[Guide: Assembly](../user-guide/02_assembly.md#backmapping-a-cg-model).

## Topology when needed

The assembled world has atoms and bonds only. Generate the angle and dihedral
topology once, when an MD writer needs it:

```python
assert not list(chain.angles)
chain.generate_topology(gen_angle=True, gen_dihedral=True)
assert list(chain.angles)
```

A walk of architectures from one ethylene-oxide kit — linear, block, ring,
star, comb, telechelic — is the user-guide section
[Polymer Topologies](../user-guide/topology/index.md) (paired with
`examples/topology/`). Statistical crosslinking of melts is not available.

## Nanostructure topology

The nanostructure builders are molrs's and keep their lattice planning
private. They expose two products: the structure as a `Frame`, and the
simulation cell it was laid out in; `mp.Atomistic.from_frame` makes the frame
a graph.

```python
from molpy.builder import CarbonTubeBuilder

tube_builder = CarbonTubeBuilder(6, 6, cells=2, periodic=True)
tube = mp.Atomistic.from_frame(tube_builder.build())
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
from molpy.builder import (
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
        mp.io.smiles.CGSmilesIR(
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

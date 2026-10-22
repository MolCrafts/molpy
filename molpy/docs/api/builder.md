# Builder

System construction. `mp.builder` mirrors `molrs.builder` by identity —
site-graph assembly (`Assembler` with its placers and orienter), the
`Coarsener`, and the graphene / nanotube builders — and adds molpy's own: the
polymer planning layer, crystals, virtual sites, `PackingTemplate`, and the
AmberTools polymer builder, which makes a GAFF chain without the assembler:
prepgen cuts one antechamber-typed oligomer and tleap `sequence` joins the
residues. Every name has one path, `mp.builder.<Name>`; the modules behind it
are implementation modules, outside `mp.builder.__all__`.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.builder.Assembler` | `Assembler(library, placer, orienter=None).assemble(sites, cls=None)`: one copy of `library[bead_type]` per site of an `mp.CoarseGrain`, each site bond joining one accepting port of each end; returns the world as `cls` (`mp.MolGraph` by default) | Every site-graph build: chains, blocks, rings, stars, combs, backmapping |
| `mp.builder.GrowthPlacer` | Grows each molecule breadth-first onto its parent's ports; needs no site positions | Topologies from CGsmiles notation |
| `mp.builder.SitePlacer` | Puts each copy's centre of mass on its site | Sites with positions (a coarsened CG model) |
| `mp.builder.AxisOrienter` | Turns each copy onto its site: backbone-to-centre along the site axis and joining atoms along the bonds for chain units; port directions fitted to bond directions for branch units | Backmapping with `SitePlacer` |
| `mp.io.cgsmiles.CgSmilesIr` | `.templates()` → name → ported `mp.Atomistic`; `.to_coarsegrain()` → site graph | Writing units and topologies |
| `mp.io.smiles.read_smiles_fragment` | `read_smiles_fragment(body).to_template()` → one ported `mp.Atomistic` from a fragment body | Writing one unit |
| `mp.perceive.SubgraphMatcher` / `mp.builder.Coarsener` | Find bead groups in a CG model; turn them into sites with a position and an axis | Site graphs from a CG model |
| `SystemPlanner` / `PolydisperseChainGenerator` / `Chain` / `SystemPlan` | Sample a polydisperse chain plan to a target mass | Bulk / MW-distributed systems |
| `SchulzZimmPolydisperse` / `FlorySchulzPolydisperse` / `PoissonPolydisperse` / `UniformPolydisperse` | Chain-length distributions (`MassDistribution` / `DpDistribution` protocols) | Choosing a molecular-weight distribution |
| `WeightedSequenceGenerator` / `BlockSequenceGenerator` / `AlternatingSequenceGenerator` | Monomer sequences (`SequenceGenerator` protocol) | Copolymer composition |
| `CarbonTubeBuilder` | `CarbonTubeBuilder(n, m, ...)` → `.build()` `Frame` + `.cell()` box | Zigzag, armchair, and chiral nanotubes |
| `GrapheneBuilder` | `GrapheneBuilder(nx, ny, ...)` → `.build()` `Frame` + `.cell()` box | Rectangular graphene honeycomb sheet |
| `Lattice` / `Site` / `SpaceGroup` | `Lattice(box, basis)`: a `mp.Box` unit cell (lattice vectors are the columns of `box.h`) plus basis sites in fractional coordinates; `Lattice.from_cif(frame)` expands a CIF's asymmetric unit by the operators `read_cif` keeps; `.build(region, repeats=...)` tiles it | Crystals |
| `mp.core.Cuboid` / `mp.core.Sphere` / `mp.core.HalfSpace` / `mp.core.Region` | Native geometric regions with `mask(block)`, on the molpy root | Selecting a volume, clipping a crystal |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` | Virtual-site augmentation (Drude parameters from `mp.ff.params.clpol_polarizability`) | Polarizable / 4-site models |
| `AmberPolymerBuilder` | `AmberPolymerBuilder(library, cuts, parameter_set="gaff2").assemble(sites)` → `AmberBuildResult` (`chain`, `forcefield`, prmtop / inpcrd paths): antechamber (once, `-fo ac`) + parmchk2 on each oligomer, prepgen per residue, tleap `sequence` over a linear site graph; needs AmberTools | GAFF / GAFF2 polymer chains; `forcefield` merges with a natively typed (`GaffTypifier`) small molecule's |
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
    mp.io.smiles.read_smiles_fragment("[<]OCC[>]").to_template()
)[0]
assert eo.n_ports == 2

sites = mp.io.cgsmiles.read_cgsmiles("{[#EO]|5}").to_coarsegrain()
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

## Crystals

A `Lattice` is a `mp.Box` unit cell and a basis of `Site`s in fractional
coordinates. The box's cell matrix `box.h` holds the lattice vectors as
columns, so `lattice.a1` is `lattice.box.h[:, 0]`, and the box does every
fractional ↔ Cartesian conversion. `build` tiles the basis over `repeats`
cells from the origin, or over every cell, negative indices included, that
reaches a region's bounding box, and keeps the atoms inside the region.

```python
from molpy.builder import Lattice

nickel = Lattice.fcc(a=3.52, species="Ni")
assert len(nickel.build(repeats=(2, 2, 2)).atoms) == 32
cluster = nickel.build(mp.core.Sphere([0.0, 0.0, 0.0], 5.0))  # centred on an atom
box = nickel.supercell((2, 2, 2))  # the tile's simulation cell
```

A CIF lists the asymmetric unit and the space group's operators.
`mp.io.read_cif` keeps the operators verbatim in `frame.meta["symops"]`
(`mp.core.keys.SYMOPS`) and the Hermann–Mauguin symbol in
`frame.meta["space_group"]` (`mp.core.keys.SPACE_GROUP`), each only when the
file states it. `Lattice.from_cif` builds the whole conventional cell from
that frame: the cell is `frame.box`, each atom of the `atoms` block is a site
(label from `name`, species from `element`), and every symmetry image is a
basis site labelled `"<name>_<k>"`. Images on a special position collapse
within `symprec`.

```python
# docs: skip — reads an offline CIF; Lattice.from_cif is unit-tested with fixtures
frame = mp.io.read_cif("NaCl.cif")  # Fm-3m: Na1 at 4a, Cl1 at 4b
salt = Lattice.from_cif(frame)
assert len(salt.basis) == 8
structure = salt.build(repeats=(4, 4, 4))
frame = structure.to_frame()
frame.box = salt.supercell((4, 4, 4))
```

`from_cif` refuses a frame it would have to guess about. With no box, or no
`symops` because the file has no operator loop, it raises `ValueError` naming
what is missing rather than assume P1; list `x,y,z` in the file for a P1
structure. A partially occupied site (an `occupancy` below 1) is rejected
too, because a lattice places one atom per site: resolve the disorder in the
file first.

Without a CIF, give the cell, the asymmetric unit and a `SpaceGroup`, from an
explicit operator list or closed from generators:

```python
from molpy.builder import Site, SpaceGroup

fm3m = SpaceGroup.from_generators(
    ["x+1/2,y+1/2,z", "x+1/2,y,z+1/2", "-x,-y,z", "-x,y,-z", "z,x,y", "y,x,-z", "-x,-y,-z"]
)
assert fm3m.order == 192
salt = Lattice.from_spacegroup(
    mp.Box(mp.core.Cell.lengths([5.64] * 3)),
    [Site("Na1", "Na", (0.0, 0.0, 0.0)), Site("Cl1", "Cl", (0.5, 0.5, 0.5))],
    fm3m,
)
assert len(salt.basis) == 8
```

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
        mp.io.cgsmiles.read_cgsmiles(
            "{" + "".join(f"[#{m}]" for m in chain.monomers) + "}"
        ).to_coarsegrain(),
        mp.Atomistic,
    )
    for chain in plan.chains[:2]
]
assert len(chains) == 2
```

`plan_system` returns a plan whose total mass is within `max_rel_error` of the
target, or raises `RuntimeError` saying how far short it fell and why (trimming
off, no shorter chain fits, or `max_chains` reached). A mass distribution's
sampled mass becomes one whole `generate_sequence(dp)` call, so alternating and
block generators keep their pattern.

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)
- [Polymer Topologies](../user-guide/topology/index.md)

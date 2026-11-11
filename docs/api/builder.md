# Builder

System construction. `mp.builder` holds the native builders —
site-graph assembly (`Assembler` with its placers and orienter), the
`Coarsener`, the graphene / nanotube builders, crystals (`Lattice`, `Site`,
`SpaceGroup`, `SymOp`) and the virtual-site builders — polymer ensembles (`Fragment`, `StochasticObject`, `Molecule`, `System`: the
generative-BigSMILES model) — and adds molpy's own: `PackingTemplate` and the
AmberTools polymer builder, which makes a GAFF chain without the assembler:
prepgen cuts one antechamber-typed oligomer and tleap `sequence` joins the
residues. Every name has one path, `mp.builder.<Name>`; the modules behind it
are implementation modules, outside `mp.builder.__all__`.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.builder.Assembler` | `Assembler(library, placer, orienter=None).assemble(sites)`: one copy of `library[bead_type]` per site of an `mp.CoarseGrain`, each site bond joining one accepting port of each end; returns the world as the library's graph type (`mp.MolGraph`, `mp.Atomistic` or `mp.CoarseGrain`) | Every site-graph build: chains, blocks, rings, stars, combs, backmapping |
| `mp.builder.GrowthPlacer` | Grows each molecule breadth-first onto its parent's ports; needs no site positions | Topologies from CGsmiles notation |
| `mp.builder.SitePlacer` | Puts each copy's centroid on its site | Sites with positions (a coarsened CG model) |
| `mp.builder.AxisOrienter` | Turns each copy onto its site: backbone-to-centre along the site axis and joining atoms along the bonds for chain units; port directions fitted to bond directions for branch units | Backmapping with `SitePlacer` |
| `mp.io.cgsmiles.CgSmilesIr` | `.templates()` → name → ported `mp.Atomistic`; `.to_assembly_graph()` → exact resolved site graph; `.to_coarsegrain()` → simplified projection | Writing units and topologies |
| `mp.io.smiles.read_smiles_fragment` | `read_smiles_fragment(body).to_template()` → one ported `mp.Atomistic` from a fragment body | Writing one unit |
| `mp.perceive.SubgraphMatcher` / `mp.builder.Coarsener` | Find bead groups in a CG model; turn them into sites with a position and an axis | Site graphs from a CG model |
| `mp.builder.Fragment` | `Fragment(name, template, weights=None)`: one ported template and a generation weight per port (a float, its reactivity; a list, its pairing weight with each port of the object) | Repeat units, end groups, caps |
| `mp.builder.StochasticObject` / `ChainSize` / `Distribution` | Repeat units and end groups grown to a size drawn from `ChainSize(Distribution.SchulzZimm(mw=…, mn=…), "mass")` (`FlorySchulz`, `Gauss`, `Uniform`, `LogNormal`, `Poisson`; measures `"repeat_units"`, `"mass"`, `"heavy_atom_mass"`) | Chain-size distributions, copolymer composition |
| `mp.builder.Molecule` / `System` / `Amount` | Fragments and objects joined end to end; molecule kinds sampled to `Amount.Total(value=…)` / `Amount.Percent(percent=…)` of a total. `sample(seed)` → site graph (`System`: `SystemSample` with `.sites` and a `MoleculeRecord` per chain); `library()` → the assembler's templates | Bulk / MW-distributed systems |
| `mp.io.bigsmiles.read_bigsmiles` | `read_bigsmiles(text).to_system()`: the same model from one generative-BigSMILES string (sizes in heavy-atom mass) | Writing an ensemble as notation |
| `CarbonTubeBuilder` | `CarbonTubeBuilder(n, m, ...)` → `.build()` `Frame` + `.cell()` box | Zigzag, armchair, and chiral nanotubes |
| `GrapheneBuilder` | `GrapheneBuilder(nx, ny, ...)` → `.build()` `Frame` + `.cell()` box | Rectangular graphene honeycomb sheet |
| `Lattice` / `Site` / `SpaceGroup` | `Lattice(box, basis)`: a `mp.Box` unit cell (lattice vectors are the columns of `box.h`) plus basis sites in fractional coordinates; `Lattice.from_cif(frame)` expands a CIF's asymmetric unit by the operators `read_cif` keeps; `.build(region, repeats=...)` tiles it | Crystals |
| `mp.core.Cuboid` / `mp.core.Sphere` / `mp.core.HalfSpace` / `mp.core.Region` | Native geometric regions in `mp.core`, with `mask(block)` | Selecting a volume, clipping a crystal |
| `DrudeBuilder` / `Tip4pBuilder` | Virtual-site augmentation (Drude parameters: `DrudeBuilder().polarizability`) | Polarizable / 4-site models |
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
chain = assembler.assemble(sites)

atoms = chain.to_frame()["atoms"]
assert sorted(set(atoms["frag_id"].tolist())) == [0, 1, 2, 3, 4]  # one per site
assert set(atoms["mol_id"].tolist()) == {1}  # one connected component
assert chain.n_ports == 2  # the chain ends keep their ports and hydrogens
```

Each bond joins one accepting port of each end. Supply `src_port` and
`dst_port` to `sites.add_bond(a, b, ...)` to fix the template port ordinals;
without them the assembler chooses from the remaining compatible ports.
The leaving handles are removed and their partial charge folds onto the
anchors. Formal charges must already describe the product fragments. A build
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

The nanostructure builders keep their lattice planning
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
cif = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
x,y+1/2,z+1/2
x+1/2,y,z+1/2
x+1/2,y+1/2,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""  # rock salt: the face-centring translations give the 8-site cell
frame = mp.io.read_cif(mp.io.Text(cif))
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

Grow chains of random size from fragments, sample them to a total, and build
each sampled chain with one assembler:

```python
eo = mp.builder.Fragment("EO", mp.io.smiles.read_smiles_fragment("[<]OCC[>]").to_template())
cap = mp.builder.Fragment("Me", mp.io.smiles.read_smiles_fragment("[<]C").to_template())
start = mp.builder.Fragment("MeO", mp.io.smiles.read_smiles_fragment("C[>]").to_template())
peo = mp.builder.Molecule([
    mp.builder.StochasticObject(
        [eo],
        [start, cap],
        size=mp.builder.ChainSize(
            mp.builder.Distribution.SchulzZimm(mw=3000.0, mn=1500.0), "mass"
        ),
    )
])
system = mp.builder.System(
    [(peo, mp.builder.Amount.Total(value=5e3))], measure="mass"
)
sample = system.sample(42)
assert sum(m.mass for m in sample.molecules) >= 5e3

conformer = mp.conformer.Conformer(seed=42)
library = {name: conformer.generate(t)[0] for name, t in peo.library().items()}
chains = [
    mp.builder.Assembler(library, mp.builder.GrowthPlacer()).assemble(peo.sample(seed))
    for seed in range(2)
]
assert all(chain.n_ports == 0 for chain in chains)
```

Growth starts from an end group, adds repeat units by port weight until the
chain reaches a size drawn from its `ChainSize` (at least one repeat unit; a
size counted in repeat units is a whole number), and caps the last open port
with the end group that accepts it. A system samples chains until the total is
reached — it stops at the first chain reaching it, trims nothing and bounds
nothing, so the amounts are sampled as written. A model that cannot be sampled
is refused when built, as a `mp.builder.PolymerError` whose `kind` names the
reason (`"InvalidMolecule"`, `"InvalidObject"`, …; `"Stuck"` for a sample that
could not be grown). An alternating copolymer is a list weight that pairs each unit only
with the other; a block copolymer is two objects in one `Molecule`. The same
system is `mp.io.bigsmiles.read_bigsmiles(text).to_system()` for a
generative-BigSMILES `text`.

## Related

- [Guide: Assembly](../user-guide/02_assembly.md)
- [Polymer Topologies](../user-guide/topology/index.md)

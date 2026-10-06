# Assembly

Assembly builds a molecule, or a whole system, by joining copies of small units along a
**site graph**: one site per unit, one bond per join. The same call grows a chain from a
CGsmiles string and backmaps a coarse-grained (CG) model onto all-atom detail; only where
the sites come from, and how each copy is posed, differ.

Three things go in:

1. **Units** — molecules that carry **ports**, the places where they may bond.
2. **A topology** — an `mp.CoarseGrain` whose beads name units and whose bonds say which units join.
3. **A placer** (and optionally an **orienter**) — the rule that gives each copy its pose.

`mp.builder.Assembler` puts them together. The per-architecture walk-through (linear, block, ring,
star, comb, telechelic) is the [Polymer Topologies](topology/index.md) section; this page
explains the pieces.

## Units carry ports

A port is a pair *(anchor, handle)*: the anchor is the atom that forms the new bond, the
handle is a real atom bonded to it that leaves when the bond forms — usually the hydrogen
that caps the open valence. The simplest way to write a unit is a CGsmiles fragment whose
bonding descriptors are its ports; `mp.Conformer` then adds hydrogens and 3D coordinates,
and the result is an ordinary `mp.Atomistic` that still carries its ports.

```python
import molpy as mp

conformer = mp.Conformer(seed=42)


def unit(body: str) -> mp.Atomistic:
    """One CGsmiles fragment body as a 3D molecule with hydrogens and ports."""
    return conformer.generate(mp.io.SmilesIR.from_fragment(body).to_template())[0]


eo = unit("[<]OCC[>]")  # -O-CH2-CH2-
print(eo.n_atoms, eo.n_ports)  # 9 2
for port in eo.ports:
    print(port.anchor["element"], port.handle_atom["element"], port["port_kind"])
# O H <
# C H >
```

`SmilesIR.from_fragment(body).to_template()` returns the fragment body as a ported
`mp.Atomistic`; `CGSmilesIR(s).templates()` returns a dict from fragment name to one
such template for every fragment a CGsmiles string defines. Whether two ports may join is decided by their kind, label and order:

| Port | Joins |
|------|-------|
| `<` | a `>` |
| `>` | a `<` |
| `$` | a `$` |
| `!` | a `!` |

A label restricts a port further: `[>g]` joins only `[<g]`, which is how a comb keeps its
grafts off the backbone. The bond orders must agree as well. When a bond forms, both
handles are removed and their partial charges fold onto the anchors, so the net charge of
the product is the sum of the units' charges. A molecule you already have gets ports with
`def_port(anchor, handle_atom, kind, label="", order=1)`.

## Topologies are site graphs

A topology is an `mp.CoarseGrain`. Each bead's `bead_type` names a unit of the library, and
each bond between two beads is one join. You rarely build it by hand; it comes either from
CGsmiles notation or from coarsening an existing CG model.

From notation, the site graph has no coordinates:

- linear: `{[#EO]|10}`
- block: `{[#EO]|6[#PO]|4}`
- ring: `{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}`
- star: `{[#X3]([#EO][#EO])([#EO][#EO])[#EO][#EO]}`
- capped: `{[#CAPA][#EO]|6[#CAPB]}`

`|n` repeats a unit along a path; a unit that carries a branch must be written out. Several
molecules go in one site graph by merging: `sites.merge(other_sites)`.

From a CG model, the sites come from groups of beads (see
[Backmapping a CG model](#backmapping-a-cg-model) below). Each site then has a position
(the group's centre of mass) and an axis.

## Building from a topology

With a topology from notation there are no coordinates to honour, so `mp.builder.GrowthPlacer`
grows each molecule breadth-first: the first copy keeps its conformer pose, and every later
copy is turned and moved so that the anchor of its port lands on its parent's leaving
handle, pointing back along that bond.

```python
sites = mp.io.CGSmilesIR("{[#EO]|10}").to_coarsegrain()
chain = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
print(chain.n_atoms, chain.n_ports)  # 72 2
```

Ten units of seven atoms, plus the two hydrogens left on the chain-end ports. Bond lengths
between units, ring closures and overlaps are not adjusted; they are left to relaxation.

## Backmapping a CG model

When the sites come from a CG simulation, the copies must sit where the beads were. Match
the bead pattern of one repeat unit with `mp.SubgraphMatcher`, turn each match into a site
with `mp.builder.Coarsener`, and assemble with `mp.builder.SitePlacer` (each copy's centre of mass on its
site) and `mp.builder.AxisOrienter` (each copy turned to its site's axis and bonds).

The toy model below is a five-unit PMMA chain with two beads per repeat unit: a backbone
bead of type `"1"` and an ester side bead of type `"2"`.

```python
cg = mp.CoarseGrain()
previous = None
for i in range(5):
    side = (-1) ** i
    backbone = cg.def_bead(bead_type="1", mass=41.0, x=2.5 * i, y=0.4 * side, z=0.0)
    ester = cg.def_bead(bead_type="2", mass=59.0, x=2.5 * i, y=3.0 * side, z=0.0)
    cg.def_cgbond(backbone, ester)
    if previous is not None:
        cg.def_cgbond(previous, backbone)
    previous = backbone

groups = mp.SubgraphMatcher(mp.io.CGSmilesIR("{[#1][#2]}").to_coarsegrain()).find(cg)
sites = mp.builder.Coarsener(cg).coarsen(groups, ["MMA"] * len(groups))

mma = unit("[<]CC([>])(C)C(=O)OC")
pmma = mp.builder.Assembler({"MMA": mma}, mp.builder.SitePlacer(), mp.builder.AxisOrienter()).assemble(
    sites, mp.Atomistic
)
print(pmma.n_atoms, pmma.n_ports)  # 77 2
```

Each site sits at its group's mass-weighted centre, and its axis runs from the group's first
bead to that centre — here from the backbone bead toward the ester. For a chain unit (a
two-port template with `<` / `>` ports) `AxisOrienter` aligns the template's
backbone-to-centre direction with that axis and its two joining atoms with the site's bond
line; for a branch unit it fits the port directions to the bond directions. Coarsening uses
positions as stored, with no periodic imaging, so unwrap a periodic CG frame first.

A realistic model has several species: collect the groups of every pattern (one repeat-unit
pattern, one per solvent or ion) into one `coarsen` call with one name per group. A template
without ports is fine for a site with no bonds, such as a solvent molecule or an ion.

## How ports are assigned

You never pick ports. For every bond of the site graph the assembler chooses one port on
each end that accepts the other, walking each molecule breadth-first and giving every bond of
a site a distinct port. Labels are the tool for steering the choice: a port labelled `g`
can only meet another `g`. If no assignment exists the build is refused before anything is
returned, and the error names the site:

```text
site 0 ('X2') has no port for every bond: 3 bonds but the template has 2 ports
```

Ports that no bond uses stay on the result, handle atoms included. A linear chain therefore
keeps a hydrogen and an open port at each end; cap them with end-group units (the
[telechelic](topology/06_telechelic.md) page) when the ends should be something else.

## Ids on the result

Every atom of the world gets two integer ids:

- `frag_id` — the ordinal of the site it came from (0, 1, 2, … in site order);
- `mol_id` — its connected component, counted from 1.

```python
sites = mp.io.CGSmilesIR("{[#EO]|3}").to_coarsegrain()
sites.merge(mp.io.CGSmilesIR("{[#EO]|4}").to_coarsegrain())
two = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
atoms = two.to_frame()["atoms"]
print(sorted(set(atoms["frag_id"].tolist())))  # [0, 1, 2, 3, 4, 5, 6]
print(sorted(set(atoms["mol_id"].tolist())))  # [1, 2]
```

## Choosing the output class

`assemble(sites, cls)` builds the world as the class you name: `mp.Atomistic` for atomistic
units, `mp.CoarseGrain` when the units are themselves CG templates (every node carries a
`bead_type`), and `mp.Graph` when `cls` is omitted. Pass the class you intend to use next; the
typifiers, writers and minimisers take an `mp.Atomistic`.

## Polydisperse systems

`SystemPlanner` and its distributions (see [Polydisperse Systems](05_polydisperse_systems.md))
still plan the chains. Each planned chain's monomer sequence becomes a CGsmiles path,
`"{" + "".join(f"[#{m}]" for m in chain.monomers) + "}"`, which one reusable assembler builds.

## Relax before use

Neither placer asks a force field anything. Growth leaves inter-unit bonds at whatever length
the parent's leaving bond had, and ring closures at whatever distance the open path grew;
backmapping leaves each copy rigid in its conformer shape. Type the world and minimise it
before a simulation — see [Geometry Optimization](08_geometry_optimization.md) — and expect
a short equilibration to finish the job for a dense backmapped system.

## Not available yet: crosslinking and gels

Statistical crosslinking of a melt — gels, end-linked networks, dual networks, curing with
an agent — needs sites to be joined by proximity, and that is not yet a site-graph primitive.
MolPy does not currently build such networks. Every architecture whose connectivity you can
write down as a site graph (chains, blocks, rings, stars, combs, capped chains, backmapped
CG models) is covered.

## See also

- [Polymer Topologies](topology/index.md) — one page and one runnable script per architecture
- [3D Conformer Generation](07_conformers.md) — how the units get their coordinates
- [Geometry Optimization](08_geometry_optimization.md) — relaxing the assembled world
- [Packing Systems](09_packing.md) — filling a cell with assembled molecules
- [Builder API](../api/builder.md) — the assembly symbols in one table

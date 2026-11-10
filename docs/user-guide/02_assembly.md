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
bonding descriptors are its ports; `mp.conformer.Conformer` then adds hydrogens and 3D coordinates,
and the result is an ordinary `mp.Atomistic` that still carries its ports.

```python
import molpy as mp

conformer = mp.conformer.Conformer(seed=42)


def unit(body: str) -> mp.Atomistic:
    """One CGsmiles fragment body as a 3D molecule with hydrogens and ports."""
    return conformer.generate(mp.io.smiles.read_smiles_fragment(body).to_template())[0]


eo = unit("[<]OCC[>]")  # -O-CH2-CH2-
print(eo.n_atoms, eo.n_ports)  # 9 2
for port in eo.ports:
    anchor = mp.Element.by_number(port.anchor["atomic_number"]).symbol
    handle = mp.Element.by_number(port.handle_atom["atomic_number"]).symbol
    print(anchor, handle, port["port_kind"])
# O H <
# C H >
```

`read_smiles_fragment(body).to_template()` returns the fragment body as a ported
`mp.Atomistic`; `read_cgsmiles(s).templates()` returns a dict from fragment name to one
such template for every fragment a CGsmiles string defines. A fragment name is ASCII
letters and digits only (or the wildcard `*`): write `psingle` or `PBIXBr`, not `p_single`
or `pbix-br`; the parser names the character it refused. Whether two ports may join is decided by their kind, label and order:

| Port | Joins |
|------|-------|
| `<` | a `>` |
| `>` | a `<` |
| `$` | a `$` |
| `!` | a `!` |

A label restricts a port further: `[>g]` joins only `[<g]`, which is how a comb keeps its
grafts off the backbone. The bond orders must agree as well. When a bond forms, both
handles are removed and their partial charges fold onto the anchors. This does not
infer formal charge changes: a quaternary ammonium product must already carry its
positive formal charge in the fragment definition. A molecule you already have gets ports with
`add_port(anchor, handle_atom, kind, label="", order=1)`.

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
copy is turned and moved along its parent's port direction. The initial bond length is
estimated from the joining atoms' covalent radii. Parallel connections to the same parent
are fitted together; geometry does not change which ports join.

```python
sites = mp.io.cgsmiles.read_cgsmiles("{[#EO]|10}").to_coarsegrain()
chain = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer()).assemble(sites)
print(chain.n_atoms, chain.n_ports)  # 72 2
```

Ten units of seven atoms, plus the two hydrogens left on the chain-end ports. Bond lengths
between units, ring closures and overlaps are not adjusted; they are left to relaxation.

## Backmapping a CG model

When the sites come from a CG simulation, the copies must sit where the beads were. Match
the bead pattern of one repeat unit with `mp.perceive.SubgraphMatcher`, turn each match into a site
with `mp.builder.Coarsener`, and assemble with `mp.builder.SitePlacer` (each copy's centroid on its
site) and `mp.builder.AxisOrienter` (each copy turned to its site's axis and bonds).

The toy model below is a five-unit PMMA chain with two beads per repeat unit: a backbone
bead of type `"1"` and an ester side bead of type `"2"`.

```python
cg = mp.CoarseGrain()
previous = None
for i in range(5):
    side = (-1) ** i
    backbone = cg.add_bead(bead_type="1", mass=41.0, x=2.5 * i, y=0.4 * side, z=0.0)
    ester = cg.add_bead(bead_type="2", mass=59.0, x=2.5 * i, y=3.0 * side, z=0.0)
    cg.add_bond(backbone, ester)
    if previous is not None:
        cg.add_bond(previous, backbone)
    previous = backbone

groups = mp.perceive.SubgraphMatcher(mp.io.cgsmiles.read_cgsmiles("{[#1][#2]}").to_coarsegrain()).find(cg)
sites = mp.builder.Coarsener(cg).coarsen(groups, ["MMA"] * len(groups))

mma = unit("[<]CC([>])(C)C(=O)OC")
assembler = mp.builder.Assembler({"MMA": mma}, mp.builder.SitePlacer(), mp.builder.AxisOrienter())
pmma = assembler.assemble(sites)
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

## Deterministic connections

A descriptor label names a compatibility class; its position in the template's port table
identifies an actual site. A fragment can have two `[<x]` ports at different atoms. A
connection must specify their ordinals when that distinction matters. Port ordinals are
zero-based in `template.ports` order, which follows descriptor order for CGsmiles templates.
Conformer generation and copying preserve the correspondence. Editing the template's port
structure after writing a connection plan can change its meaning, so complete that editing
first.

The two optional fields on `add_bond` belong to the templates named by the two beads.
They are supplied together; the assembler refuses a connection carrying one without the
other. This example fixes the carbon-to-oxygen connection of two EO units, independent of
any later connections added to the graph:

```python
exact = mp.CoarseGrain()
left = exact.add_bead(bead_type="EO")
right = exact.add_bead(bead_type="EO")
exact.add_bond(left, right, src_port=1, dst_port=0)
dimer = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer()).assemble(exact)
assert dimer.n_ports == 2
```

The assembler reserves all explicitly chosen ports before assigning any unspecified
connections. A nonexistent, incompatible or reused port raises an error rather than being
replaced by another port. Reversing a connection's bead endpoints also requires swapping
its port fields. Multiple relations between the same beads represent separate atomistic
bonds, such as the two bonds between successive units of a ladder polymer. Two distinct
ports on one bead can also close a ring inside that instance.

For ordinary graphs without these fields, the assembler chooses compatible free ports
while walking each component. Labels steer that choice, but do not select one of two ports
with the same label. Unused single-bond ports stay on the result with their handle atoms.
A chain can therefore retain its end hydrogens. An unused port declaring a multiple bond
is refused: one capping hydrogen cannot represent that unsatisfied bond order. Add an
explicit end-group fragment to complete that chemistry before assembling.

## Resolved CGsmiles connections

A complete CGsmiles string already resolves the ports. Repeating that selection during
assembly could connect different atoms when a template has equivalent labels at distinct
positions. Use `to_assembly_graph()` to transfer the resolved choices, together with their
bond classes and all connections expanded from higher levels:

```python
ir = mp.io.cgsmiles.read_cgsmiles("{[#A]=[#A]}.{#A=[$]CCC[$]}")
resolved_library = {
    name: conformer.generate(template)[0]
    for name, template in ir.templates().items()
}
resolved_sites = ir.to_assembly_graph()
assert resolved_sites.n_bonds == 2  # Coarse '=' means two connections.
ring = mp.builder.Assembler(resolved_library, mp.builder.GrowthPlacer()).assemble(resolved_sites)
assert ring.n_ports == 0
```

The graph uses the deepest coarse layer, whose beads directly name the atomistic templates.
Every resolved pair becomes one relation, including parallel pairs and connections derived
from an upper layer. Aromatic and explicit numeric bond classes are retained. A base-only
string has no fragment definitions to resolve and cannot use this conversion.
`to_coarsegrain()` remains useful for base-only topologies and graph matching, but projects
only the coarsest layer and drops pairings and multiplicity.

Final product fragments also express charged connections. Here the nitrogen has its
positive formal charge before the new carbon–nitrogen bond forms:

```python
charged_ir = mp.io.cgsmiles.read_cgsmiles(
    "{[#M][#Q]}.{#M=C[>x],#Q=[<x][N+](C)(C)C}"
)
charged_library = {
    name: conformer.generate(template)[0]
    for name, template in charged_ir.templates().items()
}
ammonium = mp.builder.Assembler(charged_library, mp.builder.GrowthPlacer()).assemble(
    charged_ir.to_assembly_graph()
)
assert sum(int(atom.get("formal_charge", 0)) for atom in ammonium.atoms) == 1
```

This is assembly of a declared product, not prediction of a quaternization reaction.
Partial charge assignment, force-field typing and structural optimization are independent
operations performed afterwards.

## Ids on the result

Every atom of the world gets two integer ids:

- `frag_id` — the ordinal of the site it came from (0, 1, 2, … in site order);
- `mol_id` — its connected component, counted from 1.

```python
sites = mp.io.cgsmiles.read_cgsmiles("{[#EO]|3}").to_coarsegrain()
sites.merge(mp.io.cgsmiles.read_cgsmiles("{[#EO]|4}").to_coarsegrain())
two = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer()).assemble(sites)
atoms = two.to_frame()["atoms"]
print(sorted(set(atoms["frag_id"].tolist())))  # [0, 1, 2, 3, 4, 5, 6]
print(sorted(set(atoms["mol_id"].tolist())))  # [1, 2]
```

## Choosing the output class

`assemble(sites)` builds the world as the graph type of the library's templates: an
`mp.Atomistic` library gives an `mp.Atomistic` world, `mp.CoarseGrain` templates (the units are
themselves CG models) give an `mp.CoarseGrain`, and `mp.MolGraph` templates an `mp.MolGraph`.
Every template of one library must be of the same type. Build the units as the class you intend
to use next; the typifiers, writers and minimisers take an `mp.Atomistic`.

## Polydisperse systems

A polydisperse ensemble is the polymer model of `mp.builder` (see
[Polydisperse Systems](05_polydisperse_systems.md)): fragments grown into chains of random
size by a `StochasticObject`, sampled to a total by a `System`, or read from one
generative-BigSMILES string with `mp.io.bigsmiles.read_bigsmiles(text).to_system()`. A sample
is already a site graph — one site per placed fragment, `bead_type` its name — so one
reusable assembler over `library()` builds every chain.

## Relax before use

Neither placer asks a force field anything. Growth estimates inter-unit bond lengths from covalent radii, while
ring closures retain whatever distance the open path grew;
backmapping leaves each copy rigid in its conformer shape. Type the world and minimise it
before a simulation — see [Geometry Optimization](08_geometry_optimization.md) — and expect
a short equilibration to finish the job for a dense backmapped system.

## A finite crosslinked network

A crosslinked network is the same site graph with more connections: chains of ladder
units joined by two bonds per backbone step, bridge fragments whose two ports consume one
side site on each of two chains, and single-ended fragments on the sites left over. The
example `molpy/examples/pbix_deterministic.py` builds such a network for a PBIX
anion-exchange membrane, adds hydroxide counterions on a shell around the polymer, checks
every new bond against the fragment pair and template anchors that requested it, and
writes the structures with the connection table, fragment definitions and atom
provenance. What comes out is an initial topology and geometry for optimization, not a
sampled microscopic network.

Deterministic crosslinks require an explicit connection table. Proximity-based curing,
random network sampling and reaction kinetics remain separate from this builder.

## See also

- [Polymer Topologies](topology/index.md) — one page and one runnable script per architecture
- [3D Conformer Generation](07_conformers.md) — how the units get their coordinates
- [Geometry Optimization](08_geometry_optimization.md) — relaxing the assembled world
- [Packing Systems](09_packing.md) — filling a cell with assembled molecules
- [Builder API](../api/builder.md) — the assembly symbols in one table

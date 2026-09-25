# Assembly

Growing a polymer from a monomer, crosslinking a relaxed melt into a network, and closing
a chain into a macrocycle are three different jobs. You reach for them at different points
in a study, and they answer different questions. What they have in common is not the job —
it is what MolPy needs you to say in order to do it.

## Different jobs, the same three inputs

Whatever you are building, MolPy asks for the same three things.

**Which atoms are allowed to react.** You mark them on the molecule, by name.

**What the reaction does.** You write it once, as a reaction SMARTS.

**Which marked sites actually pair up.** A chain pairs each monomer with the next. A
crosslinker pairs sites that happen to be close in space. A macrocycle pairs the two ends
of something already connected.

Only the third input differs between the three jobs, and MolPy makes that literal: one
assembler does the work, and you hand it a **selector** that answers only that question.
`TopologySelector` pairs by chain adjacency. `RandomSelector` pairs within a distance cutoff.
You can write one that pairs whatever you like. Everything downstream of that choice — editing
the graph once, retyping what the edits disturbed, and optionally finalizing topology — is one
code path you never touch.

## React first; retype what the reaction disturbed; finalize when needed

**An assembler matches the reaction once, lets the selector choose which pairs react, executes
every chosen reaction as one batch on the intact world, and only then retypes the neighbourhood
of each edit.**

Retyping is needed because of how force fields work. A *force field* is the set of energy
functions and parameters a molecular-dynamics (MD) engine uses, and it looks those parameters
up by *atom type* — a label such as "ether oxygen" or "aliphatic carbon bonded to an oxygen"
that summarises an atom's chemical environment. A *typifier* is the object that assigns those
types (and, for some force fields, per-atom parameters such as charges) from each atom's
element, bonds and rings. Forming a bond changes the environment of the atoms around it, so
their types may change too.

The order matters. After the batch of reactions has run, MolPy cuts a small subgraph — an
*affected region* — around each edit and hands only that region to the typifier. A region cut
*before* the edit would have to model the edit itself, including every neighbouring reaction
close enough to change a type and the leaving atoms each one deletes; a region cut afterwards
inherits a graph on which those edits have already happened. The thousandth bond is therefore
never followed by a retype of the thousand-monomer graph.

The region's size is set by one number you declare with the typifier, `reach`: the radius, in
bonds, of the neighbourhood that decides one atom's type. Distances are counted from the atoms
the reaction *touched* — the two ends of the new bond, any atom it created, and any atom that
lost a neighbour the reaction deleted. Two radii follow from `reach`:

- the **write-back set** is every atom within $\max(\text{reach}, 2)$ bonds of a touched atom.
  The floor of 2 is there because a dihedral (a four-atom torsion term) containing the new bond
  reaches two bonds out on either side, and all four of its atoms need types before the term
  can be looked up;
- the **extracted region** reaches $\max(\text{reach}, 2) + \text{reach}$ bonds, so that the
  outermost written-back atom still sees its whole `reach`-neighbourhood.

Atoms beyond the write-back set are context only and are never written back: their own
neighbourhood is cut off at the region's edge, and a truncated environment does not fail to
match a typing rule, it matches the *wrong* one. A small ring (at most eight atoms) that the
radius would cut arrives whole, because to a typifier a cut ring is a different molecule.

What comes back is scalar per-atom data only — the atom type and any per-atom parameters the
typifier set. Bond, angle, dihedral and improper annotations are never copied out of a region.

Topology is a separate tail:

- `Finalization.ATOMS` leaves bonds and per-atom force-field data only.
- `Finalization.TOPOLOGY` (the default) generates complete angles and dihedrals once.
- `Finalization.BONDED` additionally assigns their force-field types through
  `ForceFieldParams`.

This lets a very large system defer topology until its MD writer actually needs explicit bonded
rows, without creating a second aggregation algorithm.

## What an assembler is not

It is **not a reaction engine**. It never parses SMARTS, never matches a pattern, never
rewrites a bond. A `Reaction` does all of that. An assembler decides *which* sites react;
the reaction decides *what happens* when they do.

It is **not a port system**. There is no `<` and `>`, no head and tail, no connector object
deciding that a hydroxyl may meet a carboxyl. The reaction SMARTS is the only place chemistry
is written down.

It is **not a typifier**. Any typifier it accepts is a `Typifier`; the assembler only decides
which bounded regions that typifier sees.

## A repeat unit is a molecule with a few marked atoms

There is no `RepeatUnit` class, no `Junction`, no `Port`. A repeat unit is an ordinary
molecule with a few of its atoms named. You mark the ones that may react and leave every
other atom alone.

An ethylene-oxide repeat unit is a real, capped molecule: ethylene glycol. Its two hydroxyl
oxygens are the sites; the hydroxyl hydrogens are the caps the reaction will remove.

```python
import molpy as mp
from molpy.core import fields

eo = mp.io.read_smiles("OCCO")  # ethylene glycol
eo.atoms[0][fields.SITE] = "a"  # one hydroxyl oxygen
eo.atoms[3][fields.SITE] = "b"  # the other
```

Nothing here says "head" or "tail": `a` and `b` are labels the reaction can refer to. A linear
chain is a world whose bonds happen to form a path. A four-arm crosslinker is a molecule with
four marked atoms. A macrocycle is one more bond between two atoms already connected. The same
`eo` above serves all three, unchanged.

## Charge is frozen on the template, so conservation is free

AM1-BCC charges (a standard recipe for partial atomic charges, computed by a quantum-chemistry
program such as AmberTools' `sqm`) come from a self-consistent solve over an entire molecule.
There is no such thing as the charge of a fragment, so there is no honest way to compute the
charge of a new junction by looking at the atoms around it. This is not a limitation of MolPy;
it is what "non-local" means.

The way out is to never compute charge during assembly. Solve it once on the capped repeat
unit, where the molecule is small and closed-shell and `sqm` is meaningful. Then fold each
cap's charge onto the site atom it capped. After the fold, every cap carries exactly zero.

Watch what the reaction then does: it deletes atoms whose charge is zero. **Net charge is
conserved because nothing charged was removed** — not because a correction term redistributed
the loss afterwards. Conservation stops being a heuristic and becomes an accounting identity.

MolPy will not paper over a template you forgot to freeze. If the reaction is about to delete
a charged atom, `apply` raises and says so:

```python
import pytest
from molpy.builder.assembly import (
    MonomerLibrary,
    PolymerBuilder,
    TracePlacer,
    SiteMap,
    linear_topology,
)
from molpy.conformer import Conformer

# A template with hydrogens and charges, but never frozen:
eo, _ = Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.read_smiles("OCCO")
)
SiteMap(eo).label_elements("O", "a", "b")
for atom in eo.atoms:
    atom[fields.CHARGE] = -0.3 if atom.get("element") == "H" else 0.2

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether, placer=TracePlacer())

# Unfrozen templates lose the charge carried by the atoms the reaction deletes:
with pytest.raises(ValueError, match="net charge"):
    builder.build(linear_topology(["EO"] * 3))
# ValueError: assembly changed the net charge by +0.8 e: the reaction deleted atoms
# that carry charge. Freeze the monomer templates first so each cap's charge folds
# onto its site atom.
```

When the leaving group is a single hydrogen, freezing is one call on `SiteMap`.
`SiteMap(mol).prepare_leaving_hydrogens("x")` finds, for every atom labelled `x`, its
lowest-handle hydrogen neighbour, labels that hydrogen as the leaving group (`"h"` by
default), and — because `fold_charge=True` is the default — moves the hydrogen's charge onto
the site atom. The hydrogen keeps a record of its original charge in `fields.Q0`, so an
unreacted site can be restored later. `SiteMap.every_nth(...)` takes the same `leaving=` and
`fold_charge=` arguments when it marks sites along a chain; the C–C crosslink in
[Building a Crosslinked Gel](16_crosslinked_gel.md) is prepared this way. A leaving group of
more than one atom — the ether condensation above removes a whole hydroxyl from its `b` side —
is folded the same way by hand: add the group's total charge to the atom it was bonded to and
set the group's own charges to zero, once, on the template.

This is also why AMBER prep files freeze per-residue charges. Same physics, arrived at
independently.

## You may guess a number, never an identity

When a placer lays out the pasted residues (see *Giving the chain a shape* below), it needs a
length for each bond about to form, before any force field has been consulted about that bond.
So it guesses: the sum of the two atoms' covalent radii, plus a small buffer.

The guess is legitimate, for a narrow reason. Bond length is a **continuous** quantity, there
is genuinely no prior to look up, and a geometry optimisation downstream pulls it to the right
value. If that optimisation fails to converge you get an error, not a quietly strained
structure.

Compare an atom's element. If MolPy did not know it and assumed carbon, nothing downstream
would ever notice. Bond lengths get relaxed; identities do not. So a missing element raises,
a missing atom type raises, and an unknown bond length gets a named constant and a comment
naming the optimiser that converges it.

**Guess the value, never the identity.** The test is whether some later step converges the
guess away.

## Identical junctions are typed once

Every EO–EO junction along a thousand-monomer chain has the same local chemistry, so each one
cuts out the same affected region. MolPy keys every region by a structural hash that does not
depend on atom numbering (two regions get the same key when their atoms and bonds can be
matched one-to-one, element for element), together with which of its atoms form the
write-back set. The second junction hits the cache. So does the eight-hundredth, and so does
the first junction of the next chain. The number of typing passes tracks the number of
*distinct* chemical environments, not the number of bonds formed.

The cache lives on the assembler, not on the call. Reuse one assembler across a hundred chains
and the EO–EO junction is typed exactly once for the whole melt.

## Growing a chain

A `PolymerBuilder` owns a monomer library and turns a residue topology into a polymer. Hand
`build` a `ResidueTopology` — or call a shortcut such as `build_linear`, which builds one for
you — and it stamps out one copy of each repeat unit, bonds the adjacent ones, and hands back
the polymer. Each pasted copy gets a residue id and name — a repeat unit *is* a residue, and
that identity survives all the way into a PDB or a prmtop.

`reach` is declared with the typifier, never guessed by the assembler. GAFF (the General AMBER
Force Field) sets an atom type from a one-to-two-bond environment, hence `reach=2`: the
builder writes back every atom within two bonds of each new bond and gives the typifier a
four-bond neighbourhood around it for context.

```python
# docs: skip — AmberToolsTypifier shells out; typifier unit-tested with stubs
import molpy as mp
from molpy.builder import MonomerLibrary, PolymerBuilder, TracePlacer
from molpy.builder.ambertools import AmberTools
from molpy.typifier import AmberToolsTypifier

ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
gaff = AmberToolsTypifier(AmberTools())
eo = gaff.typify(eo)  # initial types for atoms unaffected by any junction

builder = PolymerBuilder(
    MonomerLibrary({"EO": eo}),
    ether,
    typifier=gaff,
    reach=2,
    placer=TracePlacer(),  # lay the pasted copies out; omit to keep them stacked
    finalize="atoms",  # defer full topology for this very large chain
)
chain = builder.build_linear("EO", 1000)
# -> bonds + cached junction atom types/charges; no angle/dihedral table yet
```

When explicit bonded rows are needed, finalize once:

```python
from molpy.builder import Finalization, StructureFinalizer
from molpy.builder.assembly import MonomerLibrary, PolymerBuilder, TracePlacer

# Any atoms-only graph takes the same tail — nothing here is GAFF-specific:
neutral, _ = Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.read_smiles("OCCO")
)
SiteMap(neutral).label_elements("O", "a", "b")
chain = PolymerBuilder(
    MonomerLibrary({"EO": neutral}), ether, placer=TracePlacer(), finalize="atoms"
).build_linear("EO", 5)
assert not list(chain.angles)

chain = StructureFinalizer(Finalization.TOPOLOGY).apply(chain)
assert list(chain.angles)
```

The reaction reads: an `a`-site oxygen bearing a hydrogen, plus a `b`-site oxygen bearing a
hydrogen on some carbon, become an ether bridge. Atoms on the left that do not reappear on the
right are the leaving groups. Nothing in the builder knows the word "dehydration"; the SMARTS
says it, and the `%a` and `%b` predicates bind it to the atoms you marked.

## Giving the chain a shape

`MonomerLibrary` pastes every copy exactly where the template sits, so without a placer a
five-residue chain is five molecules stacked on top of each other, joined by bonds that pass
through them. Placement is therefore something you ask for: pass `placer=TracePlacer()`.

`TracePlacer` walks the residue graph from its lowest-numbered residue, which stays where it
is, and moves every other residue rigidly — once, as a whole — relative to **its own parent**
in that walk. The child's bonding atom lands one *bonding range* (the two atoms' summed
covalent radii plus a buffer) from the parent's reacting atom, along the direction pointing
out of the parent (from the parent's centroid through its reacting atom), and the child is
turned to face away from the parent. Every bond from a residue to its parent therefore starts
at bonding range, which covers linear chains, stars and combs alike.

A ring is the exception. Its closing bond joins two residues that were already placed through
the rest of the ring, so it is formed but **not placed**: after `build_ring` the closing bond
is as long as the open chain left it. Fixing that length is your job — either give the placer
an explicit ring-shaped trace, or run a geometry optimization afterwards (see
[Geometry Optimization](08_geometry_optimization.md)).

A `Trace` is a list of 3D points; its tangents (the direction from one point toward the next)
replace each parent's outward direction as the growth direction, so the chain follows the
curve's shape at bonding range rather than landing on the points themselves:

```python
# docs: skip — continues from `neutral` and `ether` above
import numpy as np
from molpy.builder import Trace, TracePlacer, LineOrienter

radius = 4.0  # Å — pick it so 2πr roughly matches the contour length of the six residues
angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]  # six points on a circle, one per residue
ring_trace = Trace([[radius * np.cos(t), radius * np.sin(t), 0.0] for t in angles])
placer = TracePlacer().with_trace(ring_trace).with_orienter(LineOrienter())
ring = PolymerBuilder(MonomerLibrary({"EO": neutral}), ether, placer=placer).build_ring("EO", 6)
```

Even with a ring-shaped trace the placer does not check or enforce closure: the closing bond
is as short as your trace made it, so a quick geometry optimization is still the reliable
finish. With a trace the residues must form a single path, or a single ring (walked as a path
from its lowest id); a truly branched topology — a residue joined to three or more others — raises
`ValueError`, as does a trace with fewer points than residues. The orienter decides how each
residue faces the growth direction: `LineOrienter` points its site axis along it, and
`TangOrienter` points a perpendicular of the site axis along it.

**Writing your own placer.** Subclass `Placer` (it may take whatever constructor arguments you
need) and implement `place(self, mol, bonds)`. `mol` is the pasted world, whose coordinates you
edit in place; `bonds` is the list of bonds about to form, as pairs of atom handles (the first
from the reaction's first reactant, the second from its second). The contract a placer honours:
move whole residues rigidly (with `translate` / `rotate`) so no bond inside a residue is
stretched, leave every residue-to-parent bond at bonding range, and raise rather than leave a
partial placement behind. The assembler calls `place` exactly once per `apply`, after the
selector has chosen the pairs and before the reaction runs.

## Crosslinking is the same machine

A `PolymerBuilder` *is* an assembler — it adds a monomer library and a residue topology to one.
Strip those two away and you have the assembler itself, which is all crosslinking needs: a
graph you already have, and a rule for which sites pair up.

```python
# docs: skip — needs AmberToolsTypifier (gaff) from offline block above
from molpy.builder import GraphAssembler, RandomSelector, Replicas

melt = Replicas(chain).grid(3, spacing=9.5, jitter=1.0, seed=7)

gel = GraphAssembler(ether, typifier=gaff, reach=2).apply(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)
```

`RandomSelector` shuffles the site pairs lying within 6 Å of each other and consumes them until
80 % of sites have reacted. It supplies a pairing rule and nothing else — the graph edit, the
retyping, and the cache are the same code that built the chain.

No `placer` is passed here, because the melt's coordinates are already meaningful and must not
be disturbed. Placement is opt-in on every assembler, `PolymerBuilder` included: you pass a
placer when your input is freshly pasted templates, and leave it out when the coordinates are
already where you want them. That is a decision about *your input*, not about which class you
reached for, which is why it is an argument.

## An end-to-end network

Build chains, pack them, crosslink, then relax — the crosslinks are bonds formed between atoms
wherever packing left them, up to the selector's cutoff apart.

```python
# docs: skip — full gel workflow (molpack + write_lammps); pack unit-tested elsewhere
import molpy as mp
from molpack import InsideBoxRestraint, Molpack, Target

# `neutral` above, not the deliberately-unfrozen `eo` — that one exists to
# demonstrate the net-charge guard, and would trip it here.
builder = PolymerBuilder(
    MonomerLibrary({"EO": neutral}), ether, typifier=gaff, reach=2, placer=TracePlacer()
)
chains = [builder.build_linear("EO", 50).to_frame() for _ in range(100)]
box = InsideBoxRestraint([0.0, 0.0, 0.0], [80.0, 80.0, 80.0])
targets = [Target(c, count=1).with_restraint(box) for c in chains]
melt = Molpack().with_seed(1).pack(targets, max_loops=200)

gel = GraphAssembler(ether, typifier=gaff, reach=2).apply(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)

frame = gel.to_frame()
opt = mp.LBFGS(gaff.forcefield.to_potentials(frame), fmax=0.05, max_steps=200)
frame, report = opt.run(frame)
mp.io.write_lammps_system("gel", frame, gaff.forcefield)
```

One `builder` builds all hundred chains, so the EO–EO junction is typed once for the entire
melt — even if the chains had different lengths, because the cache keys on local structure, not
on topology. The crosslink junctions are typed once per distinct environment. Neither count
grows with the number of chains, which is the only reason a hundred fifty-mers is a tractable
amount of typing.

The `LBFGS` line (L-BFGS, a quasi-Newton energy minimiser) is where the unphysical bond lengths
go away. It is not optional polish: the crosslinks were formed between atoms up to 6 Å apart,
and nothing before this line has asked the force field what that distance should be.

## When an assembler refuses

Three refusals, all of them loud, all of them before the real product graph is returned.

Hand it a typifier without `reach` and construction fails: the assembler cannot size an
affected region for a typifier whose neighbourhood it does not know. If a typifier uses
genuinely non-local information such as unbounded ring membership, it is not valid for
region typing; run it later as an explicit whole-graph operation instead.

Hand it two reaction sites that share an atom and `apply` raises, rather than applying one
edit on top of handles the other already invalidated.

Hand it a repeat unit whose caps still carry charge — you never folded them onto their site
atoms (`SiteMap(...).prepare_leaving_hydrogens(...)` for hydrogen caps, by hand for larger
leaving groups) — and it raises rather than silently leaking net charge into your system.

If the selector yields nothing to react, that is not an error. A cutoff can be too tight, or a
target conversion already met. You get a warning naming the selector class and the number of
matched sites — `RandomSelector selected no bindings from 12 matched sites; the world is
returned unchanged` — and an unchanged copy of your world comes back. The world you passed in
is never mutated, whether or not anything reacted.

## Writing your own pairing rule

You never subclass the assembler. You write a `Selector`, which answers the third question from
the top of this page and nothing else. Its one method, `select(context)`, receives a
`MatchContext`: `context.world` is the graph about to be edited (read it, never edit it), and
`context.occurrences` holds one list per reactant of the reaction, each entry a
`{map_number: atom handle}` dict describing one place that reactant's pattern matched.
`context.map_a` / `context.map_b` are the map numbers of the two atoms the new bond joins, and
`context.comp_a` / `context.comp_b` say which reactant list each belongs to;
`context.sites(component, map_number)` returns one occurrence per site atom, ordered by handle.
`select` yields the merged dicts it wants bonded, and no two of them may share an atom.

```python
import numpy as np
from molpy.builder.assembly import GraphAssembler, MatchContext, Selector


class NearestNeighborSelector(Selector):
    """Pair each first-reactant site with the closest still-free second-reactant site."""

    def select(self, context: MatchContext):
        a_sites = context.sites(context.comp_a, context.map_a)
        b_sites = context.sites(context.comp_b, context.map_b)
        used: set[int] = set()
        for occ_a in a_sites:
            atoms_a = set(occ_a.values())
            if atoms_a & used:
                continue
            free = [
                occ_b
                for occ_b in b_sites
                if not (set(occ_b.values()) & (used | atoms_a))
            ]
            if not free:
                return
            here = self._xyz(context, occ_a[context.map_a])
            occ_b = min(
                free,
                key=lambda occ: np.linalg.norm(self._xyz(context, occ[context.map_b]) - here),
            )
            used |= atoms_a | set(occ_b.values())
            yield {**occ_a, **occ_b}  # {map_number: atom handle}

    @staticmethod
    def _xyz(context: MatchContext, handle: int) -> np.ndarray:
        return np.array([context.world.get(handle, k) for k in ("x", "y", "z")], dtype=float)


melt = mp.Atomistic()
for i in range(3):
    melt.def_atom(element="N", x=float(i), y=0.0, z=0.0)
    melt.def_atom(element="O", x=float(i) + 0.5, y=1.0, z=0.0)

paired = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).apply(
    melt, NearestNeighborSelector()
)
assert len(list(paired.bonds)) == 3
```

The matching has already happened — the assembler does it once, in linear time — so a selector
never scans the system. It only decides. Retyping, atom write-back, the cache, the charge
check, and the non-overlapping guarantee all come for free, because none of them depend on how
you chose the pairs.

That is what it means for the three jobs on this page to be one algorithm: the part you might
want to change is the only part you can.

## See also

- **[Polymer Topologies](topology/index.md)** — the same machine as a full section:
  linear, block, ring, star, comb, telechelic, gels, end-linked, dual network, agent
  (each page ↔ `examples/topology/<name>.py`).
- **[Force Field Typification](06_typifier.md)** — which typifiers can be used during
  assembly, and how to declare `reach` for a black-box one.
- **[Geometry Optimization](08_geometry_optimization.md)** — the step that converges the
  guessed and stretched bond lengths.
- **[Building a Crosslinked Gel](16_crosslinked_gel.md)** — the workflow above, with packing
  and equilibration.
- **[Reaction SMARTS](topology/index.md#reaction-smarts-one-screen)** — reaction SMARTS
  semantics, leaving groups, and `%label` predicates.
- **[Builder API](../api/builder.md)** — every assembly symbol in one table.

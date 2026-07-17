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
You can write one that pairs whatever you like. Everything downstream of that choice — compiling
the possible local products, editing the graph once, and optionally finalizing topology — is one
code path you never touch.

## Assembly is compile, execute, finalize

**An assembler selects every reaction first, compiles its local product against intact
monomer templates, executes the full reaction batch, and only then performs the requested
whole-system finalization.**

This order is the performance contract. The thousandth bond is never followed by a retype of
the thousand-monomer graph. Before any real edit, MolPy overlays all planned forming bonds,
extracts bounded motifs around them, and runs the typifier only for distinct rooted motifs.
For a `PolymerBuilder`, each motif expands to complete user-defined residues: a dimer or short
oligomer stays a chemically complete molecule, so no artificial completion of a cut monomer is
needed. A generic atom-level `GraphAssembler` cut is completed only at the outer context shell.

The cached result contains scalar per-atom information only — atom type, charge, class, and pair
parameters. It deliberately contains no bond, angle, dihedral, or improper rows. After the real
`Reaction.apply_many` batch has formed every bond, those per-atom values are replayed by stable
atom handle. Overlapping motifs must agree; disagreement means the declared `reach` was too
small and raises instead of making write order decide the answer.

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
deciding that a hydroxyl may meet a carboxyl. Sites are unordered and undirected, and the
reaction SMARTS is the only place chemistry is written down.

It is **not a typifier**. Every accepted implementation inherits the common `molrs.Typifier`
base; the assembler only compiles the bounded graph on which it is invoked.

## A repeat unit is a molecule with a few marked atoms

There is no `RepeatUnit` class, no `Junction`, no `Port`. A repeat unit is an ordinary
molecule with a few of its atoms named. You mark the ones that may react and leave every
other atom alone.

An ethylene-oxide repeat unit is a real, capped molecule: ethylene glycol. Its two hydroxyl
oxygens are the sites; the hydroxyl hydrogens are the caps the reaction will remove.

```python
from molpy.core import fields
from molpy.parser import parse_smiles, smilesir_to_atomistic

eo = smilesir_to_atomistic(parse_smiles("OCCO"))   # ethylene glycol
eo.atoms[0][fields.SITE] = "a"                     # one hydroxyl oxygen
eo.atoms[3][fields.SITE] = "b"                     # the other
```

Nothing here says "head" or "tail", and `a` and `b` carry no direction — they are labels the
reaction can refer to. A linear chain is a world whose bonds happen to form a path. A
four-arm crosslinker is a molecule with four marked atoms. A macrocycle is one more bond
between two atoms already connected. The same `eo` above serves all three, unchanged.

## Charge is frozen on the template, so conservation is free

AM1-BCC charges come from a self-consistent solve over an entire molecule. There is no such
thing as the charge of a fragment, so there is no honest way to compute the charge of a new
junction by looking at the atoms around it. This is not a limitation of MolPy; it is what
"non-local" means.

The way out is to never compute charge during assembly. Solve it once on the capped repeat
unit, where the molecule is small and closed-shell and `sqm` is meaningful. Then fold each
cap's charge onto the site atom it capped. After the fold, every cap carries exactly zero.

Watch what the reaction then does: it deletes atoms whose charge is zero. **Net charge is
conserved because nothing charged was removed** — not because a correction term redistributed
the loss afterwards. Conservation stops being a heuristic and becomes an accounting identity.

MolPy will not paper over a template you forgot to freeze. If the reaction is about to delete
a charged atom, `assemble` raises and says so:

```python
builder.build("{[#EO]|3}")
# ValueError: assembly changed the net charge by -0.6 e: the reaction deleted atoms
# that carry charge. Freeze the monomer templates first so each cap's charge folds
# onto its site atom.
```

This is also why AMBER prep files freeze per-residue charges. Same physics, arrived at
independently.

## You may guess a number, never an identity

Assembly needs a length for the bond it is about to form, before any force field has been
consulted about that bond. So it guesses: the sum of the two atoms' covalent radii.

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

Every EO–EO junction along a thousand-monomer chain has the same local chemistry. The compiler
builds these planned product motifs before the polymer exists and keys each by structure,
chemical scalar labels, and the touched root. The second junction hits the cache. So does the
eight-hundredth, and so does the first junction of the next chain. The number of typing passes
tracks the number of *distinct* chemical environments, not the number of bonds formed.

The cache lives on the assembler, not on the call. Reuse one assembler across a hundred chains
and the EO–EO junction is typed exactly once for the whole melt.

## Growing a chain

A `PolymerBuilder` owns a monomer library and speaks CGSmiles. Hand it a notation string and it
stamps out one copy of each repeat unit, bonds the adjacent ones, and hands back the polymer.
Each pasted copy gets a residue id and name — a repeat unit *is* a residue, and that identity
survives all the way into a PDB or a prmtop.

`reach` is part of the compilation boundary, not a property guessed by the assembler. GAFF
atom types are set by a one-to-two-bond environment, hence `reach=2`. The compiler uses one
additional shell of that width as context and writes only the inner atoms back.

```python
import molpy as mp
from molpy.builder import MonomerLibrary, PolymerBuilder, ResiduePlacer
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
    finalize="atoms",  # defer full topology for this very large chain
)
chain = builder.build("{[#EO]|1000}")
# -> bonds + cached junction atom types/charges; no angle/dihedral table yet
```

When explicit bonded rows are needed, finalize once:

```python
from molpy.builder import Finalization, StructureFinalizer

chain = StructureFinalizer(Finalization.TOPOLOGY).apply(chain)
```

The reaction reads: an `a`-site oxygen bearing a hydrogen, plus a `b`-site oxygen bearing a
hydrogen on some carbon, become an ether bridge. Atoms on the left that do not reappear on the
right are the leaving groups. Nothing in the builder knows the word "dehydration"; the SMARTS
says it, and the `%a` and `%b` predicates bind it to the atoms you marked.

## Crosslinking is the same machine

A `PolymerBuilder` *is* an assembler — it adds a library and a notation to one. Strip those two
away and you have the assembler itself, which is all crosslinking needs: a graph you already
have, and a rule for which sites pair up.

```python
from molpy.builder import GraphAssembler, RandomSelector

gel = GraphAssembler(ether, typifier=gaff, reach=2).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)
```

`RandomSelector` shuffles the site pairs lying within 6 Å of each other and consumes them until
80 % of sites have reacted. It supplies a pairing rule and nothing else — the graph edit, the
retyping, and the cache are the same code that built the chain.

No `placer` is passed here, because the melt's coordinates are already meaningful and must not
be disturbed. `PolymerBuilder` places by default, because fresh template copies land on top of
one another. That is a decision about *your input*, not about which class you reached for,
which is why it is an argument.

## An end-to-end network

Build chains, pack them, crosslink, then relax — the crosslinks are the bonds whose lengths
were guessed.

```python
import molpy as mp
from molpy.optimize import LBFGS, ForceFieldPotential

builder = PolymerBuilder(
    MonomerLibrary({"EO": eo}), ether, typifier=gaff, reach=2
)
melt = mp.pack.Packmol().pack([builder.build("{[#EO]|50}")] * 100, density=0.9)

gel = GraphAssembler(ether, typifier=gaff, reach=2).assemble(
    melt, RandomSelector(conversion=0.8, cutoff=6.0, seed=1)
)

frame = gel.to_frame()
LBFGS(ForceFieldPotential(gaff.forcefield)).run(frame, fmax=0.05, steps=200)
mp.io.write_lammps_system("gel", frame, gaff.forcefield)
```

One `builder` builds all hundred chains, so the EO–EO junction is typed once for the entire
melt — even if the chains had different lengths, because the cache keys on local structure, not
on topology. The crosslink junctions are typed once per distinct environment. Neither count
grows with the number of chains, which is the only reason a hundred fifty-mers is a tractable
amount of typing.

The `LBFGS` line is where the guessed bond lengths go away. It is not optional polish: the
crosslinks were formed at covalent-radius separation, and nothing before this line has asked
the force field what that distance should be.

## When an assembler refuses

Three refusals, all of them loud, all of them before the real product graph is returned.

Hand it a typifier without `reach` and construction fails: a bounded compiler cannot infer a
black box's receptive field. If a typifier uses genuinely non-local information such as
unbounded ring membership, it is not valid for local compilation; run it later as an explicit
whole-graph operation instead.

Hand it two reaction sites that share an atom and `assemble` raises, rather than applying one
edit on top of handles the other already invalidated.

Hand it a repeat unit whose caps still carry charge — you forgot to `freeze` — and it raises
rather than silently leaking net charge into your system.

If `select` finds nothing to react, that is not an error. A cutoff can be too tight, or a
target conversion already met. You get a warning naming the candidate count and the cutoff,
and your world comes back untouched.

## Writing your own pairing rule

You never subclass the assembler. You write a `Selector`, which answers the third question from
the top of this page and nothing else. It receives the world and the sites the reaction matched,
grouped by reactant, and yields the pairs it wants bonded.

```python
from molpy.builder import Selector
from molpy.core.atomistic import Atomistic

class NearestNeighborSelector(Selector):
    def select(self, world: Atomistic, occurrences: list[list[dict[int, int]]]):
        a_sites, b_sites = occurrences
        for occ_a in a_sites:
            occ_b = self._nearest(world, occ_a, b_sites)
            yield {**occ_a, **occ_b}       # {map_number: atom handle}
```

The matching has already happened — the assembler does it once, in linear time — so a selector
never scans the system. It only decides. Product-motif compilation, atom write-back, the cache,
the charge check, and the non-overlapping guarantee all come for free, because none of them
depend on how you chose the pairs.

That is what it means for the three jobs on this page to be one algorithm: the part you might
want to change is the only part you can.

## See also

- **[Polymer Topologies](topology/index.md)** — the same machine as a full section:
  linear, block, ring, star, comb, telechelic, gels, end-linked, dual network, agent
  (each page ↔ `examples/topology/<name>.py`).
- **Force Field Typification** — which typifiers can be used during assembly, and how to
  declare `reach` for a black-box one.
- **Geometry Optimization** — the step that converges the guessed bond lengths.
- **Building a Crosslinked Gel** — the workflow above, with packing and equilibration.
- `molpy.Reaction` — reaction SMARTS semantics, leaving groups, and `%label` predicates.
</content>

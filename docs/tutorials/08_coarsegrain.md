# Coarse-Grained Structure

This page explains the `CoarseGrain` data structure, why it stays out of the way of force-field choices, and how to project an atomistic system onto a coarse-grained one.

## A coarse-grained structure is the same kind of object as an atomistic one

In MolPy, `CoarseGrain` is the analog of `Atomistic` for systems where the basic unit is a *bead* rather than an atom. A bead may correspond to a group of atoms in a fine-grained model (Martini, MARTINI 3, VOTCA-style mappings), or it may have no atomistic precursor at all (DPD, equilibrated Martini snapshots, polymer scaffolds awaiting backmapping).

The shape of `CoarseGrain` is identical to that of `Atomistic`. The same factory methods, the same spatial operations, the same composition operators, the same dict-based property access. If you know how to build an atomistic system, you already know how to build a coarse-grained one.

```python
import molpy as mp

cg = mp.CoarseGrain(name="lipid")
b1 = cg.def_bead(type="P4", x=0.0, y=0.0, z=0.0)
b2 = cg.def_bead(type="C1", x=4.7, y=0.0, z=0.0)
cg.def_cgbond(b1, b2, k=120.0)

cg.move([1, 0, 0])
print(b1["x"])   # 1.0
```

## Beads carry whatever fields you decide they should carry

`Bead` is a dict-like object with no mandatory fields. Position, mass, charge, type label, atom-level provenance — every one of these is a key you set or omit, just like on `Atom`. There is no force-field-specific schema; there is no built-in opinion about how a bead's position relates to the atoms it represents.

This deliberate absence of structure is the design point. A Martini 3 bead uses the geometric centre of its constituent heavy atoms (with hydrogens). A Martini 2 bead uses the mass-weighted centre. A VOTCA-style bead uses arbitrary per-atom weights. A DPD bead has no atoms at all and stores its position directly. MolPy refuses to pick one of these conventions for you, because each is correct in its own context.

```python
# A Martini-flavoured bead: type label + position
cg.def_bead(type="P4", x=1.0, y=2.0, z=3.0)

# A bead that remembers which atoms it represents
ato = mp.Atomistic()
c1 = ato.def_atom(element="C", x=0.0, y=0.0, z=0.0)
c2 = ato.def_atom(element="C", x=1.5, y=0.0, z=0.0)
cg.def_bead(atoms=(c1, c2), type="CG_C2")

# A bead that uses a vermouth-style residue template key
cg.def_bead(template="ALA_BB", residue_id=12)
```

None of these layouts is "correct" or "preferred". The data structure simply records what you put into it.

## One convention key gets first-class support

There is one and only one convention key the core data structure recognises: `bead["atoms"]`. When present, it is interpreted as a tuple of `Atom` references that the bead represents. This convention exists for the same reason `entity["x/y/z"]` exists — to give the spatial mixin something to operate on. Where `move(delta)` requires `x/y/z`, the reverse-lookup method `beads_of(atom)` requires `atoms`.

```python
ato = mp.Atomistic()
a = ato.def_atom(element="C")
b = ato.def_atom(element="C")
c = ato.def_atom(element="O")

bead_ab = cg.def_bead(atoms=(a, b), type="CC")
bead_c  = cg.def_bead(atoms=(c,),     type="O")

cg.beads_of(a)   # (bead_ab,)
cg.beads_of(c)   # (bead_c,)
cg.beads_of(b)   # (bead_ab,)
```

The lookup is a linear scan; for hot loops over many atoms, the user is expected to build a private `id(atom) → list[Bead]` index. The data structure deliberately does not cache, because cache invalidation would introduce coupling with every factory method on `CoarseGrain`.

`beads_of` returns multiple beads if the mapping has overlap, and an empty tuple if the atom is not referenced by any bead.

```python
shared = cg.def_bead(atoms=(a,), type="virtual")
cg.beads_of(a)   # (bead_ab, shared)
```

Shared atoms are real in production force fields. Martini uses them in fused aromatic rings; AdResS-style hybrid resolution uses them at the AA/CG boundary. The data structure does not need to know any of that — it only needs to permit the user to express it.

## Projecting from atomistic is your code, not the framework's

There is no `from_atomistic` factory and no `to_atomistic` method on `CoarseGrain`. This is not an oversight. The act of projecting an atomistic system onto a coarse-grained one bundles several independent decisions: how to partition atoms into beads, how to compute each bead's position, whether to infer CG bonds from crossing atomistic bonds or to declare them explicitly, and what additional properties to copy. Every one of those decisions has more than one defensible answer.

The framework's role is to make your projection easy to express, not to choose its policies for you.

```python
import numpy as np

def my_coarsegrain(ato, mask):
    """A simple disjoint-partition projection with COG positions and
    crossing-bond inference. Adapt freely."""
    cg = mp.CoarseGrain()
    bead_of = {}
    for idx in np.unique(mask):
        atoms = tuple(a for a, m in zip(ato.atoms, mask) if m == idx)
        pos = np.mean([[a["x"], a["y"], a["z"]] for a in atoms], axis=0)
        bead_of[int(idx)] = cg.def_bead(
            atoms=atoms,
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        )

    atom_to_idx = {id(a): i for i, a in enumerate(ato.atoms)}
    seen = set()
    for bond in ato.bonds:
        bi = int(mask[atom_to_idx[id(bond.itom)]])
        bj = int(mask[atom_to_idx[id(bond.jtom)]])
        if bi == bj:
            continue
        key = (bi, bj) if bi < bj else (bj, bi)
        if key in seen:
            continue
        seen.add(key)
        cg.def_cgbond(bead_of[bi], bead_of[bj])
    return cg
```

This is the entire AA→CG path for a basic Martini-style mapping in twenty lines. Switching to mass-weighted positions, residue-based partitioning, explicit ITP-declared bonds, or per-bead virtual sites is a matter of changing this function — not of fighting the framework's choices.

## Round-tripping is the builder's job, not the data structure's

The reverse direction — turning a coarse-grained snapshot back into an atomistic one — is also intentionally absent from `CoarseGrain`. Backmapping is a constructive operation: it requires a fragment library keyed by bead type, a placement procedure that respects bond geometry, and usually a relaxation step. Tools like *Backward*, *initram*, and *vermouth* implement this as a pipeline, not as a single method. In MolPy the same role belongs to the polymer builder layer, which consumes `CoarseGrain` snapshots and produces `Atomistic` outputs through fragment templates that the user supplies.

For the present page, the takeaway is simpler: `CoarseGrain` is a place to put beads and the bonds between them. Everything else — projection, backmapping, energetics, force-field assignment — happens around it.

## Spatial and compositional operations work exactly as on Atomistic

Because `CoarseGrain` mirrors `Atomistic`'s public surface, the spatial mixin and the system-composition operators behave identically.

```python
cg2 = cg.copy()
cg2.move([10, 0, 0])
combined = cg + cg2

cg.replicate(4, transform=lambda copy, i: copy.move([i * 5, 0, 0]))
```

You can select a subset of beads by predicate, rename bead types in bulk, or attach arbitrary metadata to the structure itself.

```python
cg.select(lambda b: b.get("type") == "P4")
cg.rename_type("P4", "Q4")
# Use .get so beads without an "x" coordinate (e.g. mapping-only beads) are
# simply not tagged, rather than raising KeyError.
cg.set_property(lambda b: b.get("x", 0.0) > 0, "region", "right")
```

These methods carry no implicit assumptions about what a "type" means, what a "region" is, or how positions relate to physical space. They are graph operations on a graph whose nodes happen to be beads.

## When to use CoarseGrain instead of Atomistic

Use `CoarseGrain` when you want the type system to make it explicit that a node is a bead, not an atom. Use it when you intend to serialize the structure as a CG topology (Martini ITP, LAMMPS DATA with a CG model, etc.) rather than as an all-atom one. Use it when a downstream builder will consume it and produce an atomistic output.

Conversely, if your beads are physically interchangeable with atoms in your workflow — for example, if you are running a single-bead-per-atom united-atom model — there is no harm in using `Atomistic` directly and treating the atoms as beads. The data structures are deliberately the same shape; the choice between them is a labelling decision, not a capability decision.

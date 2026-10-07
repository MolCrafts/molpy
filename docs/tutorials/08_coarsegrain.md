# Coarse-Grained Structure

You have a Martini lipid or a DPD scaffold. Do you need a second library API,
or the same graph tools with a coarser unit?

**`CoarseGrain` is `Atomistic` with beads instead of atoms** — the same
factories, transforms, and composition. A bead may map to a group of atoms
(Martini, VOTCA-style) or have no atomistic precursor (DPD, scaffolds awaiting
backmapping).

What it is **not**: a forced mapping convention, and not a different
connectivity model. If you can build an atomistic graph, you can build a
coarse-grained one.

```python
import molpy as mp

cg = mp.CoarseGrain(name="lipid")
b1 = cg.def_bead(type="P4", x=0.0, y=0.0, z=0.0)
b2 = cg.def_bead(type="C1", x=4.7, y=0.0, z=0.0)
cg.def_cgbond(b1, b2, k=120.0)

cg.translate([1, 0, 0])  # in place; returns cg, so calls chain
print(b1["x"])  # -> 1.0
```

## Beads carry the fields you choose

`Bead` is dict-like, like `Atom`: position, mass, charge, type, provenance —
set or omit what your workflow needs. There is no force-field schema baked into
the type.

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

There is one and only one convention key the core data structure recognises: `bead["atoms"]`. When present, it is interpreted as a tuple of `Atom` references that the bead represents. This convention exists for the same reason `entity["x/y/z"]` exists — to give the spatial mixin something to operate on. Where `translate(delta)` requires `x/y/z`, the reverse-lookup method `beads_of_atom(handle)` requires `atoms`. The lookup takes and returns integer handles (`atom.handle`, `bead.handle`).

```python
ato = mp.Atomistic()
a = ato.def_atom(element="C")
b = ato.def_atom(element="C")
c = ato.def_atom(element="O")

# Membership names atoms of *one* atomistic world, so a mapping onto a new
# world is a new coarse-grained graph, not more beads on the previous one.
mapped = mp.CoarseGrain(name="mapping")
bead_ab = mapped.def_bead(atoms=(a, b), type="CC")
bead_c = mapped.def_bead(atoms=(c,), type="O")

assert mapped.beads_of_atom(a.handle) == [bead_ab.handle]
assert mapped.beads_of_atom(c.handle) == [bead_c.handle]
assert mapped.beads_of_atom(b.handle) == [bead_ab.handle]
```

The lookup is a linear scan; for hot loops over many atoms, the user is expected to build a private `atom handle → list[bead handle]` index. The data structure deliberately does not cache, because cache invalidation would introduce coupling with every factory method on `CoarseGrain`.

`beads_of_atom` returns multiple beads (in handle order) if the mapping has overlap, and an empty list if the atom is not referenced by any bead.

```python
shared = mapped.def_bead(atoms=(a,), type="virtual")
assert mapped.beads_of_atom(a.handle) == [bead_ab.handle, shared.handle]
```

Shared atoms are real in production force fields. Martini uses them in fused aromatic rings; AdResS-style hybrid resolution uses them at the AA/CG boundary. The data structure does not need to know any of that — it only needs to permit the user to express it.

## Projecting from atomistic: you choose the partition

There is no `from_atomistic` factory and no `to_atomistic` method on `CoarseGrain`. Projecting an atomistic system onto a coarse-grained one bundles several independent decisions: how to partition atoms into beads, how to compute each bead's position, whether to infer CG bonds from crossing atomistic bonds or to declare them explicitly, and what additional properties to copy. Every one of those decisions has more than one defensible answer.

The partition is always yours. `Coarsener` covers the common remaining choices — one site per atom group at the group's mass-weighted centre, carrying the summed `mass`, `bead_type` from the names you pass, the group as its members, and one CG bond wherever an atomistic bond crosses two groups:

```python
ethanol = mp.Atomistic(name="ethanol")
ca = ethanol.def_atom(element="C", x=0.0, y=0.0, z=0.0, mass=12.011)
cb = ethanol.def_atom(element="C", x=1.5, y=0.0, z=0.0, mass=12.011)
oh = ethanol.def_atom(element="O", x=2.9, y=0.0, z=0.0, mass=15.999)
ethanol.def_bond(ca, cb)
ethanol.def_bond(cb, oh)

projected = mp.builder.Coarsener(ethanol).coarsen(
    [[ca.handle, cb.handle], [oh.handle]], ["C2", "OH"]
)
print(projected.n_beads, len(projected.cgbonds))  # -> 2 1
print(projected.bead_types(list(projected.entities())))  # -> ['C2', 'OH']
```

Switching to geometric centres, explicit ITP-declared bonds, or per-bead virtual sites is a matter of building the `CoarseGrain` yourself with `def_bead(atoms=..., x=..., ...)` and `def_cgbond` — not of fighting the framework's choices.

## Round-tripping is the builder's job, not the data structure's

The reverse direction — turning a coarse-grained snapshot back into an atomistic one — is also intentionally absent from `CoarseGrain`. Backmapping is a constructive operation: it requires a fragment library keyed by bead type, a placement procedure that respects bond geometry, and usually a relaxation step. Tools like *Backward*, *initram*, and *vermouth* implement this as a pipeline, not as a single method. In MolPy the same role belongs to `mp.builder.Assembler`: it reads a site `CoarseGrain` (one site per bead group, from `mp.builder.Coarsener`), places one copy of a user-supplied template per site and joins the copies through their ports, returning an `mp.Atomistic`.

For the present page, the takeaway is simpler: `CoarseGrain` is a place to put beads and the bonds between them. Everything else — projection, backmapping, energetics, force-field assignment — happens around it.

## Spatial and compositional operations work exactly as on Atomistic

Because `CoarseGrain` mirrors `Atomistic`'s public surface, `copy`, `translate`, and `merge` behave identically.

```python
cg2 = cg.copy()
cg2.translate([10, 0, 0])
combined = cg.copy()
combined.merge(cg2)  # in place; returns the old-to-new handle map

row = mp.CoarseGrain(name="row")
for i in range(4):
    row.merge(cg.copy().translate([i * 5, 0, 0]))
```

Selecting beads by predicate, renaming bead types in bulk, or tagging a region are plain loops over the live bead views:

```python
p4 = [b for b in cg.beads if b.get("type") == "P4"]
for b in p4:
    b["type"] = "Q4"
# Use .get so beads without an "x" coordinate (e.g. mapping-only beads) are
# simply not tagged, rather than raising KeyError.
for b in cg.beads:
    if b.get("x", 0.0) > 0:
        b["region"] = "right"
```

These operations carry no implicit assumptions about what a "type" means, what a "region" is, or how positions relate to physical space. They are graph operations on a graph whose nodes happen to be beads.

## When to use CoarseGrain instead of Atomistic

Use `CoarseGrain` when you want the type system to make it explicit that a node is a bead, not an atom. Use it when you intend to serialize the structure as a CG topology (Martini ITP, LAMMPS DATA with a CG model, etc.) rather than as an all-atom one. Use it when a downstream builder will consume it and produce an atomistic output.

Conversely, if your beads are physically interchangeable with atoms in your workflow — for example, if you are running a single-bead-per-atom united-atom model — there is no harm in using `Atomistic` directly and treating the atoms as beads. The data structures are deliberately the same shape; the choice between them is a labelling decision, not a capability decision.

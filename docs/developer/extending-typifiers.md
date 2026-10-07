# Extending Typifiers

Typifiers operate on molecular graphs. The core contract is the molrs base
`mp.ff.typifier.Typifier`:

```python
import molpy as mp


class MyTypifier(mp.ff.typifier.Typifier):
    def match(self, graph: mp.Atomistic) -> mp.ff.typifier.Match: ...
```

`typify(mol)` belongs to the base and is final: it copies `mol`, calls `match`
on the copy, writes the returned `Match` onto the copy and defines its types in
the output force field, `forcefield()`. The returned object is a typed
`Atomistic`; the input is never touched. A typifier must not return a `Frame`;
call `.to_frame()` after typification when a writer or potential compiler needs
columnar data.

## Required API Shape

- Implement `match(graph) -> Match` and, optionally, `library()`. Defining
  `typify` on a subclass raises `TypeError` at class creation.
- `library()` returns the force field the typifier matches against. The output
  starts as its empty likeness — its name, declared units and special_bonds —
  so declare them there (the AmberTools typifiers declare `real` units and the
  AMBER 1-4 scaling this way).
- `match` may write intermediate results (generated angles and dihedrals,
  perceived bond types) onto the graph it is given: `typify` always hands it a
  private copy.
- Do not add `from_forcefield(ff)`. A typifier constructor loads or builds its
  own parameter tables and keeps what it needs internally.
- `forcefield()` is the union of what `typify` assigned. Do not add a second
  way to build it.

The match must cover every topology class the force field supports. For OPLS-AA
that means atoms, bonds, angles, and dihedrals; for MMFF it also includes
out-of-plane impropers. If a force field has no improper table, do not
synthesize one just to satisfy a generic abstraction.

## The Match

`Match(nodes, links, styles=..., pairs=...)`:

- `nodes`: one mapping of `key -> annotation` per atom, positional against
  `graph.atoms`.
- `links`: relation class (`mp.Bond`, `mp.Angle`, `mp.Dihedral`, `mp.Improper`)
  to one mapping per row, positional against `graph.links.exact_bucket(cls)`.
- `styles`: `(category, style, params)` to declare.
- `pairs`: `(style, name, endpoints, params)` pair rows.

An annotation is a plain value (stamped, defines nothing) or a type
`(style, name, endpoints, params)`, which stamps `name` and every param and
defines the type on `endpoints` — atom-type names, empty for an atom type.
Names are opaque: endpoints are always given, never parsed out of a name.

```python
class ElementBondTypifier(mp.ff.typifier.Typifier):
    """Atom types from elements; one harmonic bond type per element pair."""

    def match(self, graph):
        nodes = [
            {"type": ("full", atom["element"], (), {"mass": atom["mass"]})}
            for atom in graph.atoms
        ]
        bonds = []
        for bond in graph.links.exact_bucket(mp.Bond):
            ends = sorted(atom["element"] for atom in bond.endpoints)
            bonds.append({"type": ("harmonic", "-".join(ends), ends, {"k": 300.0, "r0": 1.5})})
        return mp.ff.typifier.Match(
            nodes, {mp.Bond: bonds}, styles=[("atom", "full", {}), ("bond", "harmonic", {})]
        )


typifier = ElementBondTypifier()
typed = typifier.typify(mp.io.SmilesIR("CCO").to_atomistic())
assert sorted({bond["type"] for bond in typed.bonds}) == ["C-C", "C-O"]
assert {t.name for t in typifier.forcefield().get_style("bond", "harmonic").types} == {"C-C", "C-O"}
```

## Matcher Boundary

The matcher is an implementation detail of a typifier, not the typifier itself.
Use molrs SMARTS matching directly:

```python
mol = mp.io.SmilesIR("CCO").to_atomistic()
pattern = mp.SmartsPattern("[C:1][O:2]")
matches = pattern.find_matches(mol)
```

Matches are bindings: atom ids plus optional mapping labels. They are not
graphs and not frames. Do not use a Python igraph matcher or
MolPy-side layered matcher classes; OPLS-AA and MMFF matching live in molrs.

## Where a Typifier Lives

Force-field typifiers that decide types by SMARTS rules (OPLS-AA, MMFF94) are
native: they live in molrs and `mp.ff.typifier` re-exports them one by one. A new
rule-based force field belongs there too.

A MolPy-side typifier is one that drives an external tool, as
`AntechamberTypifier` and `TLeapTypifier` in `molpy/typifier/ambertools.py`
do: `match` writes the graph for the tool, runs it through a `molpy.wrapper`
wrapper, reads the result back and returns it as a `Match`. Keep the rules
explicit:

- Keep atom order: the tool's row *i* is graph atom *i*, and bonded terms are
  matched by their endpoint rows — raise when the two disagree.
- Raise when the tool is missing, fails, or leaves its output unwritten, with
  the tool's stderr in the message.
- Declare units and scaling in `library()`, not by patching the output.

## Tests

New typifiers need focused tests at three levels:

- Atom coverage: expected `type` and charge on representative real molecules.
- Topology coverage: expected bond, angle, dihedral, and improper types plus
  parameter columns after `typify()`.
- Force-field coverage: `forcefield()` holds exactly the types `typify`
  assigned, with their parameters.

A typifier that shells out never runs the tool in unit tests: patch
`subprocess.run` so each call copies a committed output fixture into place, as
`tests/test_ff/test_ambertools.py` does.

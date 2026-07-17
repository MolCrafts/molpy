# Extending the Data Model

MolPy's graph data model is backed directly by molrs. `Atomistic` and
`CoarseGrain` are native worlds; `Atom`, `Bond`, `Bead`, and the other Python
objects are live views over stable handles in those worlds.

!!! note "Discuss before you build"
    A new stored node or relation kind changes the molrs schema, Rust graph
    algorithms, Python bindings, frame projection, and I/O formatters. Open a
    [GitHub issue](https://github.com/MolCrafts/molpy/issues) before implementing it.

## What can be extended in Python

Python subclasses may add stateless conveniences around an existing native
kind: constructor aliases, selection helpers, callbacks, display methods, and
format-specific serialization. They must not create a second property store,
endpoint list, or handle registry.

Use a graph factory to create data:

```python
from molpy import Atomistic

mol = Atomistic(name="water")
oxygen = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0)
hydrogen = mol.def_atom(element="H", x=0.96, y=0.0, z=0.0)
bond = mol.def_bond(oxygen, hydrogen)

assert mol.atoms[0] is oxygen
assert bond.atoms == (oxygen, hydrogen)
```

Writes through a ref immediately update the native world. Collection properties
such as `.atoms`, `.bonds`, and `.impropers` are live molrs views.

## Adding a stored graph kind

There is intentionally no `TypeBucket.register_type` hook. A real new kind
requires, in order:

1. Define its storage and relation arity in molrs.
2. Teach native copy/merge/extract/topology and Frame projection about it.
3. Expose a handle view and graph factory in `molrs.views`.
4. Re-export that same object from `molpy.core`; add only Python syntax sugar.
5. Update relevant readers/writers and add Rust, binding, and molpy integration tests.

If the concept is only an annotation, prefer a typed field on an existing node
or relation. For example, assembly sites use `fields.SITE`; they do not require
a new node class.

## Checklist

- [ ] The data has one owner in molrs; no Python mirror or `_inner` facade.
- [ ] Handles remain stable across live views and writes update the world.
- [ ] Native copy/merge/extract and Frame round trips cover the new kind.
- [ ] PyO3 exports and type stubs are updated before molpy uses the API.
- [ ] MolPy re-exports the native type or adds a true native subclass only.
- [ ] Rust, molrs-python, and molpy integration tests all pass.

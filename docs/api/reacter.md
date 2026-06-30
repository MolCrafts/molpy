# Reacter

Chemical reaction modeling: anchor selection, leaving group removal, bond formation, template generation.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Reacter` | Execute a reaction between two structures | Polymer growth, bond formation |
| `BondReactReacter` | Reacter + subgraph extraction for LAMMPS `fix bond/react` | Reactive MD templates |
| `find_port(struct, name)` | Find first atom with port marker | Locating `<`/`>`/`$` ports |
| `select_self` | Use port atom itself as reaction anchor | Right-side anchor selection |
| `select_neighbor(element)` | Select neighbor of given element | Left-side anchor selection |
| `select_hydrogens(n)` | Select n hydrogen leaving atoms | Dehydration reactions |
| `select_none` | Return empty list (no leaving group) | Addition reactions |
| `form_single_bond` | Create single bond between atoms | Standard bond formation |

## Selector protocol

- **Anchor selectors**: `(struct: Atomistic, port_atom: Atom) -> Atom` —
  map the port atom to the atom that actually forms the bond.
- **Leaving selectors**: `(struct: Atomistic, anchor: Atom) -> list[Atom]` —
  return zero or more atoms to remove.

## Canonical example

```python
from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    Reacter,
    find_port,
    form_single_bond,
    select_hydrogens,
    select_self,
)


def methane_fragment(port):
    struct = Atomistic()
    carbon = Atom(element="C")
    hydrogens = [Atom(element="H") for _ in range(3)]
    struct.add_entity(carbon, *hydrogens)
    for hydrogen in hydrogens:
        struct.add_link(Bond(carbon, hydrogen))
    carbon["port"] = port
    return struct


left = methane_fragment(">")
right = methane_fragment("<")

rxn = Reacter(
    name="cc_coupling",
    anchor_selector_left=select_self,
    anchor_selector_right=select_self,
    leaving_selector_left=select_hydrogens(1),
    leaving_selector_right=select_hydrogens(1),
    bond_former=form_single_bond,
)
result = rxn.run(
    left=left,
    right=right,
    port_atom_L=find_port(left, ">"),
    port_atom_R=find_port(right, "<"),
    compute_topology=True,
)
product = result.product
assert len(result.removed_atoms) == 2
assert len(list(product.atoms)) == 6
```

`Reacter.run()` never mutates the caller's inputs — the reaction runs on
internal copies and the product is a fresh structure.

## Related

- [Guide: Stepwise Polymer Construction](../user-guide/02_polymer_stepwise.md)
- [Guide: Crosslinked Networks](../user-guide/04_crosslinking.md)

---

## Full API

### Base

::: molpy.reacter.base

### Selectors

::: molpy.reacter.selectors

### Topology Detector

::: molpy.reacter.topology_detector

### Utils

::: molpy.reacter.utils

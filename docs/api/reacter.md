# Reacter

Chemical reaction modeling: site selection, leaving group removal, bond formation, template generation.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Reacter` | Execute a reaction between two structures | Polymer growth, bond formation |
| `TemplateReacter` | Reacter + subgraph extraction for LAMMPS `fix bond/react` | Reactive MD templates |
| `find_port(struct, name)` | Find first atom with port marker | Locating `<`/`>`/`$` ports |
| `select_self` | Use port atom itself as reaction site | Right-side site selection |
| `select_neighbor(element)` | Select neighbor of given element | Left-side site selection |
| `select_hydrogens(n)` | Select n hydrogen leaving atoms | Dehydration reactions |
| `select_nothing` | Return empty list (no leaving group) | Addition reactions |
| `form_single_bond` | Create single bond between atoms | Standard bond formation |

## Selector protocol

All selectors: `(struct: Atomistic, atom: Atom) -> list[Atom]`

- **Site selectors**: return exactly one atom
- **Leaving selectors**: return zero or more atoms to remove

## Canonical example

```python
from molpy.reacter import (
    Reacter, find_port, form_single_bond,
    select_neighbor, select_self, select_hydrogens,
)

rxn = Reacter(
    name="dehydration",
    anchor_selector_left=select_neighbor("C"),
    anchor_selector_right=select_self,
    leaving_selector_left=my_leaving_selector,
    leaving_selector_right=select_hydrogens(1),
    bond_former=form_single_bond,
)
result = rxn.run(left=mol_a, right=mol_b,
    port_atom_L=find_port(mol_a, ">"),
    port_atom_R=find_port(mol_b, "<"),
    compute_topology=True)
product = result.product_info.product
```

## Related

- [Guide: Stepwise Polymer Construction](../user-guide/02_polymer_stepwise.ipynb)
- [Guide: Crosslinked Networks](../user-guide/04_crosslinking.ipynb)

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

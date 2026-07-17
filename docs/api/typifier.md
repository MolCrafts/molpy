# Typifier

Graph typification and force-field parameter assignment.

## The contract

A typifier is `MolGraph -> MolGraph`: it takes a molecular graph and returns a
new one whose elements carry force-field types and parameters. Every typifier
runs the same flow — copy, match, write the annotations back — so `typify()` is
written once, on the base class, and `match()` is the single abstract step.

```python
class Typifier[G: MolGraph](ABC):
    def typify(self, graph: G) -> G: ...      # concrete: copy, match, write back
    @abstractmethod
    def match(self, graph: G) -> Match: ...   # the only thing that differs
```

The pipeline is generic over the graph. An `Atomistic` and a `CoarseGrain` are
both molrs graph leaves, and a concrete typifier specialises `G` to the one it
understands. Nothing in the contract mentions bonds, angles or dihedrals: that
decomposition belongs to a force field, not to typification.

**Typifiers are named after the force field or the tool that decides the types.**
There is no "typifier" that merely spends a type it was given — that is a
component, `ForceFieldParams`.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Typifier` | The contract: one abstract `match` | Writing your own |
| `OPLSAATypifier` | Full OPLS-AA typing pipeline re-exported from `molrs.typifier` | OPLS-AA all-atom force fields |
| `MMFFTypifier` | Full MMFF94 typing pipeline re-exported from `molrs.typifier` | MMFF all-atom force fields |
| `ClpTypifier` | CL&P ionic-liquid overlay: molrs SMARTS types + MolPy parameters | Ionic-liquid force fields |
| `AmberToolsTypifier` | GAFF atom types via antechamber; accumulates the force field it discovers | GAFF / AmberTools |
| `ForceFieldParams` | **Not a typifier.** Annotates pair and bonded terms from node types | A graph whose types are already known |

## Canonical example

```python
import molpy as mp
from molpy.typifier import OPLSAATypifier

typifier = OPLSAATypifier(strict=True)
typed_mol = typifier.typify(mol)  # returns a new Atomistic
frame = typed_mol.to_frame()
```

## Key behavior

- `typify()` returns a **new** graph — the original is not modified
- A typifier never asks whether its graph is a *fragment*. Truncation is a fact
  about provenance, not something readable off a graph's valences: a radical is a
  perfectly good molecule. The party that cut the graph completes it — see
  `RegionTypes.of`, which caps every region it types because every region is a cut
- A term the force field does not parameterise is left **undecided**, never
  stamped with `None`
- SMARTS matching is implemented in molrs; MolPy no longer carries a matcher

## Writing a typifier

Implement `match` and stop. It returns the annotations each node and each link
should receive, positional against `graph.nodes` and `graph.links.bucket(cls)`.

```python
from molpy.typifier import ForceFieldParams, Match, Typifier

class MyTypifier(Typifier[mp.Atomistic]):
    def __init__(self, forcefield):
        self._params = ForceFieldParams(forcefield)

    def match(self, graph):
        node_types = [{"type": decide(atom)} for atom in graph.atoms]
        return self._params.match(graph, node_types)
```

`ForceFieldParams` is the tail every force-field typifier ends with. It is also
the one place in MolPy that knows a `Bond` is parameterised by a `BondType` —
arity cannot decide that, since a dihedral and an improper both span four atoms.

## Related

- [Guide: Force Field Typification](../user-guide/06_typifier.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Contract

::: molpy.typifier.base

### Force-field parameters

::: molpy.typifier.forcefield

### CL&P Typifier

::: molpy.typifier.clp

### AmberTools Typifier

::: molpy.typifier.ambertools

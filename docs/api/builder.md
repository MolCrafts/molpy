# Builder

System assembly: polymer chain construction from G-BigSMILES / CGSmiles notation and monomer libraries.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `polymer(spec, ...)` | String → chain in one call | Quick prototyping |
| `polymer_system(spec, ...)` | G-BigSMILES → polydisperse multi-chain system | Bulk system setup |
| `prepare_monomer(bigsmiles)` | BigSMILES → 3D monomer with ports | Building monomer libraries |
| `PolymerBuilder` | Build chains from CGSmiles + library + connector + placer | Full control over assembly |
| `Connector` | Port selection rules + reaction binding | Defining which ports react |
| `ReactionPresets` | Named reaction chemistries (`"dehydration"`, …) | Reusing / registering chemistry |
| `Placer` | Geometric placement (separator + orienter) | Controlling inter-monomer geometry |
| `CovalentSeparator` | Covalent radii-based distance (buffer in Å) | Default monomer spacing |
| `LinearOrienter` | Linear chain orientation | Default growth direction |

## Canonical example

The one-call entry point auto-detects the notation and returns an
`Atomistic` directly:

```python
from molpy.builder import polymer

# G-BigSMILES: monomer repeat unit + chain length in one string
chain = polymer("{[<]CCO[>]}|5|", optimize=False, random_seed=42)
assert chain.__class__.__name__ == "Atomistic"
assert len(list(chain.atoms)) > 0
```

For full control, prepare a monomer library and drive `PolymerBuilder`
step by step:

```python
from molpy.builder import prepare_monomer
from molpy.builder.polymer import (
    Connector,
    CovalentSeparator,
    LinearOrienter,
    Placer,
    PolymerBuilder,
    ReactionPresets,
)

eo = prepare_monomer("{[<]CCO[>]}", optimize=False)

builder = PolymerBuilder(
    library={"EO": eo},
    connector=Connector(
        reacter=ReactionPresets.get("dehydration"),
        port_map={("EO", "EO"): (">", "<")},
    ),
    placer=Placer(CovalentSeparator(), LinearOrienter()),
)
result = builder.build("{[#EO]|3}")
chain = result.polymer
assert len(list(chain.atoms)) > 0
```

## Custom reaction chemistry

`ReactionPresets.register()` is the extension point for naming your own
chemistry and using it via the `reaction_preset` keyword everywhere:

```python
from molpy.builder.polymer import ReactionPresets, ReactionPresetSpec
from molpy.reacter import form_single_bond, select_hydrogens, select_self

ReactionPresets.register(
    ReactionPresetSpec(
        name="cc_coupling_demo",
        description="C-C coupling, one H lost per side",
        anchor_selector_left=select_self,
        anchor_selector_right=select_self,
        leaving_selector_left=select_hydrogens(1),
        leaving_selector_right=select_hydrogens(1),
        bond_former=form_single_bond,
    )
)
assert "cc_coupling_demo" in ReactionPresets.list_presets()
```

## Related

- [Guide: Stepwise Polymer](../user-guide/02_polymer_stepwise.ipynb)
- [Guide: Topology-Driven Assembly](../user-guide/03_polymer_topology.ipynb)

---

## Full API

### Crystal

::: molpy.builder.crystal

### Polymer

::: molpy.builder.polymer

### Polymer entry functions

High-level entry functions (`polymer`, `polymer_system`,
`prepare_monomer`, `generate_3d`).

::: molpy.builder.polymer.dsl

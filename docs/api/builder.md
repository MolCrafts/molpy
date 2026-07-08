# Builder

System assembly: polymer chain construction from a monomer library plus
CGSmiles / label-sequence connectivity. You compose the real engine
classes directly — there is no `polymer()` dispatcher.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `PolymerBuilder` | Assemble a chain from a monomer library + reaction; `.build_sequence(labels)` or `.build(cgsmiles)` | The entry point for chain assembly |
| `Connector` | Port selection rules + reaction binding | Defining which ports react |
| `ReactionPresets` | Named reaction chemistries (`"dehydration"`, …) | Reusing / registering chemistry |
| `Placer` | Geometric placement (separator + orienter) | Controlling inter-monomer geometry |
| `CovalentSeparator` | Covalent radii-based distance (buffer in Å) | Default monomer spacing |
| `LinearOrienter` | Linear chain orientation | Default growth direction |
| `SystemPlanner` / `PolydisperseChainGenerator` | Sample a polydisperse chain plan | Bulk / molecular-weight-distributed systems |
| `AmberPolymerBuilder` | GAFF-parameterised build via AmberTools | AMBER/LAMMPS-bound workflows |
| `DrudeBuilder` / `Tip4pBuilder` / `VirtualSiteBuilder` | Virtual-site augmentation | Polarizable / 4-site models |

## Canonical example

Prepare the repeat-unit monomer, then feed a label sequence to
`PolymerBuilder` — no notation round-trip, no wrapper:

```python
from molpy.builder.polymer import PolymerBuilder, ReactionPresets
from molpy.conformer import Conformer
from molpy.parser import parse_monomer

# 1. repeat-unit monomer: BigSMILES -> 3D Atomistic with < / > ports
#    (Conformer is molpy's native molrs embedder)
eo, _ = Conformer(add_hydrogens=True, seed=42).generate(parse_monomer("{[<]CCO[>]}"))

# 2. assemble a chain: monomer library + a reaction, then a label sequence
builder = PolymerBuilder({"EO": eo}, reacter=ReactionPresets.get("dehydration"))
chain = builder.build_sequence(["EO"] * 5).polymer

assert chain.__class__.__name__ == "Atomistic"
assert len(list(chain.atoms)) > 0
```

For explicit port mapping and geometry, pass a `Connector` + `Placer` and
build from a CGSmiles string (reusing the `eo` monomer above):

```python
from molpy.builder.polymer import (
    Connector,
    CovalentSeparator,
    LinearOrienter,
    Placer,
    PolymerBuilder,
    ReactionPresets,
)

builder = PolymerBuilder(
    library={"EO": eo},
    connector=Connector(
        reacter=ReactionPresets.get("dehydration"),
        port_map={("EO", "EO"): (">", "<")},
    ),
    placer=Placer(CovalentSeparator(), LinearOrienter()),
)
chain = builder.build("{[#EO]|3}").polymer
assert len(list(chain.atoms)) > 0
```

## Polydisperse systems

Drive `PolymerBuilder` from the distribution + planner primitives
yourself — sample a chain plan, then loop `build_sequence`:

```python
import numpy as np
from molpy.builder.polymer import (
    PolydisperseChainGenerator,
    SchulzZimmPolydisperse,
    SystemPlanner,
    WeightedSequenceGenerator,
)

planner = SystemPlanner(
    PolydisperseChainGenerator(
        WeightedSequenceGenerator({"EO": 1.0}),
        {"EO": 44.05},
        distribution=SchulzZimmPolydisperse(1500, 3000),
    ),
    target_total_mass=5e5,
)
plan = planner.plan_system(np.random.default_rng(42))
chains = [builder.build_sequence(c.monomers).polymer for c in plan.chains]
assert len(chains) >= 1
```

## Custom reaction chemistry

`ReactionPresets.register()` is the extension point for naming your own
chemistry, then handed to `PolymerBuilder` via `ReactionPresets.get`:

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

- [Guide: Stepwise Polymer](../user-guide/02_polymer_stepwise.md)
- [Guide: Topology-Driven Assembly](../user-guide/03_polymer_topology.md)

---

## Full API

### Crystal

::: molpy.builder.crystal

### Polymer

::: molpy.builder.polymer

### Virtual Sites

Add Drude oscillators (CL&Pol) or TIP4P M-sites to a structure; each builder
copies its input, selects host atoms, builds the virtual sites, and
redistributes mass/charge without mutating the caller.

::: molpy.builder.virtualsite

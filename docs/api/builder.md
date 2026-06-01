# Builder

System assembly: polymer chain construction from CGSmiles topology and monomer libraries.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `PolymerBuilder` | Build chains from CGSmiles + library + connector + placer | Full control over assembly |
| `polymer(cgsmiles, ...)` | CGSmiles → chain in one call | Quick prototyping |
| `Connector` | Port selection rules + reaction binding | Defining which ports react |
| `Placer` | Geometric placement (separator + orienter) | Controlling inter-monomer geometry |
| `CovalentSeparator` | Covalent radii-based distance | Default monomer spacing |
| `LinearOrienter` | Linear chain orientation | Default growth direction |

## Canonical example

```python
from molpy.builder.polymer import (
    PolymerBuilder, Connector, Placer,
    CovalentSeparator, LinearOrienter,
)
from molpy.builder import polymer

builder = PolymerBuilder(
    library={"EO": eo_template},
    connector=Connector(port_map={("EO","EO"): (">","<")}, reacter=rxn),
    placer=Placer(separator=CovalentSeparator(buffer=-0.1),
                  orienter=LinearOrienter()),
)
result = builder.build("{[#EO]|10}")
chain = result.polymer

# Or use the one-call entry function:
result = polymer("{[#EO]|10}", library={"EO": eo_template}, reacter=rxn)
chain = result.polymer
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

### Polymer DSL tools

High-level polymer-building tools and entry functions (`PrepareMonomer`,
`BuildPolymer`, `PlanSystem`, `BuildSystem`, `BuildPolymerAmber`, `polymer`,
`polymer_system`, `prepare_monomer`, `generate_3d`).

::: molpy.builder.polymer.dsl

### Tool framework

`Tool` and `ToolRegistry` are the internal base classes that the builder
DSL tools are built on. They are not public top-level exports.

::: molpy.builder._tool

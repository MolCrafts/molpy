# Prepolymer + tetrafunctional agent

**Script:** [`examples/topology/11_prepolymer_agent.py`](../../../examples/topology/11_prepolymer_agent.py)

Linear EO chains keep free hydroxyl ends. A small-molecule agent (`X4`, four SITE `a`) is merged into the world; the same **ETHER** reaction couples ends to the agent through a proximity selector — agent curing is assembly, not a special API.

```python
from eo_kit import ETHER, eo_builder, full_library
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas
import molpy as mp

chain = eo_builder().build_linear("EO", 5)
agent = full_library()["X4"]
world = Replicas(chain).times(4, spacing=8.0)
# merge agent copies …
cured = GraphAssembler(mp.Reaction(ETHER)).assemble(
    world,
    ExhaustiveSelector(cutoff=10.0, exclude_same_molecule=True),
)
```

```bash
cd examples && python topology/11_prepolymer_agent.py
```

## See also

- [Star](04_star.md) (multifunctional cores) · [Section index](index.md)

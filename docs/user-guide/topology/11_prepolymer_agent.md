# Prepolymer + tetrafunctional agent

**Script:** [`examples/topology/11_prepolymer_agent.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/11_prepolymer_agent.py)

Linear EO chains keep free hydroxyl ends. A small-molecule agent (`X4`, four SITE `a`) is merged into the world; the same **ETHER** reaction couples ends to the agent through a proximity selector — agent curing is assembly, not a special API.

```python
from eo_kit import ETHER, eo_builder, full_library
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas
from molpy.core import fields
import molpy as mp

chain = eo_builder().build_linear("EO", 5)
agent = full_library()["X4"]
world = Replicas(chain).times(4, spacing=8.0)  # four chains along x, mol_id 1..4
for i in range(2):  # two agent molecules beside the chains
    copy = agent.copy().translate([i * 3.0, 2.0, 0.0])
    for atom in copy.atoms:
        atom[fields.MOL_ID] = 100 + i
    world.merge(copy)

cured = GraphAssembler(mp.Reaction(ETHER)).apply(
    world,
    ExhaustiveSelector(cutoff=10.0, exclude_same_molecule=True),
)
```

`exclude_same_molecule=True` forbids a bond between two sites that are already
connected through bonds, so a chain never closes on itself and an agent never
reacts with its own hydroxyls. It reads bond connectivity, not the `mol_id`
column; the `mol_id` values above only label the molecules for output.

```bash
cd examples && python topology/11_prepolymer_agent.py
```

## See also

- [Star](04_star.md) (multifunctional cores) · [Section index](index.md)

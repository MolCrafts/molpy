# Block and sequence copolymers

**Script:** [`examples/topology/02_block.py`](../../../examples/topology/02_block.py)

Architecture is the **label sequence**. Two library keys may share the same chemistry; only residue names differ.

```python
from eo_kit import eo_builder, ethylene_glycol

a, b = ethylene_glycol(seed=42), ethylene_glycol(seed=43)
builder = eo_builder(extra={"A": a, "B": b})
block = builder.build_sequence(["A"] * 6 + ["B"] * 4)
# → build("{[#A][#A]…[#B][#B]}")  →  residue sequence AAAAAABBBB
```

Use `build_sequence` when a polydispersity planner has already emitted a list of monomer ids ([Polydisperse Systems](../05_polydisperse_systems.md)).

```bash
cd examples && python topology/02_block.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)

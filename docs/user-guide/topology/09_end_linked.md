# End-linked network

**Script:** [`examples/topology/09_end_linked.py`](../../../examples/topology/09_end_linked.py)

Telechelic strands; only **end residue** carbons are marked for crosslinking. Mid-chain carbons never react — network vertices are the ends you chose.

```python
from eo_kit import XLINK, eo_builder, full_library, mark_residue_crosslink_sites
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas
import molpy as mp

lib = full_library()
builder = eo_builder(extra={"CAPA": lib["CAPA"], "CAPB": lib["CAPB"]})
strand = builder.build_sequence(["CAPA"] + ["EO"] * 5 + ["CAPB"])
mark_residue_crosslink_sites(strand, {"CAPA", "CAPB"}, site="x", leaving="h")
melt = Replicas(strand).times(6, spacing=5.0)

gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
    melt,
    ExhaustiveSelector(cutoff=8.0, exclude_same_molecule=True),
)
```

```bash
cd examples && python topology/09_end_linked.py
```

## See also

- [Telechelic](06_telechelic.md) · [Exhaustive gel](07_gel_exhaustive.md) · [Section index](index.md)

# Telechelic oligomer

**Script:** [`examples/topology/06_telechelic.py`](../../../examples/topology/06_telechelic.py)

Head and tail caps must carry **complementary** single sites so both path ends can form ether bonds with `EO`.

```python
from eo_kit import eo_builder, full_library

lib = full_library()
builder = eo_builder(extra={"CAPA": lib["CAPA"], "CAPB": lib["CAPB"]})
tele = builder.build_sequence(["CAPA"] + ["EO"] * 6 + ["CAPB"])
# CAPA = SITE a only · CAPB = SITE b only
```

A single monofunctional label on both ends cannot pair: the second edge would see two `b` sites and no free `a`.

```bash
cd examples && python topology/06_telechelic.py
```

## See also

- [End-linked network](09_end_linked.md) — mark only CAP residues for crosslinking
- [Section index](index.md)

# Linear homopolymer

**Script:** [`examples/topology/01_linear.py`](../../../examples/topology/01_linear.py)

A path of identical residues is the simplest ruled topology. `build_linear` is a shortcut that formats CGSmiles and calls the sole entry `build`.

```python
from eo_kit import eo_builder  # examples/topology/

builder = eo_builder()
chain = builder.build_linear("EO", 10)
# identical to:
chain = builder.build("{[#EO]|10}")
```

**Check:** 10 residues (`fields.RES_ID` 1…10), acyclic; atom count matches a direct `build("{[#EO]|10}")`.

```bash
cd examples && python topology/01_linear.py
```

## See also

- [Block / sequence](02_block.md) — non-identical labels on a path
- [Section index](index.md)

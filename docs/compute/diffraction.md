# Diffraction

Overview

| Class / entry | Description |
|---------------|-------------|
| [`StaticStructureFactorDebye`](#staticstructurefactordebye) | Static structure factor $S(k)$ via the Debye sum. |

Details

The `molpy.compute.diffraction` module: reciprocal-space structure ($S(k)$).

## `StaticStructureFactorDebye`

Static structure factor $S(k)$ via the Debye sum.

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 10.0, size=(40, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(10.0)
```

```python
import numpy as np
from molpy.compute import StaticStructureFactorDebye

k = np.linspace(0.5, 2.0, 20)
Sk = StaticStructureFactorDebye(k)(frame)
```

## See also

- [RDF](rdf.md)

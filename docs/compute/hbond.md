# HBond

Overview

| Class / entry | Description |
|---------------|-------------|
| [`HBonds`](#hbonds) | Detect H-bonds from donor `(D,H)` pairs and acceptor indices. |
| [`HBondCriterion`](#hbondcriterion) | Distance / angle cutoffs (Luzar–Chandler defaults: 3.5 Å, 150°). |

Details

The `molpy.compute.hbond` module: geometric hydrogen-bond detection.

## `HBonds`

Detect H-bonds from donor `(D,H)` pairs and acceptor indices.

```python
import numpy as np
import molpy as mp
from molpy.compute import HBonds, HBondCriterion

mol = mp.Atomistic()
o = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0)
h1 = mol.def_atom(element="H", x=1.0, y=0.0, z=0.0)
h2 = mol.def_atom(element="H", x=-0.3, y=0.9, z=0.0)
mol.def_bond(o, h1)
mol.def_bond(o, h2)
frame = mol.to_frame()
frame.box = mp.Box.cubic(20.0)

donors = np.array([[0, 1], [0, 2]], dtype=np.int64)
acceptors = np.array([0], dtype=np.int64)
crit = HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0)
bonds = HBonds(donors, acceptors, crit)([frame])
```

## `HBondCriterion`

Distance / angle cutoffs (Luzar–Chandler defaults: 3.5 Å, 150°).

```python
from molpy.compute import HBondCriterion

crit = HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0)
```

## See also

- [Persist](persist.md)
- [Distribution](distribution.md)

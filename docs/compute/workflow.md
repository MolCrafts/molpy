# Workflow

Overview

| Class / entry | Description |
|---------------|-------------|
| [`Workflow`](#workflow) | Directed graph; bind parameters with `inputs={{...}}`. |

Details

The `molpy.compute.workflow` module: chain computes in a DAG.

## `Workflow`

Directed graph; bind parameters with `inputs={{...}}`.

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
from molpy.compute import Workflow, NeighborList, RDF

wf = Workflow()
wf.add("nlist", NeighborList(cutoff=5.0), inputs={"frame": "frame"})
wf.add(
    "rdf",
    RDF(n_bins=40, r_max=5.0),
    inputs={"frames": "frame", "neighbors": "nlist"},
)
results = wf.run(frame=frame)
results["rdf"].rdf
```

## See also

- [RDF](rdf.md)
- [Cluster](cluster.md)

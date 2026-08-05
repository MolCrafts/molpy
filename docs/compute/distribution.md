# Distribution

Overview

| Class / entry | Description |
|---------------|-------------|
| [`DistanceDistribution`](#distancedistribution) | Histogram of bond distances (from the `bonds` block). |
| [`AngleDistribution`](#angledistribution) | Histogram of bond angles (ADF). |
| [`DihedralDistribution`](#dihedraldistribution) | Histogram of dihedral angles (DDF). |
| [`CombinedDistribution`](#combineddistribution) | Joint histogram over several geometric axes. |

Details

The `molpy.compute.distribution` module: histograms of internal coordinates from topology blocks.

## `DistanceDistribution`

Histogram of bond distances (from the `bonds` block).

```python
import molpy as mp

mol = mp.Atomistic()
atoms = [
    mol.def_atom(element="C", x=float(i), y=0.3 * (i % 2), z=0.0) for i in range(6)
]
for i in range(5):
    mol.def_bond(atoms[i], atoms[i + 1])
frame = mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()
```

```python
from molpy.compute import DistanceDistribution

hist = DistanceDistribution(30, 0.0, 6.0)([frame])
hist.density.shape
```

## `AngleDistribution`

Histogram of bond angles (ADF).

```python
from molpy.compute import AngleDistribution

hist = AngleDistribution(30, 0.0, 180.0)([frame])
```

## `DihedralDistribution`

Histogram of dihedral angles (DDF).

```python
from molpy.compute import DihedralDistribution

hist = DihedralDistribution(30)([frame])
```

## `CombinedDistribution`

Joint histogram over several geometric axes.

```python
from molpy.compute import CombinedDistribution

cdf = CombinedDistribution(
    [("angle", 20, 0.0, 180.0, True), ("angle", 20, 0.0, 180.0, True)]
)
result = cdf([frame])
result.ndim
```

## See also

- [Spatial](spatial.md)
- [PMFT](pmft.md)

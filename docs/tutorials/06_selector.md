# Selector

This page explains how MolPy expresses atom selections as composable predicates over `Block` columns, including filters by element, type, coordinate range, and distance.

## Selecting atoms without loops

Analysis workflows constantly need to answer questions like "give me all carbons," "give me atoms within 3 Å of this point," or "give me heavy atoms in the left half of the box." Writing manual loops with if-statements works, but it scales poorly and obscures intent.

**Selectors are composable predicates that produce boolean masks over `Block` columns.** They combine with `&`, `|`, and `~` to build complex queries without manual loops.

Every selector implements the same protocol: call it on a block to get a filtered block, or call `.mask()` to get a boolean array.


## Property-based selectors

The simplest selectors filter by a single column value. `ElementSelector` matches element symbols, `AtomTypeSelector` matches type identifiers.

```python
import molpy as mp
from molpy.core.selector import (
    ElementSelector, AtomTypeSelector,
    CoordinateRangeSelector, DistanceSelector,
)
import numpy as np

atoms = mp.Block({
    "element": ["C", "C", "H", "H", "O", "N"],
    "type":    [1, 1, 2, 2, 3, 4],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
})

carbons = ElementSelector("C")(atoms)
print(carbons.nrows)         # 2
print(carbons["element"])    # ['C', 'C']

type2 = AtomTypeSelector(2)(atoms)
print(type2["element"])      # ['H', 'H']
```


## Geometric selectors

`CoordinateRangeSelector` filters by a coordinate range along one axis. `DistanceSelector` filters by distance from a reference point. Both require `x`, `y`, `z` columns.

```python
right_half = CoordinateRangeSelector("x", min_value=2.5)(atoms)
print(right_half["element"])   # ['H', 'O', 'N']

near_origin = DistanceSelector(center=[0.0, 0.0, 0.0], max_distance=1.5)(atoms)
print(near_origin["element"])  # ['C', 'C']
```

A shell selection — atoms between a minimum and maximum distance — is a common pattern for solvation analysis.

```python
shell = DistanceSelector(
    center=[2.0, 0.0, 0.0],
    min_distance=1.0,
    max_distance=2.5,
)(atoms)
print(shell["element"])
```


## Combining selectors with logic operators

The real power of selectors comes from composition. `&` means AND, `|` means OR, `~` means NOT. The result is a new selector that can be applied or composed further.

```python
# (Carbon OR Oxygen) AND (x > 0.5)
sel = (ElementSelector("C") | ElementSelector("O")) & CoordinateRangeSelector("x", min_value=0.5)
result = sel(atoms)
print(result["element"])   # ['C', 'O']

# Everything except hydrogen
no_h = ~ElementSelector("H")
print(no_h(atoms)["element"])   # ['C', 'C', 'O', 'N']
```

Nested combinations let you express precise scientific queries concisely.

```python
# Heavy atoms near a specific point
heavy_near = (
    ~ElementSelector("H")
    & DistanceSelector(center=[2.0, 0.0, 0.0], max_distance=2.5)
)
print(heavy_near(atoms)["element"])
```


## Working with masks directly

Sometimes you need the boolean mask rather than the filtered block — for indexing into other arrays, for NumPy operations, or for combining with external logic.

```python
mask = ElementSelector("C").mask(atoms)
print(mask)                        # [ True  True False False False False]
print(np.where(mask)[0])           # [0, 1]
print(atoms["x"][mask])            # [0., 1.]
```


## When to use selectors

Use selectors whenever you need to partition atoms in a `Block` — for analysis, for assigning properties, for feeding subsets into computations. They are faster and more readable than hand-written loops, and their composability means you build complex queries from simple, tested parts.

See also: [Block and Frame](02_block_and_frame.md), [Box and Periodicity](03_box_and_periodicity.md).

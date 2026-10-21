# Selector

You need every carbon within 3 Å of a point — without a page of nested
`if` statements. How do you write that as a *query*?

**Selectors are composable predicates over `Block` rows.** They return a
filtered block, or a boolean mask via `.mask()`, and combine with `&`, `|`,
and `~`. molpy has two: `ColumnSelector` (rows whose column equals a value)
and `AtomIndexSelector` (rows by 0-based position).

What they are **not**: a second copy of the system, or a replacement for
topology. They filter rows of an existing table.

## Column selectors

`ColumnSelector(value, column=...)` matches one column against one value. The
column is named by a `mp.core.keys` constant and defaults to `keys.ELEMENT`.

```python
import molpy as mp
from molpy.core import AtomIndexSelector, ColumnSelector, keys
import numpy as np

# The Frame schema declares a dtype per field: `type` is the string label and
# `type_id` is the number. Writing integers into `type` is rejected, not coerced.
atoms = mp.Block(
 {
 "element": ["C", "C", "H", "H", "O", "N"],
 "type": ["c3", "c3", "hc", "hc", "oh", "n"],
 "type_id": np.array([1, 1, 2, 2, 3, 4], dtype=np.uint32),
 "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
 "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 }
)

carbons = ColumnSelector("C")(atoms)
print(carbons.n_rows) # 2
print(carbons["element"]) # ['C', 'C']

# Select by the string label, or point the selector at the numeric column.
hydrogens = ColumnSelector("hc", column=keys.TYPE)(atoms)
print(hydrogens["element"]) # ['H', 'H']
print(ColumnSelector(2, column=keys.TYPE_ID)(atoms)["element"]) # ['H', 'H']
```

The value must have the column's type. `type` is always a string label, so
`ColumnSelector(2, column=keys.TYPE)` raises `TypeError` when applied, instead
of silently selecting nothing.

## Selecting by row index

`AtomIndexSelector` picks rows by their 0-based position in the block. It does
not read the `id` column: `id` is a label (LAMMPS ids start at 1), never an
index.

```python
first_and_last = AtomIndexSelector([0, 5])(atoms)
print(first_and_last["element"]) # ['C', 'N']
```

## Selecting by position: regions

Where an atom is — a slab, a sphere, a shell — is a geometric region's job.
The regions are molrs's (`mp.core.Cuboid`, `mp.core.Sphere`, `mp.core.HalfSpace`, …); each has
the same `mask(block)` and is callable on a block, and they compose with `&`,
`|` and `~` into an `mp.core.Region`. They read the `x`, `y`, `z` columns.

```python
# x >= 2.5: the half-space whose outward normal points to -x
right_half = mp.core.HalfSpace([-1.0, 0.0, 0.0], [2.5, 0.0, 0.0])(atoms)
print(right_half["element"]) # ['H', 'O', 'N']

near_origin = mp.core.Sphere([0.0, 0.0, 0.0], 1.5)(atoms)
print(near_origin["element"]) # ['C', 'C']
```

A shell selection — atoms between a minimum and maximum distance — is a common
pattern for solvation analysis: the outer sphere without the inner one.

```python
center = [2.0, 0.0, 0.0]
shell = (mp.core.Sphere(center, 2.5) & ~mp.core.Sphere(center, 1.0))(atoms)
print(shell["element"])
```

## Combining selectors with logic operators

The real power of selectors comes from composition. `&` means AND, `|` means OR, `~` means NOT. The result is a new selector that can be applied or composed further.

```python
# (Carbon OR Oxygen) AND (x >= 0.5): a selector composes with a region
sel = (ColumnSelector("C") | ColumnSelector("O")) & mp.core.HalfSpace(
 [-1.0, 0.0, 0.0], [0.5, 0.0, 0.0]
)
result = sel(atoms)
print(result["element"]) # ['C', 'O']

# Everything except hydrogen
no_h = ~ColumnSelector("H")
print(no_h(atoms)["element"]) # ['C', 'C', 'O', 'N']
```

Nested combinations let you express precise scientific queries concisely.

```python
# Heavy atoms near a specific point
heavy_near = ~ColumnSelector("H") & mp.core.Sphere([2.0, 0.0, 0.0], 2.5)
print(heavy_near(atoms)["element"])
```

## Working with masks directly

Sometimes you need the boolean mask rather than the filtered block — for indexing into other arrays, for NumPy operations, or for combining with external logic.

```python
mask = ColumnSelector("C").mask(atoms)
print(mask) # [ True True False False False False]
print(np.where(mask)[0]) # [0, 1]
print(atoms["x"][mask]) # [0., 1.]
```

## When to use selectors

Use selectors whenever you need to partition atoms in a `Block` — for analysis, for assigning properties, for feeding subsets into computations. They are faster and more readable than hand-written loops, and their composability means you build complex queries from simple, tested parts.

See also: [Block and Frame](02_block_and_frame.md), [Box and Periodicity](03_box_and_periodicity.md).

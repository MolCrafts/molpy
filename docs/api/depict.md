# Depict

2D drawing coordinates for a molecular graph. `Depictor` takes an
`Atomistic` and returns a copy with `x` / `y` in the drawing plane and
`z = 0`: rings as regular polygons (fused rings share an edge, spiro rings an
atom), chains as a 120° zigzag, sp centres straight, substituents spread into
the free angle, and disconnected species side by side. Every property and
relation travels with the copy, so a 2D viewer draws formal charges and ports
from the frame `to_frame()` writes. Available via `import molpy as mp`
(`mp.depict.Depictor`).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Depictor` | Lay a graph out in 2D | A sketch of a parsed or built molecule |

## Related

- [Conformer](conformer.md) — the 3D counterpart, for simulation geometry.

---

## Full API

::: molpy.depict.Depictor

# Dual network

**Script:** [`examples/topology/10_dual_network.py`](../../../examples/topology/10_dual_network.py)

Two `assemble` steps, two SITE namespaces — complex materials are stacks of simple edits.

1. Mark `x` / `h`, partial random crosslink (`XLINK`)
2. Clear first labels, mark `y` / `k`, second `assemble` with `XLINK2`

```python
# net1: RandomSelector on XLINK (sites x/h)
# clear SITE x/h
# mark second population y/k
# net2: ExhaustiveSelector on XLINK2
```

**Note:** after the first network percolates, bond-components may be a single “molecule”. `exclude_same_molecule=True` then forbids every pair; the second pass in the example allows same-component pairs and relies on SITE + cutoff only.

```bash
cd examples && python topology/10_dual_network.py
```

## See also

- [Random gel](08_gel_random.md) · [Section index](index.md)

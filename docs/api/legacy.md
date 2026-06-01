# Legacy

Original pure-NumPy mean-squared-displacement and cross-displacement-correlation
operators. The maintained trunk lives in [Compute](compute.md) (e.g.
`molpy.compute.MSD`, `molpy.compute.MCDCompute`); the operators here are
retained for direct `NDArray` workflows that do not need trajectory coupling.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `MSD` | Mean squared displacement | Diffusion analysis on raw arrays |
| `msd(positions)` | Convenience function for MSD | Quick MSD calculation |
| `DisplacementCorrelation` | Cross-displacement correlation | Ion transport correlations |
| `displacement_correlation(...)` | Convenience function | Quick correlation calculation |

## Canonical example

```python
from molpy.legacy import MSD

msd = MSD(max_lag=3000)
msd_values = msd(unwrapped_positions)  # shape (max_lag,)
```

---

## Full API

### MSD

::: molpy.legacy.msd

### Cross-Displacement Correlation

::: molpy.legacy.cross_correlation

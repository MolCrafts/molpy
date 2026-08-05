# Distribution

Textbook guide to **geometric distribution functions**: distance, angle (ADF),
dihedral (DDF), and their joint **combined distribution function** (CDF).

Where [RDF](rdf.md) answers "how far?", these answer "at what angle, and in what
combination?"

!!! note "Conventions"
    - Distances Å; angles/dihedrals in degrees.
    - Tuples come from frame topology: `bonds`, `angles` (vertex in the middle),
      `dihedrals` — perceive with `get_topo(gen_angle=True, gen_dihe=True)`.
    - ADF reports `density_sin_corrected` to remove solid-angle $\sin\theta$ bias.

---

## 1. One-dimensional geometric histograms

For triplets, the angular distribution is

$$
p(\theta) = \frac{1}{N_\mathrm{groups}
\Big\langle\sum_{(i,j,k)}\delta(\theta-\theta_{ijk})\Big\rangle.
$$

Dihedrals replace $\theta$ with a torsion; distance distributions histogram
selected pairs without the $4\pi r^2$ RDF shell normalization.

An isotropic direction set samples $\mathrm{d}\Omega=\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$,
so raw $p(\theta)$ peaks near $90^\circ$. Always compare
`density_sin_corrected $\propto p(\theta)/\sin\theta$`.

For a backbone torsion,

$$
F(\phi) = -k_B T\ln p(\phi) + \mathrm{const}
$$

is the conformational free-energy map (gauche / anti populations).

---

## 2. Combined distribution function (CDF)

A 1-D histogram averages away correlations. The joint $p(r,\theta)$ for
donor–acceptor geometry **defines** geometric [H-bond](hbond.md) cutoffs from
data: the associated basin at short $r$ and near-linear $\theta$.

Each CDF axis is `(kind, n_bins, min, max, sin_weight)`; all axes must share a
topology kind so they sample the same tuples.

---

## 3. Usage

```python
import molpy as mp
from molpy.conformer import Conformer
from molpy.compute import (
    AngleDistribution, DihedralDistribution, DistanceDistribution,
    CombinedDistribution,
)

# docs: skip — optional if Conformer / read_smiles path is heavy in CI
mol = mp.io.read_smiles("CCO")
mol, _ = Conformer(seed=42).generate(mol)
mol.get_topo(gen_angle=True, gen_dihe=True)
frame = mol.to_frame()

adf = AngleDistribution(n_bins=180, min=0.0, max=180.0)
result = adf([frame])
result.bin_centers, result.density, result.density_sin_corrected

cdf = CombinedDistribution([
    ("angle", 90, 0.0, 180.0, True),
    ("angle", 45, 90.0, 180.0, True),
])
joint = cdf([frame])
assert joint.ndim == 2
```

---

## 4. Pitfalls

1. Wrong vertex order in angles (middle index is the vertex).
2. Forgetting sin-correction on ADF.
3. CDF axes with mismatched tuple counts.
4. Sparse 2-D sampling.

## See also

- [Spatial](spatial.md) · [HBond](hbond.md) · [RDF](rdf.md)
- [API reference](../api/compute.md)

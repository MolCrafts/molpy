---
slug: molrs-analyses-expose-02
spec: molrs-analyses-expose-02.md
created: 2026-05-28
revised: 2026-05-28
---

# Acceptance — molrs-analyses-expose-02

Binding "done" contract. `/mol:impl` must satisfy every criterion below before
deleting the spec. Format follows `plugins/mol/docs/evaluator-protocol.md`.
All `numerical` criteria are covered by their pytest `pass_when` invocations —
no separate runtime evaluator is owed (no `ui_runtime` criteria here).

## Criteria

```yaml
# ── Phase 1 — order parameters ─────────────────────────────────────

- id: order-classes-exposed
  type: code
  summary: Steinhardt / Hexatic / Nematic / SolidLiquid importable from molpy.compute and are Compute subclasses
  pass_when: |
    python -c "
    from molpy.compute import Steinhardt, Hexatic, Nematic, SolidLiquid
    from molpy.compute.base import Compute
    assert all(issubclass(c, Compute) for c in (Steinhardt, Hexatic, Nematic, SolidLiquid))
    "
  status: verified
  last_checked: 2026-05-28

- id: order-parity-with-molrs
  type: numerical
  summary: each order-parameter operator matches direct molrs.compute.order.<X>
  pass_when: |
    pytest tests/test_compute/test_order.py -k "parity" -v
  tolerance:
    numeric_outputs: 1e-12 absolute (deterministic inputs or seed-pinned)
  status: verified
  last_checked: 2026-05-28

- id: order-input-immutable
  type: code
  summary: order-parameter operators do not mutate the input frame
  pass_when: |
    pytest tests/test_compute/test_order.py -k "immutable" -v
  status: verified
  last_checked: 2026-05-28

# ── Phase 2 — density ──────────────────────────────────────────────

- id: density-classes-exposed
  type: code
  summary: LocalDensity / GaussianDensity importable from molpy.compute and are Compute subclasses
  pass_when: |
    python -c "
    from molpy.compute import LocalDensity, GaussianDensity
    from molpy.compute.base import Compute
    assert issubclass(LocalDensity, Compute) and issubclass(GaussianDensity, Compute)
    "
  status: verified
  last_checked: 2026-05-28

- id: density-parity-with-molrs
  type: numerical
  summary: LocalDensity / GaussianDensity match direct molrs.compute.density.<X>
  pass_when: |
    pytest tests/test_compute/test_density.py -k "parity" -v
  tolerance:
    numeric_outputs: 1e-12 absolute
  status: verified
  last_checked: 2026-05-28

# ── Phase 3 — diffraction / environment / pmft ─────────────────────

- id: diffraction-environment-pmft-exposed
  type: code
  summary: StaticStructureFactorDebye / BondOrder / PMFTXY importable from molpy.compute and are Compute subclasses
  pass_when: |
    python -c "
    from molpy.compute import StaticStructureFactorDebye, BondOrder, PMFTXY
    from molpy.compute.base import Compute
    assert all(issubclass(c, Compute) for c in (StaticStructureFactorDebye, BondOrder, PMFTXY))
    "
  status: verified
  last_checked: 2026-05-28

- id: diffraction-environment-pmft-parity
  type: numerical
  summary: StaticStructureFactorDebye / BondOrder / PMFTXY match direct molrs calls
  pass_when: |
    pytest tests/test_compute/test_diffraction.py tests/test_compute/test_environment.py tests/test_compute/test_pmft.py -k "parity" -v
  tolerance:
    numeric_outputs: 1e-12 absolute
  status: verified
  last_checked: 2026-05-28

- id: ssf-ideal-gas-sanity
  type: numerical
  summary: static structure factor of uniform random points approaches 1 at large k
  pass_when: |
    pytest tests/test_compute/test_diffraction.py -k "ideal_gas" -v
  tolerance:
    s_of_k_large_k: within 0.3 of 1.0 for the upper third of the k range (n_points >= 2000)
  status: verified
  last_checked: 2026-05-28

# ── Phase 4 — cluster properties ───────────────────────────────────

- id: cluster-properties-exposed-and-parity
  type: numerical
  summary: ClusterProperties importable, a Compute subclass, and matches direct molrs.compute.cluster.ClusterProperties
  pass_when: |
    python -c "
    from molpy.compute import ClusterProperties
    from molpy.compute.base import Compute
    assert issubclass(ClusterProperties, Compute)
    "
    pytest tests/test_compute/test_cluster.py -k "cluster_properties" -v
  tolerance:
    numeric_outputs: 1e-12 absolute
  status: verified
  last_checked: 2026-05-28

# ── Cross-cutting ──────────────────────────────────────────────────

- id: all-exports-listed
  type: code
  summary: all 10 new operators are in compute.__all__
  pass_when: |
    python -c "
    import molpy.compute as c
    for n in ('Steinhardt','Hexatic','Nematic','SolidLiquid','LocalDensity',
              'GaussianDensity','StaticStructureFactorDebye','BondOrder',
              'PMFTXY','ClusterProperties'):
        assert n in c.__all__, n
    "
  status: verified
  last_checked: 2026-05-28

- id: zero-extra-copy-discipline
  type: code
  summary: no defensive coordinate-array copy in the new compute modules
  pass_when: |
    ! grep -rnE '\.copy\(\)|copy=True' src/molpy/compute/order.py \
        src/molpy/compute/density.py src/molpy/compute/diffraction.py \
        src/molpy/compute/environment.py src/molpy/compute/pmft.py
  status: verified
  last_checked: 2026-05-28

- id: docs-catalog-updated
  type: docs
  summary: the molrs-backend user-guide page lists the newly exposed analyses
  pass_when: |
    grep -q 'Steinhardt' docs/developer/molrs-backend.md
    grep -q 'StaticStructureFactorDebye\|structure factor' docs/developer/molrs-backend.md
    grep -qE 'LocalDensity|PMFTXY|BondOrder' docs/developer/molrs-backend.md
  status: pending
  # Mechanical greps pass; type: docs owed to a human reviewer before close.

- id: full-test-suite-green
  type: code
  summary: full molpy test suite (excluding external) is green
  pass_when: |
    pytest tests/ -m "not external" -q
  status: verified
  last_checked: 2026-05-28
  # 1897 passed, 135 deselected (external), 1 xfailed.
```

## Notes for /mol:impl

- **Two-input call convention.** Most operators take `(frames, nlists)` and a
  few take other second args (`Nematic` → `directors`, `PMFTXY` →
  `orientations=None`, `ClusterProperties` → `clusters`, `GaussianDensity` /
  `StaticStructureFactorDebye` → `frames` only). Follow `compute.RDF`'s
  precedent: implement `__call__`, make `_compute` raise `NotImplementedError`.
- **Parity tests are the core guarantee.** Each operator forwards verbatim to
  molrs, so parity is near-tautological — its real value is that the wrapper
  exists, forwards the right args, and the frame survives unmutated. molrs owns
  the physics tests.
- **Return molrs natives.** Several operators return tuples / ndarrays / dicts,
  not `*Result` objects. Do not wrap them in molpy types.
- `docs-catalog-updated` is `type: docs` — the greps are mechanical; a human
  reviewer signs off prose quality before close.

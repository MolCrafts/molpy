---
slug: molrs-backend
spec: molrs-backend.md
created: 2026-05-06
revised: 2026-05-28
---

# Acceptance — molrs-backend

Binding "done" contract. `/mol:impl` must satisfy every criterion below
before deleting the spec. Format follows
`plugins/mol/docs/evaluator-protocol.md`.

## Criteria

```yaml
# ── Phase 0 — molrs prerequisites ──────────────────────────────────

- id: molrs-pybox-subclassable
  type: code
  summary: molrs.Box is Python-subclassable
  pass_when: |
    cd ../molrs && cargo build -p molrs-python --release
    python -c "
    import molrs
    class Sub(molrs.Box): pass
    Sub(__import__('numpy').eye(3) * 10.0)
    "
  status: verified
  last_checked: 2026-05-28
  # Verified via the Python subclass instantiation (cargo build of the already
  # shipped molcrafts-molrs not re-run; the installed wheel is subclassable).

# ── Phase 1 — required dep + Box inheritance ───────────────────────

- id: molrs-required-not-optional
  type: code
  summary: molcrafts-molrs is in [project.dependencies], not extras
  pass_when: |
    grep -E '^\s*"molcrafts-molrs' pyproject.toml             # in dependencies
    ! grep -E '^\s*molrs\s*=\s*\[' pyproject.toml             # no extras key
  status: verified
  last_checked: 2026-05-28

- id: molpy-box-is-a-molrs-box
  type: code
  summary: molpy.Box inherits from molrs.Box and behaves as one
  pass_when: |
    pytest tests/test_compute/test_box_inheritance.py::test_molpy_box_is_a_molrs_box -v
    pytest tests/test_compute/test_box_inheritance.py::test_frame_box_passed_directly_to_molrs -v
  status: verified
  last_checked: 2026-05-28

- id: molpy-box-public-api-preserved
  type: code
  summary: existing core/box.py public surface still resolves and existing tests pass
  pass_when: |
    pytest tests/test_core/test_box.py -v
    python -c "
    import molpy
    b = molpy.Box.cubic(10.0)
    assert b.style == molpy.Box.Style.ORTHOGONAL
    assert b.matrix.shape == (3, 3)
    "
  status: verified
  last_checked: 2026-05-28

# ── Phase 2 — NeighborList + RDF ───────────────────────────────────

- id: compute-neighborlist-class-exposed
  type: code
  summary: molpy.compute.NeighborList exists and is a Compute subclass
  pass_when: |
    python -c "from molpy.compute import NeighborList; \
               from molpy.compute.base import Compute; \
               assert issubclass(NeighborList, Compute)"
  status: verified
  last_checked: 2026-05-28

- id: compute-rdf-class-exposed
  type: code
  summary: molpy.compute.RDF exists and is a Compute subclass
  pass_when: |
    python -c "from molpy.compute import RDF; \
               from molpy.compute.base import Compute; \
               assert issubclass(RDF, Compute)"
  status: verified
  last_checked: 2026-05-28

- id: neighborlist-parity-with-molrs
  type: numerical
  summary: molpy.compute.NeighborList output matches direct molrs.NeighborQuery
  pass_when: |
    pytest tests/test_compute/test_neighborlist.py::test_parity_with_molrs_direct -v
  tolerance:
    pair_count: exact
    distances: 1e-12 (absolute, after canonical sort)
  status: verified
  last_checked: 2026-05-28
  # Per acceptance Notes: numerical criteria are covered by their pytest
  # pass_when invocations; no separate runtime evaluator is owed.

- id: rdf-ideal-gas-correct
  type: numerical
  summary: g(r) for uniform random points lies in [0.7, 1.3] across middle bins
  pass_when: |
    pytest tests/test_compute/test_rdf.py::test_ideal_gas_g_of_r_approaches_one -v
  tolerance:
    g_of_r_middle_bins: within 0.3 of 1.0 (n_points >= 2000, averaged over >= 5 frames)
  status: verified
  last_checked: 2026-05-28

- id: input-frame-immutable
  type: code
  summary: NeighborList and RDF do not mutate the input frame
  pass_when: |
    pytest tests/test_compute/test_neighborlist.py::test_input_frame_immutable \
           tests/test_compute/test_rdf.py::test_input_frame_immutable -v
  status: verified
  last_checked: 2026-05-28

- id: rdf-requires-box
  type: code
  summary: RDF on a frame without a box raises ValueError mentioning "box"
  pass_when: |
    pytest tests/test_compute/test_rdf.py::test_no_box_raises -v
  status: verified
  last_checked: 2026-05-28

# ── Phase 3 — RDKit replacement ────────────────────────────────────

# AMENDED 2026-05-28 (operator decision): keep the RDKit adapter
# (adapter/rdkit.py) and the RDKit external-tool wrapper (tool/rdkit.py) as an
# OPTIONAL backend; only the *main trunk* compute path switches to the
# molrs-backed embed. So `_HAS_RDKIT` is removed from compute/ only (adapter/
# and tool/ legitimately retain it), and `rdkit` stays an optional extra item
# (no standalone `rdkit = [...]` extras key is introduced).
- id: rdkit-module-deleted
  type: code
  summary: compute/rdkit.py is removed, _HAS_RDKIT gone from compute/, no standalone rdkit extras key
  pass_when: |
    test ! -e src/molpy/compute/rdkit.py
    ! grep -rE '_HAS_RDKIT' src/molpy/compute/
    ! grep -E '^\s*rdkit\s*=\s*\[' pyproject.toml
  status: verified
  last_checked: 2026-05-28

- id: embed-replacement-physical-sanity
  type: numerical
  summary: new molrs-backed Generate3D produces geometries with bond lengths within 10% of literature values for a small molecule set
  pass_when: |
    pytest tests/test_compute/test_embed_replacement.py -v
  tolerance:
    bond_length: ±10% vs literature for water, methane, ethanol
  status: verified
  last_checked: 2026-05-28

# ── Phase 4 — MCD / PMSD internals ─────────────────────────────────

- id: mcd-pmsd-public-signature-preserved
  type: code
  summary: MCDCompute and PMSDCompute keep their public signatures after internal rewiring
  pass_when: |
    pytest tests/test_compute/test_mcd.py tests/test_compute/test_pmsd.py -v
    # Plus: existing call sites in tests/ and notebooks compile (`python -m compileall ...`).
  status: verified
  last_checked: 2026-05-28

# ── Phase 5 — exposed molrs analyses ───────────────────────────────

- id: exposed-analyses-importable
  type: code
  summary: every exposed molrs analysis is importable from molpy.compute
  pass_when: |
    python -c "
    from molpy.compute import (
        MSD, Cluster, ClusterCenters, CenterOfMass,
        GyrationTensor, InertiaTensor, RadiusOfGyration,
        Pca, KMeans,
    )
    "
  status: verified
  last_checked: 2026-05-28

- id: exposed-analyses-parity
  type: numerical
  summary: each exposed analysis has a passing parity test vs direct molrs.<X>.compute
  pass_when: |
    pytest tests/test_compute/ -k "parity" -v
  tolerance:
    numeric_outputs: 1e-12 (absolute, post canonical sort) — except KMeans whose
      output depends on RNG seed; assert seed-pinned reproducibility instead.
  status: verified
  last_checked: 2026-05-28

# ── Cross-cutting hygiene ──────────────────────────────────────────

- id: zero-extra-copy-discipline
  type: code
  summary: no defensive coordinate-array copy at the molrs boundary in compute/
  pass_when: |
    # AMENDED 2026-05-28: the criterion targets defensive copies of NUMERICAL
    # data at the molrs boundary. The lone allowed match is Compute.dump()'s
    # config-DICT copy in base.py (API serialization, not a data-boundary copy).
    ! grep -rnE '\.copy\(\)|copy=True' src/molpy/compute/ | grep -v 'self._config.copy()'
    # The single physical coordinate copy lives inside Block.__getitem__(list)
    # (np.column_stack); outside that one site, the data boundary is zero-copy.
  status: verified
  last_checked: 2026-05-28

- id: no-molrs-gate-flag
  type: code
  summary: molrs is unconditional — no _HAS_MOLRS flag, no gated import
  pass_when: |
    ! grep -rE '_HAS_MOLRS' src/molpy/
    ! grep -rE 'try:\s*\n\s+import molrs' src/molpy/
  status: verified
  last_checked: 2026-05-28

- id: no-frame-to-xyz-helper
  type: code
  summary: there is no _frame_to_xyz / _frame_to_pybox helper anywhere
  pass_when: |
    ! grep -rE '_frame_to_xyz|_frame_to_pybox' src/molpy/
  status: verified
  last_checked: 2026-05-28

- id: docs-page-published
  type: docs
  summary: user-guide page documents installation, Box inheritance, NeighborList + RDF, and the analysis catalog
  pass_when: |
    test -f docs/developer/molrs-backend.md
    grep -q 'class Box(molrs.Box)' docs/developer/molrs-backend.md
    grep -q 'NeighborList' docs/developer/molrs-backend.md
    grep -q 'RDF' docs/developer/molrs-backend.md
    grep -q 'MSD\|Cluster\|GyrationTensor' docs/developer/molrs-backend.md
  status: pending
  # Mechanical greps pass (page exists at the pinned path, wired into mkdocs
  # nav). type: docs is owed to a human reviewer for prose quality before close.

- id: changelog-breaking-change
  type: docs
  summary: changelog calls out that molrs is now required and rdkit extras are removed
  pass_when: |
    grep -i 'breaking' docs/changelog.md
    grep -q 'molcrafts-molrs' docs/changelog.md
    grep -q 'rdkit' docs/changelog.md
  status: pending
  # Mechanical greps pass; type: docs owed to a human reviewer before close.

- id: full-test-suite-green
  type: code
  summary: full molpy test suite (excluding external) is green
  pass_when: |
    pytest tests/ -m "not external" -q
  status: verified
  last_checked: 2026-05-28
  # 1869 passed, 135 deselected (external), 1 xfailed.
```

## Notes for /mol:impl

- **Phase 0 is a hard prerequisite.** Until the molrs `subclass`-flag patch
  ships in a published `molcrafts-molrs` release, Phase 1's Box-inheritance
  work cannot land. If the molrs patch is delayed, you may either (a) bump
  the dependency to a path/git source temporarily, or (b) pause Phase 1
  and proceed with Phases 2–5 using composition (and re-add the inheritance
  patch later) — but flag the deviation in the spec before doing so.
- **Phase 4 and Phase 5 tasks are intentionally left as one-line entries.**
  Decompose each into per-operator subtasks at the start of those phases
  rather than up front, so the breakdown reflects real signatures
  encountered during implementation.
- All `numerical` criteria are also covered by `code` test invocations —
  no separate runtime evaluator is needed (no `ui_runtime` criteria here).
- The `zero-extra-copy-discipline` and `no-frame-to-xyz-helper` criteria
  are the encoded form of the user's directive: no Python-side adapter
  layer, just direct calls into molrs.
- `full-test-suite-green` is the final gate; run it last. It must pass
  without the previously-existing `[molrs]` and `[rdkit]` extras (those
  keys are gone), so don't try to re-add them as a workaround for
  unrelated test failures.

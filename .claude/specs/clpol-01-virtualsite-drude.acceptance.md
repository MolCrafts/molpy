---
slug: clpol-01-virtualsite-drude
criteria:
  - id: ac-001
    summary: VirtualSite/DrudeParticle/MasslessSite exist as Atom subclasses
    type: code
    pass_when: |
      In src/molpy/core/atomistic.py, VirtualSite subclasses Atom,
      and DrudeParticle and MasslessSite each subclass VirtualSite;
      all three importable from molpy.core.
    status: verified
    last_checked: 2026-06-10
  - id: ac-002
    summary: VirtualSite injectable via add_atom and marker survives molrs round-trip
    type: runtime
    pass_when: |
      A DrudeParticle added to an Atomistic via add_atom is retrievable
      and identifiable as a virtual site after passing through the molrs
      backend (test in tests/test_core/test_virtualsite.py passes).
    status: verified
    last_checked: 2026-06-10
  - id: ac-003
    summary: VirtualSiteBuilder.apply does not mutate input structure
    type: code
    pass_when: |
      After DrudeBuilder().apply(struct), struct's atom count and per-atom
      charges are identical to before the call, and the return value is a
      different object (is not struct).
    status: verified
    last_checked: 2026-06-10
  - id: ac-004
    summary: Drude count equals polarizable-heavy-atom count; H excluded
    type: code
    evaluator_hint: "fixture: clp_typed [C4C1im]+"
    pass_when: |
      For a CL&P-typed cation, the number of DrudeParticle sites added equals
      the number of atoms whose alpha.ff type has k_D > 0, and no hydrogen
      receives a DrudeParticle. (H carries alpha=0.323 for Thole only but
      k_D=0, so selection is on k_D > 0, not alpha > 0.)
    status: verified
    last_checked: 2026-06-10
  - id: ac-005
    summary: Each Drude has one core-shell harmonic bond with K = k_D = 4184
    type: code
    pass_when: |
      Every DrudeParticle is connected to its core by exactly one
      BondHarmonic-style bond whose force constant K == 4184.0
      (no factor-of-2 division, matching U = 1/2 K r^2 convention).
    status: verified
    last_checked: 2026-06-10
  - id: ac-006
    summary: alpha recovered from assigned q_D, k_D within tolerance
    type: scientific
    evaluator_hint: "fixture: clp_typed [C4C1im]+"
    pass_when: |
      For every Drude-augmented atom, q_D**2 / (4*pi*eps0 * k_D) equals the
      alpha.ff input alpha for that CL&P type within relative tolerance 1e-6
      (4*pi*eps0 = 0.0007197587 e^2/(kJ/mol*A); the molpy-units form of
      alpha = q_D^2/k_D).
    status: verified
    last_checked: 2026-06-11
  - id: ac-007
    summary: Ion net charge stays the original integer after augmentation
    type: scientific
    pass_when: |
      For each ion, sum of (core charge + shell charge) over its atoms
      equals the original integer ionic charge (+-1) within 1e-9.
    status: verified
    last_checked: 2026-06-11
  - id: ac-008
    summary: Tip4pBuilder emits a MasslessSite on the bisector with O charge moved to M, no spring
    type: code
    evaluator_hint: "fixture: single water molecule"
    pass_when: |
      Tip4pBuilder().apply(water) returns a structure with exactly one
      MasslessSite on the HOH bisector, the O charge transferred to the
      M-site, and no new bond added.
    status: verified
    last_checked: 2026-06-10
  - id: ac-009
    summary: alpha.ff data file present and resolvable
    type: code
    pass_when: |
      get_forcefield_path("alpha.ff") resolves to an existing file under
      src/molpy/data/forcefield/ containing CL&P type -> alpha (and a_thole)
      entries transcribed from paduagroup/clandpol.
    status: verified
    last_checked: 2026-06-10
  - id: ac-010
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      ruff format --check src tests && ruff check src tests &&
      ty check src/molpy/ && pytest tests/ -m "not external" -v
      all succeed.
    status: verified
    last_checked: 2026-06-10
---

# Acceptance criteria

- **ac-001 / ac-002** — 数据模型存在且可注入 molrs（标记可往返）。ac-002 是 runtime，因为它要求实际经 molrs 后端实例化并检索。
- **ac-003** — 不变性：`apply` 返回新结构、入参不被改动。
- **ac-004 / ac-005** — Drude 计数与 H 排除；core-shell 弹簧存在且 `K = 4184`（固化 factor-of-2 核验结果，molpy harmonic 为 `U = 1/2 K r^2`，故不除 2）。
- **ac-006 / ac-007** — 科学验证：`alpha = q_D^2/k_D` 反算一致；离子整数净电荷守恒。两者均依赖 ac-009 的 `alpha.ff` 数据文件落地。
- **ac-008** — `Tip4pBuilder` 证明基类对刚性几何位点的通用性（M 位点、电荷转移、无弹簧）。
- **ac-009** — 极化率数据文件作为科学验收项的前置条件。
- **ac-010** — 全量门禁。

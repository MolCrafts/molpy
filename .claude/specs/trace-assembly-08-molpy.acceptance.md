---
slug: trace-assembly-08-molpy
created: 2026-09-27
criteria:
  - id: ac-001
    summary: Preconditions hold, including branch and section VII scope
    type: code
    pass_when: |
      In molpy, `git rev-parse --abbrev-ref HEAD` prints ci/narrow-gitignore
      and .claude/notes/law.md § VII "molrs 0.15 co-development pin" Scope
      names that branch; molrs-015-align 01-03 are committed (`git status
      --porcelain -- src/` empty at start; base SHA recorded); the active
      venv's python imports molrs.perceive.Coarsener, molrs.Trace,
      molrs.builder.Assembler, molrs.builder.TracePlacer and
      molrs.ff.typifier.ElementTypifier from a release build of molrs
      feat/cgsmiles with links 01-05 and 07 (06 dropped); backmap.py line 25 reads
      `mp.Box(frame.box.h * f, pbc=frame.box.pbc, origin=frame.box.origin *
      f)`.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-002
    summary: The script's names are molpy identities
    type: code
    pass_when: |
      A one-off script asserts mp.Coarsener is molrs.perceive.Coarsener,
      mp.Trace is molrs.Trace, mp.Assembler is molrs.builder.Assembler,
      mp.TracePlacer is molrs.builder.TracePlacer, mp.Perceive is
      molrs.perceive.Perceive, mp.typifier.ElementTypifier is
      molrs.ff.typifier.ElementTypifier, and the four root names are in
      molpy.__all__.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-003
    summary: mp.typifier imports and exposes only native typifiers
    type: runtime
    pass_when: |
      `python -c "import molpy.typifier as t; print(sorted(t.__all__))"`
      exits 0 and prints ['ElementTypifier', 'MMFFTypifier',
      'OPLSAATypifier']; notes.md holds the Law-10 narrowing entry (Scope
      src/molpy/typifier/, removal condition naming the deferred typifier
      link) and the Law-10 write_lammps_data entry routed /mol:refactor.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-004
    summary: The script runs end to end within the wall-time and RSS bars
    type: performance
    evaluator_hint: "/usr/bin/time -v around the script run"
    pass_when: |
      With the active venv, `python /home/jicli594/work/backmap_pe_pma/backmap.py`
      exits 0, prints "wrote /home/jicli594/work/backmap_pe_pma/pe_pma_aa.data:
      6170200 atoms", and `/usr/bin/time -v` reports elapsed wall time < 15 min
      and maximum resident set size < 64 GB; both numbers, the host and the
      molrs commit are recorded in notes.md 2026-09-27.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "exit 0, 6170200 atoms, wall 37.9 s, RSS 11.1 GB"
  - id: ac-005
    summary: The written file is a bond-capable LAMMPS data file that reads back
    type: runtime
    pass_when: |
      The file's Atoms header reads "# molecular"; mp.io.read_lammps_data(OUT,
      atom_style="molecular") yields 6,170,200 atoms, 460,100 distinct
      molecule ids and a bond count equal to the header's `bonds` line; atom
      type labels are exactly the elements present and every bond label is a
      canonical element pair; if `lmp` is on PATH, `lmp -in <input>` with
      `units real`, `atom_style molecular`, `read_data OUT` exits 0 (else the
      absence of lmp is recorded).
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "Atoms # molecular; read-back counts match; 460100 mol ids; lmp read_data OK"
  - id: ac-006
    summary: The box is the scaled CG box and every atom lies inside it
    type: runtime
    pass_when: |
      The written xlo/ylo/zlo equal the CG frame.box.origin * 4.2 and the
      extents equal the CG box lengths * 4.2 (1e-9 Å); every atom's x/y/z
      satisfies lo <= coordinate < hi on each axis.
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-007
    summary: The run's measurements are recorded
    type: docs
    pass_when: |
      molpy notes.md 2026-09-27 records wall time, max RSS, host, molrs
      commit, atom/bond counts, header style, the box check and the lmp
      result (or its absence).
    status: verified
    verified_by: impl-chain
    last_checked: 2026-09-28
    note: "cargo mrs-test (test_single) green at link end"
  - id: ac-008
    summary: No molpy behaviour beyond re-exports
    type: code
    pass_when: |
      `git diff --name-only <P3 base SHA> -- src/ tests/` lists only src/molpy/__init__.py and
      src/molpy/typifier/__init__.py; no class,
      function, test file or `regressions/` path is added.
    status: pending
    note: chain-end gate
---

# Acceptance criteria

`ac-004` to `ac-006` are the operator's acceptance: the binding script on the
real data. The file it writes is checked to be usable by LAMMPS, not only
readable by molpy (review 🔴). `ac-008` is measured against the committed
align base (review 🟡).

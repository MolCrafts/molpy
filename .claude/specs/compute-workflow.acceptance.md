---
slug: compute-workflow
criteria:
  - id: ac-001
    summary: Workflow unit test suite passes (≥8 tests green)
    type: code
    pass_when: |
      `pytest tests/test_compute/test_workflow.py -v` exits 0 and reports
      at least 8 passed tests covering linear chain, diamond reuse, cycle,
      missing input, duplicate name, multi-input RDF, predecessors filter,
      and rerun-non-mutation.
    status: verified
    last_checked: 2026-05-12

  - id: ac-002
    summary: Public surface exports Workflow + four exceptions
    type: code
    pass_when: |
      `python -c "from molpy.compute import Workflow, WorkflowError,
      WorkflowCycleError, WorkflowMissingInputError,
      WorkflowDuplicateNodeError; Workflow()"` exits 0.
    status: verified
    last_checked: 2026-05-12

  - id: ac-003
    summary: Diamond reuse contract — upstream runs exactly once
    type: code
    pass_when: |
      `pytest tests/test_compute/test_workflow.py::test_diamond_reuse_runs_upstream_once -v`
      exits 0; the counting upstream's `_compute` invocation count equals 1
      after a single `run()` with two downstream consumers.
    status: verified
    last_checked: 2026-05-12

  - id: ac-004
    summary: Cycle detection raises WorkflowCycleError at add() time
    type: code
    pass_when: |
      `pytest tests/test_compute/test_workflow.py::test_cycle_raises -v`
      exits 0; the exception type is `molpy.compute.WorkflowCycleError`
      and the workflow state is unchanged after the rejected add.
    status: verified
    last_checked: 2026-05-12

  - id: ac-005
    summary: Multi-input node chains correctly (NeighborList → RDF)
    type: code
    pass_when: |
      `pytest tests/test_compute/test_workflow.py::test_multi_input_node_rdf -v`
      exits 0; `results["rdf"]` is a `molrs.RDFResult` whose `.rdf` array has
      length == n_bins and contains at least one positive entry.
    status: verified
    last_checked: 2026-05-12

  - id: ac-006
    summary: No regression in existing molpy local tests
    type: code
    pass_when: |
      `pytest tests/ -m "not external" -v` exits 0; the count of passed
      tests is ≥ the pre-change baseline.
    status: verified
    last_checked: 2026-05-12

  - id: ac-007
    summary: molrs-compute test suite green after Graph removal
    type: code
    pass_when: |
      `cd /Users/roykid/work/molcrafts/molrs && cargo test -p molcrafts-molrs-compute`
      exits 0; no test references `Graph`, `Slot`, `Store`, `Inputs`, or `NodeId`.
    status: verified
    last_checked: 2026-05-12

  - id: ac-008
    summary: Clean clippy on molrs-compute with -D warnings
    type: code
    pass_when: |
      `cd /Users/roykid/work/molcrafts/molrs && cargo clippy
      -p molcrafts-molrs-compute -- -D warnings` exits 0.
    status: verified
    last_checked: 2026-05-12

  - id: ac-009
    summary: Graph re-exports and module declaration removed from lib.rs
    type: code
    pass_when: |
      `rg -nq 'pub use graph::|pub mod graph'
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/` exits 1
      (no matches).
    status: verified
    last_checked: 2026-05-12

  - id: ac-010
    summary: graph directory and graph_tests.rs / graph.rs benches deleted
    type: code
    pass_when: |
      `test ! -d /Users/roykid/work/molcrafts/molrs/molrs-compute/src/graph
      && test ! -f /Users/roykid/work/molcrafts/molrs/molrs-compute/tests/graph_tests.rs
      && test ! -f /Users/roykid/work/molcrafts/molrs/molrs-compute/benches/graph.rs`
      exits 0.
    status: verified
    last_checked: 2026-05-12

  - id: ac-011
    summary: Benches build after graph.rs removal
    type: code
    pass_when: |
      `cd /Users/roykid/work/molcrafts/molrs && cargo bench
      -p molcrafts-molrs-compute --no-run` exits 0; benchmarks.rs no longer
      contains `mod graph` (verify with `! rg -nq '\bmod graph\b'
      molrs-compute/benches/benchmarks.rs`).
    status: verified
    last_checked: 2026-05-12

  - id: ac-012
    summary: Tutorial page exists and mkdocs builds
    type: docs
    pass_when: |
      `test -f /Users/roykid/work/molcrafts/molpy/docs/user/compute/workflow.md`
      and `cd /Users/roykid/work/molcrafts/molpy && mkdocs build` exits 0;
      the rendered site contains a `workflow` page reachable from nav.
    status: verified
    last_checked: 2026-05-12

  - id: ac-013
    summary: No stale Graph references across either repo
    type: code
    pass_when: |
      `rg -n 'molrs_compute::Graph|molrs\.compute\.Graph'
      /Users/roykid/work/molcrafts/molpy/
      /Users/roykid/work/molcrafts/molrs/` returns no hits (rg exits 1).
    status: verified
    last_checked: 2026-05-12

  - id: ac-014
    summary: Stateless-Compute smoke audit on Rust analysis modules
    type: code
    confidence: medium
    evaluator_hint: "Loose grep — &mut self on helper methods (not the Compute trait impl) is acceptable; criterion flags surviving on-Compute mutation candidates for human review."
    pass_when: |
      `rg -nq '&mut self'
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/rdf
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/msd
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/cluster
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/kmeans
      /Users/roykid/work/molcrafts/molrs/molrs-compute/src/pca`
      exits 1; OR each surviving match has been annotated `// audit: ok —
      helper, not Compute impl` in the same file (manual review note).
    status: verified
    last_checked: 2026-05-12

  - id: ac-015
    summary: Pre-commit clean on molpy after the Python changes
    type: code
    pass_when: |
      `cd /Users/roykid/work/molcrafts/molpy && pre-commit run --all-files`
      exits 0.
    status: verified
    last_checked: 2026-05-12
---

# Acceptance criteria

This contract has 15 binding gates split across:

- **ac-001 – ac-006 / ac-012 / ac-015**: Python (molpy) side — Workflow
  exists, behaves per spec, no regressions, tutorial shipped, formatter
  clean.
- **ac-007 – ac-011 / ac-013 / ac-014**: Rust (molrs-compute) side —
  Graph removed cleanly, build/test/clippy green, no stale references
  in either repo, stateless audit recorded.

ac-014 (`confidence: medium`) is a deliberately loose grep — a positive
hit triggers a follow-up sub-task during `/mol:impl`, not a blocker.
The deeper per-module audit lives in the Phase-C "Audit each
per-analysis module" task.

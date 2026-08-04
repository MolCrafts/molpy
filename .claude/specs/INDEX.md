# Specs

## release-0-12-molpy

| Slug | Status |
|---|---|
| [release-0-12-molpy-compute-sink](release-0-12-molpy-compute-sink.md) | **done** |
| [release-0-12-molpy-01-api-cleanup](release-0-12-molpy-01-api-cleanup.md) | **done** |
| [release-0-12-molpy-05-docs-harness](release-0-12-molpy-05-docs-harness.md) | **done** (ac-005 doc-blocks: run when env has molrs) |

**Superseded / do not implement as written:**

| Slug | Why |
|---|---|
| `release-0-12-molpy-02-units-fs` | SI units live in molrs; do not re-fix prefactors in molpy |
| `release-0-12-molpy-03-charge-transport` | no molpy charge-weighting — use molrs `GreenKuboConductivity` / `EinsteinConductivity` |
| `release-0-12-molpy-04-compute-tests` | rewrite as identity asserts against molrs types |

> Closed historically: graph-assembler-01..04 + molrs-core-cutover (2026-07-22).

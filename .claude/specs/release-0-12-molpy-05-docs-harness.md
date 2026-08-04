---
slug: release-0-12-molpy-05-docs-harness
status: done
created: 2026-08-04
closed: 2026-08-04
grilled: false
depends_on:
  - release-0-12-molpy-01-api-cleanup
  - release-0-12-molpy-compute-sink
---

# release-0-12-molpy-05-docs-harness — docs + blueprint for 0.12

## Summary

Rewrite user-facing and agent-facing documentation so they match the post-parser-sink tree and 0.12 science/API contracts. Refresh `.claude/notes/architecture.md` inventory.

## Tasks

- [x] Refresh `.claude/notes/architecture.md` inventory to live packages
- [x] Rewrite README + CLAUDE package table + pin 0.12
- [x] Rewrite indexes / architecture-overview / glossary / external-tools
- [x] Fix VACF/compute index off `compute_acf` (compose / VACF)
- [x] Align testing.md / ci.md / docs-style / performance notes
- [x] Grep-gate dead surfaces documented in regression
- [x] Regression `regressions/release-0-12-molpy-05-docs-harness.md`

## Out of scope

- Tagging/publishing (release skill)
- molrs docs

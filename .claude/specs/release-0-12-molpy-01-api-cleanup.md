---
slug: release-0-12-molpy-01-api-cleanup
status: done
created: 2026-08-04
closed: 2026-08-04
grilled: false
---

# release-0-12-molpy-01-api-cleanup — delete dual APIs

## Summary

Delete remaining dual public names and factory constructors in molpy: `AmberResult.ff`, unused `get_topo` kwargs, and `get_packer`. Hard break for 0.12.

## Tasks

- [x] Dual names gone (`AmberResult.forcefield` only; no `.ff`)
- [x] `get_topo` has no `entity_type` / `link_type`
- [x] `get_packer` not exported from `molpy.pack`
- [x] Tests: `tests/test_release/test_api_cleanup_01.py`
- [x] Regression: `regressions/release-0-12-molpy-01-api-cleanup.py`

## Out of scope

- Science (02–03)
- Docs sweep (05)

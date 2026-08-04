---
spec: release-0-12-molpy-01-api-cleanup
created: 2026-08-04
criteria:
  - id: ac-001
    summary: AmberResult.ff removed
    type: code
    pass_when: "AmberResult has forcefield only; no .ff property"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-002
    summary: get_topo has no entity_type/link_type kwargs
    type: code
    pass_when: "get_topo signature has no entity_type or link_type parameters"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-003
    summary: get_packer removed
    type: code
    pass_when: "get_packer is not exported from molpy.pack"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-004
    summary: suite green
    type: runtime
    pass_when: "pytest tests/test_release/test_api_cleanup_01.py passes"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
out_of_scope:
  - Science unit migration
---

# Acceptance — release-0-12-molpy-01-api-cleanup

Dual APIs and factories listed are gone; callers use the single names.

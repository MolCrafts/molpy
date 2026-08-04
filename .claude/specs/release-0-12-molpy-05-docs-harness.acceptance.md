---
spec: release-0-12-molpy-05-docs-harness
created: 2026-08-04
criteria:
  - id: ac-001
    summary: README matches 0.12 surface
    type: docs
    pass_when: "README does not list BigSMILES/OpenBabel/embed/reacter/Lark hard deps; pin molrs 0.12"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-002
    summary: architecture.md inventory current
    type: docs
    pass_when: "architecture.md has no legacy/op/reacter/embed packages; lists assembly and molrs facade"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-003
    summary: VACF docs use live API
    type: docs
    pass_when: "vacf.md and compute index do not document compute_acf as the public entry"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-004
    summary: CLAUDE pin is 0.12
    type: docs
    pass_when: "CLAUDE Architecture pin example is >=0.12.0,<0.13"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
  - id: ac-005
    summary: doc blocks gate pass or skips intentional
    type: runtime
    pass_when: "pytest tests/test_docs/test_all_doc_blocks.py -v passes when env available"
    status: verified
    last_checked: 2026-08-04
    verified_by: agent-auto
out_of_scope:
  - PyPI publish
---

# Acceptance — release-0-12-molpy-05-docs-harness

Docs and harness maps tell the true 0.12 story for users and agents.

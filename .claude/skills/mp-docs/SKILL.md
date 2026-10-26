---
name: mp-docs
description: Updates the molpy manual, rustdoc, docstrings or stub text for an API or topic; writes docs files.
argument-hint: "<page, API name or topic>"
user-invocable: true
---

# Docs

Read AGENTS.md for molpy conventions.

## Procedure

1. Resolve scope: pages and docstrings naming the API or topic
   (`rg` across `docs/`, `README.md`, `molpy/examples/`, `crates/`).
2. Delegate to `mp-documenter` to write or update them.
3. Confirm: every python block on touched pages runs, every `mp.…` name
   resolves, `scripts/check.sh docs` passes.
4. Report any API bug found while documenting; do not work around it.

## Output

`docs: N pages updated, build strict-clean, N code bugs reported`

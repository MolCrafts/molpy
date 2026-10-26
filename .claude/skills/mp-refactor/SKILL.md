---
name: mp-refactor
description: Restructures code without changing behaviour (moves, renames, core sinks, deletions); writes code, tests and docs that name moved symbols.
argument-hint: "<scope and goal>"
user-invocable: true
---

# Refactor

Read AGENTS.md for molpy conventions.

## Procedure

1. State the invariant to preserve (public behaviour, outputs, error types).
2. Delegate to `mp-architect` for the target layout before moving anything.
3. Run the relevant gates to record the green baseline.
4. Move/rename/delete in one sweep; update every caller, test, doc page and
   stub in the same change (no aliases or shims).
5. Re-run the same gates; behaviour must be unchanged.
6. Delegate to `mp-architect` on the result; for binding-facing moves also
   `mp-bindings` (one message).
7. `rg` the old names across the repo; zero hits outside history notes.

## Output

`refactor: <scope> — N files moved/renamed, old names 0 hits, gates green`

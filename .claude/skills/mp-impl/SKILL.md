---
name: mp-impl
description: Implements a spec link or feature end to end — failing tests first, then code, tests and docs; writes source, tests and docs.
argument-hint: "<spec slug[/link]> or <feature>"
user-invocable: true
---

# Implement

Read AGENTS.md for molpy conventions and .claude/NOTES.md for open decisions.

## Procedure

1. Load the spec link (or restate the feature); list acceptance checks.
2. Scope check — delegate to `mp-architect` with the planned file list;
   resolve every violation before writing code.
3. RED — delegate to `mp-tester` to write failing tests for each acceptance
   behaviour; confirm each fails for the intended reason.
4. GREEN — implement the smallest change that passes, in the layer the
   architect named (core first, then each binding that exposes it).
5. Run the gates the change touches (`scripts/check.sh` with the relevant
   gates; `verify` before handing back a multi-layer change).
6. Post-implementation check — delegate to `mp-architect` on the diff; for
   binding changes also `mp-bindings`, for numeric changes `mp-science`
   (all in one message).
7. Docs — delegate to `mp-documenter` for every page, docstring and stub
   naming the changed API.
8. Update the spec's state in `.claude/specs/INDEX.md`; capture any new
   decision with `mp-note`.

## Output

`impl: <slug> — N files, gates <list> green, findings fixed N / open N`

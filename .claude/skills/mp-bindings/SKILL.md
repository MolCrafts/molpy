---
name: mp-bindings
description: Single-axis bindings deep dive on a path or diff via the mp-bindings agent; read-only.
argument-hint: "[path | diff range, default: dev...HEAD]"
user-invocable: true
---

# Bindings review

Read AGENTS.md for molpy conventions.

## Procedure

1. Resolve scope: the given path, or `git diff --name-only dev...HEAD`.
2. Delegate to `mp-bindings` with the scope.
3. Verify each CRITICAL/HIGH finding by reading the cited line.

## Output

The agent's `[SEVERITY] file:line — message` lines, then
`bindings: N findings (C/H/M/L)`.

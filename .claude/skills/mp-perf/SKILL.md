---
name: mp-perf
description: Single-axis performance deep dive on a path or diff via the mp-perf agent; read-only.
argument-hint: "[path | diff range, default: dev...HEAD]"
user-invocable: true
---

# Performance review

Read AGENTS.md for molpy conventions.

## Procedure

1. Resolve scope: the given path, or `git diff --name-only dev...HEAD`.
2. Delegate to `mp-perf` with the scope.
3. Verify each CRITICAL/HIGH finding by reading the cited line.

## Output

The agent's `[SEVERITY] file:line — message` lines, then
`performance: N findings (C/H/M/L)`.

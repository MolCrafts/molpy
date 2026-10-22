---
name: mp-review
description: Runs a parallel multi-axis review of a diff or path and renders a severity table with a verdict; read-only.
argument-hint: "[path | diff range, default: dev...HEAD]"
user-invocable: true
---

# Review

Read AGENTS.md for molpy conventions.

## Procedure

1. Resolve scope: the given path, or `git diff --name-only dev...HEAD`.
2. Fan out in ONE message to `mp-architect`, `mp-science`, `mp-perf`,
   `mp-bindings` and `mp-tester` (audit mode), each with the same scope.
   Add `mp-documenter` (audit mode) when public API or docs changed.
3. Merge findings; drop duplicates (keep the highest severity); verify each
   CRITICAL/HIGH by reading the cited line.
4. Verdict: BLOCK if any CRITICAL; REQUEST CHANGES if any HIGH; else APPROVE.

## Output

| Severity | Count | Findings |
|---|---|---|
| Critical | | |
| High | | |
| Medium | | |
| Low | | |

`review: <verdict> — C/H/M/L = a/b/c/d`

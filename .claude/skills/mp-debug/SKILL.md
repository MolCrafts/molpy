---
name: mp-debug
description: Diagnoses a symptom to a root cause and never edits any file.
argument-hint: "<symptom, error text or failing command>"
user-invocable: true
---

# Debug (diagnose only)

Read AGENTS.md for molpy conventions.

This skill NEVER edits, writes or deletes repository files. Scratch
reproducers go in the session scratchpad only.

## Procedure

1. Reproduce the symptom; capture the exact output.
2. Form hypotheses; test each with a read, a grep or a scratch run — one
   variable at a time.
3. Narrow to the root cause with `path:line` evidence and the input that
   triggers it.
4. Name the fix location and the regression test that would catch it, then
   hand off to `mp-fix` (do not apply it).

## Output

`debug: root cause <file:line> — <one clause>; reproducer <command>; next: /mp-fix`

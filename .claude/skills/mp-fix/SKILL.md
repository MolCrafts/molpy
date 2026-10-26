---
name: mp-fix
description: Fixes one bug with a minimal diff after a regression test reproduces it; writes code and one test.
argument-hint: "<bug description or failing command>"
user-invocable: true
---

# Fix

Read AGENTS.md for molpy conventions.

## Procedure

1. Reproduce: run the failing command or write a scratch reproducer; record
   the exact error.
2. Locate the root cause (inline diagnosis; read the code path, do not guess).
3. Delegate to `mp-tester` for one regression test that fails now for this
   reason.
4. Apply the minimal fix at the root cause — no refactors, no unrelated
   cleanups, no special case that only satisfies the test.
5. Run the regression test, then the gates for the touched layer.
6. If the same defect class appears elsewhere, list the locations; do not
   fix them silently.

## Output

`fix: <root cause in one clause> — <file:line>; regression test <path>; gates green`

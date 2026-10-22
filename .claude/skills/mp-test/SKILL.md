---
name: mp-test
description: Audits or extends test coverage for a path — finds tests that cannot fail and missing behaviours; writes test files only.
argument-hint: "<path or feature> [--audit-only]"
user-invocable: true
---

# Test

Read AGENTS.md for molpy conventions.

## Procedure

1. Resolve scope: source files and their tests.
2. Delegate to `mp-tester` in audit mode: which tests cannot fail, which
   behaviours (happy, edge, failure) have no test.
3. Unless `--audit-only`, delegate to `mp-tester` to write the missing tests;
   each must be shown RED against a deliberately broken build, then restored.
4. Run the affected test gates.

## Output

`test: N tests added, N cannot-fail tests found, gates green`

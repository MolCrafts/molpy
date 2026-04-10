---
name: molpy-test
description: Test coverage analysis, gap identification, and scientific validation audit. Use after implementing a feature or before release.
argument-hint: "[path or module]"
user-invocable: true
---

Analyze test coverage for: $ARGUMENTS

If no path given, analyze the entire project.

**Step 1 — Run Tests**

```bash
# Full project
pytest tests/ --cov=src/molpy --cov-report=term-missing -v -m "not external"

# Specific package
pytest tests/test_<pkg>/ --cov=src/molpy/<pkg> --cov-report=term-missing -v
```

**Step 2 — Coverage Analysis**

Report coverage by package. Flag any module below 80% (90% for core/).

**Step 3 — Scientific Test Audit**

For each module implementing a physical model (in potential/, compute/, typifier/):
- [ ] Numerical validation against known analytical values
- [ ] Unit consistency test
- [ ] Limiting case tests (r→∞, r→0, single atom)
- [ ] Force consistency: F = -dV/dr (numerical gradient check)
- [ ] Immutability test (input not mutated)

For each I/O module:
- [ ] Round-trip test (write → read → compare)
- [ ] Malformed input handling

For each parser:
- [ ] Known molecules parse correctly
- [ ] Invalid input produces clear errors
- [ ] Round-trip (parse → serialize → parse → compare)

**Step 4 — Report**

```
TEST COVERAGE REPORT

Overall: XX% (target: 80%)

By Package:
  core/:       XX% ✅/⚠️  (target: 90%)
  io/:         XX% ✅/⚠️
  parser/:     XX% ✅/⚠️
  potential/:  XX% ✅/⚠️
  compute/:    XX% ✅/⚠️
  builder/:    XX% ✅/⚠️
  typifier/:   XX% ✅/⚠️

Scientific Validation:
  ✅ <module>: <tests present>
  ❌ <module>: <missing tests>

Suggested Tests:
  1. <file>: <what to test>
```

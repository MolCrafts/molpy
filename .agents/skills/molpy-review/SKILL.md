---
name: molpy-review
description: Comprehensive code review aggregating architecture, performance, documentation, scientific correctness, and quality checks. Use after writing code or during PR review.
argument-hint: "[path or module]"
user-invocable: true
---

Review code for: $ARGUMENTS

If no path given, review all files modified in `git diff --name-only HEAD`.

**Invoke all dimensions in parallel:**

1. **Architecture** → invoke `/molpy-arch` on $ARGUMENTS
2. **Performance** → invoke `/molpy-perf` on $ARGUMENTS
3. **Documentation** → invoke `/molpy-docs` on $ARGUMENTS
4. **Scientific Correctness** (for potential/, compute/, typifier/, core/ops/):
   - Equations match cited paper
   - Units documented and consistent throughout
   - Physical limiting cases satisfied (V(r→∞)→0, F(r_eq)=0)
   - Convention documented when ambiguous (LAMMPS K vs standard k/2)
   - Force consistency: F = -dV/dr
   - Combining rules match declared force field
5. **Code Quality** (inline):
   - Functions < 50 lines, files < 800 lines
   - No deep nesting (> 4 levels)
   - No hardcoded magic numbers
   - Type annotations on all public APIs
   - Google-style docstrings with units
   - Black formatting (88 char lines)
6. **Immutability** (inline):
   - No mutation of input `Atomistic`, `ForceField`, `Entity`, `Link` objects
   - Transformations return new objects
   - No `list.append()` on input collections
   - `copy()` before modification

**Severity levels**:
- CRITICAL — must fix (architecture violations, scientific errors)
- HIGH — should fix (missing tests, performance issues)
- MEDIUM — fix when possible (style, documentation gaps)
- LOW — nice to have

**Output**: Merged report:
```
CODE REVIEW: <path>
ARCHITECTURE: ✅/❌ per check
PERFORMANCE: ✅/⚠️ per check
DOCUMENTATION: ✅/⚠️ per check
SCIENTIFIC CORRECTNESS: ✅/❌ per check
CODE QUALITY: ✅/⚠️ per check
IMMUTABILITY: ✅/❌ per check
SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
```

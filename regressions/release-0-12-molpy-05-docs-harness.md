# Regression — release-0-12-molpy-05-docs-harness

Grep gates (must stay empty / intentional):

```bash
# Forbidden as live public docs / package claims
! grep -RInE 'compute_acf|get_packer|IonicConductivity|DielectricSusceptibility|ACFAnalyzer' docs/ src/molpy/compute/ || true
# BigSMILES / OpenBabel not listed as hard parser/adapter surfaces in README package table
! grep -nE 'BigSMILES|OpenBabel' README.md | grep -vE 'not |removed|no ' || true
# Pin
grep -n 'molcrafts-molrs>=0.12.0,<0.13' pyproject.toml CLAUDE.md
```

Verified 2026-08-04:

- README capabilities match live packages; deps: numpy + molrs 0.12 line.
- `docs/compute/msd.md` / `docs/compute/index.md` use VACF / compose API (no `compute_acf`).
- `.claude/notes/architecture.md` inventory rewritten for post-parser-sink tree.
- CLAUDE Architecture pin example `>=0.12.0,<0.13`.

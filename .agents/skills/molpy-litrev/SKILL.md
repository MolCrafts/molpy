---
name: molpy-litrev
description: Literature review to verify scientific basis before implementation. Use before implementing any potential, force field parameter set, compute operator, or typifier rule.
argument-hint: <method or topic name>
user-invocable: true
---

Perform a literature review for: $ARGUMENTS

**Step 1 — Search Literature**
Search arXiv, journals, and code repositories:
- `"$ARGUMENTS" molecular dynamics force field`
- `"$ARGUMENTS" computational chemistry`
- `"$ARGUMENTS" molecular simulation`
Find the original publication and follow-up papers.

**Step 2 — Extract Key Information**
From the primary paper:
- **Equations**: All key mathematical formulations with exact notation
- **Parameters**: Default values, valid ranges, units
- **Conventions**: Which convention is used (LAMMPS vs GROMACS vs AMBER)
- **Approximations**: What is approximated and why
- **Limitations**: Known failure modes or accuracy limits

**Step 3 — Find Reference Implementations**
Search for existing code:
- LAMMPS source (pair_style, bond_style implementations)
- OpenMM / GROMACS reference code
- AmberTools source (for force field parameters)
- RDKit / OpenBabel (for SMARTS patterns and atom typing)

**Step 4 — Identify Validation Targets**
From papers or reference code:
- Known analytical values for simple systems (e.g., argon LJ, SPC/E water)
- Published benchmark results
- Reference parameter values (bond lengths, angles, force constants)

**Step 5 — Report**
Output:

```markdown
# Literature Review: <Method Name>

## Primary Reference
- **Paper**: Author et al., "Title", Journal Year
- **DOI**: URL

## Key Equations
[equations from the paper, with units]

## Convention Notes
[which software convention is followed, any factor-of-2 issues]

## Reference Implementations
- [code/software] — [notes on implementation details]

## Validation Targets
- Simple system: expected numerical values
- Parameter values: published reference data

## Known Limitations

## Recommendations for MolPy
- Which existing base classes to use
- Unit convention to follow
- Numerical stability concerns
```

If no credible paper is found, report clearly and ask the user for a reference.

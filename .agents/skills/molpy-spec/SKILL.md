---
name: molpy-spec
description: Convert natural language requirements into a detailed technical specification with literature grounding. Use before implementing new features. Supports Chinese and English.
argument-hint: <natural language description>
user-invocable: true
---

Generate a detailed technical spec for: $ARGUMENTS

**Step 1 — Understand Intent**
Parse the description to identify: what capability is added/changed, which packages are affected (core, io, parser, builder, typifier, compute, reacter, potential, pack, engine, wrapper, adapter), what new classes or functions are needed.

**Step 2 — Literature Search**
For any physical model, potential, or algorithm:
- Search for the original paper and extract key equations
- Identify unit conventions and known approximations
- Find reference implementations (LAMMPS, GROMACS, AmberTools, RDKit)
- Document expected validation targets

**Step 3 — Codebase Analysis**
Read relevant existing code to understand:
- 9-layer architecture and where the new code fits
- Existing similar implementations for consistency
- Base classes to inherit from
- Data model (Entity/Link/Struct, Block/Frame)
- ForceField I/O patterns (formatter registry)

**Step 4 — Generate Spec**
Produce a structured markdown document:

```markdown
# Spec: <Feature Name>

## Summary
One-paragraph description.

## Scientific Basis (if applicable)
- Paper: [Author et al., "Title", Journal Year](DOI)
- Key equations (with units)
- Convention followed (LAMMPS/GROMACS/AMBER)
- Known approximations

## Design

### New Types
Classes with constructor signatures, method signatures, type hints. Include units for all physical quantities.

### Modified Types
Existing types that need changes (before/after).

### Module Changes
| File | Action | Description |

### Layer Validation
Which layer, what can/cannot be imported.

## Data Flow
How data moves through the new code (input → transform → output).

## Dependencies
Internal modules, external packages (optional or required).

## Testing Strategy
Unit tests, scientific validation, round-trip tests, edge cases. Estimated count and coverage target.

## Performance Considerations
Algorithm complexity, vectorization opportunities, memory scaling.
```

**Step 5 — Save or Present**
If an OpenSpec change is appropriate, scaffold under `openspec/changes/<kebab-case-name>/`.
Otherwise present the spec inline for review.

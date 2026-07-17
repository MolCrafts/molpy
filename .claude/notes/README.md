# .claude/notes/

Passive project knowledge — kept across features, consumed by agents during spec/impl/review.

- `notes.md` — evolving decisions captured by `/mol:note`
- `architecture.md` — project blueprint, populated by `/mol:map`, consumed by `librarian` during `/mol:spec`
- `open-questions.md` — things uncertain during bootstrap; fill over time
- `performance.md` — NumPy/algorithm performance standards (hot paths, vectorization, memory)
- `testing.md` — coverage targets + scientific validation standards (units, physical limits)
- `docs-style.md` — docstring/tutorial standards (agent-first vs human-first, unit conventions)
- `ci.md` — CI / pre-commit parity policy and intentional exemptions
- `crosslinking-syntax-design.md`, `incremental-typification-design.md` — design narratives
- `assembly-guide-draft.md` — narrative for the `graph-assembler` chain's API; **ships as
  `docs/user-guide/02_assembly.md`** in `graph-assembler-03-purge` T11. Read this first to
  understand `GraphAssembler` / `TypeScope` / `RegionPatch` before reading the specs.
- `polymer-topology-editing-design.md` — catalog of topology/network editing from a shared
  PEO monomer kit (`PolymerBuilder` + `GraphAssembler` + Selectors); example suite + PR plan.

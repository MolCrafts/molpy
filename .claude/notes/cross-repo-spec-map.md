# molpy ↔ molrs

**Release order (2026-07-22):** molrs **first**, then molpy. Master landings
for a version need a **tag**. No monorepo merge; no pin-parity scripts.
See `.claude/notes/release.md`.

## Live specs

| molrs | molpy |
|-------|-------|
| (see molrs `.claude/specs/INDEX.md`) | (see molpy `.claude/specs/INDEX.md`) |

## Version

Exact pin in `pyproject.toml`: `molcrafts-molrs==X.Y.Z` must match a **published**
molrs tag, not only a local rebuild with the same version string.

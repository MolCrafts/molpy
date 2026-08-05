# molpy ↔ molrs

**Release order (2026-07-22):** molrs **first**, then molpy. Master landings
for a version need a **tag**. No monorepo merge; no pin-parity scripts.
See `.claude/notes/release.md`.

## Live specs

| molrs | molpy |
|-------|-------|
| (see molrs `.claude/specs/INDEX.md`) | (see molpy `.claude/specs/INDEX.md`) |
| **smiles-emit-01-ir-write** → **02-from-atomistic** → **03-local-smarts** → **04-python** | **smiles-emit-01-io-surface** (after molrs tag + pin) |

### smiles-emit cross-repo contract

- **Order:** molrs chain fully landed + **tagged publish** → bump molpy `molcrafts-molrs` pin → molpy io surface.
- **Dependency direction:** `io` / `parser` → `core` only. **Forbidden:** `Atomistic.to_smiles` / `from_smiles` / `to_smarts` on core (either repo).
- **Flags:** all science/representation knobs are explicit options (`SmilesEmitOptions`, `LocalSmartsOptions` / kwargs); no silent policy in core.
- **Engine ownership:** parse/write/local SMARTS live in molrs `io::smiles`; molpy only delegates.

## Version

Minor-line pin in `pyproject.toml`: `molcrafts-molrs>=X.Y.0,<X.(Y+1)` must
include a **published** molrs release on that minor, not only a local rebuild
with a matching version string. Runtime check is major.minor only.

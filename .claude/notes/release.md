# Release discipline — molrs then molpy

**Supersedes:** monorepo merge discussion (discarded); automated pin-parity
scripts / pre-commit API probes (forbidden).

## Rule (agents must obey)

1. **Order:** Always ship **molrs first**, then **molpy**. Never land molpy
   code that needs molrs APIs not already on the **published** pin
   (`molcrafts-molrs==X.Y.Z` on the package index).
2. **Master carries a tag:** A commit that is allowed onto `master` for a
   versioned release must be **reachable from a version tag** (`vX.Y.Z`).
   Do not treat untagged tip-of-master as a released molrs binary for molpy.
3. **No release scripts for this.** Do not add `scripts/check_*_pin*`, CI-only
   git-install hacks, or pre-commit hooks that reinstall wheels to “guess”
   PyPI. **Manual checklist every time**, enforced by agent process (this
   note + CLAUDE.md), not by more automation.
4. **Local maturin ≠ release.** An editable/local rebuild that reuses the
   same version string as PyPI is **not** a release. Agents must not claim
   “pin satisfied” from a local build alone.

## Manual checklist (operator or agent before molpy lands)

- [ ] molrs change merged to its `master` **and** tagged `vX.Y.Z`
- [ ] Published artifacts for that tag are live (crates.io / PyPI as applicable)
- [ ] molpy `pyproject.toml` exact pin == that `X.Y.Z`
- [ ] molpy `version` matches the shared version line when co-releasing
- [ ] Clean install path (or CI log) proves tests against **index** pin, not only editable

## Agent hard-stops

- About to use a molrs symbol not in the published pin → **stop**, release
  molrs first (or drop the dependency).
- About to push molpy that only passes with local molrs → **stop**.
- About to invent a script/hook to paper over pin drift → **stop**; update
  this note only if the *rule* changes.

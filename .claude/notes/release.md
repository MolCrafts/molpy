# Release discipline — molrs then molpy

**Supersedes:** monorepo merge discussion (discarded); the 2026-07-22 ban on
*any* pin automation (too weak — editable hid missing PyPI APIs on 2026-07-29).

## Rule (agents must obey)

1. **Order:** Always ship **molrs first**, then **molpy**. Never land molpy
   code that needs molrs APIs not already on the **published** pin
   (`molcrafts-molrs==X.Y.Z` on the package index).
2. **Master carries a tag:** A commit that is allowed onto `master` for a
   versioned release must be **reachable from a version tag** (`vX.Y.Z`).
   Do not treat untagged tip-of-master as a released molrs binary for molpy.
3. **Pre-push enforces pin existence on PyPI.** Hook `molrs-pin-on-pypi` in
   `.pre-commit-config.yaml` (pre-push only; script under `.pre-commit/`) fails
   if the exact `molcrafts-molrs==X.Y.Z` pin is not on the index. That is the
   **only** place for this check — not tox, not conftest.
4. **No papering over.** Do not add git-install hacks, path deps, or
   `language: system` pytest that can see sibling molrs. Do not claim
   “pin satisfied” from a local rebuild alone.

## Manual checklist (operator or agent before molpy lands)

- [ ] molrs change merged to its `master` **and** tagged `vX.Y.Z`
- [ ] Published artifacts for that tag are live (crates.io / PyPI as applicable)
- [ ] molpy `pyproject.toml` exact pin == that `X.Y.Z`
- [ ] molpy `version` matches the shared version line when co-releasing
- [ ] Local `prek run molrs-pin-on-pypi --all-files --hook-stage pre-push` passes
- [ ] `tox -e py` green against **index** pin (not only editable)

## Agent hard-stops

- About to use a molrs symbol not in the published pin → **stop**, release
  molrs first (or drop the dependency).
- About to push molpy that only passes with local molrs → **stop**.
- About to reintroduce `language: system` pytest as the CI mirror → **stop**.

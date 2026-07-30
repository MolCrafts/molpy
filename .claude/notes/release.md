# Release discipline — molrs then molpy

**Supersedes:** monorepo merge discussion (discarded); the 2026-07-22 ban on
*any* pin automation (too weak — editable hid missing PyPI APIs on 2026-07-29);
exact `==X.Y.Z` pin + exact runtime match (superseded by minor-line policy).

## Rule (agents must obey)

1. **Order:** Always ship **molrs first**, then **molpy**. Never land molpy
   code that needs molrs APIs not already on a **published** minor line
   (`molcrafts-molrs>=X.Y.0,<X.(Y+1)` on the package index).
2. **Master carries a tag:** A commit that is allowed onto `master` for a
   versioned release must be **reachable from a version tag** (`vX.Y.Z`).
   Do not treat untagged tip-of-master as a released molrs binary for molpy.
3. **Pre-push enforces a published minor line on PyPI.** Hook `molrs-pin-on-pypi`
   in `.pre-commit-config.yaml` (pre-push only; script under `.pre-commit/`)
   fails if no published molrs release matches the minor range in
   `pyproject.toml`. That is the **only** place for this check — not tox, not
   conftest.
4. **Runtime = major.minor only.** `check_molrs_version()` accepts patch drift
   within the same minor; a different major or minor fails import.
5. **No papering over.** Do not add git-install hacks, path deps, or
   `language: system` pytest that can see sibling molrs. Do not claim
   “pin satisfied” from a local rebuild alone.
6. **No hand-written CHANGELOG.** History is git tags / GitHub Releases.

## Manual checklist (operator or agent before molpy lands)

- [ ] molrs change merged to its `master` **and** tagged `vX.Y.Z`
- [ ] Published artifacts for that tag are live (crates.io / PyPI as applicable)
- [ ] molpy `pyproject.toml` minor range includes that `X.Y.*`
      (`>=X.Y.0,<X.(Y+1)`)
- [ ] molpy `version` shares major.minor when co-releasing
- [ ] Local `prek run molrs-pin-on-pypi --all-files --hook-stage pre-push` passes
- [ ] `tox -e py` green against **index** pin (not only editable)

## Agent hard-stops

- About to use a molrs symbol not in the published minor line → **stop**, release
  molrs first (or drop the dependency).
- About to push molpy that only passes with local molrs → **stop**.
- About to reintroduce `language: system` pytest as the CI mirror → **stop**.
- About to reintroduce exact `==X.Y.Z` pin or exact-patch runtime gate → **stop**.

# Release Process

This page is the practical checklist for cutting a MolPy release.


## Version source of truth

Version metadata lives in `src/molpy/version.py`. Update both fields before release:

```python
version = "X.Y.Z"
release_date = "YYYY-MM-DD"
```


## Pre-release checks

Run all three validation steps locally before creating the release branch.

```bash
pytest tests/ -v -m "not external"    # tests pass
zensical build                         # docs build
python -m build && twine check dist/*  # package is valid
```


## Release workflow

**master is branch-protected**, and the release/publish workflow refuses a tag
that is not reachable from `master`. So the release commit must land on `master`
**before** the tag is pushed — otherwise you get an orphan tag and the publish
job fails. Order matters:

```bash
# 1. Bump version.py + CHANGELOG on dev, commit.
# 2. Get the release commit onto master via a PR (direct pushes are rejected):
gh pr create --base master --head dev --title "Release vX.Y.Z"
gh pr merge --merge            # after checks pass

# 3. Only after master has the release commit, tag it and push the tag:
git fetch main master
git tag -a vX.Y.Z -m "Release vX.Y.Z"    # on the merged master commit
git push main vX.Y.Z
```

Do **not** `git push <remote> master --tags`: if the protected-master push is
rejected, the tag still goes out as an orphan and publish refuses it.

On tag push (`v*`), GitHub Actions runs `.github/workflows/release.yml`. It
validates the tag against `molpy.version.version`, runs the test suite, builds
artifacts, and publishes to PyPI.


## Nightly releases

Nightlies are **independent** of the tagged release flow above. They ship to a
separate PyPI project, `molcrafts-molpy-nightly`, and never touch the stable
`molcrafts-molpy`.

- **Trigger:** every push to the `nightly` branch, or a manual run of the
  *Nightly* workflow (`.github/workflows/nightly.yml`) via
  `workflow_dispatch`.
- **Versioning:** the workflow reads the current `molpy.version.version` and
  appends a UTC timestamp → `X.Y.Z.dev<YYYYMMDDHHMM>` (a PEP 440 dev release).
  No manual version bump or tag is needed; do **not** edit `version.py` for a
  nightly.
- **Distribution rename:** the build rewrites the PyPI name to
  `molcrafts-molpy-nightly` in-flight (the commit on `nightly` is unchanged).
- **Publishing:** PyPI Trusted Publishing (OIDC) into the `pypi-nightly`
  GitHub Environment — no API token, no required reviewers (so nightlies never
  block on manual approval).

To cut a nightly, fast-forward `nightly` to the commit you want and push:

```bash
git push origin master:nightly      # or push your integration branch onto nightly
```

Install a nightly with `pip install --pre molcrafts-molpy-nightly`. It imports
as `molpy` and therefore conflicts with the stable package — test it in a
dedicated virtual environment.


## Release notes

Use this structure on the [GitHub Releases page](https://github.com/MolCrafts/molpy/releases):

```markdown
## MolPy vX.Y.Z

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Breaking Changes
- ... (or "None")
```


## Hotfix

For critical fixes on a released version:

```bash
git checkout -b hotfix/vX.Y.Z vA.B.C
# fix, test, update version.py
git commit -am "fix: ..."
git tag -a vX.Y.Z -m "Hotfix vX.Y.Z"
git push origin --tags
```

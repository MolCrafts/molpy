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
mkdocs build                           # docs build
python -m build && twine check dist/*  # package is valid
```


## Release workflow

Create a release branch, update the version, and merge back to master with a tag.

```bash
git checkout master && git pull origin master
git checkout -b release/vX.Y.Z

# update version.py, commit
git add src/molpy/version.py
git commit -m "chore(release): vX.Y.Z"

git checkout master
git merge release/vX.Y.Z
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin master --tags
```

On tag push (`v*`), GitHub Actions runs `.github/workflows/release.yml`. It validates the tag against `molpy.version.version`, runs the test suite, builds artifacts, and publishes to PyPI.


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

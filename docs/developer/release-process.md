# Release Process

This page is the practical checklist for cutting a MolPy release.


## Version source of truth

Version metadata lives in `src/molpy/version.py`. Update both fields before release:

```python
version = "X.Y.Z"
release_date = "YYYY-MM-DD"
```

### molrs co-release: major.minor only

MolPy and [molrs](https://github.com/MolCrafts/molrs) share a **major.minor**
line when co-released. Patch may drift.

| Mechanism | Rule |
|-----------|------|
| `pyproject.toml` | `molcrafts-molrs>=X.Y.0,<X.(Y+1)` (not `==X.Y.Z`) |
| Import-time check | `check_molrs_version()` accepts any installed molrs with the same major.minor |
| Release gate | `release.yml` runs `scripts/check_partners_on_pypi.py` (a published release in every partner range: molrs on that minor line, mollog and molcfg in theirs), then the test matrix against the partners from PyPI (`uv run --no-sources`) |

**Order:** ship molrs first (`master` + tag `vX.Y.Z` + publish), then land molpy
APIs that need the new surface. Editable local molrs does not count as a release.
There is no hand-written `CHANGELOG.md` and no release-notes page; the history
is git.

### development pins partner commits; a release tests PyPI

Development builds use the molrs commit pinned in `.github/partners.env`: `pyproject.toml`'s
`[tool.uv.sources]` builds `molcrafts-molrs` from the sibling `../molrs`, which
CI and the hooks check out at the commit `scripts/partners.py` resolves
(`.github/partners.env`; see [Development Setup](development-setup.md)). That
table only steers uv; the wheel declares the minor-line range alone.

mollog and molcfg are partners too, pinned to full commits
(`[tool.uv.sources]` builds them from `../mollog` / `../molcfg`) and declared
by major range (`molcrafts-mollog>=1.3.1,<2`). A molpy that needs a new
mollog or molcfg API waits for that release: mollog and molcfg ship before
molpy, and molpy's lower bound moves to it.

A release keeps the table and is judged against PyPI instead: `release.yml`
checks that every declared partner range has a published release
(`scripts/check_partners_on_pypi.py`) and runs the test matrix with
`uv run --no-sources`, i.e. against the partners from PyPI, exactly what
users of the wheel get. `uv.lock` (which records the dev build's partners) is
not used there.



## Pre-release checks

Run all three validation steps locally before creating the release branch.

```bash
uv run --extra dev python -m pytest tests/ -n auto   # tests pass
uv run --extra doc zensical build --clean            # docs build
python -m build && twine check dist/*                # package is valid
```


## Release workflow

**master is branch-protected**, and the release/publish workflow refuses a tag
that is not reachable from `master`. So the release commit must land on `master`
**before** the tag is pushed — otherwise you get an orphan tag and the publish
job fails. Order matters:

```bash
# 1. Bump version.py on dev, commit (history lives in git, no CHANGELOG).
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
checks that the tag is `v` + `molpy.version.version` on `master`, runs the
test matrix against molrs from PyPI, builds the sdist and wheel, publishes
them to PyPI, and creates the GitHub Release. Dispatching **release** on a
branch (a fork is fine) is the dry run: the same tests and build, no upload.
PyPI's trusted publisher names `release.yml` and the `pypi` environment.


## Nightly releases

Nightlies are **independent** of the tagged release flow above. They ship to a
separate PyPI project, `molcrafts-molpy-nightly`, and never touch the stable
`molcrafts-molpy`.

- **Trigger:** every push to the `nightly` branch of MolCrafts/molpy, or a
  dispatch of `.github/workflows/nightly.yml` on that branch. (Its scheduled
  run on `master` only measures tests and coverage for the CI dashboard.)
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


## Hotfix

For critical fixes on a released version:

```bash
git checkout -b hotfix/vX.Y.Z vA.B.C
# fix, test, update version.py
git commit -am "fix: ..."
git tag -a vX.Y.Z -m "Hotfix vX.Y.Z"
git push origin --tags
```

# Release Process

This document describes the complete process for making a MolPy release.

## Release Goals

- **Predictable** - Releases follow a consistent process
- **Documented** - Users can see what changed
- **Stable** - Releases are well-tested
- **Semantic** - Version numbers convey meaning

## Versioning

MolPy follows [Semantic Versioning](https://semver.org/) (SemVer):

```
MAJOR.MINOR.PATCH
```

- **MAJOR** - Incompatible API changes (e.g., 1.0.0 → 2.0.0)
- **MINOR** - New features, backwards-compatible (e.g., 1.0.0 → 1.1.0)
- **PATCH** - Bug fixes, backwards-compatible (e.g., 1.0.0 → 1.0.1)

### Version Examples

- `0.1.0` → `0.2.0` - Added new IO module (minor)
- `0.2.0` → `0.2.1` - Fixed PDB reader bug (patch)
- `0.9.0` → `1.0.0` - First stable release (major)
- `1.5.0` → `2.0.0` - Changed Frame API (major, breaking)

### Pre-release Versions

For development versions:

- `1.0.0a1` - Alpha release
- `1.0.0b1` - Beta release
- `1.0.0rc1` - Release candidate

## Release Checklist

### 1. Pre-release Preparation

**Code Quality:**
- [ ] All tests pass on all supported Python versions
- [ ] Coverage meets threshold (>80%)
- [ ] No critical bugs in issue tracker
- [ ] Code style checks pass (Black, isort)
- [ ] Type hints are complete

**Documentation:**
- [ ] Documentation builds without errors
- [ ] All examples run successfully
- [ ] API documentation is up-to-date
- [ ] Changelog is updated
- [ ] Migration guide written (if breaking changes)

**Testing:**
- [ ] Run full test suite: `pytest`
- [ ] Test on multiple platforms (Linux, macOS, Windows)
- [ ] Test with minimum and maximum supported Python versions
- [ ] Manual testing of key workflows
- [ ] Test installation from source

### 2. Update Version Number

Update version in `src/molpy/version.py`:

```python
__version__ = "0.3.0"
```

Update version in `pyproject.toml`:

```toml
[project]
name = "molcrafts-molpy"
version = "0.3.0"
```

### 3. Update Changelog

Update `changelog/releases.md` with release notes:

```markdown
## [0.3.0] - 2024-01-15

### Added
- New `compute` module for molecular property calculations
- Support for XYZ trajectory files
- RDF calculation in `compute.rdkit`

### Changed
- Improved PDB reader performance by 2x
- Updated `Frame.merge()` to preserve metadata

### Fixed
- Fixed atom indexing bug in topology detection
- Corrected periodic boundary handling in packmol

### Deprecated
- `old_function()` is deprecated, use `new_function()` instead

### Removed
- Removed deprecated `legacy_module`

### Breaking Changes
- `Frame.blocks()` now returns a list instead of dict
  - Migration: Use `Frame.block_names()` for names
```

### 4. Create Release Branch

```bash
# Ensure main is up-to-date
git checkout main
git pull upstream main

# Create release branch
git checkout -b release/v0.3.0

# Commit version changes
git add src/molpy/version.py pyproject.toml changelog/releases.md
git commit -m "[release] prepare v0.3.0"

# Push release branch
git push origin release/v0.3.0
```

### 5. Final Testing

Run comprehensive tests on the release branch:

```bash
# Full test suite
pytest -v

# Test with coverage
pytest --cov=molpy --cov-report=html tests/

# Test documentation build
mkdocs build

# Test package build
python -m build

# Test installation from built package
pip install dist/molcrafts_molpy-0.3.0-py3-none-any.whl
```

### 6. Create Pull Request

Create a PR from `release/v0.3.0` to `main`:

- **Title:** "Release v0.3.0"
- **Description:** Include changelog highlights
- **Labels:** `release`

Wait for:
- [ ] CI checks to pass
- [ ] Code review approval
- [ ] Final testing confirmation

### 7. Merge and Tag

Once PR is approved:

```bash
# Merge to main
git checkout main
git merge release/v0.3.0

# Create annotated tag
git tag -a v0.3.0 -m "Release v0.3.0"

# Push to GitHub
git push upstream main --tags
```

### 8. Build and Publish

Build the package:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Verify contents
tar -tzf dist/molcrafts-molpy-0.3.0.tar.gz
unzip -l dist/molcrafts_molpy-0.3.0-py3-none-any.whl
```

Publish to PyPI:

```bash
# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ molcrafts-molpy

# If successful, upload to PyPI
python -m twine upload dist/*
```

### 9. Create GitHub Release

On GitHub:

1. Go to [Releases](https://github.com/MolCrafts/molpy/releases)
2. Click "Draft a new release"
3. Select tag `v0.3.0`
4. Set title: "MolPy v0.3.0"
5. Copy changelog content to description
6. Attach built distributions (optional)
7. Click "Publish release"

### 10. Post-release Tasks

**Update Documentation:**
- [ ] Verify docs deployed to GitHub Pages
- [ ] Update README if needed
- [ ] Announce on GitHub Discussions

**Prepare for Next Release:**
- [ ] Create new "Unreleased" section in changelog
- [ ] Update version to next development version (e.g., `0.4.0-dev`)
- [ ] Create milestone for next release

**Communication:**
- [ ] Announce release on GitHub Discussions
- [ ] Update project README with new version
- [ ] Notify downstream projects of breaking changes

## Release Schedule

MolPy follows a time-based release schedule:

- **Minor releases** - Every 2-3 months
- **Patch releases** - As needed for critical bugs
- **Major releases** - When significant breaking changes accumulate

## Breaking Changes

When introducing breaking changes:

### 1. Deprecation Period

Deprecate old API before removing:

```python
import warnings

def old_function():
    """Old function (deprecated).

    .. deprecated:: 0.3.0
        Use :func:`new_function` instead.
    """
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

### 2. Migration Guide

Provide clear migration instructions:

```markdown
## Migration Guide: v0.2 → v0.3

### Frame.blocks() returns list instead of dict

**Before (v0.2):**
```python
blocks = frame.blocks()
for name, block in blocks.items():
    print(name, len(block))
```

**After (v0.3):**
```python
names = frame.block_names()
for name in names:
    block = frame[name]
    print(name, len(block))
```
```

### 3. Changelog Highlighting

Clearly mark breaking changes:

```markdown
### Breaking Changes

⚠️ **Frame.blocks() API changed**
- `Frame.blocks()` now returns `list[str]` instead of `dict[str, Block]`
- Use `Frame.block_names()` for block names
- Use `Frame[name]` to access individual blocks
- See migration guide for details
```

## Hotfix Process

For critical bugs in released versions:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/v0.3.1 v0.3.0

# Fix the bug
# ... make changes ...

# Update version to 0.3.1
# Update changelog

# Commit and tag
git commit -am "[hotfix] fix critical bug"
git tag -a v0.3.1 -m "Hotfix v0.3.1"

# Merge to main
git checkout main
git merge hotfix/v0.3.1

# Push
git push upstream main --tags
```

## CI/CD Integration

### Automated Checks

On every push and PR:
- Run full test suite
- Check code style (Black, isort)
- Verify documentation builds
- Check coverage threshold

### Automated Publishing

On tag push (e.g., `v0.3.0`):
- Build package
- Run tests
- Publish to PyPI (if tests pass)
- Deploy documentation to GitHub Pages

### GitHub Actions Example

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install build twine

      - name: Run tests
        run: |
          pip install -e ".[dev]"
          pytest

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

## Version Support

### Supported Versions

- **Latest minor version** - Full support
- **Previous minor version** - Security fixes only
- **Older versions** - No support

Example:
- Current: `0.3.x` - Full support
- Previous: `0.2.x` - Security fixes
- Older: `0.1.x` - No support

### Python Version Support

Follow [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html):

- Support Python versions released in last 42 months
- Drop support with 6 months notice

## Troubleshooting

### Build Fails

**Issue:** Package build fails

**Solutions:**
- Check `pyproject.toml` syntax
- Verify all files are included in MANIFEST.in
- Clean build artifacts: `rm -rf dist/ build/`

### Tests Fail on CI

**Issue:** Tests pass locally but fail on CI

**Solutions:**
- Check Python version matches
- Verify all dependencies in `pyproject.toml`
- Check for platform-specific issues

### PyPI Upload Fails

**Issue:** Upload to PyPI fails

**Solutions:**
- Verify PyPI token is correct
- Check version doesn't already exist
- Ensure package name is available

## Summary Checklist

Before releasing:

- [ ] All tests pass
- [ ] Documentation builds
- [ ] Changelog updated
- [ ] Version numbers updated
- [ ] Migration guide written (if breaking changes)
- [ ] Release branch created
- [ ] PR reviewed and approved
- [ ] Tag created
- [ ] Package built and tested
- [ ] Published to PyPI
- [ ] GitHub release created
- [ ] Documentation deployed
- [ ] Announcement posted

## Further Reading

- [Semantic Versioning](https://semver.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)

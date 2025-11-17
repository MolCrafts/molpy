# Release Process

This document describes the high‑level process for making a MolPy release.

The goals are:

- Keep releases predictable.
- Make it easy for users to see what changed.
- Avoid breaking downstream projects unexpectedly.

---

## 1. Pre‑release checklist

Before cutting a release:

1. Ensure the test suite passes on all supported Python versions.
2. Check that **documentation builds cleanly** (no broken links or obvious gaps).
3. Review `changelog/releases.md` and add entries for:
   - New features
   - Bug fixes
   - Breaking changes
4. Verify that core examples in the docs still run:
   - Quickstart
   - Key tutorials and user guides

---

## 2. Versioning

MolPy follows a semantic‑style versioning scheme:

- **MAJOR** – incompatible API changes
- **MINOR** – new functionality in a backwards‑compatible manner
- **PATCH** – backwards‑compatible bug fixes

When you change the version:

1. Update the version in the appropriate module (e.g. `molpy.version`).
2. Make sure the new version is reflected in packaging metadata.

---

## 3. Tagging and publishing

Typical steps:

```bash
git status          # clean working tree
pytest              # all tests pass

git commit -am "Bump version to X.Y.Z"
git tag -a vX.Y.Z -m "MolPy X.Y.Z"
git push origin main --tags
```

Publishing to PyPI or internal registries depends on your deployment setup,
but generally involves building wheels/sdist and uploading them with `twine`
or similar tools.

---

## 4. Post‑release

After a release:

- Update `changelog/releases.md` with a heading for the **next** unreleased
  version.
- Create issues or milestones for follow‑up work discovered during the release.
- Consider updating examples in the docs to highlight new features.

Keeping this loop tight makes each release smaller and easier to manage.

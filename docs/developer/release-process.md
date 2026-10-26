# Release Process

Releases are cut from `master` of MolCrafts/molpy, the project repository;
MolCrafts/molrs is retired. 0.20.0 is the first release of the unified
project, and the downstream repositories move to it after that release
(the repository's `.claude/specs/molpy-unification.md`, link 04). The last
releases before it were 0.19.0 (`v0.19.0`) of the two former repositories.

One version and one tag cover the whole project.

## Before the first release from MolCrafts/molpy

The registries trusted the former repositories, so each needs, once:

- crates.io `molcrafts-molrs`: a trusted publisher for MolCrafts/molpy,
  workflow `release.yml`, environment `crates-io`;
- npm `@molcrafts/molwasm`: a new package. npm configures trusted
  publishing only on a package that exists, so its first version is
  published by hand, then the trusted publisher (MolCrafts/molpy,
  `release.yml`, environment `npm`) is added;
- PyPI `molcrafts-molpy`: its trusted publisher already names MolCrafts/molpy,
  `release.yml` and environment `pypi`;
- the GitHub environments `crates-io` and `npm` on MolCrafts/molpy (`pypi`
  and `pypi-nightly` exist).

## One version

One version appears in:

- `[workspace.package].version` in `Cargo.toml`, inherited by `crates/molrs/`;
- `version`, and the `molcrafts-molrs` / `molcrafts-molrs-ffi` dependency
  versions, in the standalone manifests `crates/molrs-ffi/`, `molpy/`,
  `molwasm/`, `molc/` and `molcxx/Cargo.toml`. molpy's wheel takes its
  version from `molpy/Cargo.toml`, and npm's `package.json` is generated
  from `molwasm/Cargo.toml` by wasm-pack;
- the `molcrafts-molrs*` entries of every committed `Cargo.lock`;
- the version-pinned examples in `README.md` and this manual
  (`version = "X.Y"` dependency lines).

## Local verification

Run the pre-push gate from the repository root:

```bash
scripts/check.sh verify
```

It is the script, on the pinned `rust-toolchain.toml` compiler, that the prek
hooks and the CI workflows call. No gate compiles the crates.io archive:
`cargo publish` in `release.yml` verifies it once, and the dispatch dry run
(`cargo publish --dry-run`) rehearses that. Inspect
`cargo package --list --manifest-path crates/molrs/Cargo.toml` for files left
out.

## Publishing

`.github/workflows/release.yml` holds every channel:

- crates.io: `molcrafts-molrs`;
- npm: `@molcrafts/molwasm`;
- PyPI: `molcrafts-molpy`, as desktop abi3 wheels and the Pyodide wheel;
- the GitHub Release, carrying the C API archives.

| run | what it does |
| --- | --- |
| `v*` tag on MolCrafts | `release / guard` (tag = `v` + the workspace version, on `master`; every job of the commit's full `test.yml` tier succeeded), the builds, then every upload. No lint or test run of its own: the commit was proven before the tag |
| dispatch (any branch, fork or upstream) | the dry run: the same guard, the builds, `cargo publish --dry-run` and `npm pack --dry-run`; no upload and no Release |

1. Finish the checks and review the release diff. Push it to `master` and
   wait for its `test.yml` run: the tag's guard reads it.
2. Dispatch **release** on the branch for a rehearsal (a fork is fine).
3. Merge the reviewed revision into `master`, then create and push `vX.Y.Z`,
   matching `[workspace.package].version`.
4. Wait for **release** to finish, then smoke-test the published artifacts.

Every upload skips a version the registry already has: retry a partial
release by re-running the tag's run. Trusted publishing on crates.io, PyPI
and npm names the workflow file `release.yml` and one environment per
registry: `crates-io`, `npm` and `pypi`.

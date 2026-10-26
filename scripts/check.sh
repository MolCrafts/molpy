#!/usr/bin/env bash
# The one definition of every gate. prek (.pre-commit-config.yaml) and CI
# (.github/workflows/test.yml) both call this script, and rust-toolchain.toml
# pins the compiler for both, so a gate that passes locally is the same
# command on the same rustc / clippy / rustdoc that CI runs.
#
#   scripts/check.sh fmt clippy     # run the named gates, in order
#   scripts/check.sh verify         # the pre-push gate: every check, once
#   scripts/check.sh --report .ci-out test python
#                                   # + each test gate's numbers, for
#                                   # MolCrafts/molcrafts-ci/actions/report
#
# Gates: fmt ruff partners clippy doc test ffi cxx capi python types wasm
# mrec docs. Every check runs once, in one place:
#   - static checks (fmt ruff partners clippy doc, and the hooks' workflows /
#     os-cfg) run only locally: pre-commit, and `verify` at pre-push. CI does
#     not repeat them.
#   - the tests run once locally in `verify` and once per OS in test.yml;
#     the molpy wheel `python` built and tested is the one `types`, `mrec`
#     and `docs` use.
#   - every build is the dev profile; release/LTO artifacts are built only by
#     release.yml, and `cargo publish` there verifies the crate tarball once.
# Root-workspace cargo calls go through the `cargo mrs-*`
# aliases (.cargo/config.toml) so they share one feature set and one build.
# Every cargo / maturin / wasm-pack call is --locked and every uv call runs
# on CI's Python (3.12) against the committed lock: a gate that would have to
# change a lock file fails instead.
#
# Dispatch: `fmt`, `ruff` and `partners` compile nothing and run wherever this script
# is called. When the environment names a runner in MOLCRAFTS_HOOK_RUNNER and
# this is not already a Slurm job, any other gate hands the whole call to it.
# The MolCrafts cluster's shared git hooks set it to a launcher that runs its
# arguments on a compute node, so a commit touching only cheap gates never
# waits for Slurm. CI and other machines leave it unset: nothing changes there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_PYTHON=3.12
# Run from a git hook, GIT_DIR, GIT_INDEX_FILE, ... name the hooked
# repository; a gate's own git calls (scripts/partners.py fetching molrec, uv
# fetching a git dependency) must not inherit them.
unset $(git rev-parse --local-env-vars)
CLEANUP=()
trap '[ "${#CLEANUP[@]}" -eq 0 ] || rm -rf "${CLEANUP[@]}"' EXIT

# Every standalone workspace root besides the root one: the ABI layer and the
# language bindings.
ROOTS=(crates/molrs-ffi molpy molwasm molc molcxx)
# wasm-opt release that release.yml's npm build runs (.github/actions/setup-wasm
# reads it from here). The wasm gate is `wasm-pack test`, which runs no wasm-opt.
BINARYEN_VERSION=version_133
WASM_PACK_VERSION=0.15.0
TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}
# --report <dir>: the test gates also write what they ran into <dir> --
# cargo-test.log (every `cargo test` of test/ffi/cxx; its `test result:`
# lines are the totals), junit.xml and coverage.json (python). Nothing reads
# them here; CI turns them into the run's summary.
REPORT=
# The molpy wheel `types`, `mrec` and `docs` install instead of building their
# own. `python` sets it to the wheel it just built and tested. CI's test.yml
# passes it in the environment to the `mrec` and `docs` jobs: the Linux wheel
# the `test / python (ubuntu-latest)` job of the SAME run built and tested,
# downloaded as an artifact. It must never name any other wheel (a stale local
# build, a PyPI download): these gates would then judge a binary no test ran
# against.
MOLPY_TESTED_WHEEL=${MOLPY_TESTED_WHEEL:-}
# Set by CI's `test / python (ubuntu-latest)`: `python` writes the wheel it
# tests here, kept after exit, so the job can upload that same file.
MOLPY_WHEEL_DIR=${MOLPY_WHEEL_DIR:-}

# `cargo test` that also appends its output to the report log.
cargo_test() {
    if [ -n "$REPORT" ]; then
        cargo "$@" | tee -a "$REPORT/cargo-test.log"
    else
        cargo "$@"
    fi
}

clippy_binder() {
    cargo clippy --locked --manifest-path "$1/Cargo.toml" --all-targets "${@:2}" -- -D warnings
}

# Sets WORK to a fresh temp dir, removed when the script exits.
scratch() {
    WORK=$(mktemp -d "${TMPDIR:-/tmp}/molrs-check.XXXXXX")
    if command -v cygpath >/dev/null; then WORK=$(cygpath -m "$WORK"); fi
    CLEANUP+=("$WORK")
}

gate_fmt() {
    cargo fmt --all --check
    for crate in "${ROOTS[@]}"; do
        cargo fmt --manifest-path "$crate/Cargo.toml" --check
    done
}

# Python lint (ruff.toml): text-mode file I/O names its encoding, so a read
# that passes on Linux cannot fail only on Windows (cp1252).
gate_ruff() {
    uvx ruff@0.16.5 check .
}

# Every partner in .github/partners.env resolves, no path dependency points
# where CI has no checkout, no workflow spells a partner commit of its own;
# every binder manifest carries the workspace version (scripts/versions.py).
gate_partners() {
    python3 scripts/partners.py check
    python3 scripts/versions.py
    uv --directory molpy lock --check
}

# Native clippy of the core and every binder, except molwasm: that one is a
# wasm32 crate, linted only for wasm32.
gate_clippy() {
    cargo --locked mrs-clippy -- -D warnings
    for crate in crates/molrs-ffi molcxx molpy molc; do
        clippy_binder "$crate"
    done
    clippy_binder molwasm --target wasm32-unknown-unknown
}

gate_doc() {
    RUSTDOCFLAGS="-D warnings" cargo --locked mrs-doc
}

# --lib does not run rustdoc examples, so the doctests are their own step.
gate_test() {
    cargo_test --locked mrs-test
    cargo_test --locked mrs-doctest
}

gate_ffi() {
    cargo_test test --locked --manifest-path crates/molrs-ffi/Cargo.toml
}

# The build regenerates molcxx/src/bridge.rs from build.rs's schema; the
# committed copy (what Atomiverse's CMake reads) must be what it regenerates.
gate_cxx() {
    cargo_test test --locked --manifest-path molcxx/Cargo.toml
    git diff --exit-code -- molcxx/src/bridge.rs ||
        { echo "molcxx/src/bridge.rs differs from build.rs's schema: commit the regenerated file" >&2; return 1; }
}

# molpy's locked environment (molpy/uv.lock) with one molpy wheel installed:
# the one `python` built and tested (MOLPY_TESTED_WHEEL), or, run on its own,
# a dev wheel built here. uv never builds molpy through `sync`. Sets
# MOLPY_PYTHON and MOLPY_TESTED_WHEEL.
molpy_env() {
    uv --directory molpy sync --locked --no-install-project --extra dev "$@"
    # Ask the selected environment for its real executable, including .exe.
    MOLPY_PYTHON=$(uv --directory molpy run --no-sync python -c \
        'import sys; print(sys.executable, end="")')
    uv pip check --python "$MOLPY_PYTHON"
    if [ -z "$MOLPY_TESTED_WHEEL" ]; then
        local work
        if [ -n "$MOLPY_WHEEL_DIR" ]; then
            mkdir -p "$MOLPY_WHEEL_DIR"
            work=$(cd "$MOLPY_WHEEL_DIR" && pwd)
            if command -v cygpath >/dev/null; then work=$(cygpath -m "$work"); fi
            rm -f "$work"/molcrafts_molpy-*.whl
        else
            scratch
            work=$WORK
        fi
        uv --directory molpy run --no-sync maturin build --locked \
            --interpreter "$MOLPY_PYTHON" --out "$work"
        MOLPY_TESTED_WHEEL=$(ls "$work"/molcrafts_molpy-*.whl)
    fi
    uv pip install -q --python "$MOLPY_PYTHON" --no-deps --reinstall "$MOLPY_TESTED_WHEEL"
    uv pip check --python "$MOLPY_PYTHON"
}

# Build one non-editable dev-profile molpy wheel (no --release, no LTO), then
# check molpy's format and types and run the whole Python suite on it.
# `pytest`, not `python -m pytest`. The suite imports the installed wheel.
# The package source is python/molpy (maturin python-source). `types`,
# `mrec` and `docs` reuse this wheel.
gate_python() {
    MOLPY_TESTED_WHEEL=
    molpy_env
    uv --directory molpy run --no-sync python -I -c \
        'import molpy, pathlib; p=pathlib.Path(molpy.__file__).resolve(); assert "site-packages" in str(p), p'
    uv --directory molpy run --no-sync ruff format --check python tests
    uv --directory molpy run --no-sync ty check python/molpy
    local report=()
    [ -z "$REPORT" ] || report=("--junitxml=$REPORT/junit.xml" --cov=molpy --cov-branch
        "--cov-report=json:$REPORT/coverage.json")
    PYTHONWARNDEFAULTENCODING=1 uv --directory molpy run --no-sync pytest -q -n auto \
        ${report[@]+"${report[@]}"}
}

# The stubs are the Python type contract of the native module:
# python/molpy/_core.pyi and the side stubs of the modules that copy its
# names at import time (python/molpy/core/{keys,schema,constants}.pyi) are
# held to themselves and to the runtime they describe, with
# molpy/pyproject.toml's [tool.mypy] (strict, the source-tree stub first on
# the path). `mypy` type-checks the stub; `mypy.stubtest` imports molpy._core
# and checks every name, signature, parameter kind and default the stub
# declares against the compiled module, and every public runtime name the
# stub misses. stubtest-allowlist.txt holds the structural exemptions only.
#
# stubtest runs from an empty directory, so the runtime it imports is the
# installed wheel. MYPYPATH points mypy at python/, where the source package
# lives; that tree has no _core, or a stale one.
gate_types() {
    molpy_env
    uv --directory molpy run --no-sync python -m mypy python/molpy/_core.pyi \
        python/molpy/core/keys.pyi python/molpy/core/schema.pyi \
        python/molpy/core/constants.pyi
    scratch
    local src=$PWD/molpy
    (cd "$WORK" && MYPYPATH="$src/python" "$MOLPY_PYTHON" -m mypy.stubtest \
        molpy._core molpy.core.keys molpy.core.schema molpy.core.constants \
        --mypy-config-file "$src/pyproject.toml" \
        --allowlist "$src/stubtest-allowlist.txt" --ignore-disjoint-bases)
}

# Profile and target dir are explicit: CMake caches both, and a build-test/
# configured once for release would otherwise keep rebuilding the release lib.
gate_capi() {
    cargo test --locked --manifest-path molc/Cargo.toml
    cmake -S molc/tests/cpp -B molc/build-test \
        -DCARGO_PROFILE=debug -DCARGO_TARGET_DIR="$TARGET_DIR"
    cmake --build molc/build-test --config Debug
    ctest --test-dir molc/build-test --build-config Debug --output-on-failure
}

# `wasm-pack test` compiles the wasm32 crate once and runs the Node suite on
# it. No `wasm-pack build --release`: release.yml builds the npm artifact.
gate_wasm() {
    local have
    have=$(wasm-pack --version | awk '{print $NF}')
    [ "$have" = "$WASM_PACK_VERSION" ] || { echo "wasm-pack must be $WASM_PACK_VERSION (found $have)" >&2; return 1; }
    # Node 24 (the LTS) is the floor; every newer release, up to the latest,
    # must work too.
    have=$(node --version)
    local major=${have#v}
    major=${major%%.*}
    [ "$major" -ge 24 ] 2>/dev/null || { echo "Node >= 24 is required (found $have)" >&2; return 1; }
    (cd molwasm && wasm-pack test --node -- --locked)
}

# molrec's conformance suite through molpy.io.mrec (test.yml's `test / mrec`;
# nightly.yml's conformance snapshot runs the same suite): molrec at the
# commit scripts/partners.py resolves (fetched into
# a temp dir, never a sibling's working tree), the tested wheel (or, run on
# its own, the extension built by `maturin develop`) in a fresh venv on CI's
# Python, scripts/ci-conformance.py judging every case. Any case that does not
# pass fails the gate.
gate_mrec() {
    scratch
    local work=$WORK
    python3 scripts/partners.py fetch MOLREC "$work/molrec"
    uv venv -q --seed "$work/venv"
    # shellcheck disable=SC1091
    local bin="$work/venv/bin"
    [ -d "$bin" ] || bin="$work/venv/Scripts"
    source "$bin/activate"
    uv pip install -q "$work/molrec" \
        "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$(sed -n 's/^CI_REF=//p' .github/partners.env)"
    if [ -n "$MOLPY_TESTED_WHEEL" ]; then
        uv pip install -q "$MOLPY_TESTED_WHEEL"
    else
        uv pip install -q maturin
        maturin develop --locked --manifest-path molpy/Cargo.toml
    fi
    python scripts/ci-conformance.py --suite "$work/molrec/tests" --out "$work/conformance.json"
    deactivate
}

# The manual (zensical.toml, docs/): every ```python block run against the
# tested molpy wheel (scripts/run_doc_blocks.py), then the site built
# --strict, its API reference imported from the same wheel.
gate_docs() {
    molpy_env --extra doc
    uv --directory molpy run --no-sync python -I ../scripts/run_doc_blocks.py
    # The build runs from molpy/; the theme finds a figure's `$file` data under
    # DOCS_DIR, else under ./docs, which from molpy/ does not exist.
    DOCS_DIR="$PWD/docs" uv --directory molpy run --no-sync zensical build --clean --strict -f ../zensical.toml
}

# The pre-push gate. The pre-commit stage (the CI profile's workflows, fmt,
# ruff, os-cfg, partners) on every file, then every remaining gate once.
# `partners` is not repeated here: the pre-commit stage ran it.
VERIFY=(clippy doc test ffi cxx capi python types wasm mrec docs)
gate_verify() {
    uvx pre-commit run --all-files --hook-stage pre-commit --show-diff-on-failure
    for gate in "${VERIFY[@]}"; do
        "gate_$gate"
    done
}

ALL=(fmt ruff partners "${VERIFY[@]}")
# Gates that compile nothing; everything else goes to MOLCRAFTS_HOOK_RUNNER.
CHEAP=(fmt ruff partners)

usage="usage: $0 [--report <dir>] <gate>... | all   (gates: ${ALL[*]})"
report_args=()
if [ "${1:-}" = --report ]; then
    [ "$#" -gt 1 ] || { echo "$usage" >&2; exit 2; }
    report_args=(--report "$2")
    mkdir -p "$2"
    REPORT=$(cd "$2" && pwd)
    # Python on Windows reads D:/a/..., not Git Bash's /d/a/...
    if command -v cygpath >/dev/null; then REPORT=$(cygpath -m "$REPORT"); fi
    # A previous run's numbers must not pass as this run's.
    rm -f "$REPORT/cargo-test.log" "$REPORT/junit.xml" "$REPORT/coverage.json"
    shift 2
fi
[ "$#" -gt 0 ] || { echo "$usage" >&2; exit 2; }
[ "$1" = all ] && set -- "${ALL[@]}"

if [ -n "${MOLCRAFTS_HOOK_RUNNER:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    for gate in "$@"; do
        if [[ " ${CHEAP[*]} " != *" $gate "* ]]; then
            exec "$MOLCRAFTS_HOOK_RUNNER" "$PWD/scripts/check.sh" ${report_args[@]+"${report_args[@]}"} "$@"
        fi
    done
fi

for gate in "$@"; do
    if ! declare -F "gate_$gate" >/dev/null; then
        echo "unknown gate: $gate (gates: ${ALL[*]})" >&2
        exit 2
    fi
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::check $gate"; else echo "== check $gate"; fi
    "gate_$gate"
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::endgroup::"
done

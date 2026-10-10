#!/usr/bin/env bash
# The one definition of every gate. prek (.pre-commit-config.yaml) and CI
# (.github/workflows/{lint,test,docs}.yml) both call this script, and rust-toolchain.toml
# pins the compiler for both, so a gate that passes locally is the same
# command on the same rustc / clippy / rustdoc that CI runs.
#
#   scripts/check.sh fmt clippy     # run the named gates, in order
#   scripts/check.sh all            # every gate (CI parity)
#   scripts/check.sh --report .ci-out test python
#                                   # + each test gate's numbers, for
#                                   # MolCrafts/molcrafts-ci/actions/report
#
# Gates: fmt ruff partners clippy doc test features package ffi cxx ext python capi
# wasm mrec docs. Root-workspace cargo calls go through the `cargo mrs-*`
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

BINDERS=(molrs-ffi molrs-python molrs-wasm molrs-capi molrs-cxxapi)
# Every standalone workspace root besides the root one: the binders, and the
# force-field IR extension proof crate.
ROOTS=("${BINDERS[@]}" molrs-ext-example)
# wasm-opt release the wasm gate runs; CI installs exactly this one.
BINARYEN_VERSION=version_133
WASM_PACK_VERSION=0.15.0
TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target}
# --report <dir>: the test gates also write what they ran into <dir> --
# cargo-test.log (every `cargo test` of test/ffi/cxx/ext; its `test result:`
# lines are the totals), junit.xml and coverage.json (python). Nothing reads
# them here; CI turns them into the run's summary.
REPORT=
# Only a wheel built and tested in this invocation may be reused.
MOLRS_TESTED_WHEEL=

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
# where CI has no checkout, no workflow spells a partner commit of its own.
gate_partners() {
    python3 scripts/partners.py check
    uv --directory molrs-python lock --check
}

gate_clippy() {
    cargo --locked mrs-clippy -- -D warnings
    for crate in molrs-ffi molrs-cxxapi molrs-ext-example molrs-python molrs-capi; do
        clippy_binder "$crate"
    done
    clippy_binder molrs-wasm --target wasm32-unknown-unknown
}

gate_doc() {
    RUSTDOCFLAGS="-D warnings" cargo --locked mrs-doc
}

# --lib does not run rustdoc examples, so the doctests are their own step.
gate_test() {
    cargo_test --locked mrs-test
    cargo_test --locked mrs-doctest
}

# Each sub-system must build on its own, without molrs's native defaults,
# and so must its tests: --all-targets, so a test that reaches past its
# feature (an `ff` test naming `io::mrec`, which is `zarr`'s) fails here.
features_clippy() {
    cargo clippy --locked -p molcrafts-molrs --all-targets "$@" -- -D warnings
}

gate_features() {
    features_clippy
    features_clippy --no-default-features
    for feature in io smiles signal compute ff conformer md builder serde stream zarr filesystem voronoi full; do
        features_clippy --no-default-features --features "$feature"
    done
}

# Compiles the unpacked crates.io archive, catching files left out of it.
gate_package() {
    cargo package --locked --allow-dirty --manifest-path molrs/Cargo.toml
    python3 - "$TARGET_DIR" <<'PY'
import sys
import tarfile
import tomllib
from pathlib import Path

version = tomllib.loads(Path("Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]
name = f"molcrafts-molrs-{version}"
with tarfile.open(Path(sys.argv[1]) / "package" / f"{name}.crate") as package:
    license_file = package.extractfile(f"{name}/LICENSE")
    assert license_file is not None, "Package is missing LICENSE"
    assert license_file.read() == Path("LICENSE").read_bytes(), "Package license differs from repository license"
PY
}

gate_ffi() {
    cargo_test test --locked --manifest-path molrs-ffi/Cargo.toml
}

gate_cxx() {
    cargo_test test --locked --manifest-path molrs-cxxapi/Cargo.toml
}

# The force-field IR as a protocol (ff-ir-02-protocol, P-Rust): a third
# party's crate extending it through molrs's pub API alone — a pair style, a
# new category, an expression style — priced against pinned LAMMPS numbers
# (scripts/ff_ir_extension_lammps_check.sh), persisted, and refused by name.
gate_ext() {
    cargo_test test --locked --manifest-path molrs-ext-example/Cargo.toml
}

# Build one non-editable release wheel and test it in the locked uv environment.
# No second tox/pip resolver or second copy of pytest/numpy/maturin.
gate_python() {
    uv --directory molrs-python sync --locked --no-install-project --extra dev
    # Ask the selected environment for its real executable, including .exe.
    # Validate it before compiling: Bash's implicit .exe matching is not uv's.
    local interpreter work wheel
    interpreter=$(uv --directory molrs-python run --no-sync python -c \
        'import sys; print(sys.executable, end="")')
    uv pip check --python "$interpreter"
    work=$(uv --directory molrs-python run --no-sync python -c \
        'import tempfile; print(tempfile.mkdtemp(prefix="molrs-wheel-"), end="")')
    CLEANUP+=("$work")
    uv --directory molrs-python run --no-sync maturin build --release --locked \
        --interpreter "$interpreter" --out "$work"
    wheel=$(ls "$work"/molcrafts_molrs-*.whl)
    uv pip install -q --python "$interpreter" --no-deps --reinstall "$wheel"
    uv pip check --python "$interpreter"
    uv --directory molrs-python run --no-sync python -c \
        'import molrs, pathlib; p=pathlib.Path(molrs.__file__).resolve(); assert "site-packages" in str(p), p'
    local report=()
    [ -z "$REPORT" ] || report=("--junitxml=$REPORT/junit.xml" --cov=molrs --cov-branch
        "--cov-report=json:$REPORT/coverage.json")
    uv --directory molrs-python run --no-sync python -X warn_default_encoding -m pytest -q \
        ${report[@]+"${report[@]}"}
    MOLRS_TESTED_WHEEL=$wheel
}

# Profile and target dir are explicit: CMake caches both, and a build-test/
# configured once for release would otherwise keep rebuilding the release lib.
gate_capi() {
    cargo test --locked --manifest-path molrs-capi/Cargo.toml
    cmake -S molrs-capi/tests/cpp -B molrs-capi/build-test \
        -DCARGO_PROFILE=debug -DCARGO_TARGET_DIR="$TARGET_DIR"
    cmake --build molrs-capi/build-test --config Debug
    ctest --test-dir molrs-capi/build-test --build-config Debug --output-on-failure
}

# Building proves the wasm compiles; the Node suite proves it works.
gate_wasm() {
    local have
    have=$(wasm-pack --version | awk '{print $NF}')
    [ "$have" = "$WASM_PACK_VERSION" ] || { echo "wasm-pack must be $WASM_PACK_VERSION (found $have)" >&2; return 1; }
    have=$(node --version)
    [[ "$have" == v24.* ]] || { echo "Node 24 is required (found $have)" >&2; return 1; }
    # Release builds print "version 133 (version_133)", Homebrew builds "version 133".
    have=version_$(wasm-opt --version | awk '{print $3}')
    if [ "$have" != "$BINARYEN_VERSION" ]; then
        echo "wasm-opt is $have, the gate pins $BINARYEN_VERSION:" >&2
        echo "https://github.com/WebAssembly/binaryen/releases/tag/$BINARYEN_VERSION" >&2
        return 1
    fi
    (cd molrs-wasm && wasm-pack build --release --target bundler --scope molcrafts --out-name molrs -- --locked)
    (cd molrs-wasm && wasm-pack test --node -- --locked)
}

# molrec's conformance suite through molrs.io.mrec (test.yml's `test / mrec`;
# nightly.yml's conformance snapshot runs the same suite): molrec at the
# commit scripts/partners.py resolves (fetched into
# a temp dir, never a sibling's working tree), the extension built by `maturin
# develop` into a fresh venv on CI's Python, scripts/ci-conformance.py judging
# every case. Any case that does not pass fails the gate.
gate_mrec() {
    scratch
    local work=$WORK
    python3 scripts/partners.py fetch MOLREC "$work/molrec"
    uv venv -q --seed "$work/venv"
    # shellcheck disable=SC1091
    local bin="$work/venv/bin"
    [ -d "$bin" ] || bin="$work/venv/Scripts"
    source "$bin/activate"
    uv pip install -q maturin "$work/molrec" \
        "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$(sed -n 's/^CI_REF=//p' .github/partners.env)"
    if [ -n "$MOLRS_TESTED_WHEEL" ]; then
        uv pip install -q "$MOLRS_TESTED_WHEEL"
    else
        maturin develop --locked --manifest-path molrs-python/Cargo.toml
    fi
    python scripts/ci-conformance.py --suite "$work/molrec/tests" --out "$work/conformance.json"
    deactivate
}

# Strict docs in the same locked tool environment. Reuse the tested wheel
# on pre-push; standalone docs builds one dev wheel for API imports.
gate_docs() {
    uv --directory molrs-python sync --locked --no-install-project --extra dev --extra doc
    local interpreter
    interpreter=$(uv --directory molrs-python run --no-sync python -c \
        'import sys; print(sys.executable, end="")')
    uv pip check --python "$interpreter"
    local wheel=$MOLRS_TESTED_WHEEL
    if [ -z "$wheel" ]; then
        scratch
        uv --directory molrs-python run --no-sync maturin build --locked \
            --interpreter "$interpreter" --out "$WORK/wheels"
        wheel=$(ls "$WORK"/wheels/molcrafts_molrs-*.whl)
    fi
    uv pip install -q --python "$interpreter" --no-deps --reinstall "$wheel"
    uv pip check --python "$interpreter"
    uv --directory molrs-python run --no-sync zensical build --clean --strict
}

gate_verify() {
    uvx pre-commit run --all-files --hook-stage pre-commit --show-diff-on-failure
    for gate in partners clippy doc test features package ffi cxx ext python capi wasm mrec docs; do
        "gate_$gate"
    done
}

ALL=(fmt ruff partners clippy doc test features package ffi cxx ext python capi wasm mrec docs)
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

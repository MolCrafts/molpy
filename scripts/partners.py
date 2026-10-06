#!/usr/bin/env python3
"""The partner repositories CI builds this one against, at pinned commits.

``.github/partners.env`` is the one place a pin lives. The workflows read it
(``grep '^[A-Z]' .github/partners.env >> "$GITHUB_ENV"``) and the git hooks
read it through this script, so a local gate and CI judge against the same
partner commit -- never the developer's sibling checkout, whatever branch that
happens to be on.

    partners.py check             cheap pre-push gate (login node is fine):
                                  every pin exists on its remote, every path
                                  dependency resolves inside the layout CI
                                  checks out, and no workflow spells a partner
                                  commit of its own.
    partners.py fetch NAME DEST   check partner NAME out at its pin into DEST.
    partners.py run -- CMD...     run CMD in CI's sibling layout: a copy of
                                  this working tree at <root>/<SELF> next to
                                  <root>/<partner> for every partner, with
                                  <root>/<SELF> as the working directory and
                                  $PARTNERS_SOURCE naming this checkout (to
                                  copy a result back, e.g. a relocked file).

``run`` keeps <root> under $MOLCRAFTS_PARTNER_CACHE when that is set (one
directory per set of pins, so the partner builds stay warm between pushes);
otherwise <root> is a temp directory removed afterwards.

partners.env holds ``KEY=VALUE`` lines and comments, nothing else (it is
appended to $GITHUB_ENV as is, minus the comments). ``SELF`` names this
repository's directory in the layout; each partner is a
``<NAME>_REPOSITORY=owner/repo`` + ``<NAME>_REF=<commit, tag or branch>``
pair, checked out as the directory ``<name>`` (lower case).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / ".github" / "partners.env"
SHA = re.compile(r"[0-9a-f]{40}")
# Build state inside the layout copy that a sync must not wipe.
KEEP = {".venv", ".tox", "target", ".pytest_cache", ".ruff_cache", "site", ".cache"}


def die(msg: str) -> None:
    print(f"partners: {msg}", file=sys.stderr)
    raise SystemExit(1)


def git(*args: str, cwd: Path | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
    )


def load() -> dict[str, str]:
    env: dict[str, str] = {}
    if not PINS.is_file():
        return env
    for n, raw in enumerate(PINS.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or value != value.strip():
            die(f"{PINS.relative_to(ROOT)}:{n}: expected KEY=VALUE, got {raw!r}")
        env[key] = value
    return env


def partners(env: dict[str, str]) -> dict[str, tuple[str, str]]:
    found = {}
    for key, repo in env.items():
        if key.endswith("_REPOSITORY"):
            name = key.removesuffix("_REPOSITORY")
            ref = env.get(f"{name}_REF") or die(f"{name}_REPOSITORY has no {name}_REF")
            found[name] = (repo, ref)
    return found


def url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def fetch(name: str, repo: str, ref: str, dest: Path) -> None:
    """Check REPO out at REF into DEST, reusing DEST when it already is that commit."""
    if (dest / ".git").is_dir() and SHA.fullmatch(ref):
        head = git("rev-parse", "HEAD", cwd=dest, quiet=True).stdout.strip()
        if (
            head == ref
            and not git(
                "status", "--porcelain", "--untracked-files=no", cwd=dest, quiet=True
            ).stdout
        ):
            return
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        git("init", "-q", cwd=dest)
    print(f"partners: fetching {repo}@{ref} -> {dest}", file=sys.stderr)
    if git("fetch", "-q", "--depth=1", url(repo), ref, cwd=dest).returncode:
        die(f"cannot fetch {name} {repo}@{ref}")
    if git("checkout", "-q", "--force", "--detach", "FETCH_HEAD", cwd=dest).returncode:
        die(f"cannot check out {name} {repo}@{ref}")


def ref_exists(repo: str, ref: str) -> bool:
    if SHA.fullmatch(ref):
        # ls-remote lists ref tips only; a pinned commit is proven by fetching
        # that one commit object (no trees, no blobs).
        with tempfile.TemporaryDirectory() as tmp:
            git("init", "-q", "--bare", cwd=Path(tmp))
            got = git(
                "fetch",
                "-q",
                "--depth=1",
                "--filter=tree:0",
                url(repo),
                ref,
                cwd=Path(tmp),
                quiet=True,
            )
            return got.returncode == 0
    got = git(
        "ls-remote", "--exit-code", url(repo), f"refs/heads/{ref}", f"refs/tags/{ref}", quiet=True
    )
    return got.returncode == 0


def tracked(basename: str) -> list[Path]:
    out = git("ls-files", "-z", cwd=ROOT, quiet=True).stdout
    return [
        ROOT / p for p in out.split("\0") if p and Path(p).name == basename and (ROOT / p).is_file()
    ]


def path_deps() -> list[tuple[Path, str, str]]:
    """(manifest, dependency, path) for every path dependency in a tracked manifest."""
    found = []
    for manifest in tracked("pyproject.toml"):
        sources = (
            tomllib.loads(manifest.read_text()).get("tool", {}).get("uv", {}).get("sources", {})
        )
        for dep, spec in sources.items():
            for entry in spec if isinstance(spec, list) else [spec]:
                if isinstance(entry, dict) and "path" in entry:
                    found.append((manifest, dep, entry["path"]))

    def tables(doc: dict) -> list[dict]:
        out = [doc.get(k, {}) for k in ("dependencies", "dev-dependencies", "build-dependencies")]
        out.append(doc.get("workspace", {}).get("dependencies", {}))
        for target in doc.get("target", {}).values():
            out += [
                target.get(k, {})
                for k in ("dependencies", "dev-dependencies", "build-dependencies")
            ]
        out += list(doc.get("patch", {}).values())
        return out

    for manifest in tracked("Cargo.toml"):
        for table in tables(tomllib.loads(manifest.read_text())):
            for dep, spec in table.items():
                if isinstance(spec, dict) and "path" in spec:
                    found.append((manifest, dep, spec["path"]))
    return found


def check() -> int:
    env = load()
    pinned = partners(env)
    failures = []
    for name, (repo, ref) in pinned.items():
        if ref_exists(repo, ref):
            print(f"ok: {name} {repo}@{ref} exists")
        else:
            failures.append(
                f"{name}: {repo}@{ref} does not exist on the remote (CI cannot check it out)"
            )

    layout = {name.lower() for name in pinned}
    for manifest, dep, raw in path_deps():
        target = Path(os.path.normpath(manifest.parent / raw))
        where = f"{manifest.relative_to(ROOT)}: {dep} = {{ path = {raw!r} }}"
        try:
            target.relative_to(ROOT)
            continue
        except ValueError:
            pass
        try:
            first = target.relative_to(ROOT.parent).parts[0]
        except ValueError:
            first = None
        if first in layout:
            continue
        failures.append(
            f"{where} resolves outside this repository, to {first or target!s}, which CI "
            f"never checks out (partners in .github/partners.env: {sorted(layout) or 'none'})"
        )

    for wf in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        for n, line in enumerate(wf.read_text().splitlines(), 1):
            if re.match(r"\s*ref:\s*['\"]?[0-9a-f]{40}\b", line):
                failures.append(
                    f"{wf.relative_to(ROOT)}:{n}: a literal partner commit; pin it in "
                    ".github/partners.env and read it from there"
                )

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def sync(dest: Path) -> None:
    """Make DEST a copy of this working tree (tracked + untracked, not ignored)."""
    listed = git(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard", cwd=ROOT, quiet=True
    ).stdout
    files = {p for p in listed.split("\0") if p and os.path.lexists(ROOT / p)}
    dest.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(dest, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in KEEP]
        for f in filenames:
            rel = os.path.relpath(os.path.join(dirpath, f), dest)
            if rel not in files:
                os.unlink(os.path.join(dirpath, f))
    for rel in sorted(files):
        src, dst = ROOT / rel, dest / rel
        if src.is_symlink():
            if dst.is_symlink() and os.readlink(dst) == os.readlink(src):
                continue
            dst.unlink(missing_ok=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(src), dst)
            continue
        s = src.stat()
        if dst.is_file() and not dst.is_symlink():
            d = dst.stat()
            if (d.st_size, d.st_mtime_ns) == (s.st_size, s.st_mtime_ns):
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run(cmd: list[str]) -> int:
    env = load()
    pinned = partners(env)
    me = env.get("SELF") or die("partners.env names no SELF")
    cache = os.environ.get("MOLCRAFTS_PARTNER_CACHE")
    key = hashlib.sha256(repr(sorted(pinned.items())).encode()).hexdigest()[:12]
    root = (
        Path(cache) / f"{me}-{key}" if cache else Path(tempfile.mkdtemp(prefix=f"{me}-partners-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        with open(root / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for name, (repo, ref) in pinned.items():
                fetch(name, repo, ref, root / name.lower())
            sync(root / me)
            print(f"partners: running in {root / me}: {' '.join(cmd)}", file=sys.stderr)
            env_out = dict(os.environ, PARTNERS_SOURCE=str(ROOT))
            return subprocess.run(cmd, cwd=root / me, env=env_out).returncode
    finally:
        if not cache:
            shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"] and len(argv) == 1:
        return check()
    if argv[:1] == ["fetch"] and len(argv) == 3:
        pinned = partners(load())
        name = argv[1].upper()
        if name not in pinned:
            die(f"no partner {name} in .github/partners.env (have: {sorted(pinned)})")
        fetch(name, *pinned[name], Path(argv[2]))
        return 0
    if argv[:2] == ["run", "--"] and len(argv) > 2:
        return run(argv[2:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

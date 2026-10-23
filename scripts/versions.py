#!/usr/bin/env python3
"""Every manifest carries the one version: ``[workspace.package].version``.

The root Cargo.toml's ``[workspace.package].version`` is molpy's version
(AGENTS.md). Each standalone binder manifest repeats it in ``[package]``, and
every path dependency on a molrs crate pins it; a manifest left behind would
publish nothing new (the registries skip an existing version) while the
release reports success. ``scripts/check.sh partners`` and the release run
this check.

    python3 scripts/versions.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = (
    "crates/molrs-ffi/Cargo.toml",
    "molpy/Cargo.toml",
    "molwasm/Cargo.toml",
    "molc/Cargo.toml",
    "molcxx/Cargo.toml",
)
CORE_PACKAGES = ("molcrafts-molrs", "molcrafts-molrs-ffi")


def load(path: str) -> dict:
    with (ROOT / path).open("rb") as handle:
        return tomllib.load(handle)


def disagreements(version: str) -> list[str]:
    """Every manifest entry that names another version."""
    found: list[str] = []
    for path in MANIFESTS:
        manifest = load(path)
        own = manifest["package"].get("version")
        if own != version:
            found.append(f"{path}: [package].version is {own!r}")
        for table in ("dependencies", "build-dependencies", "dev-dependencies"):
            for name, spec in manifest.get(table, {}).items():
                if not isinstance(spec, dict):
                    continue
                if (
                    spec.get("package", name) in CORE_PACKAGES
                    and spec.get("version") != version
                ):
                    found.append(
                        f"{path}: [{table}].{name} pins {spec.get('version')!r}"
                    )
    return found


def main() -> int:
    version = load("Cargo.toml")["workspace"]["package"]["version"]
    found = disagreements(version)
    for line in found:
        print(f"{line}, not the workspace's {version!r}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())

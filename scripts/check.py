#!/usr/bin/env python3
"""Orthogonal gates, shared by hooks and CI. Run from the repository root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERSION = os.environ.get("UV_PYTHON", "3.12")
GATES = {
    "lint": [
        ["uvx", "ruff@0.16.1", "format", "--check", "src", "tests"],
        ["uvx", "ruff@0.16.1", "check", "src", "tests", "scripts"],
        ["uvx", "ty@0.0.65", "check", "src/molpy"],
    ],
    "dependencies": [
        [sys.executable, "scripts/partners.py", "check"],
        ["uv", "lock", "--check"],
        [
            "uv",
            "sync",
            "--locked",
            "--python",
            PYTHON_VERSION,
            "--extra",
            "dev",
        ],
        ["uv", "pip", "check"],
    ],
    "test": [
        [
            "uv",
            "run",
            "--locked",
            "--python",
            PYTHON_VERSION,
            "--extra",
            "dev",
            "python",
            "-m",
            "pytest",
            "tests/",
            "-n",
            "auto",
        ]
    ],
    "docs": [
        [
            "uv",
            "run",
            "--locked",
            "--python",
            PYTHON_VERSION,
            "--extra",
            "dev",
            "--extra",
            "doc",
            "zensical",
            "build",
            "--clean",
            "--strict",
        ]
    ],
}


def main() -> int:
    gates = sys.argv[1:]
    if gates == ["verify"]:
        # Install/lock once and keep one resolved partner layout for every gate.
        subprocess.run(
            [
                "uvx",
                "pre-commit",
                "run",
                "--all-files",
                "--hook-stage",
                "pre-commit",
                "--show-diff-on-failure",
            ],
            cwd=ROOT,
            check=True,
        )
        return subprocess.run(
            [
                sys.executable,
                "scripts/partners.py",
                "run",
                "--",
                sys.executable,
                "scripts/check.py",
                "dependencies",
                "test",
                "docs",
            ],
            cwd=ROOT,
            check=False,
        ).returncode
    if not gates or any(gate not in GATES for gate in gates):
        print("usage: check.py verify | " + " ".join(GATES), file=sys.stderr)
        return 2
    env = dict(
        os.environ,
        MATURIN_PEP517_ARGS="--locked",
        PYTHONWARNDEFAULTENCODING="1",
        PYTHONUTF8="1",
    )
    for gate in gates:
        print(f"== {gate}", flush=True)
        for command in GATES[gate]:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the shared dependency resolver at the commit declared in partners.env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    values = dict(
        line.split("=", 1)
        for line in (ROOT / ".github/partners.env")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    )
    ref = values["CI_REF"]
    if len(ref) != 40 or any(c not in "0123456789abcdef" for c in ref):
        raise ValueError("CI_REF must pin a full molcrafts-ci commit")
    package = f"molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@{ref}"
    names = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.split()
    env = {key: value for key, value in os.environ.items() if key not in names}
    env["MOLCRAFTS_PARTNERS_ROOT"] = str(ROOT)
    return subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            package,
            "python",
            "-m",
            "molcrafts_ci.partners",
            *sys.argv[1:],
        ],
        cwd=ROOT,
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

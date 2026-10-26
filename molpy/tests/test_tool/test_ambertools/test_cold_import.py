"""``molpy.tool.ambertools`` imports on a cold interpreter.

``molpy.ff`` is built on ``molpy.tool``; the tool package must therefore import
without ``molpy.ff`` having been imported first. The suite as a whole imports
``molpy.ff`` early, so only a fresh interpreter can show the order problem.
"""

import subprocess
import sys


def test_tool_ambertools_imports_before_molpy_ff():
    proc = subprocess.run(
        [sys.executable, "-I", "-c", "import molpy.tool.ambertools"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

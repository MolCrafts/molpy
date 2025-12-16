"""External tool wrappers for MolPy.

This module provides minimal wrappers around external binaries and CLIs.
Wrappers handle tool invocation, working directories, and environment setup.
Domain logic and workflow decisions belong in Compute nodes.

**Architecture:**
- Wrapper: Minimal shell around external tools (this module)
- Compute: Domain logic and operation orchestration (uses wrappers)
- Adapter: Data structure synchronization (separate from wrappers)

**Example Usage:**
    >>> from molpy.tools import AntechamberWrapper, TLeapWrapper
    >>> from pathlib import Path
    >>>
    >>> # Create wrapper
    >>> ante = AntechamberWrapper(name="antechamber", workdir=Path("tmp_ante"))
    >>> # Compute node builds args and calls wrapper
    >>> proc = ante.run(args=["-i", "lig.mol2", "-fi", "mol2", "-o", "lig.gaff.mol2", "-fo", "mol2"])
    >>>
    >>> # TLeap wrapper
    >>> tleap = TLeapWrapper(name="tleap", workdir=Path("tmp_tleap"))
    >>> script = "source leaprc.protein.ff14SB\nloadamberparams lig.frcmod\n"
    >>> proc = tleap.run_script(script_text=script)
"""

from .wrapper import Wrapper
from .antechamber_wrapper import AntechamberWrapper
from .tleap_wrapper import TLeapWrapper

__all__ = [
    "Wrapper",
    "AntechamberWrapper",
    "TLeapWrapper",
]

"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.

Environment isolation (``env`` / ``env_manager``) is owned by
:class:`~molpy.wrapper.EnvironmentSpec` — the shared infrastructure for every
wrapper and any facade that shells out through one.
"""

from ._wrapper import Wrapper, run_step
from ._environment import EnvironmentSpec
from ._antechamber import AntechamberWrapper
from ._prepgen import Parmchk2Wrapper, PrepgenWrapper
from ._sander import SanderWrapper
from ._tleap import TleapWrapper

__all__ = [
    "Wrapper",
    "EnvironmentSpec",
    "AntechamberWrapper",
    "Parmchk2Wrapper",
    "PrepgenWrapper",
    "SanderWrapper",
    "TleapWrapper",
    "run_step",
]

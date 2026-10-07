"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.

Environment isolation (``env`` / ``env_manager``) is owned by
:class:`~molpy.wrapper.EnvSpec` — the shared infrastructure for every
wrapper and any facade that shells out through one.
"""

from ._base import Wrapper, run_step
from ._env import EnvSpec
from ._antechamber import AntechamberWrapper
from ._prepgen import Parmchk2Wrapper, PrepgenWrapper
from ._sander import SanderWrapper
from ._tleap import TLeapWrapper

__all__ = [
    "Wrapper",
    "EnvSpec",
    "AntechamberWrapper",
    "Parmchk2Wrapper",
    "PrepgenWrapper",
    "SanderWrapper",
    "TLeapWrapper",
    "run_step",
]

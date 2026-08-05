"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.

Environment isolation (``env`` / ``env_manager``) is owned by
:class:`~molpy.wrapper.env.EnvSpec` — the shared infrastructure for every
wrapper and any facade that shells out through one.
"""

from .base import Wrapper
from .env import EnvSpec
from .antechamber import AntechamberWrapper
from .prepgen import Parmchk2Wrapper, PrepgenWrapper, write_prepgen_control_file
from .sander import SanderWrapper
from .tleap import TLeapWrapper

__all__ = [
    "Wrapper",
    "EnvSpec",
    "AntechamberWrapper",
    "Parmchk2Wrapper",
    "PrepgenWrapper",
    "SanderWrapper",
    "TLeapWrapper",
    "write_prepgen_control_file",
]

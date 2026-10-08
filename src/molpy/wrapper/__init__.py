"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.

Each wrapper reads its executable, environment (``env`` / ``env_manager``),
environment variables and time limit from its ``wrapper.<tool>`` settings in
molpy's configuration (:mod:`molpy.config`), and logs every run to
``molpy.wrapper.<tool>`` through :mod:`mollog`. Environment isolation is owned
by :class:`~molpy.wrapper.EnvironmentSpec` — the shared infrastructure for
every wrapper and engine.
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

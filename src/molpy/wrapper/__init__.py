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

from molpy.wrapper._antechamber import AntechamberWrapper
from molpy.wrapper._environment import EnvironmentSpec
from molpy.wrapper._prepgen import Parmchk2Wrapper, PrepgenWrapper
from molpy.wrapper._sander import SanderWrapper
from molpy.wrapper._tleap import TleapWrapper
from molpy.wrapper._wrapper import Wrapper, run_step

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

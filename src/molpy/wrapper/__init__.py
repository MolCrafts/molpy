"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in their
callers (the AmberTools typifiers of :mod:`molpy.ff.typifier` and
:class:`molpy.builder.AmberPolymerBuilder`).

:class:`Tool` is the core every wrapper and engine shares: it reads the
executable, environment (``env`` / ``env_manager``), environment variables
and time limit from the tool's ``<group>.<tool>`` settings in molpy's
configuration (:mod:`molpy.config`), logs every run to
``molpy.<group>.<tool>`` through :mod:`mollog`, and raises one
:class:`ToolError` for every failure. Environment isolation is owned by
:class:`EnvironmentSpec`.
"""

from molpy.wrapper._antechamber import AntechamberWrapper
from molpy.wrapper._environment import EnvironmentSpec
from molpy.wrapper._prepgen import Parmchk2Wrapper, PrepgenWrapper
from molpy.wrapper._process import ToolError
from molpy.wrapper._sander import SanderWrapper
from molpy.wrapper._tleap import TleapWrapper
from molpy.wrapper._tool import Tool
from molpy.wrapper._wrapper import Wrapper, run_step

__all__ = [
    "Tool",
    "ToolError",
    "Wrapper",
    "EnvironmentSpec",
    "AntechamberWrapper",
    "Parmchk2Wrapper",
    "PrepgenWrapper",
    "SanderWrapper",
    "TleapWrapper",
    "run_step",
]

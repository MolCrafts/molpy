"""The external programs molpy runs, and the one core they share.

:class:`Tool` resolves a program from molpy's configuration
(:mod:`molpy.config`): its ``"<group>.<tool>"`` settings, the
:class:`EnvironmentSpec` it runs in (system, conda or venv), its executable,
environment variables and time limit, and its logger. Every subprocess goes
through the same runner, so every failure is one :class:`ToolError`.

Two families of tools are built on it: the engines of :mod:`molpy.engine`
(group ``engine``; a third-party engine subclasses
:class:`molpy.engine.Engine`), and the AmberTools programs (group
``ambertools``: antechamber, parmchk2, prepgen, tleap) behind
:mod:`molpy.ff.typifier`, :class:`molpy.ff.charge.Am1BccModel` and
:class:`molpy.builder.AmberPolymerBuilder`. The AmberTools pipeline lives in
``molpy.tool.ambertools``, an implementation package: it is reached through
those classes, not by name.
"""

from molpy.tool.environment import EnvironmentSpec
from molpy.tool.process import ToolError
from molpy.tool.tool import Tool

__all__ = ["EnvironmentSpec", "Tool", "ToolError"]

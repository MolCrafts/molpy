"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.
"""

from .base import Wrapper
from .antechamber import AntechamberWrapper
from .prepgen import Parmchk2Wrapper, PrepgenWrapper, write_prepgen_control_file
from .tleap import TLeapWrapper

__all__ = [
    "Wrapper",
    "AntechamberWrapper",
    "Parmchk2Wrapper",
    "PrepgenWrapper",
    "TLeapWrapper",
    "write_prepgen_control_file",
]

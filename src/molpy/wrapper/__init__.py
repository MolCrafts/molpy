"""Wrappers for invoking external binaries and CLIs.

Wrappers encapsulate subprocess invocation, working directory, and environment.
They MUST NOT contain high-level domain logic; orchestration belongs in compute
nodes.
"""

from .base import Wrapper
from .antechamber import AntechamberWrapper
from .prepgen import PrepgenWrapper
from .tleap import TLeapWrapper

__all__ = ["Wrapper", "AntechamberWrapper", "PrepgenWrapper", "TLeapWrapper"]

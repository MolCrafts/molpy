"""Compatibility shim for the refactor.

Import AntechamberWrapper from :mod:`molpy.wrapper` instead.
"""

from molpy.wrapper.antechamber import AntechamberWrapper

__all__ = ["AntechamberWrapper"]

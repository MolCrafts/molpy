"""Compatibility shim for the refactor.

Import TLeapWrapper from :mod:`molpy.wrapper` instead.
"""

from molpy.wrapper.tleap import TLeapWrapper

__all__ = ["TLeapWrapper"]

"""Compatibility shim for the refactor.

Import Adapter from :mod:`molpy.adapter` instead.
"""

from molpy.adapter.base import Adapter

__all__ = ["Adapter"]

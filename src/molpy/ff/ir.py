"""The force-field IR as a protocol — :mod:`molrs.ff.ir`, by identity.

Register a new style (:class:`StyleSpec`, :func:`register_style`) or a new
category (:func:`register_category`) from Python, with nothing rebuilt; every
refusal is an :class:`IrError` (a ``ValueError``), one subclass per refusal.
molpy keeps no parallel IR: ``mp.ff.ir.StyleSpec is molrs.ff.ir.StyleSpec``.
"""

from molrs.ff.ir import *  # noqa: F403
from molrs.ff.ir import __all__ as __all__

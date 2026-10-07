"""The force-field IR as a protocol — :mod:`molrs.ff.ir`, by identity.

Register a new style (:func:`register_style`, or a :class:`StyleDeclaration`
subclass) or a new category (:func:`register_category`) from Python, with
nothing rebuilt; :func:`styles` / :func:`categories` describe what is
registered as :class:`StyleSpec` / :class:`CategorySpec` records whose
parameters are :class:`ParamSpec`; :func:`unregister_style` takes a style
back. Every refusal is an :class:`IrError` (a ``ValueError``), one
``*Error`` subclass per refusal (``ArityError``, ``DimensionError``, …).
molpy keeps no parallel IR: ``mp.ff.ir.StyleSpec is molrs.ff.ir.StyleSpec``.
"""

from molrs.ff.ir import *  # noqa: F403
from molrs.ff.ir import __all__ as __all__

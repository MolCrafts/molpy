"""The force-field IR as vocabulary — :mod:`molrs.ff.ir`, by identity.

What a category (:class:`CategorySpec`), a style (:class:`StyleSpec`) and its
parameters (:class:`ParamSpec`) are, and every refusal: an :class:`IrError`
(a ``ValueError``), one ``*Error`` subclass per refusal (``ArityError``,
``DimensionError``, …). Registering a style or a category is
:mod:`molpy.ff.style_registry`'s. molpy keeps no parallel IR:
``mp.ff.ir.StyleSpec is molrs.ff.ir.StyleSpec``.
"""

from molrs.ff.ir import *  # noqa: F403
from molrs.ff.ir import __all__ as __all__

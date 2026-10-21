"""The frame schema — :mod:`molrs.core.schema`, mirrored by identity.

The block and column specs (``BlockSpec``, ``ColumnSpec``), their lookups
(``block``, ``column``, …) and the block names (``ATOMS``, ``BONDS``, …);
``mp.core.schema.ColumnSpec is molrs.core.schema.ColumnSpec``.
"""

from molrs.core.schema import *  # noqa: F403
from molrs.core.schema import __all__ as _native

__all__ = [*_native]

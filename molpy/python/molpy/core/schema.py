"""Frame vocabulary, projected from the compiled Rust tables.

``ColumnSpec`` / ``BlockSpec`` live here as a real package path so pickle and
``from molpy.core.schema import ColumnSpec`` resolve the same class the binder
declares (``module = "molpy.core.schema"``).

Block-name constants (``ATOMS``, ``BONDS``, … and ``TOPOLOGY``) are whatever
the native module exports. Adding one in Rust adds it here with no edit.
:func:`relation_endpoints` says which block and columns a relation block's
rows point into.
"""

from molpy import _core

ColumnSpec = _core.schema.ColumnSpec

BlockSpec = _core.schema.BlockSpec

columns = _core.schema.columns

blocks = _core.schema.blocks

column = _core.schema.column

block = _core.schema.block

relation_endpoints = _core.schema.relation_endpoints

to_json = _core.schema.to_json

to_markdown = _core.schema.to_markdown

_CONSTS = [name for name in dir(_core.schema) if name.isupper()]

globals().update({name: getattr(_core.schema, name) for name in _CONSTS})

__all__ = [
    "BlockSpec",
    "ColumnSpec",
    "block",
    "blocks",
    "column",
    "columns",
    "relation_endpoints",
    "to_json",
    "to_markdown",
]
__all__.extend(_CONSTS)

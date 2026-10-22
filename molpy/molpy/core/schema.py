"""Frame vocabulary, projected from the compiled Rust tables.

``ColumnSpec`` / ``BlockSpec`` live here as a real package path so pickle and
``from molpy.core.schema import ColumnSpec`` resolve the same class the binder
declares (``module = "molpy.core.schema"``).

Block-name constants (``ATOMS``, ``BONDS``, … and ``TOPOLOGY``) are whatever
the native module exports. Adding one in Rust adds it here with no edit.
:func:`relation_endpoints` says which block and columns a relation block's
rows point into.
"""

from molpy._core import schema as _schema

ColumnSpec = _schema.ColumnSpec

BlockSpec = _schema.BlockSpec

columns = _schema.columns

blocks = _schema.blocks

column = _schema.column

block = _schema.block

relation_endpoints = _schema.relation_endpoints

to_json = _schema.to_json

to_markdown = _schema.to_markdown

_CONSTS = [name for name in dir(_schema) if name.isupper()]

globals().update({name: getattr(_schema, name) for name in _CONSTS})

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

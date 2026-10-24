"""Type stub of :mod:`molpy.core.schema`: the specs and functions it binds
from ``molpy._core.schema``, and the block-name constants it copies at
import time. The ``types`` gate's stubtest holds the names to the runtime's,
so a block added in Rust fails there until it is declared here."""

from molpy import _core

__all__ = [
    "ANGLES",
    "ATOMS",
    "BONDS",
    "BlockSpec",
    "CMAPS",
    "CONSTRAINTS",
    "ColumnSpec",
    "DIHEDRALS",
    "DRUDES",
    "EXCLUSIONS",
    "IMPROPERS",
    "MEMBERS",
    "PAIR_OVERRIDES",
    "PAIRS",
    "TOPOLOGY",
    "VIRTUAL_SITES",
    "block",
    "blocks",
    "column",
    "columns",
    "relation_endpoints",
    "to_json",
    "to_markdown",
]

ColumnSpec = _core.schema.ColumnSpec
BlockSpec = _core.schema.BlockSpec
columns = _core.schema.columns
blocks = _core.schema.blocks
column = _core.schema.column
block = _core.schema.block
relation_endpoints = _core.schema.relation_endpoints
to_json = _core.schema.to_json
to_markdown = _core.schema.to_markdown

ANGLES: str
ATOMS: str
BONDS: str
CMAPS: str
CONSTRAINTS: str
DIHEDRALS: str
DRUDES: str
EXCLUSIONS: str
IMPROPERS: str
MEMBERS: str
PAIR_OVERRIDES: str
PAIRS: str
TOPOLOGY: tuple[str, ...]
VIRTUAL_SITES: str

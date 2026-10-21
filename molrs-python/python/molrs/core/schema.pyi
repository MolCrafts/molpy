"""Type stub of :mod:`molrs.core.schema`: the specs and functions it binds
from ``molrs._native.schema``, and the block-name constants it copies at
import time. ``tests/test_stub_parity.py`` holds the names to the runtime's,
so a block added in Rust fails there until it is declared here."""

from molrs._native import schema as _schema

ColumnSpec = _schema.ColumnSpec
BlockSpec = _schema.BlockSpec
columns = _schema.columns
blocks = _schema.blocks
column = _schema.column
block = _schema.block
relation_endpoints = _schema.relation_endpoints
to_json = _schema.to_json
to_markdown = _schema.to_markdown

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
PAIRS: str
TOPOLOGY: tuple[str, ...]
VIRTUAL_SITES: str

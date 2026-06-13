"""molpy Frame/Block — re-exported from the molrs Rust backend.

After the ``frame-block-sink`` cutover (chain 4/4), molpy no longer defines its
own ``Frame``/``Block`` subclasses or any Python-side object-column overflow.
The canonical rich types live in molrs: numpy-only typed columns in the Rust
Store with a pandas-style Python API (selectors, sorting, row iteration, CSV,
and dict conversion).

Identity contract::

    molpy.core.frame.Frame is molrs.Frame
    molpy.core.frame.Block is molrs.Block

This module exists only so the existing ``from molpy.core.frame import Frame,
Block`` call sites keep resolving. Object / None / ragged columns are now
rejected fail-fast at write time (``molrs.BlockDtypeError``); callers must use
numpy-representable columns.
"""

from molrs import Block, Frame

__all__ = ["Block", "Frame"]

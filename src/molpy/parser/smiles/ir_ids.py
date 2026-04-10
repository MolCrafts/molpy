"""Unified ID generator for all SMILES/BigSMILES/CGSmiles IR nodes."""

from itertools import count

_counter = count(1)


def generate_id() -> int:
    """Generate a unique ID for IR nodes."""
    return next(_counter)

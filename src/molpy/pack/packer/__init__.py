from typing import Any

from .base import Packer
from .packmol import Packmol

__all__ = ["Packmol", "Packer", "get_packer"]


def get_packer(*args: Any, **kwargs: Any) -> Packmol:
    """
    Factory function to get a packer instance.

    Currently only supports Packmol backend.

    Args:
        *args: Positional arguments passed to Packmol constructor
        **kwargs: Keyword arguments passed to Packmol constructor

    Returns:
        Packmol instance
    """
    return Packmol(*args, **kwargs)

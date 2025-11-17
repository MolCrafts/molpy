from typing import Any

from .packmol import Packmol

__all__ = ["Packmol", "get_packer"]


def get_packer(name: str, *args: Any, **kwargs: Any) -> Packmol:
    """
    Factory function to get a packer by name.

    Args:
        name: Packer name ("packmol" or "nlopt")
        *args: Positional arguments passed to packer constructor
        **kwargs: Keyword arguments passed to packer constructor

    Returns:
        Packer instance

    Raises:
        NotImplementedError: If packer name is not recognized
    """
    if name == "packmol":
        return Packmol(*args, **kwargs)
    elif name == "nlopt":
        # Nlopt packer not yet implemented
        raise NotImplementedError(
            f"Packer '{name}' not yet implemented. Use 'packmol' instead."
        )
    else:
        raise NotImplementedError(
            f"Packer '{name}' not recognized. Available: 'packmol'"
        )

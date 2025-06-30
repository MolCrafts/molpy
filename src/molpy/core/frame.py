from typing import Any, overload, TypeAlias
from collections.abc import MutableMapping, Iterator
import numpy as np
from numpy.typing import ArrayLike

from .box import Box


BlockLike: TypeAlias = dict[str, np.ndarray | ArrayLike]

class Block(MutableMapping[str, np.ndarray]):
    """
    Lightweight container that maps variable names → NumPy arrays.

    • Behaves like a dict but auto-casts any assigned value to ndarray.
    • All built-in `dict`/`MutableMapping` helpers work out of the box.

    Examples
    --------
    >>> blk = Block()
    >>> blk["xyz"] = [[0, 0, 0], [1, 1, 1]]
    >>> "xyz" in blk
    True
    >>> len(blk)
    1
    >>> blk["xyz"].dtype
    dtype('float64')
    """

    __slots__ = ("_vars",)

    # ------------------------------------------------------------------
    def __init__(self, vars_: dict[str, np.ndarray | ArrayLike] = {}) -> None:
        # NOTE: we force every value to ndarray once, so later reads are safe.
        try:
            self._vars: dict[str, np.ndarray] = {
                k: np.asarray(v) for k, v in vars_.items()
            }
        except Exception:
            raise ValueError("Value must be a BlockLike, i.e. dict[str, np.ndarray | ArrayLike]")

    # ------------------------------------------------------------------ core mapping API

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ... 

    @overload
    def __getitem__(self, key: int | slice) -> dict[str, np.ndarray|int|float|str|Any]: ...  # type: ignore[override]

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, (int, slice)):
            return {
                k: (v[key] if v[key].ndim > 0 else v[key].item()) for k, v in self._vars.items()
            }
        elif isinstance(key, str):
            return self._vars[key]
        else:
            raise KeyError(f"Invalid key type: {type(key)}. Expected str, int, or slice.")

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        self._vars[key] = np.asarray(value)

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        del self._vars[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self._vars)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._vars)

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        """Check if a variable exists in this block."""
        return key in self._vars

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a JSON-serialisable copy (arrays → Python lists)."""
        return {k: v for k, v in self._vars.items()}

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray]) -> "Block":
        """Inverse of :meth:`to_dict`."""
        return cls({k: np.asarray(v) for k, v in data.items()})

    def copy(self) -> "Block":
        """Shallow copy (arrays are **not** copied)."""
        return Block(self._vars.copy())

    # ------------------------------------------------------------------ repr / str
    def __repr__(self) -> str:
        contents = ", ".join(f"{k}: shape={v.shape}" for k, v in self._vars.items())
        return f"Block({contents})"

    @property
    def nrows(self) -> int:
        """Return the number of rows in the first variable (if any)."""
        if not self._vars:
            return 0
        return len(next(iter(self._vars.values())))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the first variable (if any)."""
        if not self._vars:
            return ()
        return self.nrows, len(self)


class Frame:
    """
    Hierarchical numerical data container.

        Frame
        ├- blocks (dict[str, Block])
        ├- box    (simulation box)
        ├- metadata   (dict[str, Any])  # metadata
    """

    def __init__(self, *, box: Box | None = None, **props) -> None:
        # guarantee a root block even if none supplied
        self._blocks: dict[str, Block] = {}
        self.box: Box | None = box
        self.metadata = props

    # ---------- main get/set --------------------------------------------

    @overload
    def __getitem__(self, key: str) -> Block: ...  # str  → Block

    @overload
    def __getitem__(self, key: tuple[str, str]) -> np.ndarray: ...  # tuple→ ndarray

    def __getitem__(self, key: str | tuple[str, str]) -> np.ndarray | Block:
        if isinstance(key, tuple):
            grp, var = key
            return self._blocks[grp][var]
        return self._blocks[key]

    def __setitem__(self, key: str | tuple[str, str], value: BlockLike | Block):

        if isinstance(key, tuple):
            grp, var = key
            self._blocks.setdefault(grp, Block())[var] = value
        else:
            if not isinstance(value, Block):
                value = Block(value)
            self._blocks[key] = value

    def __delitem__(self, key: str | tuple[str, str]) -> None:
        del self[key]

    def __contains__(self, key: str | tuple[str, str]) -> bool:
        if isinstance(key, tuple):
            grp, var = key
            return grp in self._blocks and var in self._blocks[grp]
        return key in self._blocks

    # ---------- helpers -------------------------------------------------
    def blocks(self) -> Iterator[str]:
        return iter(self._blocks)

    def variables(self, block: str) -> Iterator[str]:
        return iter(self._blocks[block])

    # ---------- (de)serialization --------------------------------------
    def to_dict(self) -> dict:
        block_dict = {g: grp.to_dict() for g, grp in self._blocks.items()}
        meta_dict = {k: v for k, v in self.metadata.items()}
        return {
            "blocks": block_dict,
            "metadata": meta_dict,
            "box": self.box.to_dict() if self.box else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Frame":
        blocks = {g: Block.from_dict(grp) for g, grp in data["blocks"].items()}
        box = Box.from_dict(data["box"]) if data.get("box") else None
        frame = cls(blocks=blocks, box=box)
        frame.metadata = data.get("metadata", {})
        return frame

    # ---------- repr ----------------------------------------------------
    def __repr__(self) -> str:
        txt = ["Frame("]
        for g, grp in self._blocks.items():
            for k, v in grp._vars.items():
                txt.append(f"  [{g}] {k}: shape={v.shape}")
        return "\n".join(txt) + "\n)"

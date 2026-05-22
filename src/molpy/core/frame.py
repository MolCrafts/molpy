import csv
from collections.abc import Iterator, Mapping, MutableMapping
from io import StringIO
from pathlib import Path
from typing import Any, Self, overload

import molrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .selector import Selector

type BlockLike = Mapping[str, ArrayLike]


class Block(MutableMapping[str, np.ndarray]):
    """Lightweight container that maps variable names -> NumPy arrays.

    Backed by ``molrs.Block`` (Rust FFI column store). All data lives in
    the Rust Store; ``view()`` / ``insert()`` / ``remove()`` operate
    directly on that memory. Python convenience methods (``sort``,
    ``from_csv``, ``iterrows``, …) are built on top.

    • Behaves like a dict but auto-casts any assigned value to ndarray.
    • All built-in ``dict``/``MutableMapping`` helpers work out of the box.
    • Supports advanced indexing: by key, by index/slice, by mask, by list of keys.

    Parameters
    ----------
    vars_ : dict[str, ArrayLike] or None, optional
        Initial data to populate the Block. Keys are variable names,
        values are array-like data that will be converted to numpy arrays.

    Examples
    --------
    Create and access basic data:

    >>> blk = Block()
    >>> blk["x"] = [0.0, 1.0, 2.0]
    >>> blk["y"] = [0.0, 0.0, 0.0]
    >>> "x" in blk
    True
    >>> len(blk)
    2
    >>> blk["x"].dtype
    dtype('float64')

    Multiple indexing methods:

    >>> blk = Block({"id": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
    >>> blk[0]  # Access single row, returns dict
    {'id': 1, 'x': 10.0}
    >>> blk[0:2]  # Slice access
    {'id': array([1, 2]), 'x': array([10., 20.])}
    >>> blk[["id", "x"]]  # Multi-column access, returns 2D array (requires same dtype)
    Traceback (most recent call last):
        ...
    ValueError: Arrays must have the same dtype...

    Using boolean masks for filtering:

    >>> blk = Block({"id": [1, 2, 3, 4, 5], "mol_id": [1, 1, 2, 2, 3]})
    >>> mask = blk["mol_id"] < 3
    >>> sub_blk = blk[mask]
    >>> sub_blk["id"]
    array([1, 2, 3, 4])
    >>> sub_blk.nrows
    4

    Sorting:

    >>> blk = Block({"x": [3, 1, 2], "y": [30, 10, 20]})
    >>> sorted_blk = blk.sort("x")  # Returns new Block
    >>> sorted_blk["x"]
    array([1, 2, 3])
    >>> _ = blk.sort_("x")  # In-place sort, returns self
    >>> blk["x"]
    array([1, 2, 3])
    """

    __slots__ = ("_inner", "_objects")

    def __init__(self, vars_: BlockLike | None = None) -> None:
        self._inner = molrs.Block()
        self._objects: dict[str, np.ndarray] = {}
        if vars_ is not None:
            if not isinstance(vars_, dict):
                raise ValueError(f"vars_ must be a dict, got {type(vars_)}")
            for k, v in vars_.items():
                try:
                    self[k] = v
                except Exception as e:
                    raise ValueError(
                        f"Value must be a BlockLike, i.e. dict[str, np.ndarray], "
                        f"but got {type(v)} for key {k}"
                    ) from e

    # --- core mapping API

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, key: int | slice) -> "Block": ...  # type: ignore[override]

    @overload
    def __getitem__(self, key: list[str]) -> np.ndarray: ...  # type: ignore[override]

    @overload
    def __getitem__(self, key: np.ndarray) -> "Block": ...  # type: ignore[override]

    @overload
    def __getitem__(self, key: Selector) -> "Block": ...  # type: ignore[override]

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            if key in self._objects:
                return self._objects[key]
            val = self._inner.view(key)
            return np.asarray(val) if isinstance(val, list) else val
        elif isinstance(key, int):
            # Return a plain dict for single-row access (avoids molrs
            # schema conflicts when 2D columns like "xyz" produce
            # different-length rows than scalar columns).
            return {
                k: (v[key] if (v[key].ndim > 0) else v[key].item())
                for k, v in self._as_dict().items()
            }
        elif isinstance(key, slice):
            return Block({k: v[key] for k, v in self._as_dict().items()})
        elif isinstance(key, list):
            if not key:
                raise KeyError("Empty list not allowed for indexing")
            for k in key:
                if k not in self:
                    raise KeyError(f"Key '{k}' not found in Block")
            arrays = [self._view_array(k) for k in key]
            first = arrays[0]
            for i, arr in enumerate(arrays[1:], 1):
                if arr.shape != first.shape:
                    raise ValueError(
                        f"Arrays must have the same shape. Array {key[0]} has shape "
                        f"{first.shape}, but array {key[i]} has shape {arr.shape}"
                    )
                if arr.dtype != first.dtype:
                    raise ValueError(
                        f"Arrays must have the same dtype. Array {key[0]} has dtype "
                        f"{first.dtype}, but array {key[i]} has dtype {arr.dtype}"
                    )
            return np.column_stack(arrays)
        elif isinstance(key, tuple):
            return np.array([self[k] for k in key])
        elif isinstance(key, np.ndarray):
            return Block({k: self._view_array(k)[key] for k in self.keys()})
        elif isinstance(key, Selector):
            return key(self)
        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. "
                "Expected str, int, slice, list[str], or np.ndarray."
            )

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        arr = np.asarray(value)
        if arr.ndim == 0:
            return  # skip scalar columns (molrs requires ≥1D)
        if arr.dtype.kind == "O":
            self._objects[key] = arr
            return
        if len(arr) == 0:
            self._objects[key] = arr  # empty arrays go to Python-side storage
            return
        if key in self._inner.keys():
            self._inner.remove(key)
        self._objects.pop(key, None)
        self._inner.insert(key, arr)

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        if key in self._objects:
            del self._objects[key]
        else:
            self._inner.remove(key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        yield from self._inner.keys()
        yield from self._objects

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._inner.keys()) + len(self._objects)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._inner or key in self._objects

    # ------------------------------------------------------------------ helpers

    def _view_array(self, key: str) -> np.ndarray:
        """Return column *key* always as an ndarray (convert list→array)."""
        if key in self._objects:
            return self._objects[key]
        val = self._inner.view(key)
        return np.asarray(val) if isinstance(val, list) else val

    def _as_dict(self) -> dict[str, np.ndarray]:
        """Return all columns as a dict of numpy arrays (views into Rust memory)."""
        result = {k: self._view_array(k) for k in self._inner.keys()}
        result.update(self._objects)
        return result

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a JSON-serialisable copy (arrays -> Python lists)."""
        result = {k: np.asarray(self._view_array(k)) for k in self._inner.keys()}
        result.update(self._objects)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray] | molrs.Block) -> "Block":
        """Create a Block from a dict or wrap a ``molrs.Block`` (zero-copy view).

        When *data* is a ``molrs.Block``, the returned block accesses arrays
        via ``view()`` — no array data is copied.
        """
        if isinstance(data, molrs.Block):
            block = cls.__new__(cls)
            block._inner = data
            block._objects = {}
            return block
        return cls({k: np.asarray(v) for k, v in data.items()})

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path | StringIO,
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: list[str] | None = None,
        **kwargs,
    ) -> "Block":
        """Create a Block from a CSV file or StringIO.

        Parameters
        ----------
        filepath : str, Path, or StringIO
            Path to the CSV file or StringIO object
        delimiter : str, default=","
            CSV delimiter character
        encoding : str, default="utf-8"
            File encoding (ignored for StringIO)
        header : list[str] or None, default=None
            Column names. If None, first row is used as headers.
            If provided, CSV is assumed to have no header row.
        **kwargs
            Additional arguments passed to csv.reader

        Returns
        -------
        Block
            A new Block instance with data from the CSV file

        Examples
        --------
        Read from StringIO:

        >>> from io import StringIO
        >>> csv_data = StringIO("x,y,z\\n0,1,2\\n3,4,5")
        >>> block = Block.from_csv(csv_data)
        >>> block["x"]
        array([0, 3])
        >>> block.nrows
        2

        CSV without header:

        >>> csv_no_header = StringIO("0,1,2\\n3,4,5")
        >>> block = Block.from_csv(csv_no_header, header=["x", "y", "z"])
        >>> list(block.keys())
        ['x', 'y', 'z']
        >>> block.nrows
        2
        """
        if isinstance(filepath, StringIO):
            csvfile = filepath
            csvfile.seek(0)
            close_file = False
        else:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"CSV file not found: {filepath}")
            csvfile = open(filepath, encoding=encoding, newline="")
            close_file = True

        try:
            reader = csv.reader(csvfile, delimiter=delimiter, **kwargs)

            if header is None:
                try:
                    headers = next(reader)
                except StopIteration:
                    raise ValueError("CSV file is empty")
            else:
                headers = header

            raw_data = {h: [] for h in headers}

            for row in reader:
                for i, header_name in enumerate(headers):
                    raw_data[header_name].append(row[i])

            data = {}
            for k, v in raw_data.items():
                for dtype in (int, float, str):
                    try:
                        data[k] = np.array(v, dtype=dtype)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Failed to convert {k} to any of int, float, str")

            return cls(data)
        finally:
            if close_file:
                csvfile.close()

    def copy(self) -> "Block":
        """Deep copy (data is copied into a new Rust Store)."""
        new = Block()
        for k in self._inner.keys():
            new[k] = np.asarray(self._view_array(k))
        for k, v in self._objects.items():
            new._objects[k] = v.copy()
        return new

    def rename(self, old_key: str, new_key: str) -> None:
        """Rename a column key in-place.

        Args:
            old_key: Existing column name.
            new_key: New column name.

        Raises:
            KeyError: If *old_key* does not exist.
        """
        if old_key in self._objects:
            self._objects[new_key] = self._objects.pop(old_key)
        elif old_key in self._inner.keys():
            arr = self._view_array(old_key)
            self._inner.remove(old_key)
            self._inner.insert(new_key, arr)
        else:
            raise KeyError(f"Column '{old_key}' not found in Block")

    def _sort(self, key: str, *, reverse: bool = False) -> dict[str, NDArray[Any]]:
        """Sort variables by a specific key and return sorted data."""
        if self.nrows == 0:
            return {}

        if key not in self:
            raise KeyError(f"Variable '{key}' not found in block")

        sort_indices = np.argsort(self._view_array(key))
        if reverse:
            sort_indices = sort_indices[::-1]

        nrows = self.nrows
        sorted_vars: dict[str, np.ndarray] = {}
        for var_name in self.keys():
            var_data = self._view_array(var_name)
            if len(var_data) != nrows:
                raise ValueError(
                    f"Variable '{var_name}' has different length than '{key}'"
                )
            sorted_vars[var_name] = var_data[sort_indices]

        return sorted_vars

    def sort(self, key: str, *, reverse: bool = False) -> "Block":
        """Sort the block by a specific variable and return a new sorted Block.

        This method creates a new Block instance with sorted data, leaving the
        original Block unchanged.

        Args:
            key: The variable name to sort by.
            reverse: If True, sort in descending order. Defaults to False.

        Returns:
            A new Block with sorted data.

        Raises:
            KeyError: If the key variable doesn't exist in the block.
            ValueError: If any variable has different length than the key variable.
        """
        sorted_vars = self._sort(key, reverse=reverse)
        return Block(sorted_vars)

    def sort_(self, key: str, *, reverse: bool = False) -> "Self":
        """Sort the block in-place by a specific variable.

        This method modifies the current Block instance by sorting all variables
        according to the specified key. The original data is overwritten.

        Args:
            key: The variable name to sort by.
            reverse: If True, sort in descending order. Defaults to False.

        Returns:
            Self (for method chaining).

        Raises:
            KeyError: If the key variable doesn't exist in the block.
            ValueError: If any variable has different length than the key variable.
        """
        sorted_vars = self._sort(key, reverse=reverse)
        if sorted_vars:
            for k, v in sorted_vars.items():
                if k in self._objects:
                    self._objects[k] = v
                else:
                    self._inner.remove(k)
                    self._inner.insert(k, v)
        return self

    # ------------------------------------------------------------------ repr / str

    def __repr__(self) -> str:
        contents = ", ".join(
            f"{k}: shape={self._view_array(k).shape}" for k in self.keys()
        )
        return f"Block({contents})"

    @property
    def nrows(self) -> int:
        """Return the number of rows (0 if empty)."""
        n = self._inner.nrows
        if n is not None and n > 0:
            return n
        for arr in self._objects.values():
            return len(arr)
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape (nrows, ncols) of the block."""
        if self.nrows == 0:
            return ()
        return self.nrows, len(self)

    def iterrows(self, n: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
        """Iterate over rows of the block.

        Returns
        -------
        Iterator[tuple[int, dict[str, Any]]]
            An iterator yielding (index, row_data) pairs.
        """
        nrows = self.nrows if n is None else n
        if nrows == 0:
            return

        var_names = list(self.keys())

        for i in range(nrows):
            row_data = {}
            for var_name in var_names:
                var_data = self._view_array(var_name)
                if i < len(var_data):
                    if var_data.ndim == 0:
                        row_data[var_name] = var_data.item()
                    else:
                        row_data[var_name] = var_data[i]
                else:
                    row_data[var_name] = None

            yield i, row_data

    def itertuples(self, index: bool = True, name: str = "Row") -> Iterator[Any]:
        """Iterate over rows of the block as named tuples.

        Parameters
        ----------
        index : bool, default=True
            If True, include the row index as the first element
        name : str, default="Row"
            The name of the named tuple class

        Returns
        -------
        Iterator[Any]
            An iterator yielding named tuples for each row
        """
        from collections import namedtuple

        nrows = self.nrows
        if nrows == 0:
            return

        var_names = list(self.keys())
        field_names = ["Index", *var_names] if index else var_names
        RowTuple = namedtuple(name, field_names)

        for i in range(nrows):
            row_values = []
            if index:
                row_values.append(i)

            for var_name in var_names:
                var_data = self._view_array(var_name)
                if i < len(var_data):
                    if var_data.ndim == 0:
                        row_values.append(var_data.item())
                    else:
                        row_values.append(var_data[i])
                else:
                    row_values.append(None)

            yield RowTuple(*row_values)


class Frame:
    """Hierarchical numerical data container with named blocks.

    Backed by ``molrs.Frame`` (Rust FFI). Each ``Frame`` holds named
    ``Block`` objects and ``metadata: dict[str, Any]`` for arbitrary
    Python-side metadata.

    The ``box`` property delegates to the underlying ``molrs.Frame.box``
    (which accepts ``molpy.Box`` because ``molpy.Box`` inherits
    ``molrs.Box``).

    Structure:
        Frame
        ├─ _inner: molrs.Frame          # Rust storage
        ├─ metadata: dict[str, Any]     # Python-side metadata
        └─ box (via _inner.box)

    Args:
        blocks (dict[str, Block | dict] | None, optional): Initial blocks.
        **props: Arbitrary keyword arguments stored in metadata.

    Examples:
        Create Frame and add data blocks:

        >>> frame = Frame()
        >>> frame["atoms"] = Block({"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
        >>> frame["atoms"]["x"]
        array([0., 1.])
        >>> frame["atoms"].nrows
        2

        Initialize with nested dictionaries:

        >>> frame = Frame(blocks={
        ...     "atoms": {"id": [1, 2, 3], "type": ["C", "H", "H"]},
        ...     "bonds": {"atomi": [0, 0], "atomj": [1, 2]}
        ... })
        >>> list(frame._blocks)
        ['atoms', 'bonds']
        >>> frame["atoms"]["id"]
        array([1, 2, 3])

        Add metadata:

        >>> frame = Frame()
        >>> frame.metadata["timestep"] = 0
        >>> frame.metadata["description"] = "Test system"
        >>> frame.metadata["timestep"]
        0
        >>> frame.metadata["description"]
        'Test system'
    """

    __slots__ = ("_inner", "metadata", "_block_objects")

    def __init__(
        self,
        blocks: dict[str, Block | BlockLike] | None = None,
        **props: Any,
    ) -> None:
        self._inner = molrs.Frame()
        self.metadata: dict[str, Any] = dict(props)
        self._block_objects: dict[str, dict[str, np.ndarray]] = {}
        if blocks is not None:
            if not isinstance(blocks, dict):
                raise ValueError(f"blocks must be a dict, got {type(blocks)}")
            for key, value in blocks.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Block keys must be strings, got {type(key)} for key {key}"
                    )
                if isinstance(value, Block):
                    self[key] = value
                elif isinstance(value, dict):
                    self[key] = Block(value)
                else:
                    try:
                        self[key] = Block(value)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert value to Block for key '{key}' "
                            f"(type {type(value)}): {e}"
                        )

    # ---------- main get/set --------------------------------------------

    def __getitem__(self, key: str) -> Block:
        """Get a Block by name.

        Args:
            key (str): Name of the block to retrieve.

        Returns:
            Block: The requested block (wraps the molrs.Block zero-copy).

        Raises:
            KeyError: If the block name doesn't exist.
        """
        block = Block.from_dict(self._inner[key])
        if key in self._block_objects:
            block._objects.update(self._block_objects[key])
        return block

    def __setitem__(self, key: str, value: BlockLike | Block) -> None:
        """Set a Block by name.

        Args:
            key (str): Name of the block to set.
            value (Block | dict[str, ArrayLike]): Block to store.
        """
        if isinstance(value, Block):
            self._inner[key] = value._inner
            if value._objects:
                self._block_objects[key] = value._objects.copy()
            elif key in self._block_objects:
                del self._block_objects[key]
        elif isinstance(value, molrs.Block):
            self._inner[key] = value
            self._block_objects.pop(key, None)
        else:
            blk = molrs.Block()
            obj_cols: dict[str, np.ndarray] = {}
            for k, v in value.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    continue
                if arr.dtype.kind == "O" or len(arr) == 0:
                    obj_cols[k] = arr
                else:
                    blk.insert(k, arr)
            self._inner[key] = blk
            if obj_cols:
                self._block_objects[key] = obj_cols
            elif key in self._block_objects:
                del self._block_objects[key]

    def __delitem__(self, key: str) -> None:
        self._block_objects.pop(key, None)
        del self._inner[key]

    def __contains__(self, key: str) -> bool:
        return key in self._inner

    def __len__(self) -> int:
        """Return the number of blocks in the frame."""
        return len(self._inner)

    def keys(self):
        """Return block names."""
        return self._inner.keys()

    # ---------- helpers -------------------------------------------------

    @property
    def _blocks(self) -> dict[str, Block]:
        """Return a snapshot dict of block names → Block objects."""
        return {name: self[name] for name in self.keys()}

    @property
    def blocks(self) -> Iterator["Block"]:
        """Iterate over stored Block objects."""
        return iter(self._blocks.values())

    @property
    def box(self):
        """The simulation Box (FREE box when unset)."""
        raw = self._inner.box
        if raw is None:
            from molpy.core.box import Box

            return Box()  # FREE
        from molpy.core.box import Box

        return Box(matrix=raw.matrix, pbc=raw.pbc, origin=raw.origin)

    @box.setter
    def box(self, value) -> None:
        from molpy.core.box import Box

        if value is None:
            self._inner.box = None
        elif isinstance(value, Box):
            if getattr(value, "_is_free", False):
                self._inner.box = None
            else:
                self._inner.box = value  # molpy.Box IS a molrs.Box
        elif isinstance(value, molrs.Box):
            self._inner.box = value
        else:
            raise TypeError(f"Expected Box or None, got {type(value)}")

    # ---------- (de)serialization --------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert Frame to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing "blocks" and "metadata" keys.
        """
        block_dict = {name: self[name].to_dict() for name in self.keys()}
        return {"blocks": block_dict, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | molrs.Frame) -> "Frame":
        """Create a Frame from a dict or wrap a ``molrs.Frame`` (zero-copy view).

        When *data* is a ``molrs.Frame``, treats it as a nested dict of numpy
        arrays — blocks are accessed via ``view()``.  No array data is copied.
        """
        if isinstance(data, molrs.Frame):
            frame = cls.__new__(cls)
            frame._inner = data
            frame.metadata = dict(data.meta) if data.meta else {}
            frame._block_objects = {}
            return frame
        blocks = {name: Block.from_dict(blk) for name, blk in data["blocks"].items()}
        frame = cls(blocks=blocks)
        frame.metadata = data.get("metadata", {})
        return frame

    def copy(self) -> "Frame":
        """Create a deep copy of the Frame.

        Returns:
            Frame: A new Frame with copied blocks and metadata.
        """
        new_frame = Frame()
        for name in self.keys():
            new_frame[name] = self[name].copy()
        new_frame.box = self.box
        new_frame.metadata = self.metadata.copy()
        return new_frame

    def to_molrs(self) -> molrs.Frame:
        """Return the underlying ``molrs.Frame`` for compute API calls.

        This is a zero-copy operation — the returned frame shares the same
        Rust Store as this ``Frame``.
        """
        return self._inner

    # ---------- repr ----------------------------------------------------

    def __repr__(self) -> str:
        txt = ["Frame("]
        for name in self.keys():
            blk = self[name]
            for k in blk.keys():
                txt.append(f"  [{name}] {k}: shape={blk[k].shape}")
        return "\n".join(txt) + "\n)"

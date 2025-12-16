import csv
from collections.abc import Iterator, Mapping, MutableMapping
from io import StringIO
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .selector import Selector
from .topology import Topology

type BlockLike = Mapping[str, ArrayLike]


class Block(MutableMapping[str, np.ndarray]):
    """
    Lightweight container that maps variable names -> NumPy arrays.

    • Behaves like a dict but auto-casts any assigned value to ndarray.
    • All built-in `dict`/`MutableMapping` helpers work out of the box.
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

    >>> blk = Block({"id": [1, 2, 3, 4, 5], "mol": [1, 1, 2, 2, 3]})
    >>> mask = blk["mol"] < 3
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

    __slots__ = ("_vars",)

    def __init__(self, vars_: BlockLike | None = None) -> None:
        self._vars: dict[str, np.ndarray] = {k: np.asarray(v) for k, v in {}.items()}
        if vars_ is not None:
            if not isinstance(vars_, dict):
                raise ValueError(f"vars_ must be a dict, got {type(vars_)}")
            for k, v in vars_.items():
                try:
                    self._vars[k] = np.asarray(v)
                except Exception as e:
                    raise ValueError(
                        f"Value must be a BlockLike, i.e. dict[str, np.ndarray], but got {type(v)} for key {k}"
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
        if isinstance(key, (int, slice)):
            # Return a new Block containing the selected rows.
            # For integer indices, scalars are converted to ndarray via np.asarray
            return Block(
                {
                    k: (
                        v[key]
                        if (isinstance(key, slice) or v[key].ndim > 0)
                        else np.asarray(v[key].item())
                    )
                    for k, v in self._vars.items()
                }
            )
        elif isinstance(key, str):
            return self._vars[key]
        elif isinstance(key, list):
            # Handle list of column names for concatenation
            if not key:
                raise KeyError("Empty list not allowed for indexing")

            # Check if all keys exist
            for k in key:
                if k not in self._vars:
                    raise KeyError(f"Key '{k}' not found in Block")

            # Get the arrays
            arrays = [self._vars[k] for k in key]

            # Check if all arrays have the same shape and dtype
            if not arrays:
                raise ValueError("No arrays to concatenate")

            first_array = arrays[0]
            for i, arr in enumerate(arrays[1:], 1):
                if arr.shape != first_array.shape:
                    raise ValueError(
                        f"Arrays must have the same shape. Array {key[0]} has shape {first_array.shape}, but array {key[i]} has shape {arr.shape}"
                    )
                if arr.dtype != first_array.dtype:
                    raise ValueError(
                        f"Arrays must have the same dtype. Array {key[0]} has dtype {first_array.dtype}, but array {key[i]} has dtype {arr.dtype}"
                    )

            # Concatenate along the last axis
            return np.column_stack(arrays)
        elif isinstance(key, tuple):
            return np.array([self[k] for k in key])
        elif isinstance(key, np.ndarray):
            return Block({k: v[key] for k, v in self._vars.items()})
        elif isinstance(key, Selector):
            return key(self)
        else:
            raise KeyError(
                f"Invalid key type: {type(key)}. Expected str, int, slice, list[str], or np.ndarray."
            )

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
        """Return a JSON-serialisable copy (arrays -> Python lists)."""
        return {k: v for k, v in self._vars.items()}

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray]) -> "Block":
        """Inverse of :meth:`to_dict`."""
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
        """
        Create a Block from a CSV file or StringIO.

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
        # Determine type
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

            # Handle headers
            if header is None:
                # Use first row as headers
                try:
                    headers = next(reader)
                except StopIteration:
                    raise ValueError("CSV file is empty")
            else:
                # Use provided headers, no header row in CSV
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
        """Shallow copy (arrays are **not** copied)."""
        return Block(self._vars.copy())  # type: ignore[arg-type]

    def _sort(self, key: str, *, reverse: bool = False) -> dict[str, NDArray[Any]]:
        """Sort variables by a specific key and return sorted data.

        This is a private helper method that performs the actual sorting logic.

        Args:
            key: The variable name to sort by.
            reverse: If True, sort in descending order. Defaults to False.

        Returns:
            Dictionary with sorted variable data.

        Raises:
            KeyError: If the key variable doesn't exist in the block.
            ValueError: If any variable has different length than the key variable.
        """
        if not self._vars:
            return {}

        if key not in self._vars:
            raise KeyError(f"Variable '{key}' not found in block")

        # Get the sorting indices
        sort_indices = np.argsort(self._vars[key])
        if reverse:
            sort_indices = sort_indices[::-1]

        # Create sorted data
        sorted_vars: dict[str, np.ndarray] = {}
        for var_name, var_data in self._vars.items():
            if len(var_data) != len(self._vars[key]):
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

        Example:
            >>> blk = Block({"x": [3, 1, 2], "y": [30, 10, 20]})
            >>> sorted_blk = blk.sort("x")
            >>> sorted_blk["x"]
            array([1, 2, 3])
            >>> sorted_blk["y"]
            array([10, 20, 30])
            >>> # Original block is unchanged
            >>> blk["x"]
            array([3, 1, 2])
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

        Example:
            >>> blk = Block({"x": [3, 1, 2], "y": [30, 10, 20]})
            >>> _ = blk.sort_("x")  # Returns self for chaining
            >>> blk["x"]
            array([1, 2, 3])
            >>> blk["y"]
            array([10, 20, 30])
            >>> # Original data is now sorted
        """
        sorted_vars = self._sort(key, reverse=reverse)
        if sorted_vars:  # Only update if we have data to sort
            self._vars.update(sorted_vars)
        return self

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

    def iterrows(self, n: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
        """
        Iterate over rows of the block.

        Returns
        -------
        Iterator[tuple[int, dict[str, Any]]]
            An iterator yielding (index, row_data) pairs where:
            - index: int, the row index
            - row_data: dict, mapping variable names to their values for this row

        Examples
        --------
        >>> blk = Block({
        ...     "id": [1, 2, 3],
        ...     "type": ["C", "O", "N"],
        ...     "x": [0.0, 1.0, 2.0],
        ...     "y": [0.0, 0.0, 1.0],
        ...     "z": [0.0, 0.0, 0.0]
        ... })
        >>> for index, row in blk.iterrows():  # doctest: +SKIP
        ...     print(f"Row {index}: {row}")
        Row 0: {'id': 1, 'type': 'C', 'x': 0.0, 'y': 0.0, 'z': 0.0}
        Row 1: {'id': 2, 'type': 'O', 'x': 1.0, 'y': 0.0, 'z': 0.0}
        Row 2: {'id': 3, 'type': 'N', 'x': 2.0, 'y': 1.0, 'z': 0.0}

        Notes
        -----
        This method is similar to pandas DataFrame.iterrows() but returns
        a dictionary for each row instead of a pandas Series.
        """
        if not self._vars:
            return

        # Get the number of rows from the first variable
        nrows = self.nrows if n is None else n
        if nrows == 0:
            return

        # Get all variable names
        var_names = list(self._vars.keys())

        for i in range(nrows):
            row_data = {}
            for var_name in var_names:
                var_data = self._vars[var_name]
                if i < len(var_data):
                    # Handle scalar values
                    if var_data.ndim == 0:
                        row_data[var_name] = var_data.item()
                    else:
                        row_data[var_name] = var_data[i]
                else:
                    # Handle case where variable has fewer rows
                    row_data[var_name] = None

            yield i, row_data

    def itertuples(self, index: bool = True, name: str = "Row") -> Iterator[Any]:
        """
        Iterate over rows of the block as named tuples.

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

        Examples
        --------
        >>> blk = Block({
        ...     "id": [1, 2, 3],
        ...     "type": ["C", "O", "N"],
        ...     "x": [0.0, 1.0, 2.0]
        ... })
        >>> for row in blk.itertuples():
        ...     print(f"Index: {row.Index}, ID: {row.id}, Type: {row.type}")
        Index: 0, ID: 1, Type: C
        Index: 1, ID: 2, Type: O
        Index: 2, ID: 3, Type: N

        Notes
        -----
        This method is similar to pandas DataFrame.itertuples().
        """
        from collections import namedtuple

        if not self._vars:
            return

        # Get the number of rows from the first variable
        nrows = self.nrows
        if nrows == 0:
            return

        # Get all variable names
        var_names = list(self._vars.keys())

        # Create field names for the named tuple
        field_names = ["Index", *var_names] if index else var_names

        # Create the named tuple class
        RowTuple = namedtuple(name, field_names)

        for i in range(nrows):
            row_values = []
            if index:
                row_values.append(i)

            for var_name in var_names:
                var_data = self._vars[var_name]
                if i < len(var_data):
                    # Handle scalar values
                    if var_data.ndim == 0:
                        row_values.append(var_data.item())
                    else:
                        row_values.append(var_data[i])
                else:
                    # Handle case where variable has fewer rows
                    row_values.append(None)

            yield RowTuple(*row_values)


class Frame:
    """Hierarchical numerical data container with named blocks.

    Frame stores multiple Block objects under string keys (e.g., "atoms", "bonds")
    and allows arbitrary metadata to be attached. It's designed for molecular
    simulation data where different entity types need separate tabular storage.

    Structure:
        Frame
        ├─ blocks: dict[str, Block]     # Named data blocks
        └─ metadata: dict[str, Any]     # Arbitrary metadata (box, timestep, etc.)

    Args:
        blocks (dict[str, Block | dict] | None, optional): Initial blocks. If a
            dict value is not a Block, it will be converted.
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
        ...     "bonds": {"i": [0, 0], "j": [1, 2]}
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

        Chained access:

        >>> frame = Frame(blocks={"atoms": {"x": [1, 2, 3], "y": [4, 5, 6]}})
        >>> atoms = frame["atoms"]
        >>> xyz_combined = atoms[["x", "y"]]
        >>> xyz_combined.shape
        (3, 2)

        Iterate over all blocks and variables:

        >>> frame = Frame(blocks={
        ...     "atoms": {"id": [1, 2], "mass": [12.0, 1.0]},
        ...     "bonds": {"i": [0], "j": [1]}
        ... })
        >>> for block_name in frame._blocks:
        ...     print(f"{block_name}: {list(frame[block_name].keys())}")
        atoms: ['id', 'mass']
        bonds: ['i', 'j']
    """

    def __init__(
        self,
        blocks: dict[str, Block | BlockLike] | None = None,
        **props: Any,
    ) -> None:
        """Initialize a Frame with optional blocks and metadata.

        Args:
            blocks (dict[str, Block | BlockLike] | None, optional): Initial
                blocks. If a dict value is not a Block, it will be converted to
                a Block. Defaults to None.
            **props (Any): Arbitrary keyword arguments stored in metadata.
        """
        # guarantee a root block even if none supplied
        self._blocks: dict[str, Block] = {}
        if blocks is not None:
            self._blocks = self._validate_and_convert_blocks(blocks)
        self.metadata: dict[str, Any] = props

    def _validate_and_convert_blocks(
        self, blocks: dict[str, Block | BlockLike | Any]
    ) -> dict[str, Block]:
        """Validate and convert input blocks to ensure all values are Block instances.

        This method recursively processes nested dictionaries and converts
        all leaf values to Block instances.

        Args:
            blocks (dict[str, Block] | dict[str, dict] | dict[str, Any]): Input
                blocks. Can be:
                - dict[str, Block]: Already correct format
                - dict[str, dict]: Nested dictionaries that will be converted to Block
                - dict[str, Any]: Mixed format that will be validated and converted

        Returns:
            dict[str, Block]: Validated blocks where all values are Block instances.

        Raises:
            ValueError: If any leaf value cannot be converted to Block.
        """
        if not isinstance(blocks, dict):
            raise ValueError(f"blocks must be a dict, got {type(blocks)}")

        validated_blocks = {}

        for key, value in blocks.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Block keys must be strings, got {type(key)} for key {key}"
                )

            if isinstance(value, Block):
                # Already a Block, use as is
                validated_blocks[key] = value
            elif isinstance(value, dict):
                # Nested dict, convert to Block
                try:
                    validated_blocks[key] = Block(value)
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert nested dict to Block for key '{key}': {e}"
                    )
            else:
                # Try to convert to Block (e.g., list, array, etc.)
                try:
                    validated_blocks[key] = Block(value)
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert value to Block for key '{key}' (type {type(value)}): {e}"
                    )

        return validated_blocks

    # ---------- main get/set --------------------------------------------

    def __getitem__(self, key: str) -> Block:
        """Get a Block by name.

        Args:
            key (str): Name of the block to retrieve.

        Returns:
            Block: The requested block.

        Raises:
            KeyError: If the block name doesn't exist.

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1, 2], "y": [3, 4]}})
            >>> atoms = frame["atoms"]
            >>> atoms["x"]
            array([1, 2])
            >>> frame["nonexistent"]
            Traceback (most recent call last):
                ...
            KeyError: 'nonexistent'
        """
        return self._blocks[key]

    def __setitem__(self, key: str, value: BlockLike | Block) -> None:
        """Set a Block by name.

        Args:
            key (str): Name of the block to set.
            value (Block | dict[str, ArrayLike]): Block to store, or dict-like
                data that will be converted to Block.

        Examples:
            >>> frame = Frame()
            >>> frame["atoms"] = Block({"x": [1, 2, 3]})
            >>> frame["bonds"] = {"i": [0, 1], "j": [1, 2]}  # Auto-converted
            >>> isinstance(frame["bonds"], Block)
            True
        """
        if not isinstance(value, Block):
            value = Block(value)
        self._blocks[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a Block by name.

        Args:
            key (str): Name of the block to delete.

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1, 2]}})
            >>> del frame["atoms"]
            >>> "atoms" in frame
            False
        """
        del self._blocks[key]

    def __contains__(self, key: str) -> bool:
        """Check if a block exists.

        Args:
            key (str): Name of the block to check.

        Returns:
            bool: True if the block exists, False otherwise.

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1, 2]}})
            >>> "atoms" in frame
            True
            >>> "bonds" in frame
            False
        """
        return key in self._blocks

    # ---------- helpers -------------------------------------------------
    @property
    def blocks(self) -> Iterator["Block"]:
        """Iterate over stored Block objects.

        Returns:
            Iterator[Block]: Iterator over Block values stored in this Frame.

        Note:
            To iterate over block *names* use `for name in frame._blocks` or
            `frame._blocks.keys()`.

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1]}, "bonds": {"i": [0]}})
            >>> [b for b in frame.blocks]
            [Block(x: shape=(1,), i: shape=(1,))]
        """
        return iter(self._blocks.values())

    # NOTE: `variables` helper removed — use `frame[block_name].keys()` or
    # `set(frame[block_name])` to iterate variable names.

    # ---------- (de)serialization --------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Convert Frame to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing "blocks" and "metadata" keys.
                Blocks are converted to dictionaries via Block.to_dict().

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1, 2]}}, timestep=0)
            >>> data = frame.to_dict()
            >>> "blocks" in data
            True
            >>> "metadata" in data
            True
        """
        block_dict = {g: grp.to_dict() for g, grp in self._blocks.items()}
        meta_dict = {k: v for k, v in self.metadata.items()}
        return {"blocks": block_dict, "metadata": meta_dict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Frame":
        """Create a Frame from a dictionary representation.

        Args:
            data (dict[str, Any]): Dictionary containing "blocks" and optionally
                "metadata" keys.

        Returns:
            Frame: A new Frame instance reconstructed from the dictionary.

        Examples:
            >>> data = {
            ...     "blocks": {"atoms": {"x": [1, 2, 3]}},
            ...     "metadata": {"timestep": 0}
            ... }
            >>> frame = Frame.from_dict(data)
            >>> frame["atoms"]["x"]
            array([1, 2, 3])
            >>> frame.metadata["timestep"]
            0
        """
        blocks = {g: Block.from_dict(grp) for g, grp in data["blocks"].items()}
        frame = cls(blocks=blocks)
        frame.metadata = data.get("metadata", {})
        return frame

    def copy(self) -> "Frame":
        """Create a shallow copy of the Frame.

        Blocks are copied (shallow), but the underlying numpy arrays are not.

        Returns:
            Frame: A new Frame instance with copied blocks and metadata.

        Examples:
            >>> frame = Frame(blocks={"atoms": {"x": [1, 2, 3]}}, timestep=0)
            >>> frame_copy = frame.copy()
            >>> frame_copy.metadata["timestep"] = 1
            >>> frame.metadata["timestep"]  # Original unchanged
            0
        """
        # Copy blocks (shallow copy of Block objects)
        new_blocks = {name: block.copy() for name, block in self._blocks.items()}
        # Create new frame
        new_frame = Frame(blocks=new_blocks)
        # Copy metadata (shallow copy of dict)
        new_frame.metadata = self.metadata.copy()
        return new_frame

    # ---------- repr ----------------------------------------------------
    def __repr__(self) -> str:
        txt = ["Frame("]
        for g, grp in self._blocks.items():
            for k, v in grp._vars.items():
                txt.append(f"  [{g}] {k}: shape={v.shape}")
        return "\n".join(txt) + "\n)"

    def get_topology(self) -> Topology:
        """Get the topology of the frame from atoms and bonds blocks.

        Constructs a Topology object from the "atoms" and "bonds" blocks
        in the frame. The bonds block must contain "i" and "j" columns
        representing bond connections.

        Returns:
            Topology: A Topology object representing the molecular connectivity.

        Raises:
            KeyError: If the "atoms" or "bonds" blocks are missing from the frame.
            KeyError: If the "bonds" block is missing "i" or "j" columns.

        Examples:
            >>> frame = Frame(blocks={
            ...     "atoms": {"id": [1, 2, 3], "type": ["C", "H", "H"]},
            ...     "bonds": {"i": [0, 0], "j": [1, 2]}
            ... })
            >>> topo = frame.get_topology()
            >>> topo.n_atoms
            3
            >>> topo.n_bonds
            2
        """
        # Check for required blocks
        if "atoms" not in self._blocks:
            raise KeyError(
                "Frame must contain an 'atoms' block to extract topology. "
                f"Available blocks: {list(self._blocks.keys())}"
            )
        if "bonds" not in self._blocks:
            raise KeyError(
                "Frame must contain a 'bonds' block to extract topology. "
                f"Available blocks: {list(self._blocks.keys())}"
            )

        bonds_block = self["bonds"]

        # Check for required columns in bonds block
        if "i" not in bonds_block:
            raise KeyError(
                "Bonds block must contain an 'i' column. "
                f"Available columns: {list(bonds_block.keys())}"
            )
        if "j" not in bonds_block:
            raise KeyError(
                "Bonds block must contain a 'j' column. "
                f"Available columns: {list(bonds_block.keys())}"
            )

        i = bonds_block["i"]
        j = bonds_block["j"]
        bonds_list = [(int(ii), int(jj)) for ii, jj in zip(i.tolist(), j.tolist())]
        n_atoms = self["atoms"].nrows
        topo = Topology()
        topo.add_atoms(n_atoms)
        topo.add_bonds(bonds_list)
        return topo

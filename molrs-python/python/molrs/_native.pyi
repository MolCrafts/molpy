"""Type stubs for molrs Python bindings.

Hand-maintained. Keep in sync with the PyO3 source in `molrs-python/src/**.rs`
on every PR that touches `#[pyfunction]` or `#[pyclass]`. `tests/test_stub_parity.py`
is the freshness guard: every compiled `_native` export must be declared here, with
the same parameter names as the compiled signature.
"""

import os
from collections.abc import (
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    ValuesView,
)
from collections.abc import Mapping as _AbcMapping
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypeVar,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

# Type aliases — `F = f64` is invariant in molrs; Python side must match.
type ArrayF = npt.NDArray[np.float64]
type ArrayI32 = npt.NDArray[np.int32]
type ArrayBool = npt.NDArray[np.bool_]
type ArrayU8 = npt.NDArray[np.uint8]
type ArrayU32 = npt.NDArray[np.uint32]
type ArrayI64 = npt.NDArray[np.int64]
type PathInput = str | os.PathLike[str]
# A force-field param as read back: a number, a string, or a float64 array
# (a CMAP ``grid``).
type ParamValue = float | str | ArrayF
# A force-field param as given: an array may be any numpy array or nested
# sequence of numbers, stored as float64.
type ParamInput = float | str | npt.ArrayLike
_TGraph = TypeVar("_TGraph", bound=MolGraph)

__version__: str

def _ffi_abi_token() -> tuple[str, str, str, str, str]:
    """FFI ABI handshake: (abi_line, version, frameref_name, forcefield_name,
    region_name)."""

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BlockDtypeError(TypeError):
    """Raised by ``Block.insert`` for a non-numpy-representable column.

    A Block holds numeric (float, int, uint and their fixed widths, complex),
    bool and str columns. Object-dtype,
    None-bearing, and ragged/mixed arrays are rejected fail-fast; the message
    names the column and the detected dtype. Subclasses ``TypeError``.
    """

# ---------------------------------------------------------------------------
# Simulation box
# ---------------------------------------------------------------------------

class Box:
    """Simulation box with periodic boundary conditions."""

    def __init__(
        self,
        h: npt.ArrayLike | None = None,
        origin: ArrayF | None = None,
        pbc: ArrayBool | None = None,
        cell_defined: bool | None = None,
    ) -> None:
        """``h`` is a ``(3, 3)`` cell matrix (lattice vectors as columns) or
        a ``(3,)`` diagonal; ``None`` or an all-zero matrix is no cell (a free
        box). ``pbc`` defaults to whether there is a cell."""
    @staticmethod
    def cube(
        a: float,
        origin: ArrayF | None = None,
        pbc: ArrayBool | None = None,
    ) -> Box: ...
    @staticmethod
    def ortho(
        lengths: Sequence[float] | ArrayF,
        origin: ArrayF | None = None,
        pbc: ArrayBool | None = None,
    ) -> Box: ...
    @staticmethod
    def from_bounds(
        points: ArrayF | Frame,
        padding: float | Sequence[float] | ArrayF,
        pbc: ArrayBool | None = None,
    ) -> Box:
        """Tight orthorhombic box around ``points`` (``(N, 3)``, or a Frame's
        atoms ``x``/``y``/``z``), grown by ``padding`` on each side (one value
        for every axis, or one per axis; non-negative)."""
    def approx_eq(self, other: Box, tol: float) -> bool:
        """Same cell within absolute ``tol``: every matrix entry and origin
        component within ``tol``, identical ``pbc`` and ``cell_defined``."""
    @property
    def cell_defined(self) -> bool: ...
    @property
    def is_free(self) -> bool: ...
    @property
    def style(self) -> str: ...
    def volume(self) -> float: ...
    def lattice(self, index: int) -> ArrayF: ...
    @property
    def h(self) -> ArrayF: ...
    @property
    def inverse(self) -> ArrayF: ...
    @property
    def origin(self) -> ArrayF: ...
    @property
    def pbc(self) -> ArrayBool: ...
    @property
    def lengths(self) -> ArrayF: ...
    @property
    def angles(self) -> ArrayF: ...
    @staticmethod
    def matrix_from_lengths_angles(
        lengths: Sequence[float] | ArrayF, angles: ArrayF
    ) -> ArrayF: ...
    @staticmethod
    def matrix_from_lengths_tilts(
        lengths: Sequence[float] | ArrayF, tilts: ArrayF
    ) -> ArrayF: ...
    @staticmethod
    def restricted_matrix(matrix: ArrayF) -> ArrayF: ...
    @property
    def tilts(self) -> ArrayF: ...
    @property
    def nearest_plane_distance(self) -> ArrayF: ...
    @property
    def bounds(self) -> ArrayF: ...
    def corners(self) -> ArrayF: ...
    def shortest_vector(self, r1: ArrayF, r2: ArrayF) -> ArrayF: ...
    def distance_squared(self, r1: ArrayF, r2: ArrayF) -> float: ...
    def distance(self, r1: ArrayF, r2: ArrayF) -> float: ...
    def to_frac(self, xyz: ArrayF) -> ArrayF: ...
    def to_cart(self, xyzs: ArrayF) -> ArrayF: ...
    def wrap(self, xyzu: ArrayF) -> ArrayF: ...
    def images(self, xyz: ArrayF) -> ArrayI32: ...
    def unwrap(self, xyz: ArrayF, images: ArrayI32) -> ArrayF: ...
    def delta(
        self, xyzu1: ArrayF, xyzu2: ArrayF, minimum_image: bool = False
    ) -> ArrayF:
        """Displacement ``xyzu2 - xyzu1``.

        Both arguments must be shape ``(N, 3)`` (returns ``(N, 3)``) or both
        shape ``(3,)`` (returns ``(3,)``). Mixed ranks raise ``ValueError``.
        ``minimum_image`` defaults to ``False``.
        """
    def distances(self, points1: ArrayF, points2: ArrayF) -> ArrayF: ...
    def pairwise_delta(self, points1: ArrayF, points2: ArrayF) -> ArrayF: ...
    def pairwise_distances(self, points1: ArrayF, points2: ArrayF) -> ArrayF: ...
    def transformed(self, transformation: ArrayF) -> Box: ...
    def contains(self, xyz: ArrayF) -> ArrayBool: ...

# ---------------------------------------------------------------------------
# Neighbor search
# ---------------------------------------------------------------------------

class NeighborList:
    """Neighbor-search engine: index coordinates, then materialize pairs.

    ``build`` / ``update`` index and enumerate nothing; ``neighbors`` returns
    the pair table. A self search is half-shell (each unordered pair once,
    ``i < j``).
    """

    def __init__(
        self,
        cutoff: float,
        points: ArrayF | None = None,
        box: Box | None = None,
        brute_force: bool = False,
    ) -> None: ...
    @staticmethod
    def brute_force(cutoff: float) -> NeighborList: ...
    @property
    def cutoff(self) -> float: ...
    def build(self, points: ArrayF, box: Box) -> None: ...
    def update(self, points: ArrayF) -> None: ...
    def neighbors(self, dist_sq: bool = ..., disp: bool = ...) -> Neighbors: ...

class Neighbors:
    """Materialized pair table — read-only columns, one row per pair.

    ``dist_sq`` and ``disp`` are opt-in: a column the search was told not to
    store reads back as ``None``, never as a fabricated zero array.
    """

    def __init__(
        self,
        is_self_query: bool,
        n_points: int,
        n_query_points: int,
        idx_i: list[int],
        idx_j: list[int],
        dist_sq: list[float] | None = None,
        disp: list[list[float]] | None = None,
    ) -> None: ...
    def query_point_indices(self) -> ArrayU32: ...
    def point_indices(self) -> ArrayU32: ...
    def dist_sq(self) -> ArrayF | None: ...
    def disp(self) -> ArrayF | None: ...
    @property
    def n_pairs(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    @property
    def n_query_points(self) -> int: ...
    @property
    def is_self_query(self) -> bool: ...

class NeighborQuery:
    """Cross-query against a fixed reference point set (directed)."""

    def __init__(self, box: Box, points: ArrayF, cutoff: float) -> None: ...
    @staticmethod
    def unbounded(points: ArrayF, cutoff: float) -> NeighborQuery: ...
    def query(self, query_points: ArrayF) -> Neighbors: ...
    def query_self(self) -> Neighbors: ...

class VerletSkin:
    """A ``NeighborList`` with Verlet skin, for the MD integrators.

    ``neighbors`` must be an engine built with cutoff ``cutoff + skin``; it is
    **moved** into this object, and passing the skin into an integrator moves
    it again (afterwards every access raises "has already been moved").
    """

    def __init__(
        self,
        neighbors: NeighborList,
        cutoff: float,
        positions: ArrayF,
        box: Box,
        skin: float = 0.0,
        every: int = 1,
        delay: int = 0,
        check: bool = True,
        ago: int = 0,
        n_rebuilds: int = 0,
        ndanger: int = 0,
    ) -> None: ...
    @property
    def cutoff(self) -> float: ...
    @property
    def skin(self) -> float: ...
    @property
    def n_edges(self) -> int: ...
    @property
    def n_rebuilds(self) -> int: ...
    @property
    def ago(self) -> int: ...
    def update(self, positions: ArrayF) -> bool: ...
    def rebuild(self, positions: ArrayF) -> None: ...

# ---------------------------------------------------------------------------
# Block / Frame
# ---------------------------------------------------------------------------

# A column key: a plain name or a ``molrs.core.keys.Key``.
type ColumnKey = str | keys.Key

@final
class Block:
    """Heterogeneous column store (dict of typed numpy arrays) — the one Block.

    Columns are dense. A per-row component that only some rows carry is stored
    with a validity mask beside it, read back with :meth:`validity`. Every
    column-key argument accepts a ``str`` or a ``molrs.core.keys.Key``. A block read
    from a frame (``frame["atoms"]``) is a handle on the stored block.
    """

    def __init__(
        self, data: _AbcMapping[ColumnKey, npt.ArrayLike] | None = None
    ) -> None:
        """Build from a mapping of column name -> array (each value goes through
        ``__setitem__``, so a canonical key adopts its schema dtype). Copying a
        block is :meth:`copy`; ``Block(block)`` raises ``TypeError``."""
    def insert(self, key: ColumnKey, array: npt.NDArray | Sequence[str]) -> None:
        """Store a column at the given width. Raises ``BlockDtypeError`` for
        object/None/ragged."""
    def insert_nullable(
        self,
        key: ColumnKey,
        array: npt.NDArray | Sequence[str],
        validity: ArrayBool | Sequence[bool],
    ) -> None:
        """Store a column together with a per-row validity mask.

        The write side of :meth:`validity`, and otherwise :meth:`insert`: same
        dtypes, same row-count rule.

        Parameters
        ----------
        key : str | Key
            Column name.
        array : numpy.ndarray | Sequence[str]
            Column data.
        validity : numpy.ndarray | Sequence[bool]
            1-D bool mask, one entry per row of *array*; ``False`` marks a row
            that holds no value. An all-``True`` mask states nothing
            :meth:`insert` does not and is dropped.

        Raises
        ------
        TypeError
            If the array dtype is unsupported, or *validity* is neither a 1-D
            bool array nor a sequence of bools.
        ValueError
            If the row count does not match existing columns, or *validity*
            does not cover exactly the rows of *array*.
        """
    def set_validity(self, key: ColumnKey, mask: ArrayBool | Sequence[bool]) -> None:
        """Attach a validity mask to an existing column of any dtype.

        ``mask[i]`` is ``False`` where row ``i`` holds no value; the mask
        replaces any the column had, and an all-``True`` mask clears it.

        Raises
        ------
        KeyError
            If ``key`` names no column.
        TypeError
            If *mask* is neither a 1-D bool array nor a sequence of bools.
        ValueError
            If *mask* does not have exactly one entry per row.
        """
    def set_precision(self, key: ColumnKey, precision: float | None) -> None:
        """Declare (``None``: withdraw) the precision of a ``float64`` column.

        An absolute tolerance in the column's units. A record writer stores
        the column rounded to the largest power of two not above it (ties to
        even), within ``precision / 2`` of the values; memory is untouched.
        The declaration reads back from ``*.mrec``.

        Raises
        ------
        KeyError
            If ``key`` names no column.
        ValueError
            If the column is not ``float64`` or *precision* is not finite and
            within ``[2**-1000, 2**1000]``.
        """
    def set_target(self, key: ColumnKey, target: str | None) -> None:
        """Declare (``None``: withdraw) that a ``uint64`` column holds row
        indices into *target* (``"<block>"`` or ``"/<section>/<block>"``).

        Raises
        ------
        KeyError
            If ``key`` names no column.
        ValueError
            If the column is not ``uint64`` or *target* is malformed or names
            a trajectory block.
        """
    def target(self, key: ColumnKey) -> str | None:
        """The declared target of a column, or ``None``. ``KeyError`` for an
        absent column."""
    def targets(self) -> dict[str, str]:
        """Every declared target, ``{column: target}``."""
    def precision(self, key: ColumnKey) -> float | None:
        """The declared precision of a column, or ``None``.

        Raises
        ------
        KeyError
            If ``key`` names no column.
        """
    @staticmethod
    def stack(parts: Sequence[Block]) -> Block:
        """Row-wise union of *parts* under the union of their columns
        (first-seen order). A column a part lacks is filled with the dtype's
        default for that part's rows and marked null in its validity mask.

        Raises
        ------
        ValueError
            If two parts carry one column under different dtypes or per-row
            shapes.
        """
    @property
    def coords(self) -> ArrayF:
        """``(n_rows, 3)`` float64 copy of the ``x`` / ``y`` / ``z`` columns.
        Raises ``KeyError`` if one is missing."""
    @coords.setter
    def coords(self, value: npt.ArrayLike) -> None:
        """Write an ``(N, 3)`` array into ``x`` / ``y`` / ``z`` (float64).
        Raises ``ValueError`` for a non-``(N, 3)`` array or a row mismatch."""
    def copy_column(self, key: ColumnKey) -> npt.NDArray:
        """Owned copy of one column, shape included.

        ``block[key]`` returns the column for every dtype. Numeric, bool and
        complex columns are a zero-copy view there, so this copies them. A
        string column is already a copy under ``block[key]`` (numpy ``str``,
        the column's shape), and this returns another one.
        """
    def validity(self, key: ColumnKey) -> ArrayBool | None:
        """The validity mask of a column, or ``None`` when it has no holes.

        Returns
        -------
        numpy.ndarray | None
            A 1-D ``bool`` array in row order — ``True`` where the cell holds a
            real value, ``False`` where it is a hole — or ``None`` when the
            column carries no mask. ``None`` means "every cell is a stated
            value", not "every cell is a hole".

        Raises
        ------
        KeyError
            If ``key`` names no column of this block — the same answer
            indexing and :meth:`dtype` give, so a misspelled key cannot
            read as a dense column.
        """
    @overload
    def __getitem__(self, key: ColumnKey) -> npt.NDArray: ...
    @overload
    def __getitem__(
        self, key: tuple[ColumnKey, ...] | list[ColumnKey]
    ) -> npt.NDArray: ...
    @overload
    def __getitem__(
        self, key: slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer]
    ) -> Block: ...
    def __setitem__(
        self,
        key: ColumnKey | tuple[ColumnKey, ...] | list[ColumnKey],
        value: npt.ArrayLike,
    ) -> None:
        """Store a column (schema dtype adopted; a scalar is refused), or spread
        an ``(N, k)`` array over ``k`` named columns."""
    def __delitem__(self, key: ColumnKey) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object) -> bool: ...
    @property
    def n_rows(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def structural_shape(self) -> list[int] | None: ...
    def resize(self, n_rows: int) -> None: ...
    def set_shape(self, shape: Sequence[int]) -> None: ...
    def keys(self) -> list[str]: ...
    def remove(self, key: ColumnKey) -> None: ...
    def rename(self, old_key: ColumnKey, new_key: ColumnKey) -> None:
        """Rename in place; a canonical ``new_key`` adopts its schema dtype."""
    def select_rows(self, indices: Sequence[int]) -> Block: ...
    def sort(self, key: ColumnKey, reverse: bool = False) -> Block: ...
    def copy(self) -> Block:
        """Deep copy: no buffer shared with this block."""
    def dtype(self, key: ColumnKey) -> str: ...
    def has_f64(self, key: ColumnKey) -> bool: ...
    def has_int(self, key: ColumnKey) -> bool: ...
    def has_uint(self, key: ColumnKey) -> bool: ...
    def has_string(self, key: ColumnKey) -> bool: ...
    def __reduce__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...

class MetaValue:
    """Exact-dtype frame metadata value."""

    def __init__(self, dtype: str, value: Any) -> None: ...
    @property
    def dtype(self) -> str: ...
    @property
    def value(
        self,
    ) -> (
        bool
        | int
        | float
        | str
        | None
        | tuple[bool | int | float, ...]
        | dict[str, Any]
        | list[Any]
    ):
        """Stored payload.

        Fixed-length vectors are tuples. A ``json`` payload stays plain
        (``dict`` / ``list`` / scalar) — this is the pickle argument, not a
        ``frame.meta`` door.
        """

# A value handed out by ``frame.meta``: scalars unwrap, fixed-length vectors
# and JSON arrays are tuples, a JSON object is a MetaDocument.
type FrozenMetaValue = bool | int | float | str | None | tuple[Any, ...] | MetaDocument

class MetaDocument:
    """Frozen JSON object read from ``frame.meta``.

    Every door of ``frame.meta`` hands back a frozen value: a fixed-length
    vector is a ``tuple``, and a JSON object is a ``MetaDocument``. Nested
    arrays are tuples; nested objects are documents. Item assignment raises
    ``TypeError``. ``copy()`` is a deep plain ``dict`` (nested documents become
    dicts, nested arrays become lists). ``json.dumps`` rejects a document; use
    ``json.dumps(frame.meta["run"].copy())``.

    Iteration order is unspecified. ``frame.meta`` itself enumerates in
    insertion order; the two levels differ.
    """

    __hash__: ClassVar[None]
    def __getitem__(self, key: object) -> FrozenMetaValue: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key: object) -> bool: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[FrozenMetaValue]: ...
    def items(self) -> ItemsView[str, FrozenMetaValue]: ...
    def get(self, key: object, default: Any = None) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def copy(self) -> dict[str, Any]: ...

class FrameMeta:
    """Live, write-through ``frame.meta`` mapping.

    Every door hands back a frozen value: scalars unwrap, a fixed-length
    vector is a ``tuple``, a JSON array is a ``tuple``, and a JSON object is
    a :class:`MetaDocument`. ``frame.meta["run"]["step"] = 3`` raises
    ``TypeError``. ``json.dumps`` rejects a document; use
    ``json.dumps(frame.meta["run"].copy())``. ``copy()``, ``|``, and ``|=``'s
    merge partner return a plain ``dict``; values inside it are still frozen.
    :meth:`MetaDocument.copy` is the deep plain unfreeze one level down.

    ``dtype(k)`` reports the tag of the value stored right now; any plain write
    re-infers it. :class:`MetaValue` fixes the dtype of that write only — it
    does not pin the key. A tag survives a round trip through a ``*.mrec``
    frame or system (stored in ``_meta_types``), a declared sequence schema
    and the serde frame document. NaN and infinities survive as ``f64``.

    Enumeration follows insertion order. ``popitem`` returns the last-inserted
    key. Order inside a nested :class:`MetaDocument` is unspecified.

    ``keys``, ``values``, and ``items`` are live ``collections.abc`` views in
    insertion order. A non-``str`` lookup is absent; a non-``str`` write raises
    ``TypeError``. Deleting a not-yet-visited key while iterating ``values()``
    or ``items()`` raises ``KeyError``.
    """

    def __getitem__(self, key: object) -> FrozenMetaValue: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: object) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    def dtype(self, key: str) -> str | None: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[FrozenMetaValue]: ...
    def items(self) -> ItemsView[str, FrozenMetaValue]: ...
    def get(self, key: object, default: Any = None) -> FrozenMetaValue | Any: ...
    def pop(self, key: object, *default: Any) -> FrozenMetaValue | Any: ...
    def popitem(self) -> tuple[str, FrozenMetaValue]: ...
    def clear(self) -> None: ...
    def setdefault(self, key: str, default: Any = None) -> FrozenMetaValue | Any: ...
    def update(self, other: Any = None, **kwargs: Any) -> None: ...
    def copy(self) -> dict[str, Any]: ...
    def typed(self) -> dict[str, MetaValue]: ...
    def __or__(self, other: Any) -> dict[str, Any]: ...
    def __ror__(self, other: Any) -> dict[str, Any]: ...
    def __ior__(self, other: Any) -> Self: ...

@final
class Frame:
    """Dictionary of Blocks with optional ``box`` and typed ``meta`` — the one Frame."""

    def __init__(
        self,
        blocks: _AbcMapping[str, Block | _AbcMapping[ColumnKey, npt.ArrayLike]]
        | None = None,
        *,
        meta: _AbcMapping[str, Any] | None = None,
        box: Box | None = None,
    ) -> None:
        """Copying a frame is :meth:`copy`; ``Frame(frame)`` raises ``TypeError``."""
    def __getitem__(self, key: str) -> Block:
        """A handle on the stored block: its members read and write this frame."""
    def __setitem__(
        self, key: str, value: Block | _AbcMapping[ColumnKey, npt.ArrayLike]
    ) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    @property
    def box(self) -> Box | None: ...
    @box.setter
    def box(self, value: Box | None) -> None: ...
    @property
    def meta(self) -> FrameMeta: ...
    @meta.setter
    def meta(self, value: Any) -> None: ...
    def validate(self) -> None: ...
    def copy(self) -> Frame:
        """Deep copy: blocks (new buffers), box and typed meta."""
    def subset(self, rows: npt.ArrayLike, block: str = "atoms") -> Frame:
        """A new frame holding the selected rows of ``block``: a 1-D bool mask
        (``True`` rows, in order) or 1-D int rows (``-n_rows <= i`` wraps). Every
        relation block indexing ``block`` keeps only the rows whose endpoints
        are all selected, renumbered. Other blocks, the box and ``meta`` are
        copied; values keep their units (Å). This frame is never modified.

        Raises
        ------
        KeyError
            no block ``block``.
        IndexError
            a mask of the wrong length, an index below ``-n_rows``,
            a selector that is not 1-D.
        TypeError
            a selector that is neither bool nor integer.
        ValueError
            a row past the end or repeated; a relation block
            without ``UInt`` endpoints; a ``members`` block.
        """
    def replicate(self, count: int) -> Frame:
        """``count`` copies of this frame, concatenated block by block (the
        inverse of :meth:`subset`). Relation endpoints of copy ``c`` are offset
        by ``c`` times the row count of the block they index; every other
        column — ``id`` / ``mol_id`` included — is copied verbatim. Validity
        masks, ``meta`` and the box travel. This frame is never modified.

        Raises
        ------
        ValueError
            a relation block indexing a missing block or lacking
            ``UInt`` endpoints; a ``members`` block.
        """
    @staticmethod
    def concat(frames: Sequence[Frame]) -> Frame:
        """The frames joined end to end, block by block (:meth:`replicate`
        for parts that differ). Relation endpoints of part ``p`` are offset by
        the rows the indexed block has in the parts before it; a column one
        part lacks is null on its rows; every other column is copied verbatim.
        ``meta`` and the box are the first part's. No part is modified.

        Raises
        ------
        ValueError
            a relation block indexing a block its part lacks or lacking
            ``UInt`` endpoints; one column under two dtypes.
        """
    @property
    def coords(self) -> ArrayF:
        """``(N, 3)`` float64 copy of ``atoms`` ``x`` / ``y`` / ``z``. Raises
        ``KeyError`` without an ``atoms`` block or one of the columns."""
    @coords.setter
    def coords(self, value: npt.ArrayLike) -> None:
        """Write an ``(N, 3)`` array into ``atoms`` (created if absent).
        Raises ``ValueError`` for a non-``(N, 3)`` array or a row mismatch."""
    def __reduce__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def _ffi_frameref_capsule(self) -> object: ...
    @staticmethod
    def _from_ffi_frameref_capsule(capsule: object) -> Frame: ...

# ---------------------------------------------------------------------------
# Live Frame streaming (molrs::stream)
# ---------------------------------------------------------------------------

class ControlCommand:
    """A control message from a streaming viewer back to the producer."""

    @staticmethod
    def pause() -> ControlCommand: ...
    @staticmethod
    def resume() -> ControlCommand: ...
    @staticmethod
    def set_frame_rate(hz: float) -> ControlCommand: ...
    @staticmethod
    def set_subset(atom_ids: Sequence[int]) -> ControlCommand: ...
    @staticmethod
    def request_key_frame() -> ControlCommand: ...
    @property
    def kind(
        self,
    ) -> Literal[
        "pause", "resume", "set_frame_rate", "set_subset", "request_key_frame"
    ]: ...
    @property
    def hz(self) -> float | None: ...
    @property
    def atom_ids(self) -> list[int] | None: ...
    def to_bytes(self, format: Literal["json", "msgpack"] = "json") -> bytes: ...
    @staticmethod
    def from_bytes(
        data: bytes, format: Literal["json", "msgpack"] = "json"
    ) -> ControlCommand: ...

class Publisher:
    """WebSocket server broadcasting frames to connected viewers.

    Native-only: a Pyodide build compiles `molrs::stream::publisher` out, so this
    class is absent there.
    """

    def __init__(
        self,
        address: str = "127.0.0.1:0",
        *,
        format: Literal["msgpack", "json"] = "msgpack",
        buffer_size: int = 4,
        token: str | None = None,
    ) -> None: ...
    @property
    def address(self) -> str: ...
    @property
    def n_clients(self) -> int: ...
    def send(self, frame: Frame) -> None: ...
    def recv_command(self, timeout: float = 0.0) -> ControlCommand | None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

# ---------------------------------------------------------------------------
# Regions: solids with a signed distance, composed with &, |, ~
# ---------------------------------------------------------------------------

type RegionLike = (
    Sphere
    | Cuboid
    | Parallelepiped
    | HalfSpace
    | Cylinder
    | Ellipsoid
    | Polyhedron
    | SphereUnion
    | Region
)

class TriMesh:
    """Triangle surface mesh: a vertex table plus faces indexing into it."""

    def __init__(self, vertices: ArrayF, faces: ArrayU32) -> None: ...
    def vertices(self) -> ArrayF: ...
    def faces(self) -> ArrayU32: ...
    @property
    def n_vertices(self) -> int: ...
    @property
    def n_faces(self) -> int: ...
    def is_watertight(self) -> bool: ...
    def scaled(self, factor: float) -> TriMesh: ...
    def bounds(self) -> ArrayF: ...

class Sphere:
    """Solid sphere region."""

    def __init__(self, center: Sequence[float] | ArrayF, radius: float) -> None: ...
    @property
    def center(self) -> ArrayF: ...
    @property
    def radius(self) -> float: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Cuboid:
    """Axis-aligned cuboid (box) region.

    A point is inside when `origin[d] <= p[d] <= origin[d] + lengths[d]` on
    every axis.
    """

    def __init__(
        self, origin: Sequence[float] | ArrayF, lengths: Sequence[float] | ArrayF
    ) -> None: ...
    @staticmethod
    def cube(edge: float, origin: Sequence[float] | ArrayF = ...) -> Cuboid:
        """The axis-aligned cube of edge ``edge`` at minimum corner ``origin``
        (default the origin)."""
    @property
    def origin(self) -> ArrayF: ...
    @property
    def lengths(self) -> ArrayF: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Parallelepiped:
    """General parallelepiped (oblique box) geometric region.

    Pure containment — not a periodic ``Box``. Columns of ``h`` are the
    three edge vectors; fractional coordinates must lie in ``[0, 1]³``.
    ``distance`` is measured perpendicular to the bounding planes.
    """

    def __init__(self, h: ArrayF, origin: Sequence[float] | ArrayF) -> None: ...
    @staticmethod
    def ortho(
        lengths: Sequence[float] | ArrayF, origin: Sequence[float] | ArrayF
    ) -> Parallelepiped: ...
    @staticmethod
    def cube(a: float, origin: Sequence[float] | ArrayF) -> Parallelepiped: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def volume(self) -> float: ...
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class HalfSpace:
    """One side of a plane: inside where ``normal · (x - point) <= 0``."""

    def __init__(
        self, normal: Sequence[float] | ArrayF, point: Sequence[float] | ArrayF
    ) -> None: ...
    def normal(self) -> ArrayF: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Cylinder:
    """Finite capped cylinder from ``base`` along ``axis``."""

    def __init__(
        self,
        base: Sequence[float] | ArrayF,
        axis: Sequence[float] | ArrayF,
        radius: float,
        length: float,
    ) -> None: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Ellipsoid:
    """Axis-aligned ellipsoid with semi-axes along x, y, z."""

    def __init__(
        self, center: Sequence[float] | ArrayF, semi_axes: Sequence[float] | ArrayF
    ) -> None: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Polyhedron:
    """Solid bounded by a watertight :class:`TriMesh`."""

    def __init__(self, mesh: TriMesh) -> None: ...
    def mesh(self) -> TriMesh: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class SphereUnion:
    """Union of spheres — atoms as a region; ``~SphereUnion`` is the void."""

    def __init__(
        self, centers: ArrayF, radii: float | ArrayF, box: Box | None = None
    ) -> None: ...
    @property
    def n_spheres(self) -> int: ...
    @property
    def box(self) -> Box: ...
    def centers(self) -> ArrayF: ...
    def radii(self) -> ArrayF: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Region:
    """Composed geometric region (result of &, |, ~ operators)."""

    def __init__(self, source: RegionLike) -> None: ...
    @staticmethod
    def _from_tree(tree: tuple[Any, ...]) -> Region: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike) -> Region: ...
    def __or__(self, other: RegionLike) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

# ---------------------------------------------------------------------------
# Trace: an ordered path of 3D points
# ---------------------------------------------------------------------------

@final
class Trace:
    """An ordered path of 3D points, with no chemistry. Frozen.

    A trace says *where* consecutive units of a chain sit (for example the
    site positions of one coarse-grained chain), not *what* sits there.

    Parameters
    ----------
    points : numpy.ndarray, shape (k, 3), float64
        Every point, in order (Å). ``(0, 3)`` is the empty trace.

    Raises
    ------
    ValueError
        If ``points`` is not ``(k, 3)``.
    """

    def __init__(self, points: ArrayF) -> None: ...
    @property
    def points(self) -> ArrayF:
        """Every point, in order.

        Returns
        -------
        numpy.ndarray, shape (k, 3), float64
            A copy of the points (Å).
        """
    def __len__(self) -> int:
        """The number of points, k."""

# ---------------------------------------------------------------------------
# Molecular graph
# ---------------------------------------------------------------------------

class Topology:
    """Bond graph of a frame: atoms are rows ``0..n`` of ``frame["atoms"]``,
    edges the ``atomi`` / ``atomj`` rows of ``frame["bonds"]``."""

    @classmethod
    def from_frame(cls, frame: Frame) -> Topology:
        """Read the bond graph of ``frame``.

        Raises
        ------
        ValueError
            ``frame`` has no atoms, the bonds block lacks
            ``atomi`` / ``atomj``, or a bond names a row outside the frame.
        """
    @property
    def n_atoms(self) -> int: ...
    @property
    def n_bonds(self) -> int: ...
    @property
    def n_components(self) -> int: ...
    def connected_components(self) -> npt.NDArray[np.uint32]:
        """Per-atom connected-component label, ``0..n_components`` in order
        of each component's first atom."""

class BondOrder:
    """The chemical class of a bond, stored under ``keys.BOND_TYPE`` as its
    code (``int(member)``): ``Unknown`` 0, ``Single`` 1, ``Double`` 2,
    ``Triple`` 3, ``Aromatic`` 4."""

    Unknown: ClassVar[BondOrder]
    Single: ClassVar[BondOrder]
    Double: ClassVar[BondOrder]
    Triple: ClassVar[BondOrder]
    Aromatic: ClassVar[BondOrder]
    @property
    def code(self) -> int: ...
    @staticmethod
    def from_code(code: int) -> BondOrder: ...
    def is_aromatic(self) -> bool: ...
    def implied_number(self) -> BondNumber | None: ...
    def __int__(self) -> int: ...

class BondNumber:
    """The integer bond number of a localized (Kekulé) structure, stored
    under ``keys.BOND_NUMBER`` as its code: ``Unknown`` 0, ``Single`` 1,
    ``Double`` 2, ``Triple`` 3, ``Quadruple`` 4."""

    Unknown: ClassVar[BondNumber]
    Single: ClassVar[BondNumber]
    Double: ClassVar[BondNumber]
    Triple: ClassVar[BondNumber]
    Quadruple: ClassVar[BondNumber]
    @property
    def code(self) -> int: ...
    @staticmethod
    def from_code(code: int) -> BondNumber: ...
    def __int__(self) -> int: ...

class Element:
    """Immutable chemical-element record backed by the Rust periodic table."""

    def __init__(self, identifier: str | int) -> None: ...
    @property
    def number(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def symbol(self) -> str: ...
    @property
    def mass(self) -> float: ...
    @property
    def vdw(self) -> float: ...
    @property
    def covalent(self) -> float: ...
    @classmethod
    def get_symbols(cls, identifiers: Iterable[str | int]) -> list[str]: ...
    @classmethod
    def get_atomic_number(cls, identifier: str | int) -> int: ...

class UnitsError(ValueError): ...

class SmilesError(ValueError):
    """Raised when a SMILES / SMARTS / CGsmiles string is refused.

    Subclasses ``ValueError``; ``str(e)`` is the message the Rust error
    renders, caret line included. The four attributes are set on every
    instance.

    Attributes
    ----------
    kind : str
        Variant name of the rule that was broken, payload dropped —
        ``"UnclosedBranch"``, ``"UnexpectedEnd"``, ``"CgNotExpandable"``, ...
    span : tuple[int, int]
        Byte range of the offending text within ``input``; the end is clamped
        to ``len(input)``, since the scanner reports end-of-input one byte
        past the text.
    input : str
        The offending string, empty for errors raised past the parser (the
        expansion and emit stages are handed an IR, not the text).
    notation : str
        Which notation was being read or written: ``"smiles"``, ``"smarts"``
        or ``"cgsmiles"``.
    """

    kind: str
    span: tuple[int, int]
    input: str
    notation: str

class Unit:
    def __init__(
        self,
        factor: float,
        offset: float,
        dimension: tuple[int, int, int, int, int, int, int],
        name: str,
    ) -> None: ...
    @property
    def dimension(self) -> tuple[int, int, int, int, int, int, int]: ...
    @property
    def dimensionality(self) -> tuple[int, int, int, int, int, int, int]: ...
    def is_affine(self) -> bool: ...
    def factor_to(self, other: Unit) -> float: ...
    def __rmul__(self, value: float) -> Quantity: ...
    def __mul__(self, value: float) -> Quantity: ...

class Quantity:
    def __init__(self, magnitude: float, unit: Unit) -> None: ...
    @property
    def magnitude(self) -> float: ...
    @property
    def value(self) -> float: ...
    @property
    def units(self) -> Unit: ...
    @property
    def unit(self) -> Unit: ...
    def to(self, target: Unit | str) -> Quantity: ...
    def to_base_units(self) -> Quantity: ...
    def __add__(self, rhs: Quantity) -> Quantity: ...
    def __sub__(self, rhs: Quantity) -> Quantity: ...
    def __mul__(self, rhs: Quantity | float) -> Quantity: ...
    def __rmul__(self, lhs: float) -> Quantity: ...
    def __truediv__(self, rhs: Quantity | float) -> Quantity: ...

class UnitPreset:
    def __init__(self, name: str) -> None: ...
    @staticmethod
    def real() -> UnitPreset: ...
    @staticmethod
    def names() -> list[str]:
        """Every registered preset name, sorted (the LAMMPS styles,
        ``openmm``, and any :meth:`register` added)."""
    @staticmethod
    def register(
        name: str,
        units: _AbcMapping[str, str],
        *,
        boltzmann: float,
        coulomb: float,
        overwrite: bool = False,
    ) -> UnitPreset:
        """Register a preset (one unit expression per each of the ten
        dimensions, plus its two constants) process-wide and return it.
        Raises ``ValueError`` for a missing/unknown dimension or a taken
        name without ``overwrite``."""
    @property
    def name(self) -> str: ...
    def boltzmann(self) -> float: ...
    def coulomb(self) -> float: ...
    def mass(self) -> str: ...
    def length(self) -> str: ...
    def time(self) -> str: ...
    def energy(self) -> str: ...
    def temperature(self) -> str: ...
    def charge(self) -> str: ...
    def pressure(self) -> str: ...
    def velocity(self) -> str: ...
    def force(self) -> str: ...
    def density(self) -> str: ...

class UnitRegistry:
    def __init__(
        self,
        definitions: list[
            tuple[
                str,
                list[str],
                str,
                float,
                float,
                tuple[int, int, int, int, int, int, int],
                bool,
            ]
        ]
        | None = None,
        *,
        empty: bool = False,
    ) -> None: ...
    def factor(self, from_unit: str, to_unit: str) -> float: ...
    def parse(self, expression: str) -> Unit: ...
    def quantity(self, value: float, expression: str) -> Quantity: ...
    def define(
        self,
        name: str,
        factor: float,
        dimension: tuple[int, int, int, int, int, int, int],
        *,
        aliases: list[str] = ...,
        symbol: str | None = None,
        offset: float = 0.0,
        prefixable: bool = False,
    ) -> None: ...
    def define_lj_sigma(self, sigma: Quantity) -> None:
        """Define the reduced-LJ length unit ``lj_sigma`` alone.

        Raises
        ------
        UnitsError
            ``sigma`` is not a finite positive length, or
            ``lj_sigma`` is already defined.
        """
    def define_lj_units(
        self, mass: Quantity, sigma: Quantity, epsilon: Quantity
    ) -> None: ...

class FragmentScaling:
    def __init__(
        self,
        name: str,
        q: float,
        mu: float,
        alpha: float,
        polarizable: bool = False,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def q(self) -> float: ...
    @property
    def mu(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def polarizable(self) -> bool: ...

def compute_k_ij(fr_i: FragmentScaling, fr_j: FragmentScaling, r: float) -> float: ...
def fragment_table() -> dict[str, FragmentScaling]: ...
def scale_lj(
    ff: ForceField,
    fragments: dict[
        str, tuple[list[str], list[tuple[float, float, float]], list[float]]
    ],
    fragment_table: dict[str, FragmentScaling] | None = None,
    scale_sigma: bool = False,
) -> ForceField: ...

class MolGraph:
    """Domain-agnostic ECS *world*. Base of the hierarchy.

    Entities are stable opaque ``int`` handles (generational slotmap keys);
    removing one entity never invalidates another and a stale handle raises.
    Components live in aligned columns addressed by convention keys (see
    :data:`keys`); ``column`` exposes a zero-copy numpy view. Topology is a
    kind-tagged relation API (``register_kind`` / ``add_relation`` / …).

    Rigid-body moves (``translate`` / ``rotate`` / ``scale``) and ``center``
    are methods of the :class:`Atomistic` / :class:`CoarseGrain` leaves,
    not of the base; perception lives in
    :mod:`molrs.perceive`. Chemistry vocabulary (``add_atom`` / ``add_bond``
    / ``add_bead``) lives on the :class:`Atomistic` / :class:`CoarseGrain`
    leaves.

    Pickles by content: a restored graph has fresh handles in the same row
    order.
    """

    def __init__(self) -> None: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    # --- entities ---
    def spawn(self) -> int: ...
    def despawn(self, h: int) -> None: ...
    def entities(self) -> list[int]: ...
    def has_entity(self, h: int) -> bool: ...
    @property
    def n_nodes(self) -> int: ...
    # --- components (convention keys; typed; missing/type-mismatch raises) ---
    def get(self, h: int, key: str) -> int | float | str | None: ...
    def set(self, h: int, key: str, value: float | str) -> None: ...
    def has(self, h: int, key: str) -> bool: ...
    def delete(self, h: int, key: str) -> None: ...
    def node_keys(self, h: int) -> list[str]: ...
    def column(self, key: str) -> npt.NDArray[Any]:
        """Column ``key`` over every entity, typed as the component (``f64`` is a
        zero-copy write-through view; ``i32``/``bool``/``str`` are copies).
        ``KeyError`` when any entity lacks the component — never zero-filled."""
    def columns(self) -> list[str]:
        """Names of every component column registered on the node table."""
    def validity(self, key: str) -> ArrayBool: ...
    # --- relations (generic, kind-tagged) ---
    def register_kind(self, kind: str, arity: int) -> None: ...
    def kinds(self) -> list[str]: ...
    def kind_arity(self, kind: str) -> int: ...
    def add_relation(self, kind: str, nodes: list[int]) -> int: ...
    def relation_nodes(self, kind: str, rh: int) -> list[int]: ...
    def incident_relations(self, nh: int, kind: str) -> list[tuple[int, int]]: ...
    def get_relation_prop(
        self, kind: str, rh: int, key: str
    ) -> int | float | str | None: ...
    def set_relation_prop(
        self, kind: str, rh: int, key: str, value: float | str
    ) -> None: ...
    def relation_keys(self, kind: str, rh: int) -> list[str]: ...
    def delete_relation_prop(self, kind: str, rh: int, key: str) -> None: ...
    def remove_relation(self, kind: str, rh: int) -> None: ...
    def n_relations(self, kind: str) -> int: ...
    def relation_ids(self, kind: str) -> list[int]: ...
    # --- adopt (zero-copy move) ---
    def adopt(self, other: MolGraph) -> None: ...

    # ---- ports: named attachment points any graph may carry ----
    def add_port(
        self,
        anchor: int,
        handle: int,
        kind: str,
        label: str = "",
        order: int = 1,
    ) -> int:
        """Record a descriptor (``"$"``, ``"<"``, ``">"``, ``"!"``) on the
        ``(anchor, handle)`` valence; registers the ``ports`` kind on first
        use. Returns the port's relation handle.

        Raises
        ------
        ValueError
            an unknown glyph, ``handle`` not bonded to
            ``anchor``, an indefinite order, a valence that already
            carries a port, or a stale handle.
        """
    @property
    def n_ports(self) -> int: ...
    def set_frag_id(self, node: int, id: int) -> None: ...
    def frag_id(self, node: int) -> int | None: ...
    def inherit_frag_ids(self) -> int: ...
    def link(self, a: int, b: int) -> int:
        """Join port ``a`` to port ``b``: remove both leaving groups, fold
        their charge (e) onto the anchors, bond the anchors with the port
        order. Returns the new bond handle.

        Raises
        ------
        ValueError
            a handle naming no live port, a stale or incompatible
            port pair, a shared anchor, anchors already bonded, or
            overlapping leaving groups; the graph is unchanged.
        OverflowError
            a negative handle.
        """

class Atomistic(MolGraph):
    """All-atom leaf — holds a core ``Atomistic`` from construction.

    Registers the ``bonds``/``angles``/``dihedrals``/``impropers`` kinds and
    exposes the atom/bond/angle/dihedral/improper builders. The generic
    :class:`MolGraph` API operates on this leaf's own graph. Owns its
    :meth:`to_frame` / :meth:`from_frame` (domain conversions); it is never
    *converted* from a bare :class:`MolGraph`. Subclassable.

    ``Atomistic(**props)`` — the keywords are :attr:`props`. Nodes and
    relations are read and edited through live views (:attr:`atoms`,
    :meth:`def_atom`, …).
    """

    def __init__(self, **props: Any) -> None: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    @property
    def props(self) -> dict[str, Any]:
        """Whole-graph annotations; carried by copies, pickles and every graph
        derived from this one."""
    @property
    def links(self) -> RelationBuckets: ...
    def remove_link(self, *links: RelationRef) -> None: ...
    @property
    def atoms(self) -> Refs[Atom]: ...
    @property
    def bonds(self) -> Refs[Bond]: ...
    @property
    def angles(self) -> Refs[Angle]: ...
    @property
    def dihedrals(self) -> Refs[Dihedral]: ...
    @property
    def impropers(self) -> Refs[Improper]: ...
    @property
    def ports(self) -> Refs[Port]: ...
    def def_atom(self, mapping: Any = None, /, **attrs: Any) -> Atom: ...
    def def_virtual_site(
        self,
        mapping: Any = None,
        /,
        *,
        kind: type[VirtualSite] | None = None,
        **attrs: Any,
    ) -> VirtualSite: ...
    def def_bond(self, a: Atom, b: Atom, /, **attrs: Any) -> Bond: ...
    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle: ...
    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral: ...
    def def_improper(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Improper: ...
    def del_atom(self, *atoms: Atom) -> None: ...
    def def_port(
        self,
        anchor: Atom,
        handle_atom: Atom,
        kind: str,
        label: str = "",
        order: int = 1,
    ) -> Port: ...
    def add_atom(
        self,
        symbol: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> int: ...
    def add_bond(self, a: int, b: int) -> int: ...
    def add_angle(self, i: int, j: int, k: int) -> int: ...
    def add_dihedral(self, i: int, j: int, k: int, l: int) -> int: ...
    def add_improper(self, i: int, j: int, k: int, l: int) -> int: ...
    def generate_topology(
        self,
        gen_angle: bool = True,
        gen_dihedral: bool = True,
        gen_improper: bool = False,
        clear_existing: bool = False,
    ) -> tuple[int, int, int]: ...
    def topo_distances(
        self, source: int, max_hops: int | None = None
    ) -> list[tuple[int, int]]: ...
    @property
    def n_atoms(self) -> int: ...
    @property
    def n_bonds(self) -> int: ...
    def to_frame(self, atom_fields: Sequence[str] | None = None) -> Frame:
        """Export to a :class:`Frame`; ``atom_fields`` keeps only those
        ``atoms`` columns (a missing one raises ``ValueError``)."""
    @staticmethod
    def from_frame(frame: Frame) -> Atomistic: ...
    # --- graph-edit conveniences ---
    def remove_atom(self, handle: int) -> None: ...
    def remove_bond(self, handle: int) -> None: ...
    def set_bond_class(self, handle: int, bond_type: int, bond_number: int) -> None:
        """Set a bond's chemical class (0 unknown, 1 single, 2 double,
        3 triple, 4 aromatic) and its localized bond number (0 unknown,
        1-4) together.

        Raises
        ------
        ValueError
            ``handle`` is stale.
        """
    def set_bond_type(self, handle: int, bond_type: int) -> None:
        """Set a plain (non-aromatic) bond class, whose class implies its
        number. Aromatic (4) implies none and leaves the number ``0``; set it
        through :meth:`set_bond_class` instead.

        Raises
        ------
        ValueError
            ``handle`` is stale.
        """
    def bond_type(self, handle: int) -> int:
        """The bond's chemical class code; ``0`` when it has none."""
    def bond_number(self, handle: int) -> int:
        """The bond's localized (Kekulé) bond number; ``0`` when it has none."""
    def copy(self) -> Atomistic: ...
    def merge(self, other: Atomistic) -> dict[int, int]: ...
    def replicate(
        self,
        template: Atomistic,
        rotations: ArrayF,
        translations: ArrayF,
        frag_ids: ArrayI32,
    ) -> list[int]:
        """Grow this graph by one rigid copy of ``template`` per transform
        (``rotations (N,3,3)``, ``translations (N,3)``), copy ``c`` stamped
        ``frag_id = frag_ids[c]``; returns the new handles, copy-major.

        Raises
        ------
        ValueError
            a wrong shape or count; this graph is unchanged.
        """
    def induced_subgraph(
        self, nodes: list[int]
    ) -> tuple[Atomistic, dict[int, int]]: ...
    def extract_subgraph(
        self,
        centers: list[int],
        radius: int,
        *,
        regenerate_topology: bool = False,
        max_ring_size: int | None = None,
    ) -> ExtractedSubgraph: ...
    # --- structural graph hash (Weisfeiler-Lehman) ---
    def structural_hash(self) -> int: ...
    def canonical_order(self) -> list[int]: ...
    def is_isomorphic(self, other: Atomistic) -> bool: ...
    def center(self) -> ArrayF:
        """Mass-weighted centre ``sum(m_i r_i) / sum(m_i)`` of every atom, from
        ``x``/``y``/``z`` (Å) and ``mass`` (g/mol); a float64 ``(3,)`` array
        in Å. No periodic imaging: unwrap first (:meth:`Box.unwrap`).

        Raises
        ------
        ValueError
            no atoms; an atom without finite ``x``/``y``/``z`` or
            a finite, non-negative ``mass`` (names its int handle); a
            non-positive total mass.
        """
    def translate(self, delta: Sequence[float] | ArrayF) -> Self:
        """Translate every node that has coordinates by ``delta``; returns self."""
    def rotate(
        self, axis: list[float], angle: float, about: list[float] | None = None
    ) -> Self:
        """Rotate every node that has coordinates by ``angle`` radians about
        ``axis``, pivoting on ``about`` (default: the origin); returns self.

        Raises
        ------
        ValueError
            ``axis`` has no direction or ``angle`` is not finite;
            nothing moves then.
        """
    def scale(self, factor: list[float], about: list[float] | None = None) -> Self:
        """Scale every node that has coordinates by a per-axis ``factor``
        about ``about`` (default: the origin); returns self. Pass
        ``[s, s, s]`` for a uniform scale."""

class ExtractedSubgraph:
    """Result of :meth:`Atomistic.extract_subgraph` / :meth:`CoarseGrain.extract_subgraph`."""

    def __init__(
        self,
        graph: Atomistic | CoarseGrain,
        boundary: list[int],
        parent_of: dict[int, int],
        hops: dict[int, int],
        node_map: dict[int, int],
    ) -> None: ...
    @property
    def graph(self) -> Atomistic | CoarseGrain: ...
    @property
    def boundary(self) -> list[int]: ...
    @property
    def parent_of(self) -> dict[int, int]: ...
    @property
    def hops(self) -> dict[int, int]: ...
    @property
    def node_map(self) -> dict[int, int]: ...

class CoarseGrain(MolGraph):
    """Coarse-grained leaf — holds a core ``CoarseGrain`` from construction.

    ``add_bead`` writes ``bead_type``; registers the CG ``bonds`` kind. Owns its
    :meth:`to_frame` / :meth:`from_frame`. Subclassable.

    ``CoarseGrain(**props)`` — the keywords are :attr:`props`. A bead made by
    ``def_bead(atoms=...)`` groups atom views of one source graph;
    ``bead["atoms"]`` answers with them.
    """

    def __init__(self, **props: Any) -> None: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    @property
    def props(self) -> dict[str, Any]: ...
    @property
    def links(self) -> RelationBuckets: ...
    def remove_link(self, *links: RelationRef) -> None: ...
    @property
    def beads(self) -> Refs[Bead]: ...
    @property
    def cgbonds(self) -> Refs[CgBond]: ...
    def def_bead(self, mapping: Any = None, /, **attrs: Any) -> Bead: ...
    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CgBond: ...
    def add_bead(
        self,
        bead_type: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> int: ...
    def add_bond(self, a: int, b: int) -> int: ...
    @property
    def n_beads(self) -> int: ...
    def to_frame(self, atom_fields: Sequence[str] | None = None) -> Frame:
        """Export to a :class:`Frame`; ``atom_fields`` keeps only those
        ``atoms`` columns (a missing one raises ``ValueError``)."""
    @staticmethod
    def from_frame(frame: Frame) -> CoarseGrain: ...
    def set_bead_members(self, bead: int, atoms: list[int]) -> None: ...
    def bead_members(self, bead: int) -> list[int]: ...
    def beads_of_atom(self, atom: int) -> list[int]: ...
    def copy(self) -> CoarseGrain: ...
    def merge(self, other: CoarseGrain) -> dict[int, int]: ...
    def replicate(
        self,
        template: CoarseGrain,
        rotations: ArrayF,
        translations: ArrayF,
        frag_ids: ArrayI32,
    ) -> list[int]:
        """As :meth:`Atomistic.replicate`; bead membership is not copied."""
    def induced_subgraph(
        self, nodes: list[int]
    ) -> tuple[CoarseGrain, dict[int, int]]: ...
    def extract_subgraph(
        self, centers: list[int], radius: int
    ) -> ExtractedSubgraph: ...
    # --- structural graph hash (Weisfeiler-Lehman) ---
    def structural_hash(self) -> int: ...
    def canonical_order(self) -> list[int]: ...
    def is_isomorphic(self, other: CoarseGrain) -> bool: ...
    def center(self, group: Sequence[int]) -> ArrayF:
        """Mass-weighted centre ``sum(m_i r_i) / sum(m_i)`` of the bead group
        ``group`` (a bead listed twice counts twice), from ``x``/``y``/``z``
        (Å) and ``mass`` (g/mol); a float64 ``(3,)`` array in Å. ``add_bead``
        writes no ``mass``; set one first. No periodic imaging: a group
        straddling a box face must be unwrapped first (:meth:`Box.unwrap`).

        Raises
        ------
        ValueError
            an empty group; a handle that is not a live bead, or a
            bead without finite ``x``/``y``/``z`` or a finite,
            non-negative ``mass`` (names its int handle); a non-positive
            total mass.
        OverflowError
            a negative handle.
        """
    def translate(self, delta: Sequence[float] | ArrayF) -> Self:
        """Translate every node that has coordinates by ``delta``; returns self."""
    def rotate(
        self, axis: list[float], angle: float, about: list[float] | None = None
    ) -> Self:
        """Rotate every node that has coordinates by ``angle`` radians about
        ``axis``, pivoting on ``about`` (default: the origin); returns self.

        Raises
        ------
        ValueError
            ``axis`` has no direction or ``angle`` is not finite;
            nothing moves then.
        """
    def scale(self, factor: list[float], about: list[float] | None = None) -> Self:
        """Scale every node that has coordinates by a per-axis ``factor``
        about ``about`` (default: the origin); returns self. Pass
        ``[s, s, s]`` for a uniform scale."""
    def positions(self, beads: Sequence[int]) -> ArrayF:
        """The positions of ``beads``, one row per listed bead, in the listed
        order (a bead listed twice appears twice).

        Parameters
        ----------
        beads : Sequence[int]
            Bead handles, e.g. ``list(cg.atoms)``.

        Returns
        -------
        numpy.ndarray, shape (k, 3), float64
            ``x`` / ``y`` / ``z`` as stored (Å).

        Raises
        ------
        ValueError
            If a handle is not a live bead, or a bead lacks a finite
            ``x`` / ``y`` / ``z``; the message names its int handle.
        """
    def axes(self, beads: Sequence[int]) -> ArrayF:
        """The site axes of ``beads``, one row per listed bead, in the listed
        order. A site from ``Coarsener.coarsen`` carries the vector from the
        first bead of its group to the site; a one-bead site's axis is zero.

        Returns
        -------
        numpy.ndarray, shape (k, 3), float64
            ``axis_x`` / ``axis_y`` / ``axis_z`` as stored (Å).

        Raises
        ------
        ValueError
            If a handle is not a live bead of this graph, or a bead lacks a
            finite axis; the message names its int handle.
        """
    def bead_types(self, beads: Sequence[int]) -> list[str]:
        """The ``bead_type`` of each of ``beads``, in the listed order (a bead
        listed twice appears twice).

        Parameters
        ----------
        beads : Sequence[int]
            Bead handles.

        Returns
        -------
        list[str]
            One type per listed bead.

        Raises
        ------
        ValueError
            If a handle is not a live bead, or a bead carries no
            ``bead_type``; the message names its int handle.
        """

_TRef = TypeVar("_TRef")

class NodeRef:
    """Live view of one node: its fields as a mapping. Made by the graph,
    never constructed directly; interned while alive."""

    @property
    def world(self) -> Atomistic | CoarseGrain: ...
    @property
    def handle(self) -> int: ...
    def __getitem__(self, key: str | tuple[str, ...]) -> Any: ...
    def __setitem__(self, key: str | tuple[str, ...], value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[Any]: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def _restore(cls, world: MolGraph, row: int) -> NodeRef: ...

class Atom(NodeRef): ...
class VirtualSite(Atom): ...
class DrudeParticle(VirtualSite): ...
class MasslessSite(VirtualSite): ...
class Bead(NodeRef): ...

class RelationRef:
    """Live view of one relation: interned endpoint views plus its fields as a
    mapping. Made by the graph, never constructed directly."""

    @property
    def world(self) -> Atomistic | CoarseGrain: ...
    @property
    def kind(self) -> str: ...
    @property
    def handle(self) -> int: ...
    @property
    def endpoints(self) -> tuple[NodeRef, ...]: ...
    def __getitem__(self, key: str | tuple[str, ...]) -> Any: ...
    def __setitem__(self, key: str | tuple[str, ...], value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[Any]: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def _restore(cls, world: MolGraph, kind: str, row: int) -> RelationRef: ...

class Bond(RelationRef):
    @property
    def itom(self) -> Atom: ...
    @property
    def jtom(self) -> Atom: ...

class Angle(RelationRef): ...
class Dihedral(RelationRef): ...
class Improper(Dihedral): ...
class CgBond(RelationRef): ...

class Port(RelationRef):
    @property
    def anchor(self) -> Atom: ...
    @property
    def handle_atom(self) -> Atom: ...

class Refs[TRef]:
    """Ordered views of one kind of one graph; ``refs["x"]`` is a column."""

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> _TRef: ...
    @overload
    def __getitem__(self, key: slice) -> Refs[_TRef]: ...
    @overload
    def __getitem__(self, key: str | tuple[str, ...]) -> npt.NDArray[Any]: ...
    def __iter__(self) -> Iterator[_TRef]: ...
    def __contains__(self, item: object) -> bool: ...
    @classmethod
    def _restore(cls, world: MolGraph, kind: str | None, rows: list[int]) -> Refs[Any]: ...

class RelationBuckets:
    """A graph's relations selected by view class (``graph.links``)."""

    def exact_bucket(self, cls: type[_TRef]) -> Refs[_TRef]: ...

class op:
    """``molrs::op`` — superposition, centroids and NeRF placement.

    Superposition finds the rotation ``R`` and translation ``t`` that best lay
    matched points ``reference[i]`` onto ``target[i]`` (weighted least
    squares, Horn's quaternion method).
    Coordinates are in the caller's length unit (Å in molrs).
    ``DEFAULT_GAP_TOL`` is the default ``gap_tol``: below this scale-free
    eigen-gap the rotation is reported ``"spin"`` (under-determined).
    """

    DEFAULT_GAP_TOL: float

    class Superposition:
        """Best-fit proper rigid motion ``target ≈ rotation @ reference +
        translation`` from :func:`molrs.op.superpose`. Frozen."""

        @property
        def rotation(self) -> ArrayF:
            """Shape ``(3, 3)``, a proper rotation (determinant +1)."""
        @property
        def translation(self) -> ArrayF:
            """Shape ``(3,)``, in the coordinates' length unit (Å)."""
        @property
        def rmsd(self) -> float:
            """Weighted root-mean-square deviation of the fit, in the
            coordinates' length unit (Å)."""
        @property
        def rho(self) -> float:
            """Scale-free eigen-gap (dimensionless); 0 when ``freedom == "free"``."""
        @property
        def center(self) -> ArrayF:
            """Weighted target centroid (Å): the point a spin axis passes through."""
        @property
        def freedom(self) -> Literal["unique", "spin", "free"]:
            """``"unique"``; ``"spin"`` (rotation about :attr:`axis` is
            undetermined); ``"free"`` (no rotation determined, ``rotation`` is
            the identity)."""
        @property
        def axis(self) -> ArrayF | None:
            """The unit spin axis; ``None`` unless ``freedom == "spin"``."""

    @staticmethod
    def superpose(
        reference: ArrayF,
        target: ArrayF,
        weights: ArrayF | None = None,
        *,
        gap_tol: float = ...,
    ) -> Superposition:
        """Best-fit proper rigid motion mapping ``reference`` onto ``target``
        (both shape ``(k, 3)``); ``weights`` shape ``(k,)``, uniform when
        omitted, zero-weight points dropped.

        Raises
        ------
        ValueError
            a shape other than ``(k, 3)``, a length mismatch, a
            negative or non-finite weight, a non-finite coordinate, or no
            positive weight.
        """
    @staticmethod
    def centroid(points: ArrayF, weights: ArrayF | None = None) -> ArrayF | None:
        """Weighted centroid ``Σ wᵢ pᵢ / Σ wᵢ`` (uniform weights when
        omitted), in the length unit of ``points`` (Å in molrs); ``None``
        when the lengths differ or the total weight is not positive and
        finite.

        Raises
        ------
        ValueError
            ``points`` is not shape ``(k, 3)`` or ``weights`` is
            not 1-D.
        """
    @staticmethod
    def place_from_internal_coords(
        a: Sequence[float] | ArrayF,
        b: Sequence[float] | ArrayF,
        c: Sequence[float] | ArrayF,
        bond: float,
        angle: float,
        torsion: float,
    ) -> ArrayF:
        """The point ``d`` (shape ``(3,)``) at distance ``bond`` from ``c``,
        with ∠b–c–d ``angle`` and dihedral a–b–c–d ``torsion`` (radians):
        NeRF (Parsons et al. 2005). Collinear ``a, b, c`` give no off-axis
        component."""

class SmartsMatch:
    """One SMARTS embedding."""

    @property
    def atoms(self) -> list[int]: ...
    @property
    def mapping(self) -> dict[int, int]: ...

class SmartsPattern:
    """Compiled, atom-map-aware SMARTS query over an :class:`Atomistic`.

    Wraps the core Rust SMARTS engine (non-uniquified, RDKit
    ``uniquify=False``). Daylight atom maps (``[C:1]``) add no match
    constraint; a match's :attr:`SmartsMatch.mapping` is its
    ``{map_number: atom_handle}`` dict.
    """

    def __init__(self, smarts: str) -> None: ...
    def has_match(
        self,
        mol: Atomistic,
        *,
        labels: dict[int, str] | None = None,
        root: int | None = None,
    ) -> bool: ...
    def find_matches(
        self,
        mol: Atomistic,
        *,
        labels: dict[int, str] | None = None,
        root: int | None = None,
        limit: int | None = None,
    ) -> list[SmartsMatch]: ...
    @property
    def n_query_atoms(self) -> int: ...
    def map_label(self, query_atom: int) -> int | None: ...
    @property
    def max_bond_depth(self) -> int: ...
    @property
    def ring_primitives(self) -> list[tuple[str, int | None]]: ...
    @classmethod
    def from_environment(
        cls,
        mol: Atomistic,
        center: int,
        *,
        reach: int = 1,
        atomic_number: bool = True,
        include_degree: bool = True,
        include_h_count: bool = True,
        include_charge: bool = True,
        include_aromatic: bool = True,
        include_ring_membership: bool = False,
        include_ring_size: bool = False,
        include_explicit_h_atoms: bool = False,
        include_bond_orders: bool = True,
        neighbor_style: Literal["chain", "recursive"] = "chain",
        canonical_neighbor_order: bool = True,
    ) -> SmartsPattern:
        """The pattern that states the local environment of ``center`` in
        ``mol`` out to ``reach`` bonds; it matches ``mol`` at ``center``."""
    def __str__(self) -> str:
        """The pattern's SMARTS text."""

class Reaction:
    """Compiled Daylight reaction SMARTS (SMIRKS) transform.

    Parses ``reactants >> products`` (tolerating an ignored ``>agent>`` field),
    derives the graph edit from the atom-map diff, and applies it to one matched
    occurrence in place. Reacting atoms may carry SMARTS queries (RDKit-style);
    only concrete product atoms are addable.
    """

    def __init__(self, reaction_smarts: str) -> None: ...
    @property
    def reactant_patterns(self) -> list[SmartsPattern]: ...
    @property
    def forming_bonds(self) -> list[tuple[int, int]]: ...
    def apply(
        self,
        mol: Atomistic,
        binding: dict[int, int],
        labels: dict[int, str] | None = None,
        refresh: bool = True,
    ) -> list[int]: ...
    def apply_many(
        self,
        mol: Atomistic,
        bindings: list[dict[int, int]],
        labels: dict[int, str] | None = None,
        refresh: bool = True,
    ) -> list[list[int]]: ...
    def apply_many_detailed(
        self,
        mol: Atomistic,
        bindings: list[dict[int, int]],
        labels: dict[int, str] | None = None,
        refresh: bool = True,
    ) -> tuple[list[list[int]], list[list[int]]]: ...

@final
class SubgraphMatcher:
    """``molrs.perceive.SubgraphMatcher`` — bead-pattern occurrences in a
    :class:`CoarseGrain`. Beads match on equal ``bead_type``. ``find`` does
    not partition: overlapping groups are all returned. Frozen.
    """

    def __init__(self, pattern: CoarseGrain) -> None:
        """Snapshot ``pattern`` (copied; later edits do not affect it).

        Raises
        ------
        TypeError
            ``pattern`` is not a :class:`CoarseGrain`.
        """
    def find(self, target: CoarseGrain) -> list[list[int]]:
        """Every induced occurrence, one group of target bead handles per
        distinct bead set, in pattern bead order; ``[]`` when none. Releases
        the GIL.

        Raises
        ------
        TypeError
            ``target`` is not a :class:`CoarseGrain`.
        """

@final
class Coarsener:
    """``molrs.builder.Coarsener`` — node groups of a held source graph
    mapped onto the sites of a new :class:`CoarseGrain`. Frozen.

    Site ``I`` stands for ``groups[I]``: it sits at the group's mass-weighted
    centre (Å; no periodic imaging, so unwrap first), carries the group's
    summed ``mass`` and ``bead_type = names[I]``, and records the group's
    handles as its members. Two sites are bonded once when a source bond
    joins their groups.

    Parameters
    ----------
    source : CoarseGrain or Atomistic
        The graph whose nodes are grouped; held, not copied, and read at each
        :meth:`coarsen` call.

    Raises
    ------
    TypeError
        If ``source`` is neither a :class:`CoarseGrain` nor an
        :class:`Atomistic`.
    """

    def __init__(self, source: CoarseGrain | Atomistic) -> None: ...
    def coarsen(
        self, groups: Sequence[Sequence[int]], names: Sequence[str]
    ) -> CoarseGrain:
        """A new :class:`CoarseGrain` with one site per group, in group
        order. Releases the GIL.

        Parameters
        ----------
        groups : Sequence[Sequence[int]]
            Disjoint, non-empty node-handle groups of the source.
        names : Sequence[str]
            One site ``bead_type`` per group.

        Returns
        -------
        CoarseGrain
            The sites; empty when ``groups`` is empty.

        Raises
        ------
        ValueError
            If ``groups`` and ``names`` differ in length, a group is empty, a
            handle is listed twice, or a group has no centre (a stale handle,
            a missing coordinate or mass, a non-positive total mass); handles
            are named by their int value.
        """

# ---------------------------------------------------------------------------
# Chemical perception — the builder (graph in / graph out, non-mutating)
# ---------------------------------------------------------------------------

def assign_rings(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` with ``is_in_ring`` / ``n_rings`` on every atom and bond."""
def assign_aromaticity(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` in the standard aromatic representation:
    ``is_aromatic`` on atoms, ``bond_type`` / ``bond_number`` on bonds."""
def add_hydrogens(mol: Atomistic) -> Atomistic:
    """New graph: ``mol`` plus the hydrogens its open valences imply."""
def assign_stereo(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` with ``stereo`` on each perceived stereocentre / bond."""
def assign_rotatable_bonds(
    mol: Atomistic,
    *,
    unknown_bond: Literal["not_rotatable", "single"] = "not_rotatable",
) -> Atomistic:
    """Clone of ``mol`` with ``is_rotatable`` (0/1) on every bond.
    ``unknown_bond`` says what a bond with no ``bond_type`` counts as:
    ``"not_rotatable"`` (never guess) or ``"single"``."""
def assign_bond_orders(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` carrying antechamber's Kekulé structure, judged from
    the connectivity (``bond_number`` / ``bond_type``)."""
def assign_kekule_bond_orders(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` whose aromatic bonds carry a legal localized
    ``bond_number``."""
def assign_bcc_bond_types(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` with ``bcc_bond_type`` on every bond, from the stated
    bond orders."""
def assign_bcc_bond_types_from_connectivity(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` with ``bcc_bond_type`` on every bond, from bond orders
    judged from the connectivity (antechamber's ``bondtype -j full``)."""
def assign_equivalence_classes(mol: Atomistic) -> Atomistic:
    """Clone of ``mol`` with ``equiv_class`` on every atom (antechamber
    ``-eq 1``)."""
def perceive_rings(mol: Atomistic) -> RingSet:
    """The SSSR rings of ``mol`` as a :class:`RingSet`; ``mol`` is untouched."""

# ---------------------------------------------------------------------------
# Frame vocabulary (molrs.core.keys / molrs.core.schema, mirrors
# molrs::core::keys / molrs::core::schema)
# ---------------------------------------------------------------------------

class keys:
    """``molrs.core.keys``: the canonical column and frame-meta names,
    projected from the Rust key tables; ordered groups are lists."""

    class Key:
        """A canonical name; ``str(key)`` / ``.key`` is the plain string."""

        @property
        def key(self) -> str: ...
        def __hash__(self) -> int: ...

    ALTLOC: Key
    ANGLE_TYPE_LABELS: Key
    ATOMI: Key
    ATOMIC_NUMBER: Key
    ATOMJ: Key
    ATOMK: Key
    ATOML: Key
    ATOMM: Key
    ATOM_MAP: Key
    ATOM_TYPE_LABELS: Key
    AXIS: list[Key]
    AXIS_X: Key
    AXIS_Y: Key
    AXIS_Z: Key
    BEAD_TYPE: Key
    BOND_NUMBER: Key
    BOND_TYPE: Key
    BOND_TYPE_LABELS: Key
    B_FACTOR: Key
    CHAIN: Key
    CHARGE: Key
    CMAP_TYPE_LABELS: Key
    COORDS: list[Key]
    DIHEDRAL_TYPE_LABELS: Key
    DIPOLE: list[Key]
    ELEMENT: Key
    ENDPOINTS: list[Key]
    EXCLUDE_14: Key
    FORCES: list[Key]
    FORMAL_CHARGE: Key
    FREE: Key
    FX: Key
    FY: Key
    FZ: Key
    IBEAD: Key
    ICODE: Key
    ID: Key
    IMAGES: list[Key]
    IMPROPER_TYPE_LABELS: Key
    IS_14: Key
    IX: Key
    IY: Key
    IZ: Key
    MASS: Key
    MOL_ID: Key
    MUX: Key
    MUY: Key
    MUZ: Key
    NAME: Key
    OCCUPANCY: Key
    QUAT: list[Key]
    QUATI: Key
    QUATJ: Key
    QUATK: Key
    QUATW: Key
    RES_ID: Key
    RES_NAME: Key
    STYLE: Key
    TYPE: Key
    TYPE_ID: Key
    UNITS: Key
    VELOCITIES: list[Key]
    VX: Key
    VY: Key
    VZ: Key
    X: Key
    Y: Key
    Z: Key
    # Molecular-graph keys.
    BCC_BOND_TYPE: Key
    BEAD_ATOMS: Key
    EQUIV_CLASS: Key
    FRAG_ID: Key
    PORTS: Key
    REACT_ID: Key
    VSITE: Key
    # LAMMPS frame-meta keys.
    LAMMPS_COEFFS_TEXT: Key
    LAMMPS_UNITS: Key

class constants:
    """``molrs.core.constants``: every constant of ``molrs::core::constants``."""

    AVOGADRO: float
    BOLTZMANN: float
    GAS_CONSTANT: float
    ELEMENTARY_CHARGE: float
    PLANCK: float
    BOHR_RADIUS: float
    HARTREE_ENERGY: float
    ATOMIC_MASS_CONSTANT: float
    COULOMB_CONSTANT: float
    COULOMB_REAL: float
    COULOMB_METAL: float
    AMBER_COULOMB: float
    AMBER_CHARGE_FACTOR: float
    CHARMM_COULOMB: float
    OPENMM_ONE_4PI_EPS0: float
    GROMACS_ONE_4PI_EPS0: float
    SPEED_OF_LIGHT: float
    SECOND_RADIATION_CONSTANT: float
    MMFF_MDYNE_A_TO_KCAL_MOL: float
    PARMCHK2_PI: float
    VACUUM_DIELECTRIC: float
    UFF_COULOMB: float
    MMFF_COULOMB: float
    OPLS_LJ_14: float
    OPLS_COULOMB_14: float
    AMBER_SCEE: float
    AMBER_SCNB: float

class schema:
    """``molrs.core.schema``: the Frame vocabulary's blocks and columns,
    projected from the Rust tables."""

    class ColumnSpec:
        """One canonical column of the Frame vocabulary."""

        def __init__(
            self,
            key: str,
            const_name: str,
            dtype: str,
            shape: str,
            dimension: str,
            unit: str,
            doc: str,
        ) -> None: ...
        @property
        def key(self) -> str: ...
        @property
        def const_name(self) -> str: ...
        @property
        def dtype(self) -> str: ...
        @property
        def shape(self) -> str: ...
        @property
        def dimension(self) -> str: ...
        @property
        def unit(self) -> str: ...
        @property
        def doc(self) -> str: ...
        @property
        def numpy_dtype(self) -> str: ...

    class BlockSpec:
        """One canonical block of the Frame vocabulary."""

        def __init__(
            self,
            name: str,
            row_kind: str,
            endpoint_target: str | None,
            endpoint_columns: list[str],
            required: list[str],
            optional: list[str],
            open: bool,
            doc: str,
            declared_endpoints: list[str],
        ) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def row_kind(self) -> str: ...
        @property
        def endpoint_target(self) -> str | None: ...
        @property
        def endpoint_columns(self) -> list[str]: ...
        @property
        def declared_endpoints(self) -> list[str]: ...
        @property
        def required(self) -> list[str]: ...
        @property
        def optional(self) -> list[str]: ...
        @property
        def open(self) -> bool: ...
        @property
        def doc(self) -> str: ...

    columns: list[ColumnSpec]
    blocks: list[BlockSpec]
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
    VIRTUAL_SITES: str
    TOPOLOGY: tuple[str, ...]

    @staticmethod
    def column(key: ColumnKey) -> ColumnSpec | None: ...
    @staticmethod
    def block(name: str) -> BlockSpec | None: ...
    @staticmethod
    def to_json() -> str: ...
    @staticmethod
    def to_markdown() -> str: ...
    @staticmethod
    def relation_endpoints(
        name: str, columns: list[str], targets: dict[str, str] | None = None
    ) -> list[tuple[str, str]]: ...

# ---------------------------------------------------------------------------
# SMILES
# ---------------------------------------------------------------------------

class SmilesIr:
    """Intermediate representation of a parsed SMILES string (or SMILES
    fragment body).

    ``to_atomistic()`` is the plain conversion: it refuses SMARTS query atoms
    and, since it will not drop them silently, any node carrying a bonding
    descriptor — which is what a ``CgFragmentDef.body`` from the last CGsmiles
    block holds. Build such a body's ported unit with ``to_template()``
    (parse it with ``SmilesIr.from_fragment``), or expand a whole string
    through ``CgSmilesIr.to_atomistic``.
    """

    def __init__(self, smiles: str) -> None: ...
    @classmethod
    def from_fragment(cls, body: str) -> SmilesIr:
        """Parse a CGsmiles fragment body (SMILES plus bonding descriptors,
        e.g. ``"[<]OCC[>]"``); the plain constructor refuses descriptors."""
    def to_template(self) -> Atomistic:
        """The ported unit of this body: heavy atoms plus one hydrogen handle
        and one port per bonding descriptor; no coordinates, no ``frag_id``.
        ``SmilesIr.from_fragment("[<]OCC[>]").to_template()`` equals
        ``CgSmilesIr("{[#EO]}.{#EO=[<]OCC[>]}").templates()["EO"]``."""
    @property
    def n_components(self) -> int: ...
    def to_atomistic(self) -> Atomistic: ...
    def components(self) -> list[Atomistic]: ...
    @classmethod
    def from_atomistic(
        cls,
        mol: Atomistic,
        *,
        canonical: bool = True,
        root: int | None = None,
        aromatic: Literal["as_marked", "kekule_only"] = "as_marked",
        hydrogens: Literal[
            "organic_subset", "explicit_all", "as_stored"
        ] = "organic_subset",
        include_stereo: bool = False,
        multi_component: Literal[
            "error_if_multiple", "join_dot", "first_only"
        ] = "error_if_multiple",
        organic_subset: bool = True,
    ) -> SmilesIr: ...

# ---------------------------------------------------------------------------
# CGsmiles — one front door plus the read-only records it hands out
#
# `CgSmilesIr` parses; every other class here is a read-only view over one
# record of the value it returns and has no constructor of its own. There is
# no `CgSmilesReader`: "Reader" here means a lazy, path-backed trajectory
# cursor, and a text-in / IR-out parser is not that — see the `molrs.io`
# module docstring.
#
# Every enum crosses as a name, not as its numeric storage code: those codes
# are not injective over these enums, so a number could not be read back as
# what the notation wrote. The exception is a coarse edge's multiplicity,
# which *is* a count and crosses as one.
#
# A descriptor kind crosses as its grammar glyph (`$`, `<`, `>`, `!`) — the
# spelling a user writes and the one a stored port's `port_kind` holds, so
# there is no third vocabulary between notation, column and boundary. The
# enums the notation does not spell out (`BondingDescriptor.order`,
# `ResolvedPair.kind`, `PairEnd.end`) cross as the lowercase spelling of their
# Rust variant.
# ---------------------------------------------------------------------------

# The nine lowercase `BondKind` spellings — shared by `BondingDescriptor.order`
# and `ResolvedPair.kind`, which name the same enum.
type BondKindName = Literal[
    "single", "double", "triple", "quadruple", "aromatic", "up", "down", "any", "ring"
]

class BondingDescriptor:
    """One bonding descriptor: a site at which a fragment may later be joined.

    ``kind`` is the operator written, as the glyph itself — the same spelling
    a stored port's ``port_kind`` uses. A ``"<"`` pairs only with a ``">"``, a
    ``"$"`` only with a ``"$"``, and the labels must match exactly. ``order``
    is the bond order written beside the bracket, ``None`` when none was —
    which counts as ``"single"`` for pairing.
    """

    @property
    def kind(self) -> Literal["$", "<", ">", "!"]: ...
    @property
    def label(self) -> str: ...
    @property
    def order(self) -> BondKindName | None: ...

class CgNode:
    """One coarse-grained node: ``[#PEO]``, ``[#A;q=-0.5]``.

    ``charge`` is a *partial* charge in elementary-charge units ``e`` (the
    ``q`` annotation), never a formal charge. ``parent`` indexes the previous
    level's ``nodes``, and is ``None`` in ``levels[0]`` and in a fragment body.
    """

    @property
    def name(self) -> str: ...
    @property
    def charge(self) -> float | None: ...
    @property
    def annotations(self) -> list[tuple[str, str]]: ...
    @property
    def descriptors(self) -> list[BondingDescriptor]: ...
    @property
    def parent(self) -> int | None: ...

class CgEdge:
    """One coarse edge, joining ``nodes[i]`` and ``nodes[j]`` of its level.

    ``multiplicity`` is how many bonds the edge stands for (1–4, from ``-``
    ``=`` ``#`` ``$``), never a bond kind. ``derived_from`` is the
    ``(level, pair)`` of the resolved pair that induced the edge, or ``None``
    when the notation wrote it; a derived edge always has multiplicity 1.
    """

    @property
    def i(self) -> int: ...
    @property
    def j(self) -> int: ...
    @property
    def multiplicity(self) -> int: ...
    @property
    def derived_from(self) -> tuple[int, int] | None: ...

class CgGraph:
    """One resolution level: coarse-grained nodes and the edges between them.

    Both lists are in parse order — nodes as their brackets were read, edges
    as the notation formed them, with every derived edge appended after the
    written ones. A node is addressed by its index in ``nodes``.
    """

    @property
    def nodes(self) -> list[CgNode]: ...
    @property
    def edges(self) -> list[CgEdge]: ...

class CgFragmentDef:
    """One entry of a fragment block: ``#PEO=[$]COC[$]``.

    ``body`` is a ``CgGraph`` in an intermediate block and a ``SmilesIr`` in
    the last one — the Python type is the tag, so dispatch with ``isinstance``.
    A ``SmilesIr`` body keeps its bonding descriptors, so its own
    ``to_atomistic()`` refuses it; expand through ``CgSmilesIr.to_atomistic``.
    Its ``repr`` echoes the fragment-table entry as written —
    ``#PEO=[$]COC[$]``, name and ``=`` included — not a bare SMILES, because
    the entry's span is the text the IR records.
    """

    @property
    def name(self) -> str: ...
    @property
    def body(self) -> CgGraph | SmilesIr: ...

class PairEnd:
    """One end of a :class:`ResolvedPair`: the port, and who offered it.

    For a pair read from ``ir.pairs[k]``: when ``end == "sub"`` ``index``
    indexes ``levels[k + 1].nodes`` (the child node carrying the port) and
    ``port`` indexes that node's ``descriptors``; when ``end == "body"``
    ``index`` indexes ``levels[k].nodes`` and ``port`` indexes that node's
    atomistic body's descriptor map.
    """

    @property
    def end(self) -> Literal["sub", "body"]: ...
    @property
    def index(self) -> int: ...
    @property
    def port(self) -> int: ...

class ResolvedPair:
    """One bond a written coarse edge stands for, with the ports it consumed.

    ``kind`` is the chemistry of that bond as a lowercase bond-kind name: the
    order written on either descriptor, ``"aromatic"`` when neither wrote one
    and both port atoms are written aromatic, ``"single"`` otherwise.
    """

    @property
    def edge(self) -> int: ...
    @property
    def bond(self) -> int: ...
    @property
    def src(self) -> PairEnd: ...
    @property
    def dst(self) -> PairEnd: ...
    @property
    def kind(self) -> BondKindName: ...

class CgSmilesIr:
    """Intermediate representation of a parsed CGsmiles string.

    Constructing it parses, validates, expands and resolves the whole string;
    the value is then read, not built. ``levels`` are the resolution levels
    (coarsest first), ``fragments[k]`` resolves the names of ``levels[k]``, and
    ``to_atomistic()`` expands the lowest level into a topology-only graph
    whose atoms carry ``frag_id``.
    """

    def __init__(self, text: str) -> None: ...
    @property
    def levels(self) -> list[CgGraph]: ...
    @property
    def fragments(self) -> list[dict[str, CgFragmentDef]]: ...
    @property
    def pairs(self) -> list[list[ResolvedPair]]: ...
    def to_atomistic(self) -> Atomistic: ...
    def templates(self) -> dict[str, Atomistic]:
        """One ported :class:`Atomistic` template per definition of the last
        fragment table, keyed by name. One body alone:
        ``SmilesIr.from_fragment(body).to_template()``."""
    def to_coarsegrain(self) -> CoarseGrain:
        """The coarsest level, ``levels[0]``, as a bead graph: one bead per
        node (``bead_type`` only, no coordinates or mass), one CG bond per
        edge.

        Raises
        ------
        SmilesError
            (a ``ValueError``) the IR breaks a reader invariant;
            no parsed string reaches this.
        """

# ---------------------------------------------------------------------------
# I/O — readers and writers (molrs.io: one function per door; each format's
# classes in its submodule)
# ---------------------------------------------------------------------------

def read_csv_block_str(
    text: str, delimiter: str = ",", header: list[str] | None = None
) -> Block:
    """CSV text as a :class:`Block`; dtypes inferred int → float → str."""
def read_csv_block(
    path: PathInput, delimiter: str = ",", header: list[str] | None = None
) -> Block:
    """:func:`read_csv_block_str` over the UTF-8 file at ``path``."""
def write_csv_block_str(block: Block, delimiter: str = ",", header: bool = True) -> str:
    """``block`` as CSV text; the inverse of :func:`read_csv_block_str`."""
def write_csv_block(
    path: PathInput, block: Block, delimiter: str = ",", header: bool = True
) -> None:
    """``block`` as a CSV file at ``path``; the inverse of :func:`read_csv_block`."""
def read_clpol_alpha_str(text: str) -> list[dict[str, str | float]]:
    """``alpha.ff`` text as its rows in file order: ``type_name``, ``m_D``,
    ``q_D_sign``, ``k_D``, ``alpha``, ``a_thole``. Raises ``ValueError`` for a
    row whose numbers do not parse."""
def read_clpol_alpha(path: PathInput) -> list[dict[str, str | float]]:
    """:func:`read_clpol_alpha_str` over the file at ``path``."""
def read_msgpack_frame_bytes(data: bytes) -> Frame: ...
def write_msgpack_frame_bytes(frame: Frame) -> bytes: ...
def read_json_frame_str(text: str) -> Frame: ...
def write_json_frame_str(frame: Frame) -> str: ...
def read_pdb(path: PathInput) -> Frame: ...
def read_xyz(path: PathInput) -> Frame: ...
def read_sdf(path: PathInput) -> Frame:
    """The first record of an MDL SDF / molfile (V2000)."""
def read_cif(path: PathInput) -> Frame:
    """The first ``data_`` block of a CIF file."""
def write_cif(path: PathInput, frame: Frame) -> None: ...
def read_poscar(path: PathInput) -> Frame: ...
def write_poscar(path: PathInput, frame: Frame) -> None: ...
def read_lammps_data(path: PathInput, atom_style: str | None = None) -> Frame:
    """Read a LAMMPS data file. Typed blocks carry ``type_id`` and the string
    ``type`` (the file's type label, or the id as a label); ``atom_style``
    fixes the ``Atoms`` layout as LAMMPS's ``atom_style`` does."""

class BondReactTemplate:
    """One ``fix bond/react`` reaction: ``pre`` / ``post`` templates
    (``Atomistic`` or ``Frame``, atoms paired by an integer ``react_id``) and
    the atoms the map file names (atoms of ``pre``, or their ``react_id``)."""

    pre: Any
    post: Any
    initiator_atoms: Any
    edge_atoms: Any
    deleted_atoms: Any
    def __init__(
        self,
        pre: Any,
        post: Any,
        initiator_atoms: Sequence[Any],
        edge_atoms: Sequence[Any] | None = None,
        deleted_atoms: Sequence[Any] | None = None,
    ) -> None: ...
    def map_text(self) -> str:
        """The map file's text. Raises ``ValueError`` for a malformed
        template."""

def write_lammps_bond_react_map(template: BondReactTemplate, base_path: PathInput) -> None:
    """Write ``{base_path}.map``. Raises ``ValueError`` for a malformed
    template."""

def write_lammps_bond_react_system(
    workdir: PathInput,
    frame: Frame,
    forcefield: ForceField,
    templates: _AbcMapping[str, BondReactTemplate] | Sequence[BondReactTemplate],
) -> None:
    """Write ``{stem}.data``, ``{stem}.ff`` and per template
    ``{name}_pre.mol`` / ``{name}_post.mol`` / ``{name}.map`` into
    ``workdir``, with one type numbering for system and templates. Warns
    for untyped template topology it leaves out."""

def read_stl(path: PathInput) -> TriMesh: ...
def read_gro(path: PathInput) -> Frame: ...
def read_xsf(path: PathInput) -> Frame: ...
def read_inpcrd(path: PathInput, frame: Frame | None = None) -> Frame:
    """Read an AMBER ASCII inpcrd / restart file. With ``frame``, its
    coordinates (``x``/``y``/``z``, ``vel``), box and meta keys go into that
    frame in place, every other column stays, and ``frame`` is returned.
    Raises ``OSError`` on an atom-count mismatch (``frame`` unchanged)."""
def read_prmtop(path: PathInput) -> Frame: ...

# The in-memory doors: each one its path twin on a str / bytes.
def read_pdb_str(text: str) -> Frame: ...
def read_pdb_bytes(data: bytes) -> Frame: ...
def write_pdb_str(frame: Frame) -> str: ...
def read_xyz_str(text: str) -> Frame: ...
def read_xyz_bytes(data: bytes) -> Frame: ...
def write_xyz_str(frame: Frame) -> str: ...
def read_sdf_str(text: str) -> Frame: ...
def read_sdf_bytes(data: bytes) -> Frame: ...
def read_gro_str(text: str) -> Frame: ...
def write_gro_str(frame: Frame) -> str: ...
def read_mol2_str(text: str) -> Frame: ...
def write_mol2_str(frame: Frame) -> str: ...
def read_cif_str(text: str) -> Frame: ...
def write_cif_str(frame: Frame) -> str: ...
def read_xsf_str(text: str) -> Frame: ...
def write_xsf_str(frame: Frame) -> str: ...
def read_cube_str(text: str) -> Frame: ...
def write_cube_str(frame: Frame) -> str: ...
def read_poscar_str(text: str) -> Frame: ...
def write_poscar_str(frame: Frame) -> str: ...
def read_chgcar_str(text: str) -> Frame: ...
def read_ac_str(text: str) -> Frame: ...
def read_inpcrd_str(text: str, frame: Frame | None = None) -> Frame: ...
def read_prmtop_str(text: str) -> Frame: ...
def read_prep_str(text: str) -> dict[str, Any]: ...
def write_prep_str(residue: dict[str, Any]) -> str: ...
def read_lammps_data_str(text: str, atom_style: str | None = None) -> Frame: ...
def read_lammps_data_bytes(data: bytes) -> Frame: ...
def write_lammps_data_str(
    frame: Frame, *, type_labels: _AbcMapping[str, Sequence[str]] | None = None
) -> str: ...
def read_lammps_dump_str(text: str) -> Frame: ...
def read_lammps_dump_bytes(data: bytes) -> Frame: ...
def write_lammps_dump_str(frame: Frame, columns: Sequence[str] | None = None) -> str: ...
def read_dcd_bytes(data: bytes, decoder_state: bytes | None = None) -> Frame: ...
def write_dcd_bytes(frame: Frame) -> bytes: ...
def read_trr_bytes(data: bytes) -> Frame: ...
def write_trr_bytes(frame: Frame) -> bytes: ...
def read_xtc_bytes(data: bytes) -> Frame: ...
def write_xtc_bytes(frame: Frame) -> bytes: ...
def read_stl_bytes(data: bytes) -> TriMesh: ...
def read_xml_forcefield_str(text: str) -> ForceField: ...
def read_openmm_forcefield_str(text: str) -> ForceField: ...
def read_lammps_forcefield_str(text: str) -> ForceField: ...
def write_frcmod_str(forcefield: ForceField) -> str: ...
def write_openmm_forcefield_str(forcefield: ForceField, precision: int | None = None) -> str: ...
def write_xml_forcefield_str(forcefield: ForceField) -> str: ...

class PdbReader:
    """Lazy, indexed reader of PDB files — ``molrs.io.pdb.PdbReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> PdbReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_pdb_trajectory(paths: PathInput | Sequence[PathInput]) -> PdbReader:
    """Open one PDB trajectory, or several whose frames are concatenated."""

class XyzReader:
    """Lazy, indexed reader of XYZ files — ``molrs.io.xyz.XyzReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> XyzReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_xyz_trajectory(paths: PathInput | Sequence[PathInput]) -> XyzReader:
    """Open one XYZ trajectory, or several whose frames are concatenated."""

class GroReader:
    """Lazy, indexed reader of GRO files — ``molrs.io.gro.GroReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> GroReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_gro_trajectory(paths: PathInput | Sequence[PathInput]) -> GroReader:
    """Open one GRO trajectory, or several whose frames are concatenated."""

class LammpsDumpReader:
    """Lazy, indexed reader of LAMMPS dump files — ``molrs.io.lammps.LammpsDumpReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> LammpsDumpReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_lammps_dump_trajectory(paths: PathInput | Sequence[PathInput]) -> LammpsDumpReader:
    """Open one LAMMPS dump trajectory, or several whose frames are concatenated."""

class DcdReader:
    """Lazy, indexed reader of DCD files — ``molrs.io.dcd.DcdReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> DcdReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_dcd_trajectory(paths: PathInput | Sequence[PathInput]) -> DcdReader:
    """Open one DCD trajectory, or several whose frames are concatenated."""

class TrrReader:
    """Lazy, indexed reader of GROMACS TRR files — ``molrs.io.trr.TrrReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> TrrReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_trr_trajectory(paths: PathInput | Sequence[PathInput]) -> TrrReader:
    """Open one GROMACS TRR trajectory, or several whose frames are concatenated."""

class XtcReader:
    """Lazy, indexed reader of GROMACS XTC files — ``molrs.io.xtc.XtcReader``.

    One path, or several whose frames are concatenated. Frames are parsed on
    demand; random access indexes a file once. Surface: ``read_frame``
    (negative indexing), ``read_frames``, ``read_range``, ``read_all``,
    ``n_frames``, integer and slice indexing, lazy iteration, ``close()``,
    and use as a context manager.
    """

    def __init__(self, paths: PathInput | Sequence[PathInput]) -> None: ...
    @property
    def n_frames(self) -> int: ...
    def build_index(self) -> None: ...
    def read_frame(self, index: int) -> Frame: ...
    def read_frames(self, indices: Sequence[int]) -> list[Frame]: ...
    def read_range(
        self, start: int = ..., stop: int | None = ..., step: int = ...
    ) -> list[Frame]: ...
    def read_all(self) -> list[Frame]: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> list[Frame]: ...
    def __iter__(self) -> XtcReader: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool: ...

def read_xtc_trajectory(paths: PathInput | Sequence[PathInput]) -> XtcReader:
    """Open one GROMACS XTC trajectory, or several whose frames are concatenated."""

def read_chgcar(path: PathInput) -> Frame: ...
def read_cube(path: PathInput) -> Frame: ...
def write_cube(path: PathInput, frame: Frame) -> None: ...
def write_pdb(path: PathInput, frame: Frame) -> None: ...
def write_pdb_trajectory(path: PathInput, frames: list[Frame]) -> None:
    """Write a list of Frames to a multi-MODEL PDB trajectory."""

def write_xyz(path: PathInput, frame: Frame) -> None: ...
def write_xyz_trajectory(path: PathInput, frames: Sequence[Frame]) -> None: ...
def write_lammps_data(
    path: PathInput,
    frame: Frame,
    *,
    type_labels: _AbcMapping[str, Sequence[str]] | None = None,
) -> None:
    """Write a LAMMPS data file. ``type_labels`` declares extra labels per
    block (``{"atoms": [...], ...}``) even when no row uses them; a Drude
    system gets a ``fix drude`` flags header comment."""
def write_lammps_dump_trajectory(
    path: PathInput, frames: Sequence[Frame], columns: Sequence[str] | None = None
) -> None: ...
def write_lammps_dump_local(path: PathInput, frames: Sequence[Frame]) -> None: ...
def write_dcd_trajectory(path: PathInput, frames: Sequence[Frame]) -> None: ...
def write_gro(path: PathInput, frame: Frame) -> None: ...
def write_gro_trajectory(path: PathInput, frames: Sequence[Frame]) -> None: ...
def write_xsf(path: PathInput, frame: Frame) -> None: ...
def read_cgsmiles_str(text: str) -> Atomistic:
    """The molecule a CGsmiles string states, its lowest level expanded into
    atoms (topology only)."""


# ---------------------------------------------------------------------------
# Structure builders (molrs::builder)
# ---------------------------------------------------------------------------

class CarbonTubeBuilder:
    def __init__(
        self,
        n: int,
        m: int,
        *,
        length: float | None = None,
        cells: int | None = None,
        bond_length: float = 1.42,
        periodic: bool = False,
        vacuum: float = 10.0,
    ) -> None: ...
    def build(self, *, atom_type: str | None = None, charge: float = 0.0) -> Frame: ...
    def cell(self, *, vacuum: float | None = None) -> Box: ...
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def cells(self) -> int: ...
    @property
    def bond_length(self) -> float: ...
    @property
    def periodic(self) -> bool: ...

class GrapheneBuilder:
    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        bond_length: float = 1.42,
        vacuum: float = 10.0,
        periodic_xy: bool = True,
    ) -> None: ...
    def build(self, *, atom_type: str | None = None, charge: float = 0.0) -> Frame: ...
    def cell(self, *, vacuum: float | None = None) -> Box: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    @property
    def bond_length(self) -> float: ...
    @property
    def periodic_xy(self) -> bool: ...

@final
class SitePlacer:
    """``molrs.builder.SitePlacer`` — translation-only placement. Frozen.

    Each copy's centre of mass (Å, weights ``mass``) lands on its site;
    rotation is the orienter's job. Takes no arguments.
    """

    def __init__(self) -> None: ...

@final
class GrowthPlacer:
    """``molrs.builder.GrowthPlacer`` — grows each molecule onto its parents'
    ports. Frozen.

    The first copy of a molecule keeps its template pose (its centre of mass
    on the site when the site graph has positions); every later copy is
    rotated and moved so the anchor of its port toward its parent lands on
    the parent's leaving handle, pointing back along that bond. Needs no site
    positions; bond lengths and overlaps are left to a later minimisation.
    Takes no arguments.
    """

    def __init__(self) -> None: ...

@final
class AxisOrienter:
    """``molrs.builder.AxisOrienter`` — turns each copy onto its site. Frozen.

    Turns each copy about its template's centre of mass. A chain site (only
    ``<`` / ``>`` ports, a two-port template) matches the template's
    backbone-to-centre direction to the site axis (``CoarseGrain.axes``) and
    its two joining atoms to the site's bond line. Any other bonded site fits
    the template's port directions to its bond directions. A site with no
    bond is not turned. Takes no arguments.
    """

    def __init__(self) -> None: ...

@final
class Assembler:
    """``molrs.builder.Assembler`` — one placed, linked world from a site
    graph. Frozen.

    Each bead of the site graph is one unit: a copy of ``library[bead_type]``,
    turned by the orienter (when given) and given its pose by the placer. Each site bond joins one
    port of each end's copy (``<`` with ``>``, ``$`` with ``$``); the leaving
    groups are removed. Any topology works: chains, branches, rings. Every
    atom gets ``frag_id`` (the site's ordinal) and ``mol_id`` (its connected
    component's ordinal + 1).

    Parameters
    ----------
    library : Mapping[str, MolGraph]
        Name → template, copied at construction: any graph (``MolGraph``,
        ``Atomistic``, ``CoarseGrain``), with ports where a site bonds. A
        template without ports can only fill an unbonded site.
    placer : SitePlacer | GrowthPlacer
        ``SitePlacer`` moves each copy's centre of mass onto its site;
        ``GrowthPlacer`` grows each molecule onto its parents' ports and needs
        no site positions.
    orienter : AxisOrienter, optional
        Turns each copy about its centre of mass before it is placed; needs
        site positions.

    Raises
    ------
    TypeError
        If ``library`` is not a mapping of ``str`` to graphs, ``placer`` is
        not a :class:`SitePlacer` or :class:`GrowthPlacer`, or ``orienter``
        is not an :class:`AxisOrienter`.
    """

    def __init__(
        self,
        library: _AbcMapping[str, MolGraph],
        placer: SitePlacer | GrowthPlacer,
        orienter: AxisOrienter | None = None,
    ) -> None: ...
    @overload
    def assemble(self, sites: CoarseGrain, cls: None = None) -> MolGraph: ...
    @overload
    def assemble(self, sites: CoarseGrain, cls: type[_TGraph]) -> _TGraph:
        """Place and join one copy per site; return the world. Releases the
        GIL.

        Parameters
        ----------
        sites : CoarseGrain
            The site graph: each bead's ``bead_type`` names its template and
            each bond joins two copies; its optional position (Å) and axis
            are read by the placer and the orienter.

        cls : type, optional
            The graph class to build the world as: ``MolGraph`` (the default),
            ``Atomistic`` or ``CoarseGrain``.

        Returns
        -------
        MolGraph
            The world, an instance of ``cls``; ports without a site bond stay
            on it. Empty when ``sites`` is empty.

        Raises
        ------
        ValueError
            Naming the site at fault: a bead lacks its type or position, a
            name is not in the library, no accepting port exists for every
            bond of a site, a site cannot be oriented or placed, or a join is
            refused.
        """

# ---------------------------------------------------------------------------
# 3D coordinate generation (embed)
# ---------------------------------------------------------------------------

class ConformerStageReport:
    """Per-stage report from the conformer generation pipeline."""

    @property
    def stage(self) -> str: ...
    @property
    def energy_before(self) -> float | None: ...
    @property
    def energy_after(self) -> float | None: ...
    @property
    def steps(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def elapsed_ms(self) -> int: ...

class ConformerReport:
    """Aggregate report from a conformer generation run."""

    @property
    def final_energy(self) -> float | None: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def stages(self) -> list[ConformerStageReport]: ...

class Conformer:
    """3D conformer generator.

    Construct with the desired generation parameters, then call
    :meth:`generate` to produce coordinates. Subclassable.
    """

    def __init__(
        self,
        speed: str = "medium",
        add_hydrogens: bool = True,
        seed: int | None = None,
    ) -> None: ...
    def generate(self, mol: Atomistic) -> tuple[Atomistic, ConformerReport]: ...

# ---------------------------------------------------------------------------
# Force field
#
# Only the native handle is declared here. The style and parameter views a
# caller reaches through it are pure Python and are declared in
# `molrs/ff/forcefield.py`, beside the code that defines them.
# ---------------------------------------------------------------------------

class ForceField:
    """A force field: styles and their types. Subclassable; pickles by
    content. Styles and types are read and written through their handles."""

    def __init__(self, name: str = "forcefield", units: str | None = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def units(self) -> str: ...
    def merge(self, other: ForceField) -> Self: ...
    def canonical(self) -> ForceField:
        """Every style of a form family in its canonical style's category
        mapped onto that style, exactly (``dihedral opls``/``charmm``/``rb``
        … → ``dihedral periodic``; ``bond class2`` → ``bond harmonic``;
        ``pair lj/class2`` → ``pair lj/cut``). Impropers stay. Idempotent.

        Raises
        ------
        molrs.ff.ir.OutOfImageError
            A row the canonical style cannot hold (charmm ``w ≠ 0``, class2
            ``k3 ≠ 0``), naming the type and the condition.
        molrs.ff.ir.FormConflictError
            A form family without exactly one canonical style."""
    def to_form(self, category: str, style: str) -> ForceField:
        """Every style of ``category`` in ``style``'s form family converted
        to ``style``, exactly, through the canonical parameters.

        Raises
        ------
        molrs.ff.ir.NoFormError
            ``style`` has no form codec.
        molrs.ff.ir.OutOfImageError
            A row is outside its image (``sin(2φ) coefficient … ≠ 0``, ``the
            constant term …``)."""
    def fit_form(
        self,
        category: str,
        style: str,
        q: Sequence[float],
        w: Sequence[float] | None = None,
        *,
        kt: float | None = None,
        offset: bool = False,
    ) -> tuple[ForceField, dict[str, Any]]:
        """Every other style of ``category`` fitted to ``style`` by least
        squares over the points ``q`` of the coordinate (``r``; ``θ``, ``φ``
        in radians), weights ``w``, Boltzmann factors at ``kt``, a free
        constant ``offset``. Returns ``(forcefield, residual)``; ``residual``
        has ``sum_sq``, ``rms``, ``max_abs`` and ``types`` (per row:
        ``style``, ``type``, ``exact``, ``sum_sq``, ``rms``, ``max_abs``,
        ``offset``). ``sum_sq`` is monotone in the metric."""
    @property
    def special_bonds(self) -> tuple[list[float], list[float]]: ...
    def set_special_bonds(self, lj: Sequence[float], coul: Sequence[float]) -> None: ...
    def materialize_one_four(self, frame: Frame) -> int:
        """Write the 1-4 pairs of ``frame`` as per-pair override cells on its
        ``pairs`` block (``epsilon``/``sigma`` from the 1-4 parameters when
        ``lj/charmm`` declares ``one_four="epsilon14"``, the scales from
        ``special_bonds``); return the number of rows filled."""
    def materialize_params(self, frame: Frame, *, prefix: str) -> dict[str, list[str]]:
        """Write the parameters this force field gives each row of ``frame``
        (relation blocks by ``type``; ``atoms`` by ``atoms.type`` under every
        atom style and every pair style's self row) as columns
        ``<prefix><parameter>``, in the force-field IR's units; a parameter a
        row's type lacks is a null cell. Return block → columns written."""
    def _ffi_forcefield_capsule(self) -> Any: ...
    def def_style(
        self,
        category: str,
        name: str,
        params: dict[str, ParamInput] | None = None,
    ) -> Style:
        """Define (or return the identical) ``category`` style ``name``: an
        ``AtomStyle`` … ``CmapStyle`` for the seven categories, a
        ``RelationStyle`` for any other the force-field IR registry declares
        or this force field already holds."""
    @property
    def styles(self) -> list[Style]: ...
    def get_style(self, category: str, name: str) -> Style | None: ...
    def get_styles(self, category: str | type[Style]) -> list[Style]: ...
    def get_types(self, category: str | type[ForceFieldType]) -> list[ForceFieldType]: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...

class Style:
    """Handle of one style of a :class:`ForceField`; equal handles name the
    same category and style of one force field."""

    @property
    def name(self) -> str: ...
    @property
    def category(self) -> str: ...
    def get_types(self, type_cls: type[ForceFieldType] | None = None) -> list[ForceFieldType]: ...
    def get_type_by_name(self, name: str) -> ForceFieldType | None: ...
    @property
    def params(self) -> dict[str, ParamValue]: ...
    def __getitem__(self, key: str) -> ParamValue | None: ...
    def __setitem__(self, key: str, value: float | str) -> None: ...

class AtomStyle(Style):
    def def_type(self, name: str, **params: ParamInput) -> AtomType: ...

class BondStyle(Style):
    def def_type(
        self, name: str, itom: AtomType, jtom: AtomType, **params: ParamInput
    ) -> BondType: ...

class AngleStyle(Style):
    def def_type(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        **params: ParamInput,
    ) -> AngleType: ...

class DihedralStyle(Style):
    def def_type(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        **params: ParamInput,
    ) -> DihedralType: ...

class ImproperStyle(Style):
    def def_type(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        **params: ParamInput,
    ) -> ImproperType: ...

class PairStyle(Style):
    def def_type(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType | None = None,
        **params: ParamInput,
    ) -> PairType: ...

class CmapStyle(Style):
    def def_type(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        mtom: AtomType,
        **params: ParamInput,
    ) -> CmapType: ...

class RelationStyle(Style):
    """The style of a category beyond the seven (a registered custom
    category, ``drude``, ``constraint``, ``virtual_site``, or one read from a
    record that nothing declares)."""

    @property
    def arity(self) -> int: ...
    def def_type(
        self, name: str, *endpoints: AtomType, **params: ParamInput
    ) -> RelationType:
        """Define the type ``name`` on exactly ``arity`` endpoints, in
        order; another count raises ``ValueError``."""

class ForceFieldType:
    """Handle of one type of a :class:`ForceField`; equal handles name the
    same category, style and type of one force field."""

    @property
    def name(self) -> str: ...
    @property
    def category(self) -> str: ...
    @property
    def params(self) -> dict[str, ParamValue]: ...
    def __getitem__(self, key: str) -> ParamValue | None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
    def __setitem__(self, key: str, value: ParamInput) -> None: ...
    def keys(self) -> list[str]: ...
    def items(self) -> list[tuple[str, ParamValue]]: ...
    @property
    def endpoints(self) -> tuple[AtomType, ...]: ...

class AtomType(ForceFieldType): ...

class BondType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...

class AngleType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...
    @property
    def ktom(self) -> AtomType: ...

class DihedralType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...
    @property
    def ktom(self) -> AtomType: ...
    @property
    def ltom(self) -> AtomType: ...

class ImproperType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...
    @property
    def ktom(self) -> AtomType: ...
    @property
    def ltom(self) -> AtomType: ...

class PairType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...

class CmapType(ForceFieldType):
    @property
    def itom(self) -> AtomType: ...
    @property
    def jtom(self) -> AtomType: ...
    @property
    def ktom(self) -> AtomType: ...
    @property
    def ltom(self) -> AtomType: ...
    @property
    def mtom(self) -> AtomType: ...

class RelationType(ForceFieldType):
    """A type of a category beyond the seven; ``endpoints`` holds as many
    atom types as the category's arity."""

class LammpsLogHeader:
    """Header lines that precede the first run of a LAMMPS log."""

    @property
    def lines(self) -> list[str]: ...
    def raw_text(self) -> str: ...

class LammpsMemoryUsage:
    """The ``Per MPI rank memory allocation`` line of a run."""

    @property
    def minimum(self) -> float: ...
    @property
    def average(self) -> float: ...
    @property
    def maximum(self) -> float: ...
    @property
    def units(self) -> str: ...
    @property
    def raw_line(self) -> str: ...

class LammpsThermo:
    """One run's thermo table; ``thermo["Step"]`` is a float64 column."""

    @property
    def columns(self) -> list[str]: ...
    @property
    def rows(self) -> ArrayF: ...
    @property
    def n_rows(self) -> int: ...
    @property
    def raw_lines(self) -> list[str]: ...
    def __getitem__(self, name: str) -> ArrayF: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...

class LammpsLoopTime:
    """The ``Loop time of ...`` line of a run."""

    @property
    def seconds(self) -> float: ...
    @property
    def procs(self) -> int: ...
    @property
    def steps(self) -> int | None: ...
    @property
    def atoms(self) -> int | None: ...
    @property
    def raw_line(self) -> str: ...

class LammpsPerformance:
    """The ``Performance: ...`` line of a run."""

    @property
    def ns_per_day(self) -> float: ...
    @property
    def hours_per_ns(self) -> float: ...
    @property
    def timesteps_per_second(self) -> float: ...
    @property
    def atom_steps_per_second(self) -> float | None: ...
    @property
    def atom_steps_units(self) -> str | None: ...
    @property
    def raw_line(self) -> str: ...

class LammpsCpuUse:
    """The ``% CPU use with N MPI tasks x M OpenMP threads`` line of a run."""

    @property
    def percent(self) -> float: ...
    @property
    def mpi_tasks(self) -> int: ...
    @property
    def omp_threads(self) -> int | None: ...
    @property
    def raw_line(self) -> str: ...

class LammpsTimingRow:
    """One row of an MPI-task or thread timing breakdown."""

    @property
    def section(self) -> str: ...
    @property
    def min_time(self) -> float: ...
    @property
    def avg_time(self) -> float: ...
    @property
    def max_time(self) -> float: ...
    @property
    def percent_varavg(self) -> float: ...
    @property
    def percent_total(self) -> float: ...
    @property
    def raw_line(self) -> str: ...

class LammpsTimingBreakdown:
    """A timing breakdown table (MPI task or thread timing)."""

    @property
    def title(self) -> str: ...
    @property
    def rows(self) -> list[LammpsTimingRow]: ...
    @property
    def raw_lines(self) -> list[str]: ...
    def __len__(self) -> int: ...

class LammpsLoadBalance:
    """One ``Nlocal`` / ``Nghost`` / ``Neighs`` load-balance block."""

    @property
    def name(self) -> str: ...
    @property
    def average(self) -> float: ...
    @property
    def maximum(self) -> float: ...
    @property
    def minimum(self) -> float: ...
    @property
    def histogram(self) -> list[int]: ...
    @property
    def raw_lines(self) -> list[str]: ...

class LammpsNeighborStatistics:
    """The neighbor-list statistics block of a run."""

    @property
    def total_neighbors(self) -> int | None: ...
    @property
    def ave_neighs_per_atom(self) -> float | None: ...
    @property
    def ave_special_neighs_per_atom(self) -> float | None: ...
    @property
    def neighbor_list_builds(self) -> int | None: ...
    @property
    def dangerous_builds(self) -> int | None: ...
    @property
    def raw_lines(self) -> list[str]: ...

class LammpsWarning:
    """A ``WARNING:`` line, with where it appeared."""

    @property
    def message(self) -> str: ...
    @property
    def raw_line(self) -> str: ...
    @property
    def line_number(self) -> int | None: ...
    @property
    def run_index(self) -> int | None: ...

class LammpsRun:
    """One ``run`` command: setup lines, thermo table, timing and warnings."""

    @property
    def index(self) -> int: ...
    @property
    def setup_log(self) -> list[str]: ...
    @property
    def memory(self) -> LammpsMemoryUsage | None: ...
    @property
    def thermo(self) -> LammpsThermo | None: ...
    @property
    def loop_time(self) -> LammpsLoopTime | None: ...
    @property
    def performance(self) -> LammpsPerformance | None: ...
    @property
    def cpu_use(self) -> LammpsCpuUse | None: ...
    @property
    def mpi_task_timing(self) -> LammpsTimingBreakdown | None: ...
    @property
    def thread_timing(self) -> LammpsTimingBreakdown | None: ...
    @property
    def load_balance(self) -> list[LammpsLoadBalance]: ...
    @property
    def neighbor_statistics(self) -> LammpsNeighborStatistics | None: ...
    @property
    def warnings(self) -> list[LammpsWarning]: ...
    @property
    def unparsed_log(self) -> list[str]: ...
    @property
    def raw_text(self) -> str: ...

class LammpsLog:
    """A parsed LAMMPS log: header, one ``LammpsRun`` per ``run``, warnings."""

    @property
    def path(self) -> str: ...
    @property
    def version(self) -> str | None: ...
    @property
    def header(self) -> LammpsLogHeader: ...
    @property
    def runs(self) -> list[LammpsRun]: ...
    @property
    def total_wall_time(self) -> str | None: ...
    @property
    def warnings(self) -> list[LammpsWarning]: ...
    @property
    def raw_text(self) -> str: ...
    @property
    def style(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...

class OptimizationReport:
    """Outcome of a minimization (geometry optimization)."""

    @property
    def converged(self) -> bool: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def final_energy(self) -> float: ...
    @property
    def final_fmax(self) -> float: ...
    @property
    def final_grad_rms(self) -> float: ...

class WeightedTerms:
    """Kernels for a neighbour-driven evaluation, each with its special-bonds weights.


    Opaque: hand it to an integrator. Taking it apart would mean

    re-deciding which member is which and how its close neighbours are

    scaled -- the two things ``PotentialCompiler.compile_typed`` decides once.

    """

    def __len__(self) -> int: ...

class Potentials:
    """Composite of the one ``Potential`` concept — itself a potential.

    ``Potentials()`` is empty; ``push`` **moves** members in (an ``PairLjCut``,
    another ``Potentials`` such as one ``ExplicitTerms.compile`` built, or an object with
    ``calc_energy_forces``). The engine is unit-agnostic: nothing scales the
    energy or forces implicitly.
    """

    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def push(self, potential: PairLjCut | Potentials | Any) -> None: ...
    def calc_energy_forces(self, arg: Frame | ArrayF) -> tuple[float, ArrayF]: ...
    def calc_energy(self, arg: Frame | ArrayF) -> float: ...
    def calc_forces(self, arg: Frame | ArrayF) -> ArrayF: ...

class PotentialCompiler:
    """Compiles a ``ForceField`` into evaluable kernels.

    Owns a copy of the force field taken at construction; later edits to that
    ``ForceField`` do not reach it. ``compile(frame)`` binds a typed frame now;
    ``defer()`` returns ``Potentials`` that bind the frame they are evaluated
    on; ``compile_typed(frame)`` builds the kernels of a neighbour-driven (MD)
    evaluation. Both doors price a pair only inside its style's ``cutoff``
    (``r < cutoff``, as LAMMPS; ∞ when the style states none).
    """

    def __init__(self, forcefield: ForceField) -> None: ...
    def compile(self, frame: Frame) -> Potentials: ...
    def defer(self) -> Potentials: ...
    def compile_typed(self, frame: Frame) -> WeightedTerms: ...

class Lbfgs:
    """L-BFGS geometry optimizer over a force-field Potential.

    Knobs live on ``__init__`` (defaults: Rust ``LbfgsSettings::DEFAULT``).
    Primary call is ``minimize(frame)``; array ranks dispatch single / batch
    coordinate paths.
    """

    def __init__(
        self,
        potentials: Potentials,
        *,
        fmax: float = 0.05,
        max_steps: int = 500,
        max_step: float = 0.2,
        memory: int = 8,
    ) -> None: ...
    @overload
    def minimize(self, frame: Frame) -> tuple[Frame, OptimizationReport]: ...
    # (N, 3) or (3N,) -> single structure; (B, N, 3) -> homogeneous batch.
    @overload
    def minimize(self, coords: ArrayF) -> tuple[ArrayF, OptimizationReport]: ...
    @overload
    def minimize(self, coords: ArrayF) -> tuple[ArrayF, list[OptimizationReport]]: ...

#: A param value of a type annotation or style: numbers to the numeric side,
#: strings to the string side.
type AssignmentParamValue = float | int | str

#: What a ``TypeAssignment`` writes under one key of one graph element: a scalar is
#: stamped and defines nothing; ``(style, name, endpoints, params)`` stamps
#: ``name`` and every param and defines the type ``name`` on ``endpoints``
#: (atom-type names; empty for an atom type) under the style.
type Annotation = (
    str | bool | int | float | tuple[str, str, Sequence[str], dict[str, AssignmentParamValue]]
)

class TypeAssignment:
    """What a typifier's ``assign`` assigns to one graph.

    ``nodes`` is positional against ``graph.atoms``; ``links`` maps a relation
    kind to rows positional against that kind's own rows, so an improper never
    shifts a dihedral position. A key is a relation class (``Bond``, ``Angle``,
    ``Dihedral``, ``Improper``, ``Port``; rows as
    ``graph.links.exact_bucket(cls)``) or a kind name (``"bonds"``, or a
    custom ``"urey_bradleys"`` from ``graph.register_kind``; rows as
    ``graph.relation_ids(kind)``). A type annotation under a kind defines a
    type of the category whose block the kind is (``urey_bradleys`` →
    ``urey_bradley``). Any other key raises ``TypeError``, a kind named twice
    ``ValueError``. ``styles`` are ``(category, style, params)`` to declare,
    in order; ``pairs`` are ``(style, name, endpoints, params)`` pair rows."""

    def __init__(
        self,
        nodes: Sequence[_AbcMapping[str, Annotation]],
        links: _AbcMapping[type | str, Sequence[_AbcMapping[str, Annotation]]]
        | None = None,
        *,
        styles: Sequence[tuple[str, str, dict[str, AssignmentParamValue]]] = (),
        pairs: Sequence[tuple[str, str, Sequence[str], dict[str, AssignmentParamValue]]] = (),
    ) -> None: ...

class Typifier[TGraph: MolGraph]:
    """The base of every graph typifier: one ``assign`` hook plus the output
    force field its typing accumulates.

    A subclass implements ``assign`` (and optionally ``source_forcefield``) and nothing
    else; defining ``typify`` on a subclass raises ``TypeError`` at class
    creation. The native classes extend this base and only construct; they
    are subclassable, but a subclass of one that defines ``assign`` or
    ``source_forcefield`` raises ``TypeError`` (they run in Rust)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def assign(self, graph: _TGraph) -> TypeAssignment:
        """Type ``graph`` and return what it assigns.

        ``assign`` may write intermediate results (generated topology, perceived
        bond types) onto the graph it is given; ``typify`` always gives it a
        private copy. The base raises ``NotImplementedError``; a native class
        runs its Rust matcher."""
    @final
    def typify(self, mol: _TGraph) -> _TGraph:
        """Do not override; the only writer of ``forcefield()``.

        Copies ``mol``, calls ``assign`` on the copy, and writes the match onto
        the copy and the output. Returns the typed copy; ``mol`` is untouched.
        ``mol`` must be an ``Atomistic`` (anything else raises ``TypeError``).
        Raises ``NotImplementedError`` without an ``assign`` and ``ValueError``
        when the match does not fit the graph or contradicts the output (which
        is then unchanged)."""
    def forcefield(self) -> ForceField:
        """The accumulated output — exactly the definitions ``typify`` has
        assigned — returned as a copy.

        Edits to the copy do not reach the typifier; ``typify`` is the only
        writer. Before the first ``typify`` it is the seeded empty output."""
    def source_forcefield(self) -> ForceField:
        """The force field this typifier matches against, returned as a copy.

        The output starts as its empty likeness (name, declared units and
        special_bonds). A Python subclass that does not override it raises
        ``NotImplementedError``, and its output starts as an empty force field
        named after the class."""

class Mmff94Typifier(Typifier[Atomistic]):
    """MMFF94 (Halgren 1996) atom types, charges and bonded parameters.

    The variant is the class, never a flag. See ``Mmff94sTypifier`` for the
    "static" parameter set.

    ``typify`` labels the graph; ``PotentialCompiler(forcefield()).compile(frame)``
    compiles it. There is no one-step ``build`` — MMFF walks the same route as every other
    force field."""

    def __init__(self) -> None: ...

class Mmff94sTypifier(Typifier[Atomistic]):
    """MMFF94s (Halgren 1999) — the "static" set, for energy minimization.

    Differs from ``Mmff94Typifier`` only on delocalised trivalent nitrogen (MMFF
    numeric types 10 ``NC=O`` / 40 ``NC=C``): 11 out-of-plane rows and 42 torsion
    rows are re-parameterised so the nitrogen minimizes planar. ``typify`` bakes
    ``koop = +0.015`` (type 10) / ``+0.030`` (type 40) md*A*rad^-2 on those
    centres, where the improper kernel reads it as
    ``E_oop = 0.5 * 143.9325 * koop * chi**2`` (chi in radians). Every atom type
    and every bond / angle / stretch-bend / vdW / charge parameter is shared with
    MMFF94."""

    def __init__(self) -> None: ...

class OplsAaTypifier(Typifier[Atomistic]):
    def __init__(self, source: Any = None, *, strict: bool = True) -> None: ...

type AtdParameterSet = Literal["bcc", "abcg2", "gas", "gaff", "gaff2", "amber", "sybyl"]
type AtdBondOrders = Literal["perceive", "input"]

class AtdTypifier(Typifier[Atomistic]):
    """antechamber atom types — one rule engine over seven ``ATOMTYPE_*.DEF`` tables.

    ``parameter_set`` is the antechamber ``-at`` flag and is **required**: seven
    tables exist, they disagree, and there is no default. An atom no rule matches
    comes back labelled ``"DU"`` — the table's own catch-all row, not a fallback.

    ``bond_orders`` says which bond orders the types follow: ``"perceive"`` (the
    default) judges them from the connectivity alone, as antechamber's
    ``bondtype -j full`` does, ignoring the molecule's own; ``"input"`` keeps
    the molecule's orders."""

    def __init__(
        self,
        *,
        parameter_set: AtdParameterSet,
        bond_orders: AtdBondOrders = "perceive",
    ) -> None: ...
    @property
    def parameter_set(self) -> AtdParameterSet: ...
    @property
    def bond_orders(self) -> AtdBondOrders: ...

type GaffParameterSet = Literal["gaff", "gaff2"]

class GaffTypifier(Typifier[Atomistic]):
    """GAFF / GAFF2 bonded terms and parameters for a molecule whose atoms
    already carry GAFF types (``AtdTypifier`` with the same ``parameter_set``).

    Angles and dihedrals are regenerated from the bond graph, impropers built
    as tleap builds them (wherever tleap finds a row, or ``parmchk2`` an
    estimate at a centre ``PARMCHK.DAT`` flags as planar; tleap's atom order,
    centre third); each term is an exact ``gaff.dat`` / ``gaff2.dat`` row, a wildcard row, or a
    ``parmchk2``-style estimate whose type carries ``estimated``,
    ``estimate_penalty``, ``estimate_method`` and ``estimate_analog``.
    ``parameter_set`` is required; there is no default.

    Raises
    ------
    ValueError
        An unknown ``parameter_set``; from :meth:`typify`, an untyped atom, a
        type the table does not declare, or terms nothing covers (all listed).
    """

    def __init__(self, *, parameter_set: GaffParameterSet) -> None: ...
    @property
    def parameter_set(self) -> GaffParameterSet: ...

class ElementTypifier(Typifier[Atomistic]):
    """``molrs.ff.typifier.ElementTypifier`` — ``type`` labels from element
    symbols alone, with no force field.

    Atoms get ``type = element`` (e.g. ``"C"``); bonds, angles and dihedrals
    get their endpoint elements joined with ``-`` in the byte-wise smaller
    orientation (bond O–H is ``"H-O"``). :meth:`forcefield` stays empty.
    Exported from :mod:`molrs.ff.typifier` only.

    Raises
    ------
    ValueError
        From :meth:`typify`, when an atom has no string ``element`` or the
        molecule has impropers.
    """

    def __init__(self) -> None: ...

# ---------------------------------------------------------------------------
# Charge models — one calling convention
#
# Each model below takes a molecule (and optionally QM base charges) and
# returns one float64 charge per atom: `needs_equivalencing()` then
# `assign(mol, qm=None)`. That is a shared shape, not a shared base — the
# native classes inherit from nothing.
# ---------------------------------------------------------------------------

type BccParameterSet = Literal["bcc", "abcg2"]

class BccModel:
    """AM1-BCC / ABCG2 bond-charge corrections.

    ``assign`` is the whole model: it averages the raw QM charges over the
    topological-equivalence classes (antechamber's ``-eq 1``) and then corrects
    them. ``correct`` is the correction stage alone, for base charges that are
    already equivalenced. The increments are pairwise antisymmetric, so the total
    charge is conserved to machine precision; nothing is renormalized."""

    def __init__(self, *, parameter_set: BccParameterSet) -> None: ...
    @property
    def parameter_set(self) -> BccParameterSet: ...
    def needs_equivalencing(self) -> bool: ...
    def assign(self, mol: Atomistic, qm: ArrayF | None = None) -> ArrayF: ...
    def correct(self, mol: Atomistic, am1: ArrayF) -> ArrayF: ...

class MullikenModel:
    """The pass-through: the QM charges it was handed, bit for bit."""

    def __init__(self) -> None: ...
    def needs_equivalencing(self) -> bool: ...
    def assign(self, mol: Atomistic, qm: ArrayF | None = None) -> ArrayF: ...

class GasteigerModel:
    """Gasteiger / PEOE charges (``antechamber -c gas``) — no QM input needed.

    ``qm`` is accepted and ignored, so that a caller holding an unknown model can
    call every model the same way."""

    def __init__(self) -> None: ...
    def needs_equivalencing(self) -> bool: ...
    def assign(self, mol: Atomistic, qm: ArrayF | None = None) -> ArrayF: ...

def read_xml_forcefield(path: PathInput) -> ForceField:
    """Read a force-field XML file (one element per style, in its declared units)."""
def write_xml_forcefield(path: PathInput, forcefield: ForceField) -> None:
    """Write a force field as force-field XML — the inverse of
    :func:`read_xml_forcefield`."""
def read_openmm_forcefield(path: PathInput) -> ForceField:
    """Read an OpenMM force-field XML file into the force-field IR."""
def read_lammps_forcefield(path: PathInput) -> ForceField: ...
def write_top_system(
    path: PathInput, forcefield: ForceField, frame: Frame, *, precision: int = 6
) -> None:
    """Write ``forcefield`` and the typed ``frame`` as one GROMACS topology
    (directives, one ``[ moleculetype ]`` per molecule, ``[ system ]``,
    ``[ molecules ]``): the inverse of :func:`read_top_system`. Raises
    ``ValueError`` for what GROMACS cannot express."""

def clpol_polarizability() -> dict[str, dict[str, float]]:
    """CL&Pol Drude parameters per atom type (``m_D``, ``q_D_sign``, ``k_D``,
    ``alpha``, ``a_thole``): the shipped ``alpha.ff`` table."""

def read_lammps_data_coeffs(path: PathInput, *, units: str | None = None) -> ForceField:
    """The force field a LAMMPS data file's ``* Coeffs`` sections define,
    rows named by the file's ``* Type Labels`` (ids as written). ``units``
    defaults to the file's stated units, else ``"real"``. Raises
    ``ValueError`` for an unreadable file, one with no ``* Coeffs``
    sections, or a ``units`` that disagrees with the file's."""

def read_lammps_data_coeffs_str(text: str, *, units: str | None = None) -> ForceField:
    """:func:`read_lammps_data_coeffs` on the text of a whole data file."""

def write_lammps_forcefield(
    path: PathInput,
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    skip_special_bonds: bool = False,
    skip_units: bool = False,
    units: str = "real",
    cmap_file: str | None = None,
) -> None: ...
def write_lammps_forcefield_str(
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    skip_special_bonds: bool = False,
    skip_units: bool = False,
    units: str = "real",
    cmap_file: str | None = None,
) -> str: ...
def write_lammps_data_coeffs_str(
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    units: str = "real",
) -> str: ...
def assign_cmaps(frame: Frame, forcefield: ForceField) -> int: ...
def read_lammps_cmap_forcefield(path: PathInput) -> ForceField: ...
def write_lammps_cmap_forcefield(
    path: PathInput,
    forcefield: ForceField,
    frame: Frame,
    *,
    precision: int = 6,
    units: str = "real",
) -> None: ...
def intramolecular_pairs(
    frame: Frame, forcefield: ForceField | None = None
) -> Block: ...

# ---------------------------------------------------------------------------
# Record / Trajectory / Observables
# ---------------------------------------------------------------------------

class Trajectory:
    """An in-memory frame sequence with optional per-frame ``step`` / ``time``
    labels; a slice is the sub-trajectory with its labels."""

    def __init__(
        self,
        frames: Sequence[Frame],
        step: ArrayI64 | None = None,
        time: ArrayF | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Frame]: ...
    @overload
    def __getitem__(self, key: int) -> Frame: ...
    @overload
    def __getitem__(self, key: slice) -> Trajectory: ...
    def map(self, func: Callable[[Frame], Frame]) -> Trajectory: ...
    @property
    def frames(self) -> list[Frame]: ...
    @property
    def step(self) -> ArrayI64 | None: ...
    @property
    def time(self) -> ArrayF | None: ...

type _ObservableScalarValue = npt.NDArray | float | int | bool | str | list[str]

class ObservableRecord:
    def __init__(
        self,
        name: str,
        values: _ObservableScalarValue,
        kind: str = "scalar",
        description: str = "",
        unit: str | None = None,
        axes: list[str] | None = None,
        time_dependent: bool = False,
        sampling: str | None = None,
        domain: str | None = None,
        target: str | None = None,
    ) -> None: ...
    @staticmethod
    def scalar(
        name: str,
        values: _ObservableScalarValue,
        description: str = "",
        unit: str | None = None,
        axes: list[str] | None = None,
        time_dependent: bool = False,
        sampling: str | None = None,
        domain: str | None = None,
        target: str | None = None,
    ) -> ObservableRecord: ...
    @staticmethod
    def vector(
        name: str,
        values: _ObservableScalarValue,
        description: str = "",
        unit: str | None = None,
        axes: list[str] | None = None,
        time_dependent: bool = False,
        sampling: str | None = None,
        domain: str | None = None,
        target: str | None = None,
    ) -> ObservableRecord: ...
    @property
    def name(self) -> str: ...
    @property
    def values(self) -> npt.NDArray | list[str] | str: ...
    @property
    def kind(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def unit(self) -> str | None: ...
    @property
    def axes(self) -> list[str]: ...
    @property
    def time_dependent(self) -> bool: ...
    @property
    def sampling(self) -> str | None: ...
    @property
    def domain(self) -> str | None: ...
    @property
    def target(self) -> str | None: ...

class mrec:
    """The ``_native.mrec`` submodule, surfaced as :mod:`molrs.io.mrec`: the
    record store's reader, writer, schema and force-field section. The
    per-section doors are flat (``read_mrec_frame`` / ``write_mrec_frame`` and
    partners), as :mod:`molrs.io`'s."""

    class ForceFieldSection:
        """The ``forcefield`` section of a ``*.mrec`` record: the document and one
        :class:`Block` per style table, kept whole (molrec ``forcefield.md``)."""

        def __init__(
            self,
            document: _AbcMapping[str, Any],
            tables: _AbcMapping[str, Block] | None = None,
        ) -> None: ...
        @property
        def document(self) -> dict[str, Any]: ...
        @property
        def tables(self) -> dict[str, Block]: ...
        @property
        def name(self) -> str | None: ...
        def table(self, category: str, style: str) -> Block | None: ...
        @staticmethod
        def block_name(category: str, style: str) -> str: ...
        def validate(self) -> None: ...
        @staticmethod
        def from_forcefield(forcefield: ForceField) -> ForceFieldSection: ...
        def to_forcefield(self) -> ForceField: ...

    @staticmethod
    def section_names(path: PathInput) -> frozenset[str]: ...

    class MrecReader:
        """Lazy one-frame cursor over a ``*.mrec`` trajectory (directory or zip).

        ``molrs.io.mrec.MrecReader``; iterating it walks every frame.
        """

        def __init__(self, path: PathInput) -> None: ...
        def read_frame(self, index: int) -> Frame: ...
        def read_columns(self, index: int, columns: list[tuple[str, str]]) -> Frame: ...
        def block_update_at(self, name: str, index: int) -> int | None: ...
        def box_at(self, index: int) -> Box | None: ...
        def __len__(self) -> int: ...
        def __getitem__(self, index: int) -> Frame: ...
        def __iter__(self) -> Iterator[Frame]: ...
        @property
        def step(self) -> list[int]: ...
        @property
        def time(self) -> list[float] | None: ...
        def has_block(self, name: str) -> bool: ...
        def block_names(self) -> list[str]: ...
        def __enter__(self) -> Self: ...
        def __exit__(self, *exc: object) -> bool: ...

    class SequenceSchema:
        """A frame-sequence schema pinned before a run's frames are written.

        ``molrs.io.mrec.SequenceSchema``; every ``declare_*`` returns the schema.
        """

        def __init__(self) -> None: ...
        @staticmethod
        def from_frame(frame: Frame) -> SequenceSchema: ...
        @staticmethod
        def from_frames(frames: Sequence[Frame]) -> SequenceSchema: ...
        def declare_block(self, name: str, rows: int | None = ...) -> Self: ...
        def declare_column(
            self, block: str, column: str, dtype: str, trailing: list[int] | None = ...
        ) -> Self: ...
        def declare_structural_shape(self, block: str, shape: list[int]) -> Self: ...
        def declare_precision(self, block: str, column: str, precision: float) -> Self:
            """Pin the precision of an ``f64`` column: every frame's values are
            rounded to its binary grid before the change check and the landing."""
        def precision(self, block: str, column: str) -> float | None: ...
        def declare_target(self, block: str, column: str, target: str) -> Self:
            """Pin a ``u64`` column as a row reference into *target*; the writer
            refuses a frame whose resolved blocks break it."""
        def target(self, block: str, column: str) -> str | None: ...
        def declare_aligned(self, block: str, target: str) -> Self:
            """Pin *block*'s rows to *target*'s rows at every resolved frame; the
            writer refuses a frame that breaks it (restate *block* when *target*
            changes its row count)."""
        def aligned_with(self, block: str) -> str | None: ...
        def declare_meta(self, key: str, dtype: str) -> Self: ...
        def declare_meta_with_fill(
            self, key: str, fill: Any, dtype: str | None = ...
        ) -> Self: ...
        def block_names(self) -> list[str]: ...
        def column_names(self, block: str) -> list[str] | None: ...
        def meta_keys(self) -> list[tuple[str, str]]: ...

    class MrecWriter:
        """Append-first writer for a ``*.mrec`` trajectory store.

        ``molrs.io.mrec.MrecWriter``.
        """

        def __init__(
            self,
            path: PathInput,
            schema: SequenceSchema,
            *,
            flush_every: int | None = ...,
            compression: str | None = ...,
            durable: bool = ...,
            meta: _AbcMapping[str, Any] | None = ...,
        ) -> None: ...
        @staticmethod
        def open(
            path: PathInput, *, flush_every: int | None = ..., durable: bool = ...
        ) -> MrecWriter: ...
        def append(
            self, frame: Frame, step: int | None = ..., time: float | None = ...
        ) -> None: ...
        def flush(self) -> None: ...
        def close(self) -> None: ...
        @property
        def flush_every(self) -> int: ...
        @property
        def committed(self) -> int: ...
        def __enter__(self) -> Self: ...
        def __exit__(self, *exc: object) -> bool: ...

    @staticmethod
    def pack_mrec_zip(path: PathInput) -> str: ...

    class validation:
        """``molrs.io.mrec.validation``: the record contract's runtime checks."""

        @staticmethod
        def validate_frame(frame: Frame) -> None: ...


# --- *.mrec whole-record doors (molrs.io) ---------------------------------

def write_mrec_frame(
    path: PathInput,
    frame: Frame,
    system: Frame | None = None,
    meta: _AbcMapping[str, Any] | None = None,
    forcefield: ForceField | mrec.ForceFieldSection | None = None,
) -> None: ...
def write_mrec_system(
    path: PathInput,
    system: Frame,
    meta: _AbcMapping[str, Any] | None = None,
    forcefield: ForceField | mrec.ForceFieldSection | None = None,
) -> None: ...
def write_mrec_forcefield(
    path: PathInput,
    forcefield: ForceField | mrec.ForceFieldSection,
    meta: _AbcMapping[str, Any] | None = None,
) -> None: ...
def write_mrec_trajectory(
    path: PathInput, traj: Trajectory, meta: _AbcMapping[str, Any] | None = None
) -> None: ...
def read_mrec_frame(path: PathInput) -> Frame: ...
def read_mrec_system(path: PathInput) -> Frame: ...
def read_mrec_trajectory(path: PathInput) -> Trajectory: ...
def read_mrec_forcefield(path: PathInput) -> mrec.ForceFieldSection | None: ...
def read_mrec_meta(path: PathInput) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Analysis (molrs.compute, mirrors molrs::compute)
#
# One class per `ComputeSpec.all()` entry, under the core name. `__init__` takes
# configuration only (every optional setting `None` = the core default);
# `compute(...)` takes the data as keyword arguments named after the core
# input fields. A frames compute takes one sequence per field, an entry per
# frame. Results are read-only and are passed on to the next compute as is.
# The structural `Compute` protocol is pure Python (`molrs/compute/_protocol.py`).
# ---------------------------------------------------------------------------

# A partition: a 1-D unsigned-int label array or a partition result.
type Partition = (
    npt.NDArray[np.unsignedinteger[Any]]
    | ClusterResult
    | KmeansResult
    | EnvironmentMatchResult
    | VoronoiDomainResult
)
# A feature matrix: a 2-D float64 array or a descriptor result.
type FeatureMatrix = ArrayF | SteinhardtResult | LocalDescriptorsResult | PcaResult

class AngularSeparationGlobal:
    """Angles between ``(N, 4)`` query quaternions and ``(M, 4)`` reference
    ones (``molrs::compute::AngularSeparationGlobal``)."""
    def __init__(self, *, equivalent_orientations: bool | None = None) -> None: ...
    def compute(self, *, query: ArrayF, reference: ArrayF) -> AngularSeparationGlobalResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class AngularSeparationGlobalResult:
    """Angles between query and reference orientations
    (``molrs::compute::AngularSeparationGlobalResult``)."""
    @property
    def angles(self) -> ArrayF: ...

class AngularSeparationNeighbor:
    """Angles between the orientations of neighbouring particles
    (``molrs::compute::AngularSeparationNeighbor``)."""
    def __init__(self, *, equivalent_orientations: bool | None = None) -> None: ...
    def compute(self, *, neighbors: Neighbors, query_orientations: ArrayF, point_orientations: ArrayF) -> AngularSeparationNeighborResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class AngularSeparationNeighborResult:
    """Angles between neighbouring orientations
    (``molrs::compute::AngularSeparationNeighborResult``)."""
    @property
    def angles(self) -> ArrayF: ...

class Axis:
    """A uniform binning axis: ``n_bins`` half-open bins over ``[min, max)``
    (``molrs::compute::Axis``)."""
    def __init__(self, min: float, max: float, n_bins: int) -> None: ...
    @property
    def centers(self) -> ArrayF: ...
    @property
    def edges(self) -> ArrayF: ...
    @property
    def max(self) -> float: ...
    @property
    def min(self) -> float: ...
    @property
    def n_bins(self) -> int: ...
    @property
    def width(self) -> float: ...

class BondOrientationalOrder:
    """Bond-orientational order diagram on a ``(theta, phi)`` grid
    (``molrs::compute::BondOrientationalOrder``)."""
    def __init__(self, n_theta: int, n_phi: int) -> None: ...
    def compute(self, *, neighbors: Neighbors) -> BondOrientationalOrderResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class BondOrientationalOrderResult:
    """The bond-orientation histogram on the sphere
    (``molrs::compute::BondOrientationalOrderResult``)."""
    @property
    def bond_order(self) -> ArrayF: ...
    @property
    def phi_edges(self) -> ArrayF: ...
    @property
    def raw_counts(self) -> npt.NDArray[Any]: ...
    @property
    def theta_edges(self) -> ArrayF: ...

class CenterOfMass:
    """Mass-weighted centre of each group (``molrs::compute::CenterOfMass``)."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box | None = None, partition: Partition, masses: ArrayF) -> CenterOfMassResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class CenterOfMassResult:
    """Centres of mass (``molrs::compute::CenterOfMassResult``): one row of
    ``centers`` and one ``total_masses`` entry per group key in ``groups``."""
    @property
    def centers(self) -> ArrayF: ...
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def total_masses(self) -> ArrayF: ...

class Cluster:
    """Connected components of a neighbour table (``molrs::compute::Cluster``)."""
    def __init__(self, *, min_cluster_size: int | None = None) -> None: ...
    def compute(self, *, pairs: Neighbors, merge: Partition | None = None) -> ClusterResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class ClusterResult:
    """Connected clusters (``molrs::compute::ClusterResult``); a partition."""
    @property
    def cluster_idx(self) -> npt.NDArray[Any]: ...
    @property
    def cluster_keys(self) -> list[npt.NDArray[Any]]: ...
    @property
    def cluster_sizes(self) -> npt.NDArray[Any]: ...
    @property
    def n_clusters(self) -> int: ...
    def sizes(self) -> list[tuple[int, int]]: ...

class CombinedDistribution:
    """Joint distribution of 2 or 3 internal coordinates over frames
    (``molrs::compute::CombinedDistribution``)."""
    def __init__(self, observables: Sequence[str], axes: Sequence[Axis], *, sin_weight: Sequence[int] | None = None) -> None: ...
    def compute(self, *, positions: Sequence[ArrayF], simbox: Sequence[Box | None] | None = None, tuples: Sequence[Sequence[Sequence[Sequence[int]]]]) -> CombinedDistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class CombinedDistributionResult:
    """A joint distribution (``molrs::compute::CombinedDistributionResult``):
    the N-D ``histogram`` and normalised ``density``, with which axes are
    sin θ-corrected."""
    @property
    def density(self) -> ArrayF: ...
    def free_energy(self, temperature: float) -> ArrayF: ...
    @property
    def histogram(self) -> Histogram: ...
    def marginal(self, axis: int) -> DistributionResult: ...
    @property
    def n_frames(self) -> int: ...
    @property
    def n_raw_samples(self) -> int: ...
    @property
    def sin_corrected(self) -> ArrayBool: ...

class ComputeSpec:
    """The descriptor of one compute type (``molrs::compute::ComputeSpec``)."""
    @staticmethod
    def all() -> list[Self]: ...
    @staticmethod
    def find(name: str) -> Self | None: ...
    @property
    def form(self) -> Literal["single", "frames"]: ...
    def input(self, name: str) -> InputSpec | None: ...
    @property
    def inputs(self) -> list[InputSpec]: ...
    @property
    def name(self) -> str: ...
    def param(self, name: str) -> ParamSpec | None: ...
    @property
    def params(self) -> list[ParamSpec]: ...

class ConductivitySumRule:
    """The conductivity sum rule ``∫ σ(ω) dω`` against its static value
    (``molrs::compute::ConductivitySumRule``)."""
    def __init__(self, units: str | UnitPreset, *, tolerance: float | None = None) -> None: ...
    def compute(self, *, frequencies: ArrayF, conductivity: ArrayF, current_sq_mean: float, volume: float, temperature: float) -> SumRuleCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class ContinuousCoordination:
    """Continuous coordination number (``molrs::compute::ContinuousCoordination``)."""
    def __init__(self, power: float) -> None: ...
    def compute(self, *, qlm: QlmTable, neighbors: Neighbors) -> ContinuousCoordinationResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class ContinuousCoordinationResult:
    """Continuous coordination per particle
    (``molrs::compute::ContinuousCoordinationResult``)."""
    @property
    def coord(self) -> ArrayF: ...
    @property
    def l(self) -> int: ...

class CorrelationFunction:
    """Radial correlation of two per-point value arrays over a pair source
    (``molrs::compute::CorrelationFunction``)."""
    def __init__(self, n_bins: int, r_max: float, r_min: float) -> None: ...
    def compute(self, *, pairs: Neighbors | NeighborList, values_a: ArrayF, values_b: ArrayF) -> CorrelationFunctionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class CorrelationFunctionResult:
    """A pair correlation of per-point values on radial bins
    (``molrs::compute::CorrelationFunctionResult``)."""
    @property
    def bin_centers(self) -> ArrayF: ...
    @property
    def bin_counts(self) -> npt.NDArray[Any]: ...
    @property
    def bin_edges(self) -> ArrayF: ...
    @property
    def correlation(self) -> ArrayF: ...

class Correlator:
    """The one time-correlation estimator (``molrs::compute::Correlator``)."""
    def __init__(self, kernel: str, max_lag: int, *, l: int | None = None, origins: str | None = None, mean: str | None = None, normalization: str | None = None, reduction: str | None = None, stride: int | None = None, dimension: int | None = None) -> None: ...
    def compute(self, *, a: ArrayF, b: ArrayF | None = None, dt: float, boxes: Sequence[Box] | None = None) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class Cubatic:
    """Cubatic order of ``(N, 3)`` directors by simulated annealing
    (``molrs::compute::Cubatic``)."""
    def __init__(self, *, seed: int | None = None, initial_temp: float | None = None, cooling_rate: float | None = None, min_temp: float | None = None, n_steps: int | None = None, n_chains: int | None = None) -> None: ...
    def compute(self, *, directors: ArrayF) -> CubaticResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class CubaticResult:
    """The cubatic order and its director basis (``molrs::compute::CubaticResult``)."""
    @property
    def director_basis(self) -> ArrayF: ...
    @property
    def order(self) -> float: ...

class CumulativeTrapezoid:
    """The running trapezoid integral of one channel of a correlation
    (``molrs::compute::CumulativeTrapezoid``) — a Green–Kubo integral."""
    def __init__(self, *, n_lags: int | None = None) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, channel: int) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class CurrentAcf:
    """Current autocorrelation of a ``(T, 3)`` current series
    (``molrs::compute::CurrentAcf``)."""
    def __init__(self, max_lag: int, *, normalization: str | None = None, stride: int | None = None) -> None: ...
    def compute(self, *, current: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DebyeFit:
    """Single-exponential (Debye) relaxation fit of one channel of a correlation
    (``molrs::compute::DebyeFit``)."""
    def __init__(self) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, channel: int) -> DebyeFitResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DebyeFitResult:
    """A single-exponential fit ``C(t) = amplitude · exp(−t / tau)``
    (``molrs::compute::DebyeFitResult``) over ``n_samples`` lags."""
    @property
    def amplitude(self) -> float: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def tau(self) -> float: ...

class Derivative:
    """Finite-difference time derivative of a ``(T, C)`` series
    (``molrs::compute::Derivative``)."""
    def __init__(self, stencil: str) -> None: ...
    def compute(self, *, series: ArrayF, dt: float) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DielectricSpectrumResult:
    """A complex permittivity ε(ω) = ``eps_real`` − i ``eps_imag`` at
    ``frequencies`` (``molrs::compute::DielectricSpectrumResult``)."""
    @property
    def eps_imag(self) -> ArrayF: ...
    @property
    def eps_real(self) -> ArrayF: ...
    @property
    def epsilon_inf(self) -> float: ...
    @property
    def frequencies(self) -> ArrayF: ...

class DiffractionPattern:
    """A 2-D diffraction pattern of one snapshot projected along an axis
    (``molrs::compute::DiffractionPattern``)."""
    def __init__(self, n_grid: int, sigma: float, *, axis: int | None = None, cutoff: float | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box) -> DiffractionPatternResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DiffractionPatternResult:
    """A 2-D diffraction pattern and the projected density image it was
    transformed from (``molrs::compute::DiffractionPatternResult``)."""
    @property
    def diffraction(self) -> ArrayF: ...
    @property
    def image(self) -> ArrayF: ...

class DipoleAcf:
    """Dipole autocorrelation of a ``(T, 3)`` dipole series
    (``molrs::compute::DipoleAcf``)."""
    def __init__(self, max_lag: int, *, normalization: str | None = None, stride: int | None = None) -> None: ...
    def compute(self, *, dipoles: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DipoleAutocorrelationSpectrum:
    """ε(ω) by the dipole-autocorrelation route
    (``molrs::compute::DipoleAutocorrelationSpectrum``)."""
    def __init__(self, units: str | UnitPreset, *, window: str | None = None, pad: int | None = None, epsilon_inf: float | None = None, plateau: tuple[float, float] | None = None, plateau_value: str | None = None) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, volume: float, temperature: float, boundary: str) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DipoleMoment:
    """The dipole moment ``Σ q r`` of each group in each frame
    (``molrs::compute::DipoleMoment``)."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, charges: ArrayF, partition: Partition | None = None) -> DipoleMomentResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DipoleMomentResult:
    """Dipole series per group (``molrs::compute::DipoleMomentResult``):
    ``moments[frame, group, :]`` for the group keys ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def moments(self) -> ArrayF: ...
    def series(self, column: int) -> ArrayF: ...

class DipoleRateCrossSpectrum:
    """ε(ω) by the dipole–current cross-correlation route
    (``molrs::compute::DipoleRateCrossSpectrum``)."""
    def __init__(self, units: str | UnitPreset, *, window: str | None = None, pad: int | None = None, epsilon_inf: float | None = None) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, volume: float, temperature: float, boundary: str) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DistributionFunction:
    """Distribution of one internal coordinate over frames
    (``molrs::compute::DistributionFunction``)."""
    def __init__(self, observable: str, *, axis: Axis | None = None, n_bins: int | None = None) -> None: ...
    def compute(self, *, positions: Sequence[ArrayF], simbox: Sequence[Box | None] | None = None, tuples: Sequence[Sequence[Sequence[int]]]) -> DistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class DistributionResult:
    """A 1-D distribution (``molrs::compute::DistributionResult``): the raw
    ``histogram``, the normalised ``density`` and, for an angular
    coordinate, the sin θ-corrected density."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def density_sin_corrected(self) -> ArrayF | None: ...
    @property
    def histogram(self) -> Histogram: ...
    @property
    def n_frames(self) -> int: ...
    @property
    def n_raw_samples(self) -> int: ...

class EinsteinHelfandSpectrum:
    """ε(ω) by the Einstein–Helfand route from the fluctuation dipole ACF
    (``molrs::compute::EinsteinHelfandSpectrum``), e.g. ``DipoleAcf``'s
    curve."""
    def __init__(self, units: str | UnitPreset, *, window: str | None = None, pad: int | None = None, epsilon_inf: float | None = None) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, volume: float, temperature: float, boundary: str) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class EnvironmentMatch:
    """Cluster particles whose neighbour environments match within
    ``rmsd_threshold`` (``molrs::compute::EnvironmentMatch``)."""
    def __init__(self, rmsd_threshold: float, *, registration: bool | None = None, max_neighbors_for_registration: int | None = None) -> None: ...
    def compute(self, *, neighbors: Neighbors) -> EnvironmentMatchResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class EnvironmentMatchResult:
    """Environment clusters (``molrs::compute::EnvironmentMatchResult``); a
    partition."""
    @property
    def cluster_idx(self) -> npt.NDArray[Any]: ...
    @property
    def fingerprints(self) -> list[ArrayF]: ...
    @property
    def n_clusters(self) -> int: ...

class FiniteFieldPolarizability:
    """Polarizability by central finite differences of the group dipoles under
    ``±field`` (``molrs::compute::FiniteFieldPolarizability``)."""
    def __init__(self, field: float) -> None: ...
    def compute(self, *, plus: MolecularMoments, minus: MolecularMoments) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class GaussianDensity:
    """Gaussian-smeared density of points on a grid over the box
    (``molrs::compute::GaussianDensity``)."""
    def __init__(self, dims: Sequence[int], sigma: float, *, r_max: float | None = None, reach: str | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box) -> GaussianDensityResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class GaussianDensityResult:
    """A Gaussian-smeared density on a grid
    (``molrs::compute::GaussianDensityResult``)."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def grid(self) -> RegularGrid: ...

class GreenKuboSpectrum:
    """ε(ω) by the Green–Kubo route from the current ACF
    (``molrs::compute::GreenKuboSpectrum``), e.g. ``CurrentAcf``'s curve."""
    def __init__(self, units: str | UnitPreset, *, window: str | None = None, pad: int | None = None, epsilon_inf: float | None = None) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, volume: float, temperature: float, boundary: str) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class GyrationTensor:
    """Weighted gyration tensor of each group (``molrs::compute::GyrationTensor``)."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box | None = None, partition: Partition, weights: ArrayF) -> GyrationTensorResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class GyrationTensorResult:
    """Gyration tensors (``molrs::compute::GyrationTensorResult``): a
    ``(3, 3)`` tensor per group key in ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    def radius_of_gyration(self) -> ArrayF: ...
    @property
    def tensors(self) -> ArrayF: ...

class HBond:
    """One hydrogen bond (``molrs::compute::HBond``): donor, hydrogen and
    acceptor indices, the D···A distance (Å) and the D–H···A angle (deg)."""
    @property
    def acceptor(self) -> int: ...
    @property
    def angle(self) -> float: ...
    @property
    def distance(self) -> float: ...
    @property
    def donor(self) -> int: ...
    @property
    def hydrogen(self) -> int: ...

class HBonds:
    """Geometric hydrogen-bond detection, frame by frame
    (``molrs::compute::HBonds``)."""
    def __init__(self, *, dist_cutoff: float | None = None, dist_kind: str | None = None, angle_cutoff: float | None = None, donor_acceptor_margin: float | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, donors: Sequence[Sequence[int]], acceptors: Sequence[int], pairs: Sequence[Neighbors]) -> HBondsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class HBondsResult:
    """The hydrogen bonds of every frame (``molrs::compute::HBondsResult``):
    ``per_frame[t]`` lists frame ``t``'s ``HBond``s, ``counts[t]`` their
    number."""
    def components(self, frame: int) -> npt.NDArray[Any]: ...
    @property
    def counts(self) -> npt.NDArray[Any]: ...
    @property
    def n_atoms(self) -> int: ...
    @property
    def per_frame(self) -> list[list[HBond]]: ...
    def presence(self) -> Presence | None: ...

class Hexatic:
    """The k-atic bond order ``ψ_k`` of a 2-D system (``molrs::compute::Hexatic``)."""
    def __init__(self, k: int) -> None: ...
    def compute(self, *, neighbors: Neighbors) -> HexaticResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class HexaticResult:
    """The k-atic order ``ψ_k`` per particle (``molrs::compute::HexaticResult``)."""
    @property
    def k(self) -> int: ...
    @property
    def psi(self) -> npt.NDArray[np.complex128]: ...

class Histogram:
    """An N-D histogram (``molrs::compute::Histogram``): raw ``counts`` on
    ``axes``, the number of samples that fell in a bin (``binned``) and the
    volume of one bin."""
    @property
    def axes(self) -> list[Axis]: ...
    @property
    def bin_volume(self) -> float: ...
    @property
    def binned(self) -> float: ...
    @property
    def counts(self) -> ArrayF: ...

class InertiaTensor:
    """Moment of inertia tensor of each group (``molrs::compute::InertiaTensor``)."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box | None = None, partition: Partition, masses: ArrayF) -> InertiaTensorResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class InertiaTensorResult:
    """Inertia tensors (``molrs::compute::InertiaTensorResult``): a ``(3, 3)``
    tensor per group key in ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def tensors(self) -> ArrayF: ...

class InputSpec:
    """One field of a compute's input (``molrs::compute::InputSpec``)."""
    @property
    def choices(self) -> list[str] | None: ...
    @property
    def kind(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def optional(self) -> bool: ...
    @property
    def upstream(self) -> Any | None: ...

class KineticEnergy:
    """Kinetic energy ``½ Σ m v² · mvv2e`` (``molrs::compute::KineticEnergy``):
    from a unit preset (``units=``) or explicit constants (``mvv2e=``,
    ``boltzmann=``)."""
    def __init__(self, *, units: str | UnitPreset | None = None, mvv2e: float | None = None, boltzmann: float | None = None) -> None: ...
    def compute(self, *, masses: ArrayF, velocities: ArrayF) -> KineticEnergyResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class KineticEnergyResult:
    """A kinetic energy (``molrs::compute::KineticEnergyResult``) with the
    Boltzmann constant of its unit preset."""
    @property
    def boltzmann(self) -> float: ...
    @property
    def energy(self) -> float: ...
    def temperature(self, n_dof: int) -> float: ...

class Kmeans:
    """K-means clustering of the rows of a feature matrix
    (``molrs::compute::Kmeans``)."""
    def __init__(self, k: int, *, max_iter: int | None = None, seed: int | None = None) -> None: ...
    def compute(self, *, features: FeatureMatrix) -> KmeansResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class KmeansResult:
    """The k-means partition (``molrs::compute::KmeansResult``): one label per
    row and the ``(k, n_features)`` centroids. A partition."""
    @property
    def centroids(self) -> ArrayF: ...
    @property
    def labels(self) -> npt.NDArray[Any]: ...

class KramersKronig:
    """Kramers–Kronig consistency check of a ε(ω) spectrum
    (``molrs::compute::KramersKronig``)."""
    def __init__(self, *, relative_tolerance: float | None = None, absolute_tolerance: float | None = None) -> None: ...
    def compute(self, *, spectrum: DielectricSpectrumResult) -> KramersKronigCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class KramersKronigCheck:
    """The Kramers–Kronig consistency of a spectrum
    (``molrs::compute::KramersKronigCheck``): ε′ ``recovered`` from ε″, its
    mean absolute error against ε′, the tolerance and the verdict."""
    @property
    def mae(self) -> float: ...
    @property
    def passed(self) -> bool: ...
    @property
    def recovered(self) -> ArrayF: ...
    @property
    def tolerance(self) -> float: ...

class LinearFit:
    """Least-squares line through one channel of a correlation over a window of
    its lags (``molrs::compute::LinearFit``) — e.g. the diffusive regime of an
    MSD."""
    def __init__(self, start: float, end: float) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, channel: int) -> LinearFitResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class LinearFitResult:
    """A least-squares line ``slope · t + intercept``
    (``molrs::compute::LinearFitResult``) with its ``r2``, fitted over the
    lags ``[fit_start, fit_end)``."""
    @property
    def fit_end(self) -> int: ...
    @property
    def fit_start(self) -> int: ...
    @property
    def intercept(self) -> float: ...
    @property
    def r2(self) -> float: ...
    @property
    def slope(self) -> float: ...

class LocalBondProjection:
    """Projections of each bond onto ``(M, 3)`` axes, in the particle frame when
    ``(N, 4)`` orientations are given (``molrs::compute::LocalBondProjection``)."""
    def __init__(self) -> None: ...
    def compute(self, *, neighbors: Neighbors, proj_vectors: ArrayF, orientations: ArrayF | None = None) -> LocalBondProjectionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class LocalBondProjectionResult:
    """Bond projections onto the given axes
    (``molrs::compute::LocalBondProjectionResult``)."""
    @property
    def projections(self) -> ArrayF: ...

class LocalDensity:
    """Local number density within ``r_max`` of each point
    (``molrs::compute::LocalDensity``)."""
    def __init__(self, r_max: float, *, diameter: float | None = None) -> None: ...
    def compute(self, *, pairs: Neighbors | NeighborList) -> LocalDensityResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class LocalDensityResult:
    """Per-point neighbour count and local number density
    (``molrs::compute::LocalDensityResult``)."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def n_neighbors(self) -> ArrayF: ...

class LocalDescriptors:
    """Spherical-harmonic descriptors of every bond up to ``l_max``
    (``molrs::compute::LocalDescriptors``)."""
    def __init__(self, l_max: int) -> None: ...
    def compute(self, *, neighbors: Neighbors) -> LocalDescriptorsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class LocalDescriptorsResult:
    """Per-bond spherical-harmonic descriptors
    (``molrs::compute::LocalDescriptorsResult``); a feature matrix."""
    def descriptor(self, pair: int, l: int, m: int) -> Any | None: ...
    @property
    def features(self) -> ArrayF: ...
    @property
    def l_max(self) -> int: ...
    @property
    def n_sphs(self) -> int: ...

class MolecularMoments:
    """Per-group moments from an integrated density
    (``molrs::compute::MolecularMoments``): the ``charges``, ``dipoles``
    and ``references`` of each group key in ``groups``."""
    @property
    def charges(self) -> ArrayF: ...
    @property
    def dipoles(self) -> ArrayF: ...
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def references(self) -> ArrayF: ...

class Msd:
    """Mean-squared displacement (``molrs::compute::Msd``), the
    ``squared_displacement`` correlator."""
    def __init__(self, max_lag: int, *, origins: str | None = None, reduction: str | None = None, stride: int | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, dt: float, boxes: Sequence[Box] | None = None) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class Nematic:
    """Nematic order of ``(N, 3)`` directors (``molrs::compute::Nematic``)."""
    def __init__(self) -> None: ...
    def compute(self, *, directors: ArrayF) -> NematicResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class NematicResult:
    """The nematic order tensor and director (``molrs::compute::NematicResult``)."""
    @property
    def director(self) -> ArrayF: ...
    @property
    def eigenvalues(self) -> ArrayF: ...
    @property
    def order(self) -> float: ...
    @property
    def q_tensor(self) -> ArrayF: ...

class OneSidedSpectrum:
    """A one-sided spectrum (``molrs::compute::OneSidedSpectrum``):
    ``real`` and ``imag`` parts at ``frequencies``, from an ``n_fft``-point
    transform."""
    @property
    def frequencies(self) -> ArrayF: ...
    @property
    def imag(self) -> ArrayF: ...
    @property
    def n_fft(self) -> int: ...
    @property
    def real(self) -> ArrayF: ...

class OnsagerCorrelation:
    """Onsager cross-correlation of two ``(T, C)`` series
    (``molrs::compute::OnsagerCorrelation``)."""
    def __init__(self, max_lag: int, *, stride: int | None = None) -> None: ...
    def compute(self, *, a: ArrayF, b: ArrayF, dt: float, boxes: Sequence[Box] | None = None) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PairPresence:
    """Pair presence with hysteresis over frames
    (``molrs::compute::PairPresence``): a pair enters at distance ``≤ r0`` and
    stays while within ``r1``."""
    def __init__(self, r0: float, exclude_self: bool, *, r1: float | None = None) -> None: ...
    def compute(self, *, frames: Sequence[Neighbors | NeighborList]) -> Presence | None: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class ParamSpec:
    """One configuration parameter of a compute (``molrs::compute::ParamSpec``)."""
    @property
    def choices(self) -> list[str] | None: ...
    @property
    def default(self) -> Any: ...
    @property
    def kind(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def required(self) -> bool: ...
    @property
    def required_in(self) -> Any | None: ...
    @property
    def same_as(self) -> Any | None: ...
    @property
    def scaled(self) -> Any | None: ...

class Pca:
    """Principal component analysis of the rows of a feature matrix
    (``molrs::compute::Pca``)."""
    def __init__(self, n_components: int) -> None: ...
    def compute(self, *, features: FeatureMatrix) -> PcaResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PcaResult:
    """Principal components (``molrs::compute::PcaResult``): the
    ``(n_rows, n_components)`` projected ``coords`` and the ``variance``
    of each component. A feature matrix."""
    @property
    def coords(self) -> ArrayF: ...
    @property
    def variance(self) -> ArrayF: ...

class Plateau:
    """The mean of one channel of a correlation over a window of its lags
    (``molrs::compute::Plateau``) — e.g. the plateau of a Green–Kubo integral."""
    def __init__(self, start: float, end: float) -> None: ...
    def compute(self, *, correlation: TimeCorrelation, channel: int) -> PlateauResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PlateauResult:
    """The mean ``value`` (and its ``std``) of a curve over a window of
    ``n_samples`` lags (``molrs::compute::PlateauResult``)."""
    @property
    def n_samples(self) -> int: ...
    @property
    def std(self) -> float: ...
    @property
    def value(self) -> float: ...

class PmftR12:
    """PMFT in ``(r, θ1, θ2)`` of 2-D pairs (``molrs::compute::Pmft<R12>``)."""
    def __init__(self, r_max: float, n_r: int, n_t1: int, n_t2: int) -> None: ...
    def compute(self, *, pairs: Neighbors, orientations: ArrayF | None = None) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PmftResult:
    """A potential of mean force on a pair-coordinate grid
    (``molrs::compute::PmftResult``): pair ``density`` and ``raw_counts``
    per bin, ``pmf = −ln(ρ / ρ_ref)`` (``+inf`` where nothing was binned)
    and the bin ``edges`` of each axis."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def edges(self) -> list[ArrayF]: ...
    @property
    def pmf(self) -> ArrayF: ...
    @property
    def raw_counts(self) -> npt.NDArray[np.uint64]: ...

class PmftXy:
    """PMFT in the ``(x, y)`` frame of a 2-D particle
    (``molrs::compute::Pmft<Xy>``)."""
    def __init__(self, x_max: float, y_max: float, n_x: int, n_y: int) -> None: ...
    def compute(self, *, pairs: Neighbors, orientations: ArrayF | None = None) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PmftXyt:
    """PMFT in ``(x, y, θ)`` of 2-D pairs (``molrs::compute::Pmft<Xyt>``)."""
    def __init__(self, x_max: float, y_max: float, n_x: int, n_y: int, n_t: int) -> None: ...
    def compute(self, *, pairs: Neighbors, orientations: ArrayF | None = None) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PmftXyz:
    """PMFT in the ``(x, y, z)`` frame of a 3-D particle
    (``molrs::compute::Pmft<Xyz>``)."""
    def __init__(self, x_max: float, y_max: float, z_max: float, n_x: int, n_y: int, n_z: int) -> None: ...
    def compute(self, *, pairs: Neighbors, orientations: ArrayF | None = None) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PolarizabilityParts:
    """Split a polarizability series into its Raman invariants
    (``molrs::compute::PolarizabilityParts``)."""
    def __init__(self) -> None: ...
    def compute(self, *, alpha: ArrayF) -> PolarizabilityPartsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class PolarizabilityPartsResult:
    """The rotational invariants of a polarizability series
    (``molrs::compute::PolarizabilityPartsResult``): ``isotropic`` ``(T, 1)``
    and ``anisotropic`` ``(T, 6)``."""
    @property
    def anisotropic(self) -> ArrayF: ...
    @property
    def isotropic(self) -> ArrayF: ...

class Presence:
    """A presence series (``molrs::compute::Presence``): ``values[t, k]`` is
    1 when pair ``pairs[k]`` is present at frame ``t``. Pass ``values`` to
    ``Survival`` as its ``presence``."""
    @property
    def pairs(self) -> npt.NDArray[np.uint32]: ...
    @property
    def values(self) -> ArrayF: ...

class QlmTable:
    """Per-particle bond averages ``q_lm`` (or planar ``ψ_k``) of one order
    (``molrs::compute::QlmTable``)."""
    def bond_coherence(self, neighbors: Neighbors) -> ArrayF: ...
    @property
    def is_planar(self) -> bool: ...
    @property
    def n_particles(self) -> int: ...
    def norms(self) -> ArrayF: ...
    @property
    def order(self) -> int: ...
    def ql(self) -> ArrayF: ...
    @property
    def values(self) -> npt.NDArray[np.complex128]: ...
    def wl(self, normalize: bool) -> ArrayF: ...

class RadicalVoronoi:
    """The radical (power) Voronoi tessellation of one snapshot
    (``molrs::compute::RadicalVoronoi``)."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box, radii: ArrayF) -> VoronoiCells: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RamanSpectrum:
    """Raman spectrum from the isotropic and anisotropic polarizability
    correlations (``molrs::compute::RamanSpectrum``)."""
    def __init__(self, *, window: str | None = None, pad: int | None = None, incident_frequency: float | None = None, averaged: bool | None = None) -> None: ...
    def compute(self, *, isotropic: TimeCorrelation, anisotropic: TimeCorrelation, temperature: float | None = None) -> RamanSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RamanSpectrumResult:
    """A Raman spectrum (``molrs::compute::RamanSpectrumResult``): the
    isotropic and anisotropic intensities at ``frequencies_cm1``, the
    polarized / depolarized combinations when averaged, and the spectral
    resolution."""
    @property
    def anisotropic(self) -> ArrayF: ...
    @property
    def frequencies_cm1(self) -> ArrayF: ...
    @property
    def isotropic(self) -> ArrayF: ...
    @property
    def parallel(self) -> ArrayF | None: ...
    @property
    def perpendicular(self) -> ArrayF | None: ...
    @property
    def resolution(self) -> int: ...

class Rdf:
    """Radial distribution function g(r) (``molrs::compute::Rdf``)."""
    def __init__(self, n_bins: int, r_max: float, r_min: float, *, dimensionality: int | None = None) -> None: ...
    def compute(self, *, pairs: Sequence[Neighbors | NeighborList]) -> RdfResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RdfResult:
    """A radial distribution function (``molrs::compute::RdfResult``):
    ``rdf`` and the cumulative count ``n_r`` on the bins, with the
    normalisation it used."""
    @property
    def bin_centers(self) -> ArrayF: ...
    @property
    def bin_edges(self) -> ArrayF: ...
    @property
    def dimensionality(self) -> int: ...
    @property
    def n_frames(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    @property
    def n_query_points(self) -> int: ...
    @property
    def n_r(self) -> ArrayF: ...
    @property
    def r_min(self) -> float: ...
    @property
    def rdf(self) -> ArrayF: ...
    @property
    def self_query(self) -> bool: ...
    @property
    def volume(self) -> float: ...

class RegularGrid:
    """A regular 3-D voxel grid (``molrs::compute::RegularGrid``): ``origin``,
    the three cell ``basis`` vectors (rows), ``dims`` voxels per axis and the
    periodicity ``pbc`` of each axis."""
    def __init__(self, origin: Sequence[float], basis: Sequence[Sequence[float]], dims: Sequence[int], pbc: Sequence[bool]) -> None: ...
    @property
    def basis(self) -> ArrayF: ...
    @staticmethod
    def centered(extent: Sequence[float], dims: Sequence[int]) -> Self: ...
    @property
    def dims(self) -> list[int]: ...
    @property
    def origin(self) -> list[float]: ...
    @staticmethod
    def over_box(box: Box, dims: Sequence[int]) -> Self: ...
    @property
    def pbc(self) -> list[bool]: ...
    @property
    def voxel_volume(self) -> float: ...

class ReorientationAcf:
    """Legendre reorientation correlation ``⟨P_l(u(0)·u(t))⟩`` of ``(T, N, D)``
    vectors, each normalised (``molrs::compute::ReorientationAcf``) — the
    orientational autocorrelation (OACF); ``l <= 2`` runs by FFT."""
    def __init__(self, l: int, max_lag: int, *, reduction: str | None = None, stride: int | None = None) -> None: ...
    def compute(self, *, vectors: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RotationalAutocorrelation:
    """Rotational autocorrelation of ``(T, N, 4)`` unit quaternions
    (``molrs::compute::RotationalAutocorrelation``)."""
    def __init__(self, l: int, max_lag: int, *, reduction: str | None = None) -> None: ...
    def compute(self, *, orientations: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RouteAgreement:
    """Agreement of several ε(ω) routes on one system
    (``molrs::compute::RouteAgreement``)."""
    def __init__(self, *, part: str | None = None, tolerance: float | None = None) -> None: ...
    def compute(self, *, spectra: Sequence[DielectricSpectrumResult]) -> RouteAgreementCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class RouteAgreementCheck:
    """The agreement of several ε(ω) routes
    (``molrs::compute::RouteAgreementCheck``): the RMS difference of every
    pair ``(i, j, rms)``, the largest and the verdict."""
    @property
    def max_rms(self) -> float: ...
    @property
    def pairwise(self) -> list[tuple[int, int, float]]: ...
    @property
    def passed(self) -> bool: ...

class SolidLiquid:
    """Solid-liquid classification by bond coherence
    (``molrs::compute::SolidLiquid``)."""
    def __init__(self, *, q_threshold: float | None = None, n_threshold: int | None = None, normalize_q: bool | None = None) -> None: ...
    def compute(self, *, qlm: QlmTable, neighbors: Neighbors) -> SolidLiquidResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class SolidLiquidResult:
    """Solid-like bonds and particles (``molrs::compute::SolidLiquidResult``)."""
    @property
    def is_solid(self) -> ArrayBool: ...
    @property
    def l(self) -> int: ...
    @property
    def n_solid_bonds(self) -> npt.NDArray[Any]: ...

class SpatialDistribution:
    """Spatial distribution of target atoms in the body frame of a rigid
    reference, over frames (``molrs::compute::SpatialDistribution``)."""
    def __init__(self, grid: RegularGrid, *, bulk_density: float | None = None) -> None: ...
    def compute(self, *, positions: Sequence[ArrayF], simbox: Sequence[Box | None] | None = None, reference: Sequence[Sequence[int]], template: Sequence[ArrayF], target: Sequence[Sequence[int]], orientation: Sequence[Sequence[Sequence[int]] | None] | None = None) -> SpatialDistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class SpatialDistributionResult:
    """A spatial distribution in a reference body frame
    (``molrs::compute::SpatialDistributionResult``)."""
    @property
    def counts(self) -> ArrayF: ...
    @property
    def density(self) -> ArrayF: ...
    @property
    def g_sdf(self) -> ArrayF | None: ...
    @property
    def grid(self) -> RegularGrid: ...
    @property
    def n_frames(self) -> int: ...
    @property
    def orientation(self) -> ArrayF | None: ...
    @property
    def voxel_volume(self) -> float: ...

class Spectrum:
    """The one-sided Fourier transform of a real signal
    (``molrs::compute::Spectrum``)."""
    def __init__(self, window: str, pad: int) -> None: ...
    def compute(self, *, signal: ArrayF, dt: float) -> OneSidedSpectrum: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class SphereVoxelization:
    """Voxels covered by spheres of radius ``r_max`` around the points
    (``molrs::compute::SphereVoxelization``)."""
    def __init__(self, dims: Sequence[int], r_max: float) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box) -> SphereVoxelizationResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class SphereVoxelizationResult:
    """Occupied voxels of spheres on a grid
    (``molrs::compute::SphereVoxelizationResult``)."""
    @property
    def grid(self) -> RegularGrid: ...
    @property
    def raw_counts(self) -> npt.NDArray[Any]: ...
    @property
    def voxels(self) -> npt.NDArray[Any]: ...

class StaticDielectric:
    """Static permittivity from the fluctuation of the total dipole
    (``molrs::compute::StaticDielectric``)."""
    def __init__(self, units: str | UnitPreset, *, epsilon_inf: float | None = None) -> None: ...
    def compute(self, *, dipoles: ArrayF, volume: float, temperature: float, boundary: str) -> StaticDielectricResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class StaticDielectricResult:
    """The static permittivity (``molrs::compute::StaticDielectricResult``):
    ``epsilon``, per-axis ``epsilon_axes``, the dipole moments behind it and
    the number of frames."""
    @property
    def dipole_mean(self) -> ArrayF: ...
    @property
    def dipole_sq_mean(self) -> ArrayF: ...
    @property
    def epsilon(self) -> float: ...
    @property
    def epsilon_axes(self) -> ArrayF: ...
    @property
    def epsilon_inf(self) -> float: ...
    @property
    def fluctuation(self) -> ArrayF: ...
    @property
    def n_frames(self) -> int: ...

class StaticStructureFactorDebye:
    """Static structure factor by the Debye sum over all pairs
    (``molrs::compute::StaticStructureFactorDebye``)."""
    def __init__(self, k_values: Sequence[float], *, sinc_tolerance: float | None = None) -> None: ...
    def compute(self, *, positions: ArrayF) -> StaticStructureFactorDebyeResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class StaticStructureFactorDebyeResult:
    """S(k) by the Debye sum (``molrs::compute::StaticStructureFactorDebyeResult``)."""
    @property
    def k_values(self) -> ArrayF: ...
    @property
    def n_particles(self) -> int: ...
    @property
    def sk(self) -> ArrayF: ...

class StaticStructureFactorDirect:
    """Static structure factor from the density's Fourier components
    (``molrs::compute::StaticStructureFactorDirect``)."""
    def __init__(self, *, k_vecs: Sequence[Sequence[float]] | None = None, k_max: float | None = None, n_bins: int | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, simbox: Box | None = None) -> StaticStructureFactorDirectResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class StaticStructureFactorDirectResult:
    """S(k) on explicit or isotropically binned k-vectors
    (``molrs::compute::StaticStructureFactorDirectResult``)."""
    @property
    def k_magnitudes(self) -> ArrayF: ...
    @property
    def n_particles(self) -> int: ...
    @property
    def sk(self) -> ArrayF: ...

class Steinhardt:
    """Steinhardt bond-orientational order (``molrs::compute::Steinhardt``)."""
    def __init__(self, l: Sequence[int], *, average: bool | None = None, wl: bool | None = None, wl_normalize: bool | None = None) -> None: ...
    def compute(self, *, neighbors: Neighbors) -> SteinhardtResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class SteinhardtResult:
    """Steinhardt order parameters (``molrs::compute::SteinhardtResult``); a
    feature matrix."""
    @property
    def l(self) -> npt.NDArray[Any]: ...
    @property
    def ql(self) -> ArrayF: ...
    def qlm(self, l: int) -> QlmTable | None: ...
    @property
    def tables(self) -> list[QlmTable]: ...
    @property
    def wl(self) -> ArrayF | None: ...

class SumRuleCheck:
    """The conductivity sum rule (``molrs::compute::SumRuleCheck``): the
    integral, its expected value, their relative error and the verdict."""
    @property
    def expected(self) -> float: ...
    @property
    def integral(self) -> float: ...
    @property
    def passed(self) -> bool: ...
    @property
    def relative_error(self) -> float: ...

class Survival:
    """Survival correlation of a ``(T, C)`` presence series
    (``molrs::compute::Survival``), e.g. ``PairPresence``'s ``values``."""
    def __init__(self, kind: str, max_lag: int, *, normalization: str | None = None, origin: str | None = None) -> None: ...
    def compute(self, *, presence: ArrayF, dt: float, inner: ArrayF | None = None, n_reference: int | None = None) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class TimeCorrelation:
    """A correlation curve (``molrs::compute::TimeCorrelation``):
    ``values[lag, channel]`` at ``lag_times[lag]``."""
    def __init__(self, lag_times: ArrayF, values: ArrayF) -> None: ...
    def __reduce__(self) -> tuple[type[TimeCorrelation], tuple[ArrayF, ArrayF]]: ...
    def channel(self, channel: int) -> ArrayF: ...
    @property
    def lag_times(self) -> ArrayF: ...
    @property
    def n_channels(self) -> int: ...
    @property
    def n_lags(self) -> int: ...
    @property
    def values(self) -> ArrayF: ...

class Unwrap:
    """Unwrap a wrapped ``(T, N, 3)`` trajectory (``molrs::compute::Unwrap``),
    from int32 image flags when given, else by minimum image between frames."""
    def __init__(self) -> None: ...
    def compute(self, *, positions: ArrayF, images: ArrayI32 | None = None, boxes: Sequence[Box]) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class Vacf:
    """Velocity autocorrelation (``molrs::compute::Vacf``)."""
    def __init__(self, max_lag: int, *, normalization: str | None = None, stride: int | None = None) -> None: ...
    def compute(self, *, velocities: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class VanHove:
    """The van Hove correlation G(r, t) over a trajectory
    (``molrs::compute::VanHove``)."""
    def __init__(self, axis: Axis, lags: Sequence[int], *, origins: str | None = None, stride: int | None = None, upper_edge: str | None = None) -> None: ...
    def compute(self, *, positions: ArrayF, boxes: Sequence[Box]) -> VanHoveResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class VanHoveResult:
    """The self and distinct van Hove functions at each lag
    (``molrs::compute::VanHoveResult``)."""
    @property
    def dr(self) -> float: ...
    @property
    def g_distinct(self) -> ArrayF: ...
    @property
    def g_self(self) -> ArrayF: ...
    @property
    def lags(self) -> npt.NDArray[Any]: ...
    @property
    def r_centers(self) -> ArrayF: ...
    @property
    def r_edges(self) -> ArrayF: ...

class VoronoiCells:
    """A tessellation (``molrs::compute::VoronoiCells``): the ``volumes`` of
    the cells and the ``faces`` of each."""
    @property
    def faces(self) -> list[list[VoronoiFace]]: ...
    def neighbors(self, i: int) -> list[int]: ...
    def total_volume(self) -> float: ...
    @property
    def volumes(self) -> ArrayF: ...

class VoronoiDomainAnalysis:
    """Connected domains of the cells a partition selects
    (``molrs::compute::VoronoiDomainAnalysis``)."""
    def __init__(self) -> None: ...
    def compute(self, *, cells: VoronoiCells, partition: Partition) -> VoronoiDomainResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class VoronoiDomainResult:
    """Voronoi domains (``molrs::compute::VoronoiDomainResult``): the ``sizes``
    of the connected domains, their ``count``, the ``largest_fraction``
    and each cell's domain (``None`` outside every domain). A partition."""
    @property
    def count(self) -> int: ...
    @property
    def domain_of(self) -> list[int | None]: ...
    @property
    def largest_fraction(self) -> float: ...
    @property
    def sizes(self) -> npt.NDArray[Any]: ...

class VoronoiFace:
    """One face of a Voronoi cell (``molrs::compute::VoronoiFace``): its
    ``area`` and the ``neighbor`` across it (negative for a boundary)."""
    @property
    def area(self) -> float: ...
    @property
    def neighbor(self) -> int: ...

class VoronoiIntegration:
    """Integrate a density over the radical Voronoi cells of the atoms and sum
    the cells of each group (``molrs::compute::VoronoiIntegration``)."""
    def __init__(self, unit: str) -> None: ...
    def compute(self, *, positions: ArrayF, radii: ArrayF, nuclear_charges: ArrayF, partition: Partition, simbox: Box, grid: RegularGrid, density: ArrayF) -> MolecularMoments: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class VoronoiVoidAnalysis:
    """Cavities formed by connected void cells
    (``molrs::compute::VoronoiVoidAnalysis``)."""
    def __init__(self) -> None: ...
    def compute(self, *, cells: VoronoiCells, is_void: Sequence[bool], box_volume: float) -> VoronoiVoidResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

class VoronoiVoidResult:
    """Voids (``molrs::compute::VoronoiVoidResult``): the cavity volumes
    (descending), their total and the void fraction of the box."""
    @property
    def cavity_volumes(self) -> ArrayF: ...
    @property
    def total_void_volume(self) -> float: ...
    @property
    def void_fraction(self) -> float: ...

# ---------------------------------------------------------------------------
# molrs.ff.potential — hand-built kernels (mirrors molrs-python/src/ff/potential.rs)
# ---------------------------------------------------------------------------

class ExplicitTerms:
    """One style's terms over explicit instances
    (``molrs::ff::compile::ExplicitTerms``): ``atoms`` ``(n, arity)``, each
    per-term parameter a number (broadcast) or one value per term, as stored
    (angle values in degrees); style parameters (``cutoff``, ``coulomb``, an
    unregistered style's ``expression``) a number or a string. Works for
    every registered style, built-in or custom. A parameter the style does
    not declare raises ``TypeError``."""

    def __init__(
        self,
        category: str,
        style: str,
        atoms: Sequence[Sequence[int]] | ArrayI64 | ArrayU32,
        *,
        charges: Sequence[float] | ArrayF | None = None,
        **params: float | str | Sequence[float] | Sequence[str] | ArrayF,
    ) -> None: ...
    def compile(self) -> Potentials:
        """The terms' kernel, built as ``PotentialCompiler.compile`` builds
        it, one type per term. Refusals raise their ``molrs.ff.ir.IrError``
        subclass."""

class PairLjCut:
    """LAMMPS ``pair_style lj/cut``: the one-type cut Lennard-Jones / Mie
    kernel (``n``/``m`` exponents) a neighbour loop feeds (MD's nonbond
    kernel). A pair list with a row per pair is
    ``ExplicitTerms("pair", "lj/cut", pairs, epsilon=..., sigma=...).compile()``."""

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        *,
        n: int = 12,
        m: int = 6,
        shifted: bool = True,
        smeared: bool = False,
    ) -> None: ...
    @property
    def epsilon(self) -> float: ...
    @property
    def sigma(self) -> float: ...
    @property
    def cutoff(self) -> float: ...
    @property
    def n(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def shifted(self) -> bool: ...
    @property
    def smeared(self) -> bool: ...
    def pair_energy(self, r2: float, disp: Sequence[float]) -> float | None: ...
    def pair_force(
        self, r2: float, disp: Sequence[float]
    ) -> list[float] | None: ...
    def pair_energy_force(
        self, r2: float, disp: Sequence[float]
    ) -> tuple[float, list[float]] | None: ...
    def calc_energy_forces(self, pos: ArrayF) -> tuple[float, ArrayF]: ...
    def energy_forces_skin(self, neighbors: VerletSkin, pos: ArrayF) -> tuple[float, ArrayF]: ...
    def energy_forces_table(
        self, n_atoms: int, neighbors: Neighbors
    ) -> tuple[float, ArrayF]: ...
    def energy_forces_pairs(
        self,
        n_atoms: int,
        i: ArrayU32,
        j: ArrayU32,
        disp: ArrayF,
        dist_sq: ArrayF | None = None,
    ) -> tuple[float, ArrayF]: ...

# ---------------------------------------------------------------------------
# molrs.md — in-process MD (mirrors molrs-python/src/md.rs).
# Submodule stubbed as a class namespace, following the `keys` precedent.
# ---------------------------------------------------------------------------

class md:
    """The ``_native.md`` submodule (``molrs.md``): NVE/Langevin integrators.
    MD defines no potential; it integrates an ``PairLjCut``, a ``Potentials``
    collection, or any object with ``calc_energy_forces``.

    The engine is unit-agnostic — supply consistent units yourself; take
    constants from :class:`UnitPreset` (``UnitPreset("real").boltzmann()``).
    """

    class MdState:
        """Dynamical state (pos/vel/forces ``(N, 3)`` float64 + potential energy).

        Fields are settable wholesale (``state.vel = v``; shape-validated).
        Getters return **copies**, so slice writes (``state.vel[:] = v``) are
        silently ineffective.
        """

        def __init__(
            self, pos: ArrayF, vel: ArrayF, forces: ArrayF, energy: float
        ) -> None: ...
        @property
        def pos(self) -> ArrayF: ...
        @pos.setter
        def pos(self, value: ArrayF) -> None: ...
        @property
        def vel(self) -> ArrayF: ...
        @vel.setter
        def vel(self, value: ArrayF) -> None: ...
        @property
        def forces(self) -> ArrayF: ...
        @forces.setter
        def forces(self, value: ArrayF) -> None: ...
        @property
        def virial(self) -> tuple[float, float, float, float, float, float] | None:
            """Virial ``Sigma f (x) r`` as ``(xx, yy, zz, xy, xz, yz)``, or ``None``.

            ``None`` means the force provider does not tally one -- not that it
            is zero. A kernel resolved against a fixed pair list declines, and
            one member declining makes the whole sum ``None``.
            """
        @property
        def images(self) -> ArrayI32: ...
        @images.setter
        def images(self, value: ArrayI32) -> None: ...
        def pressure(self, kinetic: float, volume: float) -> float | None: ...
        @property
        def energy(self) -> float: ...
        @energy.setter
        def energy(self, value: float) -> None: ...

    class VelocityVerlet:
        """NVE velocity-Verlet. ``potential`` (a ``PairLjCut`` /
        ``Potentials`` / an object with ``calc_energy_forces``) and
        ``neighbors`` (a ``VerletSkin``) are
        moved in; the loop feeds fresh pairs to the potential after each
        rebuild. ``PairLjCut`` requires ``neighbors=``."""

        def __init__(
            self,
            dt: float,
            *,
            potential: PairLjCut | Potentials | WeightedTerms | Any,
            neighbors: VerletSkin | None = None,
            mass: float | ArrayF,
            box: Box | None = None,
        ) -> None: ...
        @property
        def dt(self) -> float: ...
        @property
        def removed_dof(self) -> int: ...
        @property
        def n_edges(self) -> int | None: ...
        @property
        def n_rebuilds(self) -> int | None: ...
        @property
        def ago(self) -> int | None: ...
        def initial(self, pos: ArrayF, vel: ArrayF) -> md.MdState: ...
        def advance(self, state: Any) -> md.MdState: ...
        def advance_n(self, state: Any, n_steps: int) -> md.MdState: ...

    class Langevin:
        """BAOAB Langevin; ``kbt`` is an energy in your unit system
        (``UnitPreset(style).boltzmann() * T``).

        ``advance`` draws noise from the internal seeded RNG; ``step`` takes
        the ``(N, 3)`` Gaussian draw explicitly."""

        def __init__(
            self,
            dt: float,
            *,
            gamma: float,
            kbt: float,
            potential: PairLjCut | Potentials | WeightedTerms | Any,
            neighbors: VerletSkin | None = None,
            mass: float | ArrayF,
            seed: int = 0,
            box: Box | None = None,
        ) -> None: ...
        @property
        def dt(self) -> float: ...
        @property
        def gamma(self) -> float: ...
        @property
        def c1(self) -> float: ...
        @property
        def c2(self) -> float: ...
        @property
        def sigma(self) -> ArrayF: ...
        @property
        def inv_mass(self) -> ArrayF: ...
        @property
        def removed_dof(self) -> int: ...
        @property
        def n_edges(self) -> int | None: ...
        @property
        def n_rebuilds(self) -> int | None: ...
        @property
        def ago(self) -> int | None: ...
        def initial(self, pos: ArrayF, vel: ArrayF) -> md.MdState: ...
        def step(self, state: Any, noise: ArrayF) -> md.MdState: ...
        def advance(self, state: Any) -> md.MdState: ...
        def advance_n(self, state: Any, n_steps: int) -> md.MdState: ...
        def draw_noise(self, n_atoms: int) -> ArrayF: ...

    class MaxwellBoltzmann:
        """Velocity distribution at thermal energy ``kbt`` (optionally COM-free).

        Takes ``kbt`` — the product of Boltzmann's constant and temperature in
        the caller's unit system — not a temperature: the engine does no unit
        conversion. See ``UnitPreset(...).boltzmann()``."""

        def __init__(
            self, kbt: float, *, seed: int = 0, remove_com: bool = True
        ) -> None: ...
        @property
        def kbt(self) -> float: ...
        @property
        def seed(self) -> int: ...
        @property
        def remove_com(self) -> bool: ...
        def velocities(self, pos: ArrayF, mass: float | ArrayF) -> ArrayF: ...

class ir:
    """The ``_native.ir`` submodule: the force-field IR registry
    (``molrs::ff::ir``), surfaced as :mod:`molrs.ff.ir`.

    Register a category or a style — by an expression, a vectorised Python
    kernel, or both — into the process-wide registry every compile reads;
    what does not conform raises the :class:`IrError` subclass named after
    the Rust variant."""

    class IrError(ValueError):
        """A refusal of the force-field IR; the variant's fields are
        attributes (``category``, ``style``, ``param``, …)."""

    class UnknownCategoryError(IrError): ...
    class BadNameError(IrError): ...
    class ArityError(IrError): ...
    class BlockNameError(IrError): ...
    class ReservedParamError(IrError): ...
    class DuplicateParamError(IrError): ...
    class DimensionError(IrError): ...
    class ParseError(IrError): ...
    class UnboundVariableError(IrError): ...
    class UnknownFunctionError(IrError): ...
    class FunctionArityError(IrError): ...
    class PointError(IrError): ...
    class CoordinateMismatchError(IrError): ...
    class DerivativeError(IrError): ...
    class DisagreeError(IrError): ...
    class AsymmetricError(IrError): ...
    class SealedError(IrError): ...
    class ConflictError(IrError): ...
    class NoKernelError(IrError): ...
    class NoMixingError(IrError): ...
    class MissingParamError(IrError): ...
    class BadValueError(IrError): ...
    class KernelShapeError(IrError):
        """A kernel output of the wrong shape or dtype, or a Python kernel
        that raised (the original exception is ``__cause__``)."""

    class NoEngineFormError(IrError): ...
    class FormConflictError(IrError): ...
    class NoFormError(IrError): ...
    class OutOfImageError(IrError):
        """An exact form conversion refused; ``from_``, ``to``, ``type`` and
        ``reason`` name the row and the condition."""

    class MalformedError(IrError): ...

    class ParamSpec:
        """One parameter of a style: name, dimension (``"E/L^2"``), kind,
        default, mixing rule (pair styles), indexed family."""

        def __init__(
            self,
            name: str,
            dim: str = "1",
            *,
            kind: Literal["scalar", "array", "text"] = "scalar",
            rank: int | None = None,
            choices: Sequence[str] | None = None,
            default: float | str | None = None,
            mix: str | tuple[str, str] | None = None,
            indexed: bool = False,
        ) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def dim(self) -> str: ...
        @property
        def kind(self) -> Literal["scalar", "array", "text"]: ...
        @property
        def rank(self) -> int | None: ...
        @property
        def choices(self) -> list[str] | None: ...
        @property
        def default(self) -> float | str | None: ...
        @property
        def mix(self) -> str | tuple[str, str] | None: ...
        @property
        def indexed(self) -> bool: ...

    class StyleSpec:
        """A registered style, as ``styles()`` lists it."""

        @property
        def category(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def params(self) -> list[ir.ParamSpec]: ...
        @property
        def style_params(self) -> list[ir.ParamSpec]: ...
        @property
        def expression(self) -> str | None: ...
        @property
        def kernel(
            self,
        ) -> Literal["expression", "scalar", "compound", "constructor"] | None: ...
        @property
        def builtin(self) -> bool: ...
        @property
        def source(self) -> Literal["type_rows", "per_instance"]: ...
        @property
        def special(self) -> Literal["lj", "coul"] | None: ...
        @property
        def lammps(self) -> str | None: ...

    class CategorySpec:
        """A registered category, as ``categories()`` lists it."""

        @property
        def name(self) -> str: ...
        @property
        def arity(self) -> int: ...
        @property
        def pair(self) -> bool: ...
        @property
        def block(self) -> str: ...
        @property
        def coordinate(self) -> str: ...
        @property
        def order(self) -> str: ...
        @property
        def builtin(self) -> bool: ...

    @staticmethod
    def register_category(
        name: str,
        arity: int,
        *,
        coordinate: Literal[
            "compound", "distance", "angle", "dihedral", "improper"
        ] = "compound",
        order: Literal["reversible", "ordered", "unordered"] = "reversible",
    ) -> None: ...
    @staticmethod
    def register_style(
        category: str,
        name: str,
        *,
        params: Sequence[ir.ParamSpec] | _AbcMapping[str, str] | None = None,
        style_params: Sequence[ir.ParamSpec] | _AbcMapping[str, str] | None = None,
        expression: str | None = None,
        kernel: Callable[..., tuple[ArrayF, ArrayF]] | None = None,
        compound: bool = False,
        special: Literal["lj", "coul"] | None = None,
        samples: Sequence[_AbcMapping[str, Any]] | None = None,
        replace: bool = False,
        lammps: str | None = None,
    ) -> None: ...
    @staticmethod
    def register_engine_form(engine: str, category: str, name: str, form: str) -> None: ...
    @staticmethod
    def unregister_style(category: str, name: str) -> None: ...
    @staticmethod
    def styles(category: str | None = None) -> list[ir.StyleSpec]: ...
    @staticmethod
    def categories() -> list[ir.CategorySpec]: ...
    @staticmethod
    def evaluate(
        category: str,
        name: str,
        q: npt.ArrayLike | None = None,
        *,
        x: npt.ArrayLike | None = None,
        **params: Any,
    ) -> tuple[ArrayF, ArrayF]: ...


# ---------------------------------------------------------------------------
# Exports declared from the compiled module's own signatures.
#
# These are real `_native` exports whose declarations were missing here. The
# signatures below are the PyO3 `text_signature`s verbatim; annotations are
# only added where the parameter name fixes the type beyond doubt.
# ---------------------------------------------------------------------------


class RingSet:
    """
    The ring facts of a molecule: SSSR rings and the systems they fuse into.

    Returned by :func:`perceive_rings`, which runs the perception once; every
    method reads the result.

    Not to be confused with :func:`assign_rings`, which answers a
    different question — it *decorates* a graph with ring flags and hands the
    graph back. This type *reports*, and never touches the molecule.

    Examples
    --------
    >>> rings = molrs.perceive.perceive_rings(molrs.io.smiles.SmilesIr("c1ccccc1").to_atomistic())
    >>> rings.n_rings()
    1
    >>> rings.ring_sizes()
    [6]
    """
    def is_atom_in_ring(self, /, atom): ...
    def max_ring_system_size(self) -> int:
        """Atom count of the largest fused / bridged ring system
        (naphthalene → 10); ``0`` for an acyclic molecule."""
    def n_atom_rings(self, /, atom): ...
    def n_rings(self, /): ...
    def ring_sizes(self, /): ...
    def ring_systems(self, /): ...
    def rings(self, /): ...
    def smallest_ring_containing_atom(self, /, atom): ...


def read_lammps_log_str(
    text: str, source_name: str = "<string>", style: str = "default"
) -> LammpsLog:
    """Parse a LAMMPS log from an in-memory string (no filesystem access)."""

def read_ac(path: PathInput) -> Frame:
    """Read an Antechamber ``.ac`` file into a Frame."""

def read_prmtop_forcefield(path: PathInput) -> ForceField:
    """Read AMBER prmtop force-field parameter tables into a :class:`ForceField`."""

def read_prmtop_system(path: PathInput) -> tuple[ForceField, Frame]:
    """Read a whole AMBER prmtop into a :class:`ForceField` and a typed :class:`Frame`.

    The structure of :func:`molrs.io.read_prmtop` plus a ``pairs`` block
    of the 1-4 pairs sander weighs otherwise than ``special_bonds`` (per-pair
    ``coul_scale`` / ``lj_scale``, null where they agree); no ``pairs`` block
    when every 1-4 pair agrees.
    """

def read_top_forcefield(
    path: PathInput,
    include: bool = False,
    *,
    include_dirs: Sequence[PathInput] = (),
    skip_directives: Sequence[str] = (),
) -> ForceField:
    """Read the force-field directives of a GROMACS topology into a :class:`ForceField`.

    Reads ``[ defaults ]``, ``[ atomtypes ]``, ``[ nonbond_params ]``,
    ``[ pairtypes ]`` (``lj/charmm`` 1-4 parameters), ``[ bondtypes ]``,
    ``[ angletypes ]`` (funct 5: ``angle charmm``), ``[ dihedraltypes ]``
    (funct 1, 2, 3, 4, 5, 9) and ``[ cmaptypes ]``. What the IR cannot hold
    and every molecule section raise ``ValueError`` naming them; a whole
    topology is :func:`read_top_system`. Each name in ``skip_directives``
    is read past instead of refused.
    """

def read_top_system(
    path: PathInput,
    *,
    include_dirs: Sequence[PathInput] = (),
    skip_directives: Sequence[str] = (),
) -> tuple[ForceField, Frame]:
    """Read a whole GROMACS topology into a :class:`ForceField` and a typed :class:`Frame`.

    Directives and molecule sections; each relation row typed by GROMACS's
    own lookup, ``pairs`` the intramolecular pairs GROMACS prices (``[ pairs ]``
    flagged ``is_14``, with per-pair overrides where they carry parameters).
    ``#include`` resolves relative to the including file, then against each of
    ``include_dirs``.
    """

def read_lammps_log(path: PathInput, style: str = "default") -> LammpsLog:
    """Read a LAMMPS log file into a structured ``LammpsLog``."""

def is_lammps_log(text: str) -> bool:
    """Whether ``text`` holds a LAMMPS run (a ``Per MPI rank memory allocation`` line)."""

def read_lammps_molecule(path: PathInput) -> Frame:
    """Read a LAMMPS molecule template (the native text format)."""

def read_lammps_molecule_json(path: PathInput) -> Frame:
    """Read a LAMMPS molecule template in its JSON format."""

def read_mol2(path: PathInput) -> Frame:
    """Read a Tripos MOL2 file and return the first molecule as a Frame, in
    canonical column names (``type``, ``res_id``, ``res_name`` on atoms; the
    SYBYL bond token as ``type`` on bonds)."""

def read_prep(path: PathInput) -> dict[str, Any]:
    """Read an Amber prep file into a nested dict (serde JSON shape)."""


def write_openmm_forcefield(
    path: PathInput, forcefield: ForceField, precision: int | None = None
) -> None:
    """Write a ForceField to OpenMM force-field XML — the inverse of
    :func:`read_openmm_forcefield`."""


def write_frcmod(path: PathInput, forcefield: ForceField) -> None:
    """Write a ForceField as an AMBER frcmod file.

    A style or parameter a frcmod cannot express raises ``ValueError``.
    """

def write_top_forcefield(
    path: PathInput, forcefield: ForceField, precision: int = 6
) -> None:
    """Write a ForceField as GROMACS force-field directives (no molecule sections).

    A style or parameter the directives cannot express raises ``ValueError``.
    """

def write_lammps_molecule(path: PathInput, frame: Frame) -> None:
    """Write a Frame as a LAMMPS molecule template (the native text format)."""

def write_lammps_molecule_json(path: PathInput, frame: Frame) -> None:
    """Write a Frame as a LAMMPS molecule template in its JSON format."""

def write_mol2(path: PathInput, frame: Frame) -> None:
    """Write a Frame to a Tripos MOL2 file, reading the canonical columns
    :func:`read_mol2` produces."""

def write_prep(path: PathInput, residue: dict[str, Any]) -> None:
    """Write an Amber prep residue from a nested dict."""

def read_smiles_str(smiles: str) -> Atomistic:
    """One molecule from a SMILES string: connectivity only, no implicit H, no
    coordinates. A ``'.'``-separated set raises ``SmilesError`` (a
    ``ValueError``) naming ``SmilesIr(s).components()``."""

def write_smiles_str(
    mol: Atomistic,
    *,
    canonical: bool = True,
    root: int | None = None,
    aromatic: Literal["as_marked", "kekule_only"] = "as_marked",
    hydrogens: Literal["organic_subset", "explicit_all", "as_stored"] = "organic_subset",
    include_stereo: bool = False,
    multi_component: Literal[
        "error_if_multiple", "join_dot", "first_only"
    ] = "error_if_multiple",
    organic_subset: bool = True,
) -> str:
    """A molecule as SMILES text — the inverse of :func:`read_smiles_str`."""

def write_trr_trajectory(path: PathInput, frames: Sequence[Frame]) -> None:
    """Write Frames to a GROMACS TRR trajectory file (single precision)."""

def write_xtc_trajectory(path: PathInput, frames: Sequence[Frame]) -> None:
    """Write Frames to a GROMACS XTC trajectory file (lossy compression)."""

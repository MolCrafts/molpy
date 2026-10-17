"""Type stubs for molrs Python bindings.

Hand-maintained. Keep in sync with the PyO3 source in `molrs-python/src/**.rs`
on every PR that touches `#[pyfunction]` or `#[pyclass]`. `tests/test_stub_parity.py`
is the freshness guard: every compiled `_native` export must be declared here, with
the same parameter names as the compiled signature.
"""

import os
from types import TracebackType
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
    Generic,
    Protocol,
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
# One node or relation field value: what ``**props`` takes and what a read
# answers. ``bool`` is kept apart from ``int`` at runtime.
type PropValue = bool | int | float | str
# A force-field param as read back: a number, a string, or a float64 array
# (a CMAP ``grid``).
type ParamValue = float | str | ArrayF
# A force-field param as given: an array may be any numpy array or nested
# sequence of numbers, stored as float64.
type ParamInput = float | str | npt.ArrayLike
_TGraph = TypeVar("_TGraph", MolGraph, Atomistic, CoarseGrain)

__version__: str

def _restore_pickled_state(
    obj: object, state: tuple[object, _AbcMapping[str, object]]
) -> None:
    """Pickle state setter: ``obj.__setstate__(state[0])``, then merge the
    instance ``__dict__`` ``state[1]`` back."""

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

@final
class Cell:
    """The geometry of a simulation cell — Rust's ``Cell`` enum, one
    staticmethod per variant. Frozen; checked where it is used (``Box(...)``
    raises for a singular matrix, a non-positive length or impossible
    angles)."""

    @staticmethod
    def matrix(h: npt.ArrayLike) -> Cell:
        """The ``(3, 3)`` cell matrix (Å), lattice vectors as columns."""
    @staticmethod
    def lengths(lengths: npt.ArrayLike) -> Cell:
        """An orthorhombic cell with edge lengths ``[a, b, c]`` (Å)."""
    @staticmethod
    def lengths_angles(lengths: npt.ArrayLike, angles: npt.ArrayLike) -> Cell:
        """Restricted triclinic from ``[a, b, c]`` (Å) and
        ``[alpha, beta, gamma]`` (degrees)."""
    @staticmethod
    def lengths_tilts(lengths: npt.ArrayLike, tilts: npt.ArrayLike) -> Cell:
        """Restricted triclinic from ``[lx, ly, lz]`` and tilts
        ``[xy, xz, yz]`` (Å) — the LAMMPS form."""
    @staticmethod
    def undefined() -> Cell:
        """No geometric cell (undefined, zero volume)."""
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __reduce__(self) -> tuple[Any, ...]: ...

class Box:
    """Simulation box with periodic boundary conditions: a :class:`Cell`, an
    origin and per-axis periodicity."""

    def __new__(
        cls,
        cell: Cell,
        origin: npt.ArrayLike = ...,
        pbc: npt.ArrayLike = ...,
    ) -> Self:
        """``Box(cell, origin=(0, 0, 0), pbc=(True, True, True))``.

        Raises ``ValueError`` for a singular matrix, a non-positive length or
        lengths/angles that describe no cell."""
    @staticmethod
    def bounding(points: ArrayF, padding: npt.ArrayLike, pbc: npt.ArrayLike) -> Box:
        """The orthorhombic box enclosing ``points`` ``(N, 3)``, grown by
        ``padding[d] >= 0`` on each side of axis ``d``. Raises ``ValueError``
        for no points, a negative padding or an axis of zero extent."""
    def approx_eq(self, other: Box, tol: float) -> bool:
        """Same cell within absolute ``tol``: every matrix entry and origin
        component within ``tol``, identical ``pbc`` and ``is_cell_defined``."""
    @property
    def is_cell_defined(self) -> bool: ...
    @property
    def is_free(self) -> bool: ...
    @property
    def style(self) -> str: ...
    def volume(self) -> float: ...
    def lattice(self, index: int) -> ArrayF: ...
    @property
    def matrix(self) -> ArrayF: ...
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
    def restricted_matrix(matrix: ArrayF) -> ArrayF: ...
    @property
    def tilts(self) -> ArrayF: ...
    @property
    def nearest_plane_distance(self) -> ArrayF: ...
    @property
    def bounds(self) -> ArrayF: ...
    def corners(self) -> ArrayF: ...
    def to_frac(self, xyz: ArrayF) -> ArrayF: ...
    def to_cart(self, frac: ArrayF) -> ArrayF: ...
    def wrap(self, xyz: ArrayF) -> tuple[ArrayF, ArrayI32]:
        """``(wrapped, shifts)``: coordinates wrapped on the periodic axes and
        the int32 lattice shift applied to each point (image flags)."""
    def unwrap(self, xyz: ArrayF, images: ArrayI32) -> ArrayF: ...
    def __reduce__(self) -> tuple[Any, ...]: ...
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

    def __new__(cls, cutoff: float, brute_force: bool = False) -> Self:
        """``brute_force=True`` picks the O(N²) all-pairs backend (same
        pairs; an oracle) over the O(N) cell list."""
    @property
    def cutoff(self) -> float: ...
    def build(self, points: ArrayF, box: Box) -> None: ...
    def update(self, points: ArrayF) -> None: ...
    def neighbors(self, dist_sq: bool = True, disp: bool = True) -> Neighbors: ...
    def __reduce__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: tuple[ArrayF, Box] | None) -> None: ...

class Neighbors:
    """Materialized pair table — read-only columns, one row per pair.

    ``dist_sq`` and ``disp`` are opt-in: a column the search was told not to
    store reads back as ``None``, never as a fabricated zero array.
    """

    def __new__(
        cls,
        is_self_query: bool,
        n_points: int,
        n_query_points: int,
        idx_i: list[int],
        idx_j: list[int],
        dist_sq: list[float] | None = None,
        disp: list[list[float]] | None = None,
    ) -> Self: ...
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

    def __new__(cls, box: Box, points: ArrayF, cutoff: float) -> Self:
        """A free-boundary search takes
        ``Box.bounding(points, (cutoff,) * 3, (False,) * 3)``."""
    def query(self, query_points: ArrayF) -> Neighbors: ...
    def query_self(self) -> Neighbors: ...

class VerletSkin:
    """A ``NeighborList`` with Verlet skin, for the MD integrators.

    ``neighbors`` must be an engine built with cutoff ``cutoff + skin``; it is
    **moved** into this object, and passing the skin into an integrator moves
    it again (afterwards every access raises "has already been moved").
    """

    def __new__(
        cls,
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
    ) -> Self: ...
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

    def __new__(cls, data: _AbcMapping[ColumnKey, npt.ArrayLike] | None = None) -> Self:
        """Build from a mapping of column name -> array (each value goes through
        ``__setitem__``, so a canonical key adopts its schema dtype). Copying a
        block is :meth:`copy`; ``Block(block)`` raises ``TypeError``."""
    def insert(
        self,
        key: ColumnKey,
        values: npt.NDArray[Any] | Sequence[str],
        validity: ArrayBool | Sequence[bool] | None = None,
    ) -> None:
        """Store a column at the given width, optionally with a per-row
        validity mask (the write side of :meth:`validity`).

        Parameters
        ----------
        key : str | Key
            Column name.
        values : numpy.ndarray | Sequence[str]
            Column data.
        validity : numpy.ndarray | Sequence[bool] | None
            1-D bool mask, one entry per row of *values*; ``False`` marks a row
            that holds no value. An all-``True`` mask states nothing and is
            dropped.

        Raises
        ------
        TypeError
            If the array dtype is unsupported, or *validity* is neither a 1-D
            bool array nor a sequence of bools.
        ValueError
            If the row count does not match existing columns, or *validity*
            does not cover exactly the rows of *values*.
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
    def __getitem__(self, key: ColumnKey, /) -> npt.NDArray[Any]: ...
    @overload
    def __getitem__(
        self, key: tuple[ColumnKey, ...] | list[ColumnKey], /
    ) -> npt.NDArray[Any]: ...
    @overload
    def __getitem__(
        self, key: slice | npt.NDArray[np.bool_] | npt.NDArray[np.integer], /
    ) -> Block: ...
    def __setitem__(
        self,
        key: ColumnKey | tuple[ColumnKey, ...] | list[ColumnKey],
        value: npt.ArrayLike,
        /,
    ) -> None:
        """Store a column (schema dtype adopted; a scalar is refused), or spread
        an ``(N, k)`` array over ``k`` named columns."""
    def __delitem__(self, key: ColumnKey, /) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object, /) -> bool: ...
    @property
    def n_rows(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def structural_shape(self) -> list[int] | None: ...
    def resize(self, n_rows: int) -> None: ...
    def set_shape(self, shape: Sequence[int]) -> None: ...
    def keys(self) -> list[str]: ...
    def rename(self, old_key: ColumnKey, new_key: ColumnKey) -> None:
        """Rename in place; a canonical ``new_key`` adopts its schema dtype."""
    def sort(self, key: ColumnKey, reverse: bool = False) -> Block: ...
    def copy(self) -> Block:
        """Deep copy: no buffer shared with this block."""
    def dtype(self, key: ColumnKey) -> str:
        """The dtype name of a column (``"float"``, ``"int"``, ``"i64"``,
        ``"uint"``, ``"bool"``, ``"u8"``, ``"string"``, …). ``KeyError`` for
        an absent column."""
    def __reduce__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...

class MetaValue:
    """Exact-dtype frame metadata value."""

    def __new__(cls, dtype: str, value: Any) -> Self: ...
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

    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __getitem__(self, key: object, /) -> FrozenMetaValue: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[FrozenMetaValue]: ...
    def items(self) -> ItemsView[str, FrozenMetaValue]: ...
    def get(self, key: object, default: Any = None) -> Any: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
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

    def __getitem__(self, key: object, /) -> FrozenMetaValue: ...
    def __setitem__(self, key: str, value: Any, /) -> None: ...
    def __delitem__(self, key: object, /) -> None: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __eq__(self, other: object, /) -> bool: ...
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
    def __or__(self, other: Any, /) -> dict[str, Any]: ...
    def __ror__(self, other: Any, /) -> dict[str, Any]: ...
    def __ior__(self, other: Any, /) -> Self: ...

@final
class Frame:
    """Dictionary of Blocks with optional ``box`` and typed ``meta`` — the one Frame."""

    def __new__(
        cls,
        blocks: _AbcMapping[str, Block | _AbcMapping[ColumnKey, npt.ArrayLike]]
        | None = None,
        *,
        meta: _AbcMapping[str, Any] | None = None,
        box: Box | None = None,
    ) -> Self:
        """Copying a frame is :meth:`copy`; ``Frame(frame)`` raises ``TypeError``."""
    def __getitem__(self, key: str, /) -> Block:
        """A handle on the stored block: its members read and write this frame."""
    def __setitem__(
        self, key: str, value: Block | _AbcMapping[ColumnKey, npt.ArrayLike], /
    ) -> None: ...
    def __delitem__(self, key: str, /) -> None: ...
    def __contains__(self, key: str, /) -> bool: ...
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

@final
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

@final
class PublisherConfig:
    """How a :class:`Publisher` streams: wire format, buffer size, token.

    Native-only, like :class:`Publisher`. Raises ``ValueError`` for an unknown
    ``format`` or a ``buffer_size`` of 0.
    """

    def __new__(
        cls,
        *,
        format: Literal["msgpack", "json"] = "msgpack",
        buffer_size: int = 4,
        token: str | None = None,
    ) -> Self: ...
    @property
    def format(self) -> Literal["msgpack", "json"]: ...
    @property
    def buffer_size(self) -> int: ...
    @property
    def token(self) -> str | None: ...

@final
class Publisher:
    """WebSocket publisher streaming frames to viewers: :meth:`bind` listens
    (a server), :meth:`connect` dials a relay.

    Native-only: a Pyodide build compiles `molrs::stream::publisher` out, so this
    class is absent there.
    """

    @staticmethod
    def bind(address: str, config: PublisherConfig) -> Publisher:
        """Listen on ``"host:port"`` (port 0: the OS picks; read
        :attr:`address`). Raises ``OSError`` when it cannot bind."""
    @staticmethod
    def connect(url: str, config: PublisherConfig) -> Publisher:
        """Dial the relay at ``url``. Raises ``OSError`` when it cannot."""
    @property
    def address(self) -> str:
        """The bound ``"host:port"``, or the URL a connected publisher dialed."""
    @property
    def n_clients(self) -> int: ...
    def send(self, frame: Frame) -> None: ...
    def recv_command(self, timeout: float = 0.0) -> ControlCommand | None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

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

    def __new__(cls, vertices: ArrayF, faces: ArrayU32) -> Self: ...
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

    def __new__(cls, center: Sequence[float] | ArrayF, radius: float) -> Self: ...
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
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Cuboid:
    """Axis-aligned cuboid (box) region.

    A point is inside when `origin[d] <= p[d] <= origin[d] + lengths[d]` on
    every axis.
    """

    def __new__(
        cls, origin: Sequence[float] | ArrayF, lengths: Sequence[float] | ArrayF
    ) -> Self:
        """A cube of edge ``a`` is ``Cuboid(origin, [a, a, a])``."""
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
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Parallelepiped:
    """General parallelepiped (oblique box) geometric region.

    Pure containment — not a periodic ``Box``. Built from a :class:`Cell`
    (as a ``Box`` takes one; columns of the matrix are the three edge
    vectors) and the origin corner; fractional coordinates must lie in
    ``[0, 1]³``. ``distance`` is measured perpendicular to the bounding
    planes.
    """

    def __new__(cls, cell: Cell, origin: Sequence[float] | ArrayF) -> Self:
        """A cube of edge ``a`` is ``Parallelepiped(Cell.lengths([a, a, a]),
        origin)``. Raises ``ValueError`` for ``Cell.undefined()``, a refused
        cell or a singular edge matrix."""
    @property
    def h(self) -> ArrayF:
        """The ``(3, 3)`` edge matrix (Å), edges as columns."""
    @property
    def origin(self) -> ArrayF: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def volume(self) -> float: ...
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class HalfSpace:
    """One side of a plane: inside where ``normal · (x - point) <= 0``."""

    def __new__(
        cls, normal: Sequence[float] | ArrayF, point: Sequence[float] | ArrayF
    ) -> Self: ...
    def normal(self) -> ArrayF: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Cylinder:
    """Finite capped cylinder from ``base`` along ``axis``."""

    def __new__(
        cls,
        base: Sequence[float] | ArrayF,
        axis: Sequence[float] | ArrayF,
        radius: float,
        length: float,
    ) -> Self: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Ellipsoid:
    """Axis-aligned ellipsoid with semi-axes along x, y, z."""

    def __new__(
        cls, center: Sequence[float] | ArrayF, semi_axes: Sequence[float] | ArrayF
    ) -> Self: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Polyhedron:
    """Solid bounded by a watertight :class:`TriMesh`."""

    def __new__(cls, mesh: TriMesh) -> Self: ...
    def mesh(self) -> TriMesh: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class SphereUnion:
    """Union of spheres — atoms as a region; ``~SphereUnion`` is the void."""

    def __new__(
        cls, centers: ArrayF, radii: float | ArrayF, box: Box | None = None
    ) -> Self: ...
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
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
    def __invert__(self) -> Region: ...
    def _ffi_regionref_capsule(self) -> object: ...

class Region:
    """Composed geometric region (result of &, |, ~ operators)."""

    def __new__(cls, source: RegionLike) -> Self: ...
    @staticmethod
    def _from_tree(tree: tuple[Any, ...]) -> Region: ...
    def contains(self, points: ArrayF) -> ArrayBool: ...
    def distance(self, points: ArrayF) -> ArrayF: ...
    def bounds(self) -> ArrayF: ...
    def mask(self, block: Block) -> ArrayBool:
        """Which rows of ``block`` (its ``x``/``y``/``z``) lie inside."""
    def __call__(self, block: Block) -> Block:
        """``block[self.mask(block)]``."""
    def __and__(self, other: RegionLike, /) -> Region: ...
    def __or__(self, other: RegionLike, /) -> Region: ...
    def __rand__(self, value: RegionLike, /) -> Region: ...
    def __ror__(self, value: RegionLike, /) -> Region: ...
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

    def __new__(cls, points: ArrayF) -> Self: ...
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
    """Index-only bond graph: atoms ``0..n_atoms``, bonds index pairs; angles,
    dihedrals and impropers are derived. Deletion is swap-remove (the last
    atom / bond moves into the freed slot); batch deletes take indices from
    before the call."""

    def __new__(cls, n_atoms: int) -> Self:
        """``n_atoms`` unbonded atoms (``Topology(0)`` is the empty graph)."""
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
    def n_angles(self) -> int: ...
    @property
    def n_dihedrals(self) -> int: ...
    @property
    def n_components(self) -> int: ...
    def bonds(self) -> npt.NDArray[np.uint64]:
        """``(n_bonds, 2)``; row ``b`` is bond index ``b``."""
    def angles(self) -> npt.NDArray[np.uint64]:
        """``(n_angles, 3)`` unique ``[i, j, k]`` with ``i < k``."""
    def dihedrals(self) -> npt.NDArray[np.uint64]:
        """``(n_dihedrals, 4)`` unique ``[i, j, k, l]`` with ``j < k``."""
    def impropers(self) -> npt.NDArray[np.uint64]:
        """``(n, 4)`` ``[center, i, j, k]`` per sorted 3-combination of the
        neighbours of every atom of degree >= 3."""
    def neighbors(self, idx: int) -> list[int]:
        """``IndexError`` for a non-atom index."""
    def degree(self, idx: int) -> int:
        """``IndexError`` for a non-atom index."""
    def are_bonded(self, i: int, j: int) -> bool:
        """``IndexError`` for a non-atom index."""
    def connected_components(self) -> npt.NDArray[np.uint32]:
        """Per-atom connected-component label, ``0..n_components`` in order
        of each component's first atom."""
    def distances(self, source: int) -> npt.NDArray[np.int64]:
        """Bond hops from ``source`` to each atom; ``-1`` across components.
        ``IndexError`` for a non-atom index."""
    def add_atoms(self, n: int) -> None: ...
    def add_bonds(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Add each bond not already present (either orientation).
        ``IndexError`` for a non-atom endpoint (nothing added)."""
    def del_atom(self, idx: int) -> None:
        """Delete atom ``idx`` and its bonds. ``IndexError`` out of range."""
    def del_atoms(self, idxs: Sequence[int]) -> None:
        """Delete every atom of ``idxs`` and their bonds. ``IndexError`` out
        of range (nothing deleted)."""
    def del_bond(self, idx: int) -> None:
        """Delete bond ``idx``. ``IndexError`` out of range."""
    def del_bonds(self, idxs: Sequence[int]) -> None:
        """Delete every bond of ``idxs``. ``IndexError`` out of range
        (nothing deleted)."""

@final
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

@final
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
    def implied_type(self) -> BondOrder:
        """The class this number implies (``Quadruple`` → ``Double``)."""
    def count(self) -> int:
        """The number as a valence count; ``Unknown`` counts 0."""
    def __int__(self) -> int: ...

@final
class Element:
    """Immutable chemical-element record backed by the Rust periodic table.

    Not constructed directly: look one up with :meth:`by_symbol` or
    :meth:`by_number`. The frame-level species conversions are the explicit
    static steps :meth:`fill_atomic_numbers` / :meth:`fill_symbols`.
    """

    @staticmethod
    def by_symbol(symbol: str) -> Element:
        """The element whose symbol is ``symbol`` (case-insensitive).

        Raises
        ------
        ValueError
            ``symbol`` is no element's symbol (``"*"`` included).
        """
    @staticmethod
    def by_number(z: int) -> Element:
        """The element whose atomic number is ``z`` (``1..=118``).

        Raises
        ------
        TypeError
            ``z`` is a ``bool``.
        ValueError
            ``z`` is outside ``1..=118`` (``0``, the wildcard, names none).
        """
    @staticmethod
    def fill_atomic_numbers(frame: Frame) -> None:
        """Write ``atoms.atomic_number`` from the ``atoms.element`` symbols,
        in place (``"*"`` is 0).

        Raises
        ------
        ValueError
            no ``atoms`` block, no string ``element`` column, or a symbol
            that is no element (names the row).
        """
    @staticmethod
    def fill_symbols(frame: Frame) -> None:
        """Write ``atoms.element`` symbols from ``atoms.atomic_number``, in
        place (0 is ``"*"``).

        Raises
        ------
        ValueError
            no ``atoms`` block, no unsigned ``atomic_number`` column, or a
            number outside ``0..=118`` (names the row).
        """
    @property
    def z(self) -> int: ...
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
    def __reduce__(self) -> tuple[Callable[[int], Element], tuple[int]]: ...

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
    def __new__(
        cls,
        factor: float,
        offset: float,
        dimension: tuple[int, int, int, int, int, int, int],
        name: str,
    ) -> Self: ...
    @property
    def dimension(self) -> tuple[int, int, int, int, int, int, int]: ...
    def is_affine(self) -> bool: ...
    def factor_to(self, other: Unit) -> float: ...
    def __rmul__(self, value: float, /) -> Quantity: ...
    def __mul__(self, value: float, /) -> Quantity: ...

class Quantity:
    def __new__(cls, value: float, unit: Unit) -> Self: ...
    @property
    def value(self) -> float: ...
    @property
    def unit(self) -> Unit: ...
    def to(self, target: Unit | str) -> Quantity: ...
    def to_base_units(self) -> Quantity: ...
    def __add__(self, rhs: Quantity, /) -> Quantity: ...
    def __radd__(self, value: Quantity, /) -> Quantity: ...
    def __sub__(self, rhs: Quantity, /) -> Quantity: ...
    def __rsub__(self, value: Quantity, /) -> Quantity: ...
    def __mul__(self, rhs: Quantity | float, /) -> Quantity: ...
    def __rmul__(self, lhs: float, /) -> Quantity: ...
    def __truediv__(self, rhs: Quantity | float, /) -> Quantity: ...
    def __rtruediv__(self, value: float, /) -> Quantity: ...

class UnitPreset:
    def __new__(cls, name: str) -> Self: ...
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
    def __new__(
        cls,
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
    ) -> Self: ...
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
    def __new__(
        cls,
        name: str,
        q: float,
        mu: float,
        alpha: float,
        polarizable: bool = False,
    ) -> Self: ...
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
    """Domain-agnostic ECS *world* with ``int`` handles.

    Nodes and relations are stable opaque ``int`` handles (generational
    slotmap keys); deleting one never invalidates another and a stale handle
    raises. Node fields live in aligned columns addressed by convention keys
    (see :data:`keys`); ``column`` exposes a zero-copy numpy view. Topology is
    a kind-tagged relation API (``register_kind`` / ``add_relation`` / …).

    :class:`Atomistic` and :class:`CoarseGrain` are not ``MolGraph``\\ s: they
    hold their own typed graph and speak only its vocabulary.

    Pickles by content: a restored world has fresh handles in the same row
    order.
    """

    def __init__(self) -> None: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    # --- nodes ---
    def add_node(self, **props: PropValue) -> int:
        """Add a node carrying ``props``; returns its handle.

        Raises
        ------
        TypeError
            a value that is not a bool, int, float or str.
        ValueError
            a value contradicts its column's type; the world is unchanged.
        """
    def del_node(self, node: int) -> None:
        """Delete ``node`` and every relation touching it."""
    def del_nodes(self, nodes: Sequence[int]) -> None:
        """Delete every node of ``nodes`` and every relation touching one;
        atomic (nothing is deleted on an error)."""
    def get_node(self, node: int) -> dict[str, PropValue]:
        """Every field of ``node``, as a new dict."""
    def set_node(self, node: int, key: ColumnKey, value: PropValue) -> None: ...
    def clear_node(self, node: int, key: ColumnKey) -> None:
        """Clear one field (no-op if absent)."""
    def node_ids(self) -> list[int]:
        """All live node handles, in row order."""
    @property
    def n_nodes(self) -> int: ...
    # --- relations (generic, kind-tagged) ---
    def register_kind(self, kind: str, arity: int) -> None: ...
    def kinds(self) -> list[str]: ...
    def kind_arity(self, kind: str) -> int: ...
    def add_relation(
        self, kind: str, nodes: Sequence[int], /, **props: PropValue
    ) -> int:
        """Add a relation of ``kind`` over node handles, carrying ``props``;
        returns its handle.

        Raises
        ------
        TypeError
            a value that is not a bool, int, float or str.
        ValueError
            an unregistered kind, a wrong endpoint count, a stale endpoint,
            or a value that contradicts its column's type; the world is
            unchanged.
        """
    def get_relation(
        self, kind: str, rel: int
    ) -> tuple[list[int], dict[str, PropValue]]:
        """Relation ``rel`` of ``kind``: its endpoint handles and a new dict
        of its fields."""
    def incident_relations(self, node: int, kind: str) -> list[tuple[int, int]]: ...
    def set_relation_prop(
        self, kind: str, rel: int, key: str, value: PropValue
    ) -> None: ...
    def clear_relation_prop(self, kind: str, rel: int, key: str) -> None: ...
    def del_relation(self, kind: str, rel: int) -> None: ...
    def del_relations(self, kind: str, rels: Sequence[int]) -> None:
        """Delete every relation of ``rels``; atomic."""
    def n_relations(self, kind: str) -> int: ...
    def relation_ids(self, kind: str) -> list[int]: ...
    # --- columns ---
    def column(self, key: ColumnKey) -> npt.NDArray[Any]:
        """Column ``key`` over every node, typed as the field (``f64`` is a
        zero-copy write-through view; ``i32``/``bool``/``str`` are copies).
        ``KeyError`` when any node lacks the field — never zero-filled."""
    def columns(self) -> list[str]:
        """Names of every field column registered on the node table."""
    def validity(self, key: ColumnKey) -> ArrayBool: ...
    def adopt(self, other: MolGraph) -> None:
        """Move ``other``'s storage into this world (handles kept); ``other``
        is left empty."""
    # --- ports ---
    @property
    def n_ports(self) -> int: ...
    def link(self, a: int, b: int) -> int:
        """Join port ``a`` to port ``b``: delete both leaving groups, fold
        their charge (e) onto the anchors, bond the anchors with the port
        order. Returns the new bond handle.

        Raises
        ------
        ValueError
            a handle naming no live port, a stale or incompatible port pair,
            a shared anchor, anchors already bonded, or overlapping leaving
            groups; the world is unchanged.
        """
    def inherit_frag_ids(self) -> int: ...

class Atomistic:
    """All-atom molecular graph.

    ``Atomistic(**props)`` — the keywords are :attr:`props`. Every atom,
    bond, angle, dihedral, improper, port and term is a live view (an
    :class:`Atom`, a :class:`Bond`, …); a view's mapping is the one door to
    its fields (``atom["x"] = 1.0``, ``del bond["k"]``). Every atom carries a
    valid ``atomic_number`` (0..=118, 0 = wildcard): an add, a write or an
    unpickle that would break that raises. Copy with ``copy.copy`` /
    ``copy.deepcopy`` (handles preserved). Subclassable.
    """

    def __new__(cls, **props: Any) -> Self: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, _memo: dict[int, Any], /) -> Self: ...
    @property
    def props(self) -> dict[str, Any]:
        """Whole-graph annotations; carried by copies, pickles and every graph
        derived from this one."""
    # --- atoms ---
    @property
    def atoms(self) -> Refs[Atom]: ...
    @property
    def n_atoms(self) -> int: ...
    @overload
    def add_atom(self, *, kind: None = None, **props: PropValue) -> Atom: ...
    @overload
    def add_atom[S: VirtualSite](self, *, kind: type[S], **props: PropValue) -> S: ...
    def del_atom(self, atom: Atom) -> None:
        """Delete ``atom`` and every relation touching it."""
    def del_atoms(self, atoms: Iterable[Atom]) -> None:
        """Delete every atom of ``atoms`` and every relation touching one;
        atomic."""
    # --- bonds ---
    @property
    def bonds(self) -> Refs[Bond]: ...
    @property
    def n_bonds(self) -> int: ...
    def add_bond(self, a: Atom, b: Atom, /, **props: PropValue) -> Bond:
        """Add a bond; with no ``bond_type`` in ``props`` it is a single bond."""
    def del_bond(self, bond: Bond) -> None: ...
    def del_bonds(self, bonds: Iterable[Bond]) -> None: ...
    def set_bond_class(
        self, bond: Bond, bond_type: BondOrder, bond_number: BondNumber
    ) -> None:
        """Set a bond's chemical class and its localized bond number together
        — the one door that writes ``bond_type`` / ``bond_number``."""
    def bond_between(self, a: Atom, b: Atom) -> Bond | None: ...
    def neighbor_bonds(self, node: Atom) -> list[tuple[Atom, Bond]]: ...
    def topo_distances(
        self, atom: Atom, max_hops: int | None = None
    ) -> list[tuple[Atom, int]]: ...
    # --- angles / dihedrals / impropers ---
    @property
    def angles(self) -> Refs[Angle]: ...
    @property
    def n_angles(self) -> int: ...
    def add_angle(self, a: Atom, b: Atom, c: Atom, /, **props: PropValue) -> Angle: ...
    def del_angle(self, relation: Angle) -> None: ...
    def del_angles(self, relations: Iterable[Angle]) -> None: ...
    @property
    def dihedrals(self) -> Refs[Dihedral]: ...
    @property
    def n_dihedrals(self) -> int: ...
    def add_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **props: PropValue
    ) -> Dihedral: ...
    def del_dihedral(self, relation: Dihedral) -> None: ...
    def del_dihedrals(self, relations: Iterable[Dihedral]) -> None: ...
    @property
    def impropers(self) -> Refs[Improper]: ...
    @property
    def n_impropers(self) -> int: ...
    def add_improper(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **props: PropValue
    ) -> Improper: ...
    def del_improper(self, relation: Improper) -> None: ...
    def del_impropers(self, relations: Iterable[Improper]) -> None: ...
    def generate_topology(
        self,
        gen_angle: bool = True,
        gen_dihedral: bool = True,
        gen_improper: bool = False,
        clear_existing: bool = False,
    ) -> tuple[int, int, int]: ...
    # --- terms (every other relation block) ---
    def declare_term(self, block: str, arity: int) -> None:
        """Declare a term block the Frame schema does not define."""
    def add_term(
        self, block: str, atoms: Iterable[Atom], /, **props: PropValue
    ) -> RelationRef:
        """Add a row to the term block ``block`` (``cmaps``, ``exclusions``,
        …) over ``atoms``; a block with typed methods is refused."""
    def del_term(self, term: RelationRef) -> None: ...
    def del_terms(self, terms: Iterable[RelationRef]) -> None:
        """Delete terms of ONE block; atomic."""
    def terms(self, block: str) -> Refs[RelationRef]: ...
    def n_terms(self, block: str) -> int: ...
    def term_blocks(self) -> list[str]: ...
    # --- ports ---
    @property
    def ports(self) -> Refs[Port]: ...
    @property
    def n_ports(self) -> int: ...
    def add_port(
        self,
        anchor: Atom,
        handle_atom: Atom,
        /,
        kind: str,
        label: str = "",
        order: int = 1,
    ) -> Port:
        """Record a descriptor (``"$"``, ``"<"``, ``">"``, ``"!"``) on the
        ``(anchor, handle_atom)`` valence.

        Raises
        ------
        ValueError
            an endpoint of another graph, an unknown glyph,
            ``handle_atom`` not bonded to ``anchor``, an indefinite order,
            or a valence that already carries a port.
        """
    def del_port(self, port: Port) -> None: ...
    def del_ports(self, ports: Iterable[Port]) -> None: ...
    def port_on(self, anchor: Atom, handle_atom: Atom) -> Port | None: ...
    def leaving_group(self, port: Port) -> list[Atom]: ...
    def link(self, a: Port, b: Port) -> Bond:
        """Join port ``a`` to port ``b``: delete both leaving groups, fold
        their charge (e) onto the anchors, bond the anchors with the port
        order; returns the new bond. The graph is unchanged on a refusal
        (``ValueError``)."""
    def inherit_frag_ids(self) -> int: ...
    # --- geometry ---
    def translate(self, delta: Sequence[float] | ArrayF) -> None: ...
    def rotate(
        self,
        axis: Sequence[float] | ArrayF,
        angle: float,
        about: Sequence[float] | ArrayF | None = None,
    ) -> None: ...
    def scale(
        self,
        factor: Sequence[float] | ArrayF,
        about: Sequence[float] | ArrayF | None = None,
    ) -> None: ...
    def center(self, nodes: Iterable[Atom]) -> ArrayF:
        """Mass-weighted centre (Å) of ``nodes``, a float64 ``(3,)`` array."""
    # --- composition ---
    def replicate(
        self,
        template: Self,
        rotations: ArrayF,
        translations: ArrayF,
        frag_ids: ArrayI32 | Sequence[int],
    ) -> Refs[Atom]:
        """Grow this graph by one rigid copy of ``template`` per transform;
        returns the new atoms, copy-major. Atomic."""
    def merge(self, other: Self) -> Refs[Atom]:
        """Merge ``other`` in (emptying it); returns its atoms, now here."""
    def induced_subgraph(
        self, nodes: Iterable[Atom]
    ) -> tuple[Self, dict[Atom, Atom]]: ...
    def extract_subgraph(
        self,
        centers: Iterable[Atom],
        radius: int,
        *,
        regenerate_topology: bool = False,
        max_ring_size: int | None = None,
    ) -> ExtractedSubgraph: ...
    # --- structural graph hash (Weisfeiler-Lehman) ---
    def structural_hash(self) -> int: ...
    def canonical_order(self) -> Refs[Atom]: ...
    def is_isomorphic(self, other: Self) -> bool: ...
    # --- frames ---
    def to_frame(self) -> Frame: ...
    @staticmethod
    def from_frame(frame: Frame) -> Atomistic: ...

class ExtractedSubgraph:
    """Result of :meth:`Atomistic.extract_subgraph` / :meth:`CoarseGrain.extract_subgraph`.

    Nodes are views: a parent node is a view of the graph the ball was cut
    from, a new node one of :attr:`graph`.
    """

    def __new__(
        cls,
        graph: Atomistic | CoarseGrain,
        boundary: list[Any],
        parent_of: dict[Any, Any],
        hops: dict[Any, int],
        node_map: dict[Any, Any],
    ) -> Self: ...
    @property
    def graph(self) -> Atomistic | CoarseGrain: ...
    @property
    def boundary(self) -> list[NodeRef]: ...
    @property
    def parent_of(self) -> dict[NodeRef, NodeRef]: ...
    @property
    def hops(self) -> dict[NodeRef, int]: ...
    @property
    def node_map(self) -> dict[NodeRef, NodeRef]: ...

class CoarseGrain:
    """Coarse-grained molecular graph.

    ``CoarseGrain(**props)`` — the keywords are :attr:`props`. Every bead
    carries a string ``bead_type``. A bead's membership
    (:meth:`set_bead_members`) names nodes of one source graph, the member
    world. Copy with ``copy.copy`` / ``copy.deepcopy``. Subclassable.
    """

    def __new__(cls, **props: Any) -> Self: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, _memo: dict[int, Any], /) -> Self: ...
    @property
    def props(self) -> dict[str, Any]: ...
    # --- beads ---
    @property
    def beads(self) -> Refs[Bead]: ...
    @property
    def n_beads(self) -> int: ...
    def add_bead(self, **props: PropValue) -> Bead: ...
    def del_bead(self, bead: Bead) -> None: ...
    def del_beads(self, beads: Iterable[Bead]) -> None: ...
    def set_bead_members(self, bead: Bead, members: Iterable[NodeRef]) -> None: ...
    def bead_members(self, bead: Bead) -> tuple[NodeRef, ...]: ...
    def beads_of_atom(self, node: NodeRef) -> list[Bead]: ...
    def positions(self, beads: Iterable[Bead]) -> ArrayF: ...
    def axes(self, beads: Iterable[Bead]) -> ArrayF: ...
    def bead_types(self, beads: Iterable[Bead]) -> list[str]: ...
    # --- bonds ---
    @property
    def bonds(self) -> Refs[CgBond]: ...
    @property
    def n_bonds(self) -> int: ...
    def add_bond(self, a: Bead, b: Bead, /, **props: PropValue) -> CgBond: ...
    def del_bond(self, bond: CgBond) -> None: ...
    def del_bonds(self, bonds: Iterable[CgBond]) -> None: ...
    def neighbor_bonds(self, node: Bead) -> list[tuple[Bead, CgBond]]: ...
    # --- ports ---
    @property
    def ports(self) -> Refs[Port]: ...
    @property
    def n_ports(self) -> int: ...
    def add_port(
        self,
        anchor: Bead,
        handle_atom: Bead,
        /,
        kind: str,
        label: str = "",
        order: int = 1,
    ) -> Port: ...
    def del_port(self, port: Port) -> None: ...
    def del_ports(self, ports: Iterable[Port]) -> None: ...
    def port_on(self, anchor: Bead, handle_atom: Bead) -> Port | None: ...
    def leaving_group(self, port: Port) -> list[Bead]: ...
    def link(self, a: Port, b: Port) -> CgBond: ...
    def inherit_frag_ids(self) -> int: ...
    # --- geometry ---
    def translate(self, delta: Sequence[float] | ArrayF) -> None: ...
    def rotate(
        self,
        axis: Sequence[float] | ArrayF,
        angle: float,
        about: Sequence[float] | ArrayF | None = None,
    ) -> None: ...
    def scale(
        self,
        factor: Sequence[float] | ArrayF,
        about: Sequence[float] | ArrayF | None = None,
    ) -> None: ...
    def center(self, nodes: Iterable[Bead]) -> ArrayF: ...
    # --- composition ---
    def replicate(
        self,
        template: Self,
        rotations: ArrayF,
        translations: ArrayF,
        frag_ids: ArrayI32 | Sequence[int],
    ) -> Refs[Bead]: ...
    def merge(self, other: Self) -> Refs[Bead]: ...
    def induced_subgraph(
        self, nodes: Iterable[Bead]
    ) -> tuple[Self, dict[Bead, Bead]]: ...
    def extract_subgraph(
        self, centers: Iterable[Bead], radius: int
    ) -> ExtractedSubgraph: ...
    # --- structural graph hash (Weisfeiler-Lehman) ---
    def structural_hash(self) -> int: ...
    def canonical_order(self) -> Refs[Bead]: ...
    def is_isomorphic(self, other: Self) -> bool: ...
    # --- frames ---
    def to_frame(self) -> Frame: ...
    @staticmethod
    def from_frame(frame: Frame) -> CoarseGrain: ...

_TRef = TypeVar("_TRef")

class NodeRef:
    """Live view of one node: its fields as a mapping — the one door to read
    (``view[key]``), write (``view[key] = value``) and clear
    (``del view[key]``) them. Made by the graph, never constructed directly;
    interned while alive. A write that would break the graph's invariant (an
    atom's ``atomic_number``, a bead's ``bead_type``) raises ``ValueError``;
    ``None`` is no value (``TypeError``)."""

    @property
    def world(self) -> Atomistic | CoarseGrain: ...
    @property
    def handle(self) -> int: ...
    @overload
    def __getitem__(self, key: ColumnKey, /) -> PropValue: ...
    @overload
    def __getitem__(self, key: tuple[ColumnKey, ...], /) -> list[PropValue]: ...
    @overload
    def __setitem__(self, key: ColumnKey, value: PropValue, /) -> None: ...
    @overload
    def __setitem__(
        self, key: tuple[ColumnKey, ...], value: Iterable[PropValue], /
    ) -> None: ...
    def __delitem__(self, key: ColumnKey, /) -> None: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[PropValue]: ...
    def items(self) -> list[tuple[str, PropValue]]: ...
    @overload
    def get(self, key: ColumnKey) -> PropValue | None: ...
    @overload
    def get[T](self, key: ColumnKey, default: T) -> PropValue | T: ...
    def update(
        self,
        *args: _AbcMapping[str, PropValue] | Iterable[tuple[str, PropValue]],
        **kwargs: PropValue,
    ) -> None: ...
    @classmethod
    def _restore(cls, world: Atomistic | CoarseGrain, row: int) -> NodeRef: ...

class Atom(NodeRef): ...
class VirtualSite(Atom): ...
class DrudeParticle(VirtualSite): ...
class MasslessSite(VirtualSite): ...
class Bead(NodeRef): ...

class RelationRef:
    """Live view of one relation (a bond, an angle, a port, a term, …):
    interned endpoint views plus its fields as a mapping, the one door to
    them. Made by the graph, never constructed directly. ``bond_type`` /
    ``bond_number`` are written only by :meth:`Atomistic.set_bond_class`; a
    port's fields are fixed when it is added."""

    @property
    def world(self) -> Atomistic | CoarseGrain: ...
    @property
    def kind(self) -> str: ...
    @property
    def handle(self) -> int: ...
    @property
    def endpoints(self) -> tuple[NodeRef, ...]: ...
    @overload
    def __getitem__(self, key: ColumnKey, /) -> PropValue: ...
    @overload
    def __getitem__(self, key: tuple[ColumnKey, ...], /) -> list[PropValue]: ...
    @overload
    def __setitem__(self, key: ColumnKey, value: PropValue, /) -> None: ...
    @overload
    def __setitem__(
        self, key: tuple[ColumnKey, ...], value: Iterable[PropValue], /
    ) -> None: ...
    def __delitem__(self, key: ColumnKey, /) -> None: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[PropValue]: ...
    def items(self) -> list[tuple[str, PropValue]]: ...
    @overload
    def get(self, key: ColumnKey) -> PropValue | None: ...
    @overload
    def get[T](self, key: ColumnKey, default: T) -> PropValue | T: ...
    def update(
        self,
        *args: _AbcMapping[str, PropValue] | Iterable[tuple[str, PropValue]],
        **kwargs: PropValue,
    ) -> None: ...
    @classmethod
    def _restore(
        cls, world: Atomistic | CoarseGrain, kind: str, row: int
    ) -> RelationRef: ...

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
    def anchor(self) -> NodeRef: ...
    @property
    def handle_atom(self) -> NodeRef: ...

class Refs[TRef]:
    """Ordered views of one kind of one graph; ``refs["x"]`` is a column.
    Every door that takes several nodes or relations takes a ``Refs`` too."""

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> TRef: ...
    @overload
    def __getitem__(self, key: slice, /) -> Refs[TRef]: ...
    @overload
    def __getitem__(self, key: str | tuple[str, ...], /) -> npt.NDArray[Any]: ...
    def __iter__(self) -> Iterator[TRef]: ...
    def __contains__(self, item: object, /) -> bool: ...
    @classmethod
    def _restore(
        cls, world: Atomistic | CoarseGrain, kind: str | None, rows: list[int]
    ) -> Refs[Any]: ...

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
    """One SMARTS embedding: the molecule's atoms in query-atom order, and the
    Daylight atom-map projection (``{map_number: Atom}``, empty without map
    labels)."""

    @property
    def atoms(self) -> list[Atom]: ...
    @property
    def mapping(self) -> dict[int, Atom]: ...

class SmartsPattern:
    """Compiled, atom-map-aware SMARTS query over an :class:`Atomistic`.

    Wraps the core Rust SMARTS engine (non-uniquified, RDKit
    ``uniquify=False``). Daylight atom maps (``[C:1]``) add no match
    constraint; a match's :attr:`SmartsMatch.mapping` is its
    ``{map_number: Atom}`` dict.
    """

    def __new__(cls, smarts: str) -> Self: ...
    def has_match(
        self,
        mol: Atomistic,
        *,
        labels: dict[Atom, str] | None = None,
        root: Atom | None = None,
    ) -> bool: ...
    def find_matches(
        self,
        mol: Atomistic,
        *,
        labels: dict[Atom, str] | None = None,
        root: Atom | None = None,
        limit: int | None = None,
    ) -> list[SmartsMatch]:
        """All matches. ``labels`` supplies the ``%LABEL`` context, ``root``
        pins query atom 0, ``limit`` stops after N embeddings.

        Raises
        ------
        ValueError
            A ``labels`` key or ``root`` is an atom of another molecule.
        """
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
        center: Atom,
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

    def __new__(cls, reaction_smarts: str) -> Self: ...
    @property
    def reactant_patterns(self) -> list[SmartsPattern]: ...
    @property
    def forming_bonds(self) -> list[tuple[int, int]]: ...
    def apply(
        self,
        mol: Atomistic,
        binding: dict[int, Atom],
        labels: dict[Atom, str] | None = None,
        refresh: bool = True,
    ) -> list[Atom]:
        """Apply the transform to ``mol`` in place at the occurrence
        ``binding`` (``{map_number: Atom}``) pins; return the surviving
        touched atoms."""
    def apply_many(
        self,
        mol: Atomistic,
        bindings: list[dict[int, Atom]],
        labels: dict[Atom, str] | None = None,
        refresh: bool = True,
    ) -> ReactionBatch:
        """Compile every binding against the intact graph, then apply the
        disjoint transforms as one batch, in place.

        Raises
        ------
        ValueError
            Two bindings resolve the same leaving atom, a binding misses an
            atom map, or an edit is refused.
        """

@final
class ReactionBatch:
    """The result of :meth:`Reaction.apply_many`, one entry per binding."""

    @property
    def touched(self) -> list[list[Atom]]:
        """The surviving touched atoms, as :meth:`Reaction.apply` returns
        them."""
    @property
    def created(self) -> list[list[Atom]]:
        """The atoms the product added, in product creation order."""

@final
class SubgraphMatcher:
    """``molrs.perceive.SubgraphMatcher`` — bead-pattern occurrences in a
    :class:`CoarseGrain`. Beads match on equal ``bead_type``. ``find`` does
    not partition: overlapping groups are all returned. Frozen.
    """

    def __new__(cls, pattern: CoarseGrain) -> Self:
        """Snapshot ``pattern`` (copied; later edits do not affect it).

        Raises
        ------
        TypeError
            ``pattern`` is not a :class:`CoarseGrain`.
        """
    def find(self, target: CoarseGrain) -> list[list[Bead]]:
        """Every induced occurrence, one group of the target's beads per
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

    def __new__(cls, source: CoarseGrain | Atomistic) -> Self: ...
    def coarsen(
        self, groups: Sequence[Iterable[NodeRef]], names: Sequence[str]
    ) -> CoarseGrain:
        """A new :class:`CoarseGrain` with one site per group, in group
        order; each site's :meth:`~CoarseGrain.bead_members` are its group.
        Releases the GIL.

        Parameters
        ----------
        groups : Sequence[Iterable[Atom | Bead]]
            Disjoint, non-empty node groups of the source (views or ``Refs``).
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
            node belongs to another graph or is listed twice, or a group has
            no centre (a missing coordinate or mass, a non-positive total
            mass); nodes are named by their handle.
        """

# ---------------------------------------------------------------------------
# Chemical perception — the builder (graph in / graph out, non-mutating)
# ---------------------------------------------------------------------------

@final
class BondOrderSource:
    """``molrs::perceive::BondOrderSource``: where BCC bond types take bond
    orders from — ``Stated`` (the molecule's own) or ``Connectivity`` (judged
    from the connectivity alone, antechamber's ``bondtype -j full``)."""

    Stated: ClassVar[BondOrderSource]
    Connectivity: ClassVar[BondOrderSource]
    def __int__(self) -> int: ...

@final
class UnknownBondPolicy:
    """``molrs::perceive::UnknownBondPolicy``: what a bond with no
    ``bond_type`` counts as for rotatable-bond perception — ``NotRotatable``
    (never guess) or ``AsSingle``."""

    NotRotatable: ClassVar[UnknownBondPolicy]
    AsSingle: ClassVar[UnknownBondPolicy]
    def __int__(self) -> int: ...

@final
class RotatableBond:
    """One rotatable bond ``j``–``k`` with the atoms on its ``k`` side
    (``molrs::perceive::RotatableBond``, as the molecule's atom views)."""

    @property
    def j(self) -> Atom: ...
    @property
    def k(self) -> Atom: ...
    @property
    def downstream(self) -> list[Atom]: ...

def assign_rings(mol: Atomistic) -> None:
    """Write ``is_in_ring`` / ``n_rings`` onto every atom and bond of ``mol``,
    in place."""

def assign_aromaticity(mol: Atomistic) -> None:
    """Bring ``mol`` to the standard aromatic representation, in place:
    ``is_aromatic`` on atoms, ``bond_type`` / ``bond_number`` on bonds."""

def add_hydrogens(mol: Atomistic) -> Atomistic:
    """New graph: ``mol`` plus the hydrogens its open valences imply; ``mol``
    is untouched."""

def assign_stereo(mol: Atomistic) -> None:
    """Write ``stereo`` onto each perceived stereocentre / bond of ``mol``,
    in place."""

def perceive_rotatable_bonds(
    mol: Atomistic,
    *,
    unknown_bond: UnknownBondPolicy = ...,
) -> list[RotatableBond]:
    """The rotatable bonds of ``mol`` (single, acyclic, both endpoints of
    degree > 1); ``mol`` is untouched. ``unknown_bond`` defaults to
    ``UnknownBondPolicy.NotRotatable``."""

def assign_rotatable_bonds(
    mol: Atomistic,
    *,
    unknown_bond: UnknownBondPolicy = ...,
) -> None:
    """Write ``is_rotatable`` (0/1) onto every bond of ``mol``, in place."""

def assign_bond_orders(mol: Atomistic) -> None:
    """Write antechamber's Kekulé structure, judged from the connectivity
    (``bond_number`` / ``bond_type``), onto ``mol``, in place."""

def assign_kekule_bond_orders(mol: Atomistic) -> bool:
    """Give every aromatic bond of ``mol`` a legal localized ``bond_number``,
    in place; ``False`` (and nothing changed) when no legal assignment
    exists."""

def assign_bcc_bond_types(mol: Atomistic, source: BondOrderSource) -> None:
    """Write ``bcc_bond_type`` onto every bond of ``mol``, in place, from the
    bond orders ``source`` names."""

def assign_equivalence_classes(mol: Atomistic) -> None:
    """Write ``equiv_class`` onto every atom of ``mol`` (antechamber
    ``-eq 1``), in place."""

def perceive_rings(mol: Atomistic) -> RingSet:
    """The SSSR rings of ``mol`` as a :class:`RingSet`; ``mol`` is untouched."""

# ---------------------------------------------------------------------------
# Frame vocabulary (molrs.core.keys / molrs.core.schema, mirrors
# molrs::core::keys / molrs::core::schema)
# ---------------------------------------------------------------------------

class keys:
    """``molrs.core.keys``: the canonical column and frame-meta names,
    projected from the Rust key tables; ordered groups are lists."""

    H_COUNT: keys.Key
    ISOTOPE: keys.Key
    IS_AROMATIC: keys.Key
    STEREO: keys.Key
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
    SPACE_GROUP: Key
    STYLE: Key
    SYMOPS: Key
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
    LAMMPS_TYPE_MASSES: Key

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

@final
class SmilesIr:
    """Intermediate representation of a parsed SMILES string.

    :meth:`parse` reads SMILES text, :meth:`from_atomistic` builds the IR of a
    molecule and ``str(ir)`` writes it back as text. A fragment body with
    bonding descriptors is :meth:`from_fragment`'s, a :class:`FragmentIr`.
    """

    @staticmethod
    def parse(smiles: str) -> SmilesIr:
        """Parse SMILES text; ``SmilesError`` for text that is not SMILES."""
    @staticmethod
    def from_fragment(body: str) -> FragmentIr:
        """Parse a fragment body (SMILES plus bonding descriptors, e.g.
        ``"[<]OCC[>]"``)."""
    @staticmethod
    def from_atomistic(
        mol: Atomistic,
        *,
        canonical: bool = True,
        root: Atom | None = None,
        aromatic: Literal["as_marked", "kekule_only"] = "as_marked",
        hydrogens: Literal[
            "organic_subset", "explicit_all", "as_stored"
        ] = "organic_subset",
        include_stereo: bool = False,
        multi_component: Literal[
            "error_if_multiple", "join_dot", "first_only"
        ] = "error_if_multiple",
        organic_subset: bool = True,
    ) -> SmilesIr:
        """The IR of ``mol``: ``str(SmilesIr.from_atomistic(mol))`` writes it
        as SMILES."""
    @property
    def n_components(self) -> int: ...
    def to_atomistic(self) -> Atomistic:
        """Connectivity only: no implicit hydrogens added, no coordinates."""
    def components(self) -> list[Atomistic]: ...

@final
class FragmentIr:
    """A SMILES fragment body — SMILES plus bonding descriptors — as
    :meth:`SmilesIr.from_fragment` parses it and a CGsmiles fragment table
    holds it; ``str(body)`` writes it back."""

    def to_atomistic(self) -> tuple[Atomistic, list[tuple[Atom, BondingDescriptor]]]:
        """The body's graph, and each bonding descriptor beside the atom it
        binds to (atom-visit order)."""
    def to_template(self) -> Atomistic:
        """The ported unit of this body: heavy atoms plus one hydrogen handle
        and one port per bonding descriptor; no coordinates, no ``frag_id``.
        ``SmilesIr.from_fragment("[<]OCC[>]").to_template()`` equals
        ``CgSmilesIr.parse("{[#EO]}.{#EO=[<]OCC[>]}").templates()["EO"]``."""

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

@final
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

@final
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

@final
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

@final
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

@final
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

@final
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

@final
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

@final
class CgSmilesIr:
    """Intermediate representation of a parsed CGsmiles string.

    Constructing it parses, validates, expands and resolves the whole string;
    the value is then read, not built. ``levels`` are the resolution levels
    (coarsest first), ``fragments[k]`` resolves the names of ``levels[k]``, and
    ``to_atomistic()`` expands the lowest level into a topology-only graph
    whose atoms carry ``frag_id``.
    """

    @staticmethod
    def parse(text: str) -> CgSmilesIr: ...
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
# I/O — readers and writers (molrs.io: one function per format and
# direction; each multi-frame format's classes in its submodule)
#
# A source is a file (os.PathLike), the bytes of a whole file, or a `Text`;
# a plain `str` is refused (ambiguous between a path and text). A sink is a
# file (os.PathLike) or a bytearray (cleared, then holding the file's bytes).
# ---------------------------------------------------------------------------

type _IoSource = os.PathLike[str] | bytes | Text
type _IoSink = os.PathLike[str] | bytearray

@final
class Text:
    """The text of a whole file, for a ``molrs.io`` reader to read."""

    def __new__(cls, text: str) -> Self: ...

def read_pdb(source: _IoSource) -> Frame: ...
def write_pdb(sink: _IoSink, frame: Frame) -> None: ...
def read_xyz(source: _IoSource) -> Frame: ...
def write_xyz(sink: _IoSink, frame: Frame) -> None: ...
def read_gro(source: _IoSource) -> Frame: ...
def write_gro(sink: _IoSink, frame: Frame) -> None: ...
def read_mol2(source: _IoSource) -> Frame: ...
def write_mol2(sink: _IoSink, frame: Frame) -> None: ...
def read_cif(source: _IoSource) -> Frame: ...
def write_cif(sink: _IoSink, frame: Frame) -> None: ...
def read_sdf(source: _IoSource) -> Frame: ...
def read_xsf(source: _IoSource) -> Frame: ...
def write_xsf(sink: _IoSink, frame: Frame) -> None: ...
def read_cube(source: _IoSource) -> Frame: ...
def write_cube(sink: _IoSink, frame: Frame) -> None: ...
def read_poscar(source: _IoSource) -> Frame: ...
def write_poscar(sink: _IoSink, frame: Frame) -> None: ...
def read_chgcar(source: _IoSource) -> Frame: ...
def read_ac(source: _IoSource) -> Frame: ...
def read_prmtop(source: _IoSource) -> Frame: ...
def read_inpcrd(source: _IoSource, frame: Frame | None = None) -> Frame:
    """The coordinates of an AMBER inpcrd / restrt; with ``frame``, merged into
    it in place and ``frame`` returned."""

def read_prep(source: _IoSource) -> dict[str, Any]: ...
def write_prep(sink: _IoSink, residue: _AbcMapping[str, Any]) -> None: ...
def read_lammps_data(source: _IoSource, atom_style: str | None = None) -> Frame:
    """Read a LAMMPS data file. Typed blocks carry ``type_id`` and the string
    ``type``; ``atom_style`` fixes the ``Atoms`` layout as LAMMPS's
    ``atom_style`` does."""

def write_lammps_data(
    sink: _IoSink,
    frame: Frame,
    *,
    type_labels: _AbcMapping[str, Sequence[str]] | None = None,
    atom_style: str | None = None,
    units: str | None = None,
) -> None: ...
def read_lammps_dump(source: _IoSource) -> Frame: ...
def write_lammps_dump(
    sink: _IoSink, frame: Frame, columns: Sequence[str] | None = None
) -> None: ...
def write_lammps_dump_local(sink: _IoSink, frames: Sequence[Frame]) -> None: ...
def read_lammps_molecule(source: _IoSource) -> Frame: ...
def write_lammps_molecule(sink: _IoSink, frame: Frame) -> None: ...
def read_lammps_molecule_json(source: _IoSource) -> Frame: ...
def write_lammps_molecule_json(sink: _IoSink, frame: Frame) -> None: ...
def read_dcd(source: _IoSource) -> Frame: ...
def write_dcd(sink: _IoSink, frame: Frame) -> None: ...
def read_trr(source: _IoSource) -> Frame: ...
def write_trr(sink: _IoSink, frame: Frame) -> None: ...
def read_xtc(source: _IoSource) -> Frame: ...
def write_xtc(sink: _IoSink, frame: Frame) -> None: ...
def read_stl(source: _IoSource) -> TriMesh: ...
def read_csv_block(
    source: _IoSource, delimiter: str = ",", header: list[str] | None = None
) -> Block:
    """CSV as a :class:`Block`; dtypes inferred int → float → str."""

def write_csv_block(
    sink: _IoSink, block: Block, delimiter: str = ",", header: bool = True
) -> None: ...
def read_clpol_alpha(source: _IoSource) -> list[dict[str, str | float]]:
    """``alpha.ff`` rows in file order: ``type_name``, ``m_D``,
    ``q_D_sign``, ``k_D``, ``alpha``, ``a_thole``."""

def read_wire_frame(data: bytes, encoding: Literal["msgpack", "json"]) -> Frame: ...
def write_wire_frame(frame: Frame, encoding: Literal["msgpack", "json"]) -> bytes: ...
def read_lammps_log(source: _IoSource, style: str = "default") -> LammpsLog:
    """Read a LAMMPS log into a structured ``LammpsLog``."""

def is_lammps_log(text: str) -> bool:
    """Whether ``text`` holds a LAMMPS run (a ``Per MPI rank memory allocation`` line)."""

def write_lammps_bond_react_map(sink: _IoSink, template: BondReactTemplate) -> None:
    """The ``fix bond/react`` map file of ``template``. Raises ``ValueError``
    for a malformed template."""

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

def write_mrec(
    sink: _IoSink,
    *,
    meta: _AbcMapping[str, Any] | None = None,
    frame: Frame | None = None,
    system: Frame | None = None,
    trajectory: Trajectory | None = None,
    forcefield: ForceField | mrec.ForceFieldSection | None = None,
) -> None:
    """Write a record: ``meta`` and whichever state sections are given (a
    ``bytearray`` receives the packed record)."""

@final
class BondReactTemplate:
    """One ``fix bond/react`` reaction: ``pre`` / ``post`` templates
    (``Atomistic`` or ``Frame``, atoms paired by an integer ``react_id``) and
    the atoms the map file names (atoms of ``pre``, or their ``react_id``)."""

    pre: Any
    post: Any
    initiator_atoms: Any
    edge_atoms: Any
    deleted_atoms: Any
    def __new__(
        cls,
        pre: Any,
        post: Any,
        initiator_atoms: Sequence[Any],
        edge_atoms: Sequence[Any] | None = None,
        deleted_atoms: Sequence[Any] | None = None,
    ) -> Self: ...

@final
class PdbReader:
    """Lazy, indexed reader of every frame of a PDB, one frame per ``MODEL`` file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class XyzReader:
    """Lazy, indexed reader of every frame of a (extended) XYZ file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class GroReader:
    """Lazy, indexed reader of every frame of a GROMACS GRO file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class Mol2Reader:
    """Lazy, indexed reader of every frame of a Tripos MOL2, one frame per molecule file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class CifReader:
    """Lazy, indexed reader of every frame of a CIF, one frame per ``data_`` block file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class SdfReader:
    """Lazy, indexed reader of every frame of a MDL SDF, one frame per record file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class CubeReader:
    """Lazy, indexed reader of every frame of a Gaussian cube, cubes back to back file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class LammpsDumpReader:
    """Lazy, indexed reader of every frame of a LAMMPS dump file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class DcdReader:
    """Lazy, indexed reader of every frame of a DCD file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class TrrReader:
    """Lazy, indexed reader of every frame of a GROMACS TRR file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class XtcReader:
    """Lazy, indexed reader of every frame of a GROMACS XTC file."""

    def __new__(cls, source: _IoSource) -> Self: ...
    def build_index(self) -> None: ...
    def close(self) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Frame]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> Frame: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class PdbWriter:
    """Writer of a PDB file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class XyzWriter:
    """Writer of a (extended) XYZ file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class GroWriter:
    """Writer of a GROMACS GRO file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class Mol2Writer:
    """Writer of a Tripos MOL2 file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class CifWriter:
    """Writer of a CIF file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class LammpsDumpWriter:
    """Writer of a LAMMPS dump file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def set_columns(self, columns: Sequence[str]) -> None:
        """The ``dump custom`` column line of the snapshots written next."""
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class DcdWriter:
    """Writer of a DCD file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class TrrWriter:
    """Writer of a GROMACS TRR file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class XtcWriter:
    """Writer of a GROMACS XTC file, one frame per :meth:`write`."""

    def __new__(cls, sink: _IoSink) -> Self: ...
    def write(self, frame: Frame) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None = None,
        _exc_value: BaseException | None = None,
        _traceback: TracebackType | None = None,
    ) -> bool: ...

@final
class LammpsForcefieldReader:
    """Reader of LAMMPS force-field files: the include (``*.ff``), a data
    file's ``* Coeffs`` sections, the ``fix cmap`` file."""

    def __new__(cls) -> Self: ...
    def read(self, source: _IoSource) -> ForceField: ...
    def read_data_coeffs(
        self, frame: Frame, units: str | None = None
    ) -> ForceField: ...
    def read_cmap(self, source: _IoSource) -> ForceField: ...

@final
class LammpsForcefieldWriter:
    """Writer of LAMMPS force-field files for the typed ``frame``, whose type
    labels number the rows."""

    def __new__(
        cls,
        frame: Frame,
        *,
        precision: int = 6,
        units: str = "real",
        skip_pair_style: bool = False,
        skip_special_bonds: bool = False,
        skip_units: bool = False,
        cmap_file: str | None = None,
    ) -> Self: ...
    def write(self, sink: _IoSink, forcefield: ForceField) -> None: ...
    def write_data_coeffs(self, sink: _IoSink, forcefield: ForceField) -> None: ...
    def write_cmap(self, sink: _IoSink, forcefield: ForceField) -> None: ...

@final
class GromacsTopForcefieldReader:
    """Reader of GROMACS topologies: the force-field directives (``read``) or
    the whole system (``read_system``)."""

    def __new__(
        cls,
        *,
        include: bool = False,
        include_dirs: Sequence[os.PathLike[str] | str] = ...,
        skip_directives: Sequence[str] = ...,
    ) -> Self: ...
    def read(self, source: _IoSource) -> ForceField: ...
    def read_system(self, source: _IoSource) -> tuple[ForceField, Frame]: ...

@final
class GromacsTopForcefieldWriter:
    """Writer of GROMACS topologies."""

    def __new__(cls, *, precision: int = 6) -> Self: ...
    def write(self, sink: _IoSink, forcefield: ForceField) -> None: ...
    def write_system(
        self, sink: _IoSink, forcefield: ForceField, frame: Frame
    ) -> None: ...

@final
class AmberPrmtopForcefieldReader:
    """Reader of an AMBER prmtop's force field (``read``), or force field and
    structure (``read_system``)."""

    def __new__(cls) -> Self: ...
    def read(self, source: _IoSource) -> ForceField: ...
    def read_system(self, source: _IoSource) -> tuple[ForceField, Frame]: ...

@final
class AmberFrcmodWriter:
    """Writer of an AMBER ``frcmod`` parameter file."""

    def __new__(cls) -> Self: ...
    def write(self, sink: _IoSink, forcefield: ForceField) -> None: ...

@final
class OpenmmXmlReader:
    """Reader of OpenMM force-field XML into the force-field IR."""

    def __new__(cls) -> Self: ...
    def read(self, source: _IoSource) -> ForceField: ...

@final
class OpenmmXmlWriter:
    """Writer of OpenMM force-field XML."""

    def __new__(cls, *, precision: int | None = None) -> Self: ...
    def write(self, sink: _IoSink, forcefield: ForceField) -> None: ...
    def write_system(
        self, sink: _IoSink, forcefield: ForceField, frame: Frame
    ) -> None: ...

@final
class XmlForcefieldReader:
    """Reader of molrs's force-field XML (one element per style)."""

    def __new__(cls) -> Self: ...
    def read(self, source: _IoSource) -> ForceField: ...

@final
class XmlForcefieldWriter:
    """Writer of molrs's force-field XML."""

    def __new__(cls) -> Self: ...
    def write(self, sink: _IoSink, forcefield: ForceField) -> None: ...

class mrec:
    """The ``_native.mrec`` submodule, surfaced as :mod:`molrs.io.mrec`: a
    record's per-section reads, the trajectory store's reader, writer and
    schema, and the force-field section. A whole record is written by
    ``molrs.io.write_mrec``."""

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
        def from_forcefield(forcefield: ForceField) -> mrec.ForceFieldSection: ...
        def to_forcefield(self) -> ForceField: ...

    @final
    class MrecFile:
        """A ``*.mrec`` record opened for its sections; each read decodes only
        the section it names."""

        def __new__(cls, source: _IoSource) -> Self: ...
        def sections(self) -> list[str]: ...
        def meta(self) -> dict[str, Any]: ...
        def frame(self) -> Frame | None: ...
        def system(self) -> Frame | None: ...
        def forcefield(self) -> mrec.ForceFieldSection | None: ...

    class MrecReader:
        """Lazy one-frame cursor over a record's ``trajectory`` section."""

        def __init__(self, source: _IoSource) -> None: ...
        def frame_columns(
            self, index: int, columns: list[tuple[str, str]]
        ) -> Frame: ...
        def block_update_at(self, name: str, index: int) -> int | None: ...
        def box_at(self, index: int) -> Box | None: ...
        def __len__(self) -> int: ...
        def __getitem__(self, index: int) -> Frame: ...
        def __iter__(self) -> Self: ...
        def __next__(self) -> Frame: ...
        def to_trajectory(self) -> Trajectory: ...
        @property
        def step(self) -> list[int]: ...
        @property
        def time(self) -> list[float] | None: ...
        def has_block(self, name: str) -> bool: ...
        def block_names(self) -> list[str]: ...
        def __enter__(self) -> Self: ...
        def __exit__(
            self,
            _exc_type: type[BaseException] | None = None,
            _exc_value: BaseException | None = None,
            _traceback: TracebackType | None = None,
        ) -> bool: ...

    class SequenceSchema:
        """A frame-sequence schema pinned before a run's frames are written:
        ``from_frames`` derives it (``from_frames([])`` is empty); every
        ``declare_*`` returns the schema."""

        @staticmethod
        def from_frames(frames: Sequence[Frame]) -> mrec.SequenceSchema: ...
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
        def declare_meta(self, key: str, dtype: str, fill: Any = None) -> Self:
            """A per-step meta key of ``dtype``; ``fill`` is written for the
            steps that omit it (without one, omitting it is an error)."""
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
            schema: mrec.SequenceSchema,
            *,
            flush_every: int | None = ...,
            compression: str | None = ...,
            durable: bool = ...,
            meta: _AbcMapping[str, Any] | None = ...,
        ) -> None: ...
        @staticmethod
        def open(
            path: PathInput, *, flush_every: int | None = ..., durable: bool = ...
        ) -> mrec.MrecWriter: ...
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
        def __exit__(
            self,
            _exc_type: type[BaseException] | None = None,
            _exc_value: BaseException | None = None,
            _traceback: TracebackType | None = None,
        ) -> bool: ...

    @staticmethod
    def pack_mrec_zip(path: PathInput) -> str: ...

    class validation:
        """``molrs.io.mrec.validation``: the record contract's runtime checks."""

        @staticmethod
        def validate_frame(frame: Frame) -> None: ...

# The in-memory doors: each one its path twin on a str / bytes.
# ---------------------------------------------------------------------------
# Structure builders (molrs::builder)
# ---------------------------------------------------------------------------

class CarbonTubeBuilder:
    def __new__(
        cls,
        n: int,
        m: int,
        *,
        length: float | None = None,
        cells: int | None = None,
        bond_length: float = 1.42,
        periodic: bool = False,
        vacuum: float = 10.0,
    ) -> Self: ...
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
    def __new__(
        cls,
        nx: int,
        ny: int,
        *,
        bond_length: float = 1.42,
        vacuum: float = 10.0,
        periodic_xy: bool = True,
    ) -> Self: ...
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
class Assembler[G: (MolGraph, Atomistic, CoarseGrain)]:
    """``molrs.builder.Assembler`` — one placed, linked world from a site
    graph. Frozen.

    Each bead of the site graph is one unit: a copy of ``library[bead_type]``,
    turned by the orienter (when given) and given its pose by the placer.
    Each site bond joins one port of each end's copy (``<`` with ``>``,
    ``$`` with ``$``); the leaving groups are removed. Any topology works:
    chains, branches, rings. Every atom gets ``frag_id`` (the site's ordinal)
    and ``mol_id`` (its connected component's ordinal + 1).

    Parameters
    ----------
    library : Mapping[str, MolGraph | Atomistic | CoarseGrain]
        Name → template, copied at construction, every template of ONE graph
        type — the type :meth:`assemble` builds the world as — with ports
        where a site bonds. A template without ports can only fill an
        unbonded site.
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
        If ``library`` is not a mapping of ``str`` to graphs of one type,
        ``placer`` is not a :class:`SitePlacer` or :class:`GrowthPlacer`, or
        ``orienter`` is not an :class:`AxisOrienter`.
    ValueError
        If ``library`` is empty.
    """

    def __new__(
        cls,
        library: _AbcMapping[str, G],
        placer: SitePlacer | GrowthPlacer,
        orienter: AxisOrienter | None = None,
    ) -> Assembler[G]: ...
    def assemble(self, sites: CoarseGrain) -> G:
        """Place and join one copy per site; return the world, of the
        library's graph type. Releases the GIL.

        Raises
        ------
        ValueError
            Naming the site at fault: a bead lacks its type or position, a
            name is not in the library, no accepting port exists for every
            bond of a site, a site cannot be oriented or placed, a join is
            refused, or the world breaks the graph type's invariant.
        """

# ---------------------------------------------------------------------------
# 3D coordinate generation (embed)
# ---------------------------------------------------------------------------

@final
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

@final
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

    def __new__(
        cls,
        speed: str = "medium",
        add_hydrogens: bool = True,
        seed: int | None = None,
    ) -> Self: ...
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

    def __new__(cls, name: str = "forcefield", units: str | None = None) -> Self: ...
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
    def get_style(self, category: str, name: str) -> Style | None: ...
    def get_styles(self, category: str | type[Style]) -> list[Style]:
        """The styles of a category (a name or a style class; ``Style``
        selects every category), in definition order."""
    def get_types(
        self, category: str | type[ForceFieldType]
    ) -> list[ForceFieldType]: ...
    def __reduce__(self) -> tuple[type, tuple[()], tuple[Any, ...]]: ...
    def __setstate__(self, state: tuple[Any, ...]) -> None: ...

class Style:
    """Handle of one style of a :class:`ForceField`; equal handles name the
    same category and style of one force field."""

    @property
    def name(self) -> str: ...
    @property
    def category(self) -> str: ...
    def get_types(
        self, type_cls: type[ForceFieldType] | None = None
    ) -> list[ForceFieldType]: ...
    def get_type_by_name(self, name: str) -> ForceFieldType | None: ...
    @property
    def params(self) -> dict[str, ParamValue]: ...
    def __getitem__(self, key: str, /) -> ParamValue | None: ...
    def __setitem__(self, key: str, value: ParamInput, /) -> None:
        """Set one style-level param (``Style::set_param``)."""
    def del_type(self, name: str) -> int:
        """Remove the type ``name`` (``Style::del_type``); the count removed,
        ``0`` or ``1``."""

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
    def __getitem__(self, key: str, /) -> ParamValue | None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def __contains__(self, key: str, /) -> bool: ...
    def __setitem__(self, key: str, value: ParamInput, /) -> None: ...
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

@final
class LammpsLogHeader:
    """Header lines that precede the first run of a LAMMPS log."""

    @property
    def lines(self) -> list[str]: ...
    def raw_text(self) -> str: ...

@final
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

@final
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
    def __getitem__(self, name: str, /) -> ArrayF: ...
    def __contains__(self, name: str, /) -> bool: ...
    def __len__(self) -> int: ...

@final
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

@final
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

@final
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

@final
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

@final
class LammpsTimingBreakdown:
    """A timing breakdown table (MPI task or thread timing)."""

    @property
    def title(self) -> str: ...
    @property
    def rows(self) -> list[LammpsTimingRow]: ...
    @property
    def raw_lines(self) -> list[str]: ...
    def __len__(self) -> int: ...

@final
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

@final
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

@final
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

@final
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

@final
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

class _PotentialLike(Protocol):
    """A Python force provider (``molrs.ff.potential.Potential``)."""
    def evaluate(self, pos: ArrayF, /) -> tuple[float, ArrayF]: ...

class Potentials:
    """Composite of the one ``Potential`` concept — itself a potential.

    ``Potentials()`` is empty; ``push`` **moves** members in (a ``PairLjCut``,
    another ``Potentials`` such as one ``ExplicitTerms.compile`` built, or an
    object with ``evaluate(pos) -> (energy, forces)``). The engine is
    unit-agnostic: nothing scales the energy or forces implicitly.
    """

    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def push(self, potential: PairLjCut | Potentials | _PotentialLike) -> None: ...
    def evaluate(self, arg: Frame | ArrayF) -> tuple[float, ArrayF]:
        """Energy and forces (``Potential::evaluate``): ``(energy, forces)``,
        forces ``(N, 3)``. ``arg`` is a :class:`Frame` (a deferred collection
        binds its topology first) or flat ``[x0, y0, z0, ...]`` coordinates."""

class PotentialCompiler:
    """Compiles a ``ForceField`` into evaluable kernels.

    Owns a copy of the force field taken at construction; later edits to that
    ``ForceField`` do not reach it. ``compile(frame)`` binds a typed frame now;
    ``defer()`` returns ``Potentials`` that bind the frame they are evaluated
    on; ``compile_typed(frame)`` builds the kernels of a neighbour-driven (MD)
    evaluation. Both doors price a pair only inside its style's ``cutoff``
    (``r < cutoff``, as LAMMPS; ∞ when the style states none).
    """

    def __new__(cls, forcefield: ForceField) -> Self: ...
    def compile(self, frame: Frame) -> Potentials: ...
    def defer(self) -> Potentials: ...
    def compile_typed(self, frame: Frame) -> WeightedTerms: ...

class Lbfgs:
    """L-BFGS geometry optimizer over a force-field Potential — the mirror of
    ``molrs::optimize::Lbfgs``.

    Knobs live on ``__new__`` (defaults: Rust ``LbfgsSettings::DEFAULT``).
    Every door relaxes its input **in place** and returns the report(s).
    """

    def __new__(
        cls,
        potentials: Potentials,
        *,
        fmax: float = ...,
        max_steps: int = ...,
        max_step: float = ...,
        memory: int = ...,
    ) -> Self:
        """Defaults: ``fmax=0.05``, ``max_steps=500``, ``max_step=0.2``,
        ``memory=8``."""
    def minimize(self, frame: Frame) -> OptimizationReport:
        """Relax ``frame``'s atoms in place; a bool ``atoms.free`` column marks
        the movable atoms (fixed ones still enter every evaluation)."""
    def minimize_coords(self, coords: ArrayF) -> OptimizationReport:
        """Relax a C-contiguous float64 ``(N, 3)`` array in place, every atom
        free."""
    def minimize_batch(self, coords: ArrayF) -> list[OptimizationReport]:
        """Relax a C-contiguous float64 ``(B, N, 3)`` stack in place, each
        structure independently; one report per structure."""

#: A param value of a type annotation or style: numbers to the numeric side,
#: strings to the string side.
type AssignmentParamValue = float | int | str

#: What a ``TypeAssignment`` writes under one key of one graph element: a scalar is
#: stamped and defines nothing; ``(style, name, endpoints, params)`` stamps
#: ``name`` and every param and defines the type ``name`` on ``endpoints``
#: (atom-type names; empty for an atom type) under the style.
type Annotation = (
    str
    | bool
    | int
    | float
    | tuple[str, str, Sequence[str], dict[str, AssignmentParamValue]]
)

class TypeAssignment:
    """What a typifier's ``assign`` assigns to one graph.

    ``nodes`` is positional against ``graph.atoms``; ``links`` maps a relation
    kind to rows positional against that kind's own rows, so an improper never
    shifts a dihedral position. A key is a relation view class (``Bond``,
    ``Angle``, ``Dihedral``, ``Improper``, ``Port``; rows as ``graph.bonds`` …
    ``graph.ports``) or a block name (``"bonds"``, or a term block such as
    ``"urey_bradleys"``; rows as ``graph.terms(block)``). A type annotation under a kind defines a
    type of the category whose block the kind is (``urey_bradleys`` →
    ``urey_bradley``). Any other key raises ``TypeError``, a kind named twice
    ``ValueError``. ``styles`` are ``(category, style, params)`` to declare,
    in order; ``pairs`` are ``(style, name, endpoints, params)`` pair rows."""

    def __new__(
        cls,
        nodes: Sequence[_AbcMapping[str, Annotation]],
        links: _AbcMapping[type | str, Sequence[_AbcMapping[str, Annotation]]]
        | None = None,
        *,
        styles: Sequence[tuple[str, str, dict[str, AssignmentParamValue]]] = (),
        pairs: Sequence[
            tuple[str, str, Sequence[str], dict[str, AssignmentParamValue]]
        ] = (),
    ) -> Self: ...

class Typifier(Generic[_TGraph]):
    """The base of every graph typifier: one ``assign`` hook plus the output
    force field its typing accumulates.

    A subclass implements ``assign`` (and optionally ``source_forcefield``) and nothing
    else; defining ``typify`` on a subclass raises ``TypeError`` at class
    creation. The native classes extend this base and only construct; they
    are subclassable, but a subclass of one that defines ``assign`` or
    ``source_forcefield`` raises ``TypeError`` (they run in Rust)."""

    def __new__(cls, *_args: object, **_kwargs: object) -> Self: ...
    def assign(self, graph: _TGraph) -> TypeAssignment:
        """Type ``graph`` and return what it assigns.

        ``assign`` may write intermediate results (generated topology, perceived
        bond types) onto the graph it is given; ``typify`` always gives it a
        private copy. The base raises ``NotImplementedError``; a native class
        runs its Rust matcher."""
    @final
    def typify(self, mol: _TGraph) -> _TGraph:
        """Do not override; the only writer of ``forcefield()``.

        Copies ``mol`` (as ``copy.copy(mol)``), calls ``assign`` on the copy,
        and writes the match onto the copy and the output. Returns the typed copy; ``mol`` is untouched.
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

    def __new__(cls) -> Self: ...

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

    def __new__(cls) -> Self: ...

class OplsAaTypifier(Typifier[Atomistic]):
    """OPLS-AA atom types and bonded parameters; ``source`` is an OpenMM XML
    file (a path) or its content (``bytes`` / ``molrs.io.Text``; a plain
    ``str`` is refused), ``None`` for the embedded table."""

    def __new__(
        cls,
        source: os.PathLike[str] | bytes | Text | None = None,
        *,
        strict: bool = True,
    ) -> Self: ...

type AtdParameterSet = Literal["bcc", "abcg2", "gas", "gaff", "gaff2", "amber", "sybyl"]

class AtdTypifier(Typifier[Atomistic]):
    """antechamber atom types — one rule engine over seven ``ATOMTYPE_*.DEF`` tables.

    ``parameter_set`` is the antechamber ``-at`` flag and is **required**: seven
    tables exist, they disagree, and there is no default. An atom no rule matches
    comes back labelled ``"DU"`` — the table's own catch-all row, not a fallback.

    ``bond_orders`` says which bond orders the types follow:
    ``BondOrderSource.Connectivity`` (the core default) judges them from the
    connectivity alone, as antechamber's ``bondtype -j full`` does, ignoring
    the molecule's own; ``BondOrderSource.Stated`` keeps the molecule's
    orders."""

    def __new__(
        cls,
        *,
        parameter_set: AtdParameterSet,
        bond_orders: BondOrderSource = ...,
    ) -> Self: ...
    @property
    def parameter_set(self) -> AtdParameterSet: ...
    @property
    def bond_orders(self) -> BondOrderSource: ...

type GaffParameterSet = Literal["gaff", "gaff2"]

class GaffTypifier(Typifier[Atomistic]):
    """GAFF / GAFF2 bonded terms and parameters for a molecule whose atoms
    already carry GAFF types (``AtdTypifier`` with the same ``parameter_set``).

    Angles and dihedrals are regenerated from the bond graph, impropers built
    as tleap builds them (wherever tleap finds a row, or ``parmchk2`` an
    estimate at a centre ``PARMCHK.DAT`` flags as planar; tleap's atom order,
    centre third); each term is an exact ``gaff.dat`` / ``gaff2.dat`` row, a wildcard row, or a
    ``parmchk2``-style estimate whose type carries ``estimated``,
    ``estimate_penalty``, ``estimate_method`` and ``estimate_analog``. The
    four price nothing: every engine writer (LAMMPS, GROMACS, OpenMM, frcmod)
    passes them over, and ``ForceField.merge`` ignores them.
    ``parameter_set`` is required; there is no default.

    Raises
    ------
    ValueError
        An unknown ``parameter_set``; from :meth:`typify`, an untyped atom, a
        type the table does not declare, or terms nothing covers (all listed).
    """

    def __new__(cls, *, parameter_set: GaffParameterSet) -> Self: ...
    @property
    def parameter_set(self) -> GaffParameterSet: ...

class ElementTypifier(Typifier[Atomistic]):
    """``molrs.ff.typifier.ElementTypifier`` — ``type`` labels from atomic
    numbers alone, with no force field.

    Atoms get ``type`` = the symbol their ``atomic_number`` names (e.g.
    ``"C"``); bonds, angles and dihedrals get their endpoint symbols joined
    with ``-`` in the byte-wise smaller orientation (bond O–H is ``"H-O"``).
    :meth:`forcefield` stays empty. Exported from :mod:`molrs.ff.typifier`
    only.

    Raises
    ------
    ValueError
        From :meth:`typify`, when an atom has no valid ``atomic_number`` or
        is the wildcard (``0``), or the molecule has impropers.
    """

    def __new__(cls) -> Self: ...

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

    def __new__(cls, *, parameter_set: BccParameterSet) -> Self: ...
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

def clpol_polarizability() -> dict[str, dict[str, float]]:
    """CL&Pol Drude parameters per atom type (``m_D``, ``q_D_sign``, ``k_D``,
    ``alpha``, ``a_thole``): the shipped ``alpha.ff`` table."""

def assign_cmaps(frame: Frame, forcefield: ForceField) -> int: ...
def intramolecular_pairs(
    frame: Frame,
    forcefield: ForceField | None = None,
    neighbors: Neighbors | None = None,
) -> Block:
    """The special-bonds pair list (``atomi`` / ``atomj`` / ``is_14``) of a
    typed frame; candidates from ``neighbors`` when given, else every
    ``i < j`` pair."""

# ---------------------------------------------------------------------------
# Record / Trajectory / Observables
# ---------------------------------------------------------------------------

class Trajectory:
    """An in-memory frame sequence with optional per-frame ``step`` / ``time``
    labels; a slice is the sub-trajectory with its labels."""

    def __new__(
        cls,
        frames: Sequence[Frame],
        step: ArrayI64 | None = None,
        time: ArrayF | None = None,
    ) -> Self: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> Frame: ...
    @overload
    def __getitem__(self, key: slice, /) -> Trajectory: ...
    def map(self, func: Callable[[Frame], Frame]) -> Trajectory: ...
    @property
    def frames(self) -> list[Frame]: ...
    @property
    def step(self) -> ArrayI64 | None: ...
    @property
    def time(self) -> ArrayF | None: ...

type _ObservableScalarValue = npt.NDArray[Any] | float | int | bool | str | list[str]

class ObservableRecord:
    """A named observable and its metadata: ``ObservableRecord(name, kind,
    values)`` as ``ObservableRecord::new(name, kind, values)`` in Rust.
    ``kind`` is ``"scalar"``, ``"vector"`` or another spelling carried
    verbatim."""

    def __new__(
        cls,
        name: str,
        kind: str,
        values: _ObservableScalarValue,
        description: str = "",
        unit: str | None = None,
        axes: list[str] | None = None,
        time_dependent: bool = False,
        sampling: str | None = None,
        domain: str | None = None,
        target: str | None = None,
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def values(self) -> npt.NDArray[Any] | list[str] | str: ...
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

@final
class AngularSeparationGlobal:
    """Angles between ``(N, 4)`` query quaternions and ``(M, 4)`` reference
    ones (``molrs::compute::AngularSeparationGlobal``)."""
    def __new__(cls, *, equivalent_orientations: bool | None = None) -> Self: ...
    def compute(
        self, *, query: ArrayF, reference: ArrayF
    ) -> AngularSeparationGlobalResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class AngularSeparationGlobalResult:
    """Angles between query and reference orientations
    (``molrs::compute::AngularSeparationGlobalResult``)."""
    @property
    def angles(self) -> ArrayF: ...

@final
class AngularSeparationNeighbor:
    """Angles between the orientations of neighbouring particles
    (``molrs::compute::AngularSeparationNeighbor``)."""
    def __new__(cls, *, equivalent_orientations: bool | None = None) -> Self: ...
    def compute(
        self,
        *,
        neighbors: Neighbors,
        query_orientations: ArrayF,
        point_orientations: ArrayF,
    ) -> AngularSeparationNeighborResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class AngularSeparationNeighborResult:
    """Angles between neighbouring orientations
    (``molrs::compute::AngularSeparationNeighborResult``)."""
    @property
    def angles(self) -> ArrayF: ...

@final
class Axis:
    """A uniform binning axis: ``n_bins`` half-open bins over ``[min, max)``
    (``molrs::compute::Axis``)."""
    def __new__(cls, min: float, max: float, n_bins: int) -> Self: ...
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

@final
class BondOrientationalOrder:
    """Bond-orientational order diagram on a ``(theta, phi)`` grid
    (``molrs::compute::BondOrientationalOrder``)."""
    def __new__(cls, n_theta: int, n_phi: int) -> Self: ...
    def compute(self, *, neighbors: Neighbors) -> BondOrientationalOrderResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class CenterOfMass:
    """Mass-weighted centre of each group (``molrs::compute::CenterOfMass``)."""
    def __init__(self) -> None: ...
    def compute(
        self,
        *,
        positions: ArrayF,
        simbox: Box | None = None,
        partition: Partition,
        masses: ArrayF,
    ) -> CenterOfMassResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class CenterOfMassResult:
    """Centres of mass (``molrs::compute::CenterOfMassResult``): one row of
    ``centers`` and one ``total_masses`` entry per group key in ``groups``."""
    @property
    def centers(self) -> ArrayF: ...
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def total_masses(self) -> ArrayF: ...

@final
class Cluster:
    """Connected components of a neighbour table (``molrs::compute::Cluster``)."""
    def __new__(cls, *, min_cluster_size: int | None = None) -> Self: ...
    def compute(
        self, *, pairs: Neighbors, merge: Partition | None = None
    ) -> ClusterResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class CombinedDistribution:
    """Joint distribution of 2 or 3 internal coordinates over frames
    (``molrs::compute::CombinedDistribution``)."""
    def __new__(
        cls,
        observables: Sequence[str],
        axes: Sequence[Axis],
        *,
        sin_weight: Sequence[int] | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        positions: Sequence[ArrayF],
        simbox: Sequence[Box | None] | None = None,
        tuples: Sequence[Sequence[Sequence[Sequence[int]]]],
    ) -> CombinedDistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class ComputeSpec:
    """The descriptor of one compute type (``molrs::compute::ComputeSpec``)."""
    @staticmethod
    def all() -> list[ComputeSpec]: ...
    @staticmethod
    def find(name: str) -> ComputeSpec | None: ...
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

@final
class ConductivitySumRule:
    """The conductivity sum rule ``∫ σ(ω) dω`` against its static value
    (``molrs::compute::ConductivitySumRule``)."""
    def __new__(
        cls, units: str | UnitPreset, *, tolerance: float | None = None
    ) -> Self: ...
    def compute(
        self,
        *,
        frequencies: ArrayF,
        conductivity: ArrayF,
        current_sq_mean: float,
        volume: float,
        temperature: float,
    ) -> SumRuleCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class ContinuousCoordination:
    """Continuous coordination number (``molrs::compute::ContinuousCoordination``)."""
    def __new__(cls, power: float) -> Self: ...
    def compute(
        self, *, qlm: QlmTable, neighbors: Neighbors
    ) -> ContinuousCoordinationResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class ContinuousCoordinationResult:
    """Continuous coordination per particle
    (``molrs::compute::ContinuousCoordinationResult``)."""
    @property
    def coord(self) -> ArrayF: ...
    @property
    def l(self) -> int: ...

@final
class CorrelationFunction:
    """Radial correlation of two per-point value arrays over a pair source
    (``molrs::compute::CorrelationFunction``)."""
    def __new__(cls, n_bins: int, r_max: float, r_min: float) -> Self: ...
    def compute(
        self, *, pairs: Neighbors | NeighborList, values_a: ArrayF, values_b: ArrayF
    ) -> CorrelationFunctionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class Correlator:
    """The one time-correlation estimator (``molrs::compute::Correlator``)."""
    def __new__(
        cls,
        kernel: str,
        max_lag: int,
        *,
        l: int | None = None,
        origins: str | None = None,
        mean: str | None = None,
        normalization: str | None = None,
        reduction: str | None = None,
        stride: int | None = None,
        dimension: int | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        a: ArrayF,
        b: ArrayF | None = None,
        dt: float,
        boxes: Sequence[Box] | None = None,
    ) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class CorrelatorAccumulator:
    """The streaming correlator (``molrs::compute::CorrelatorAccumulator``),
    made by :meth:`Msd.accumulator` / :meth:`Vacf.accumulator`."""
    def push(
        self,
        a: ArrayF,
        *,
        b: ArrayF | None = None,
        dt: float,
        simbox: Box | None = None,
    ) -> None:
        """Fold one ``(N, D)`` frame (``CorrelatorFrame``); a rejected frame
        leaves the state unchanged."""
    def finish(self) -> TimeCorrelation:
        """The correlation over every frame pushed so far (not consuming)."""
    @property
    def n_frames(self) -> int: ...

@final
class Cubatic:
    """Cubatic order of ``(N, 3)`` directors by simulated annealing
    (``molrs::compute::Cubatic``)."""
    def __new__(
        cls,
        *,
        seed: int | None = None,
        initial_temp: float | None = None,
        cooling_rate: float | None = None,
        min_temp: float | None = None,
        n_steps: int | None = None,
        n_chains: int | None = None,
    ) -> Self: ...
    def compute(self, *, directors: ArrayF) -> CubaticResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class CubaticResult:
    """The cubatic order and its director basis (``molrs::compute::CubaticResult``)."""
    @property
    def director_basis(self) -> ArrayF: ...
    @property
    def order(self) -> float: ...

@final
class CumulativeTrapezoid:
    """The running trapezoid integral of one channel of a correlation
    (``molrs::compute::CumulativeTrapezoid``) — a Green–Kubo integral."""
    def __new__(cls, *, n_lags: int | None = None) -> Self: ...
    def compute(
        self, *, correlation: TimeCorrelation, channel: int
    ) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class CurrentAcf:
    """Current autocorrelation of a ``(T, 3)`` current series
    (``molrs::compute::CurrentAcf``)."""
    def __new__(
        cls,
        max_lag: int,
        *,
        normalization: str | None = None,
        stride: int | None = None,
        mean: str | None = None,
    ) -> Self: ...
    def compute(self, *, current: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DebyeFit:
    """Single-exponential (Debye) relaxation fit of one channel of a correlation
    (``molrs::compute::DebyeFit``)."""
    def __init__(self) -> None: ...
    def compute(
        self, *, correlation: TimeCorrelation, channel: int
    ) -> DebyeFitResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DebyeFitResult:
    """A single-exponential fit ``C(t) = amplitude · exp(−t / tau)``
    (``molrs::compute::DebyeFitResult``) over ``n_samples`` lags."""
    @property
    def amplitude(self) -> float: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def tau(self) -> float: ...

@final
class Derivative:
    """Finite-difference time derivative of a ``(T, C)`` series
    (``molrs::compute::Derivative``)."""
    def __new__(cls, stencil: str) -> Self: ...
    def compute(self, *, series: ArrayF, dt: float) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class DiffractionPattern:
    """A 2-D diffraction pattern of one snapshot projected along an axis
    (``molrs::compute::DiffractionPattern``)."""
    def __new__(
        cls,
        n_grid: int,
        sigma: float,
        *,
        axis: int | None = None,
        cutoff: float | None = None,
    ) -> Self: ...
    def compute(
        self, *, positions: ArrayF, simbox: Box
    ) -> DiffractionPatternResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DiffractionPatternResult:
    """A 2-D diffraction pattern and the projected density image it was
    transformed from (``molrs::compute::DiffractionPatternResult``)."""
    @property
    def diffraction(self) -> ArrayF: ...
    @property
    def image(self) -> ArrayF: ...

@final
class DipoleAcf:
    """Dipole autocorrelation of a ``(T, 3)`` dipole series
    (``molrs::compute::DipoleAcf``)."""
    def __new__(
        cls,
        max_lag: int,
        *,
        normalization: str | None = None,
        stride: int | None = None,
        mean: str | None = None,
    ) -> Self: ...
    def compute(self, *, dipoles: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DipoleAutocorrelationSpectrum:
    """ε(ω) by the dipole-autocorrelation route
    (``molrs::compute::DipoleAutocorrelationSpectrum``)."""
    def __new__(
        cls,
        units: str | UnitPreset,
        *,
        window: str | None = None,
        pad: int | None = None,
        epsilon_inf: float | None = None,
        plateau: tuple[float, float] | None = None,
        plateau_value: str | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        correlation: TimeCorrelation,
        volume: float,
        temperature: float,
        boundary: str,
    ) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DipoleMoment:
    """The dipole moment ``Σ q r`` of each group in each frame
    (``molrs::compute::DipoleMoment``)."""
    def __init__(self) -> None: ...
    def compute(
        self, *, positions: ArrayF, charges: ArrayF, partition: Partition | None = None
    ) -> DipoleMomentResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DipoleMomentResult:
    """Dipole series per group (``molrs::compute::DipoleMomentResult``):
    ``moments[frame, group, :]`` for the group keys ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def moments(self) -> ArrayF: ...
    def series(self, column: int) -> ArrayF: ...

@final
class DipoleRateCrossSpectrum:
    """ε(ω) by the dipole–current cross-correlation route
    (``molrs::compute::DipoleRateCrossSpectrum``)."""
    def __new__(
        cls,
        units: str | UnitPreset,
        *,
        window: str | None = None,
        pad: int | None = None,
        epsilon_inf: float | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        correlation: TimeCorrelation,
        volume: float,
        temperature: float,
        boundary: str,
    ) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class DistributionFunction:
    """Distribution of one internal coordinate over frames
    (``molrs::compute::DistributionFunction``)."""
    def __new__(
        cls, observable: str, *, axis: Axis | None = None, n_bins: int | None = None
    ) -> Self: ...
    def compute(
        self,
        *,
        positions: Sequence[ArrayF],
        simbox: Sequence[Box | None] | None = None,
        tuples: Sequence[Sequence[Sequence[int]]],
    ) -> DistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class EinsteinHelfandSpectrum:
    """ε(ω) by the Einstein–Helfand route from the fluctuation dipole ACF
    (``molrs::compute::EinsteinHelfandSpectrum``), e.g. ``DipoleAcf``'s
    curve."""
    def __new__(
        cls,
        units: str | UnitPreset,
        *,
        window: str | None = None,
        pad: int | None = None,
        epsilon_inf: float | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        correlation: TimeCorrelation,
        volume: float,
        temperature: float,
        boundary: str,
    ) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class EnvironmentMatch:
    """Cluster particles whose neighbour environments match within
    ``rmsd_threshold`` (``molrs::compute::EnvironmentMatch``)."""
    def __new__(
        cls,
        rmsd_threshold: float,
        *,
        registration: bool | None = None,
        max_neighbors_for_registration: int | None = None,
    ) -> Self: ...
    def compute(self, *, neighbors: Neighbors) -> EnvironmentMatchResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class EnvironmentMatchResult:
    """Environment clusters (``molrs::compute::EnvironmentMatchResult``); a
    partition."""
    @property
    def cluster_idx(self) -> npt.NDArray[Any]: ...
    @property
    def fingerprints(self) -> list[ArrayF]: ...
    @property
    def n_clusters(self) -> int: ...

@final
class FiniteFieldPolarizability:
    """Polarizability by central finite differences of the group dipoles under
    ``±field`` (``molrs::compute::FiniteFieldPolarizability``)."""
    def __new__(cls, field: float) -> Self: ...
    def compute(self, *, plus: MolecularMoments, minus: MolecularMoments) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class GaussianDensity:
    """Gaussian-smeared density of points on a grid over the box
    (``molrs::compute::GaussianDensity``)."""
    def __new__(
        cls,
        dims: Sequence[int],
        sigma: float,
        *,
        r_max: float | None = None,
        reach: str | None = None,
    ) -> Self: ...
    def compute(self, *, positions: ArrayF, simbox: Box) -> GaussianDensityResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class GaussianDensityResult:
    """A Gaussian-smeared density on a grid
    (``molrs::compute::GaussianDensityResult``)."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def grid(self) -> RegularGrid: ...

@final
class GreenKuboSpectrum:
    """ε(ω) by the Green–Kubo route from the current ACF
    (``molrs::compute::GreenKuboSpectrum``), e.g. ``CurrentAcf``'s curve."""
    def __new__(
        cls,
        units: str | UnitPreset,
        *,
        window: str | None = None,
        pad: int | None = None,
        epsilon_inf: float | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        correlation: TimeCorrelation,
        volume: float,
        temperature: float,
        boundary: str,
    ) -> DielectricSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class GyrationTensor:
    """Weighted gyration tensor of each group (``molrs::compute::GyrationTensor``)."""
    def __init__(self) -> None: ...
    def compute(
        self,
        *,
        positions: ArrayF,
        simbox: Box | None = None,
        partition: Partition,
        weights: ArrayF,
    ) -> GyrationTensorResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class GyrationTensorResult:
    """Gyration tensors (``molrs::compute::GyrationTensorResult``): a
    ``(3, 3)`` tensor per group key in ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    def radius_of_gyration(self) -> ArrayF: ...
    @property
    def tensors(self) -> ArrayF: ...

@final
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

@final
class HBonds:
    """Geometric hydrogen-bond detection, frame by frame
    (``molrs::compute::HBonds``)."""
    def __new__(
        cls,
        *,
        dist_cutoff: float | None = None,
        dist_kind: str | None = None,
        angle_cutoff: float | None = None,
        donor_acceptor_margin: float | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        positions: ArrayF,
        donors: Sequence[Sequence[int]],
        acceptors: Sequence[int],
        pairs: Sequence[Neighbors],
    ) -> HBondsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class Hexatic:
    """The k-atic bond order ``ψ_k`` of a 2-D system (``molrs::compute::Hexatic``)."""
    def __new__(cls, k: int) -> Self: ...
    def compute(self, *, neighbors: Neighbors) -> HexaticResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class HexaticResult:
    """The k-atic order ``ψ_k`` per particle (``molrs::compute::HexaticResult``)."""
    @property
    def k(self) -> int: ...
    @property
    def psi(self) -> npt.NDArray[np.complex128]: ...

@final
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

@final
class InertiaTensor:
    """Moment of inertia tensor of each group (``molrs::compute::InertiaTensor``)."""
    def __init__(self) -> None: ...
    def compute(
        self,
        *,
        positions: ArrayF,
        simbox: Box | None = None,
        partition: Partition,
        masses: ArrayF,
    ) -> InertiaTensorResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class InertiaTensorResult:
    """Inertia tensors (``molrs::compute::InertiaTensorResult``): a ``(3, 3)``
    tensor per group key in ``groups``."""
    @property
    def groups(self) -> npt.NDArray[Any]: ...
    @property
    def tensors(self) -> ArrayF: ...

@final
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

@final
class KineticEnergy:
    """Kinetic energy ``½ Σ m v² · mvv2e`` (``molrs::compute::KineticEnergy``):
    the constants of the unit preset ``units`` (``KineticEnergy::new``), each
    replaceable — ``mvv2e`` (``with_mvv2e``), ``boltzmann``
    (``with_boltzmann``)."""
    def __new__(
        cls,
        units: str | UnitPreset,
        *,
        mvv2e: float | None = None,
        boltzmann: float | None = None,
    ) -> Self: ...
    def compute(self, *, masses: ArrayF, velocities: ArrayF) -> KineticEnergyResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class KineticEnergyResult:
    """A kinetic energy (``molrs::compute::KineticEnergyResult``) with the
    Boltzmann constant of its unit preset."""
    @property
    def boltzmann(self) -> float: ...
    @property
    def energy(self) -> float: ...
    def temperature(self, n_dof: int) -> float: ...

@final
class Kmeans:
    """K-means clustering of the rows of a feature matrix
    (``molrs::compute::Kmeans``)."""
    def __new__(
        cls, k: int, *, max_iter: int | None = None, seed: int | None = None
    ) -> Self: ...
    def compute(self, *, features: FeatureMatrix) -> KmeansResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class KmeansResult:
    """The k-means partition (``molrs::compute::KmeansResult``): one label per
    row and the ``(k, n_features)`` centroids. A partition."""
    @property
    def centroids(self) -> ArrayF: ...
    @property
    def labels(self) -> npt.NDArray[Any]: ...

@final
class KramersKronig:
    """Kramers–Kronig consistency check of a ε(ω) spectrum
    (``molrs::compute::KramersKronig``)."""
    def __new__(
        cls,
        *,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
    ) -> Self: ...
    def compute(self, *, spectrum: DielectricSpectrumResult) -> KramersKronigCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class LinearFit:
    """Least-squares line through one channel of a correlation over a window of
    its lags (``molrs::compute::LinearFit``) — e.g. the diffusive regime of an
    MSD."""
    def __new__(cls, start: float, end: float) -> Self: ...
    def compute(
        self, *, correlation: TimeCorrelation, channel: int
    ) -> LinearFitResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class LocalBondProjection:
    """Projections of each bond onto ``(M, 3)`` axes, in the particle frame when
    ``(N, 4)`` orientations are given (``molrs::compute::LocalBondProjection``)."""
    def __init__(self) -> None: ...
    def compute(
        self,
        *,
        neighbors: Neighbors,
        proj_vectors: ArrayF,
        orientations: ArrayF | None = None,
    ) -> LocalBondProjectionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class LocalBondProjectionResult:
    """Bond projections onto the given axes
    (``molrs::compute::LocalBondProjectionResult``)."""
    @property
    def projections(self) -> ArrayF: ...

@final
class LocalDensity:
    """Local number density within ``r_max`` of each point
    (``molrs::compute::LocalDensity``)."""
    def __new__(cls, r_max: float, *, diameter: float | None = None) -> Self: ...
    def compute(self, *, pairs: Neighbors | NeighborList) -> LocalDensityResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class LocalDensityResult:
    """Per-point neighbour count and local number density
    (``molrs::compute::LocalDensityResult``)."""
    @property
    def density(self) -> ArrayF: ...
    @property
    def n_neighbors(self) -> ArrayF: ...

@final
class LocalDescriptors:
    """Spherical-harmonic descriptors of every bond up to ``l_max``
    (``molrs::compute::LocalDescriptors``)."""
    def __new__(cls, l_max: int) -> Self: ...
    def compute(self, *, neighbors: Neighbors) -> LocalDescriptorsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
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

@final
class Msd:
    """Mean-squared displacement (``molrs::compute::Msd``), the
    ``squared_displacement`` correlator — a stream only: push each frame's
    unwrapped ``(N, 3)`` positions into :meth:`accumulator`."""
    def __new__(
        cls,
        max_lag: int,
        *,
        origins: str | None = None,
        reduction: str | None = None,
        stride: int | None = None,
    ) -> Self: ...
    def accumulator(self) -> CorrelatorAccumulator: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class Nematic:
    """Nematic order of ``(N, 3)`` directors (``molrs::compute::Nematic``)."""
    def __init__(self) -> None: ...
    def compute(self, *, directors: ArrayF) -> NematicResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
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

@final
class OnsagerCorrelation:
    """Onsager cross-correlation of two ``(T, C)`` series
    (``molrs::compute::OnsagerCorrelation``)."""
    def __new__(cls, max_lag: int, *, stride: int | None = None) -> Self: ...
    def compute(
        self, *, a: ArrayF, b: ArrayF, dt: float, boxes: Sequence[Box] | None = None
    ) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PairPresence:
    """Pair presence with hysteresis over frames
    (``molrs::compute::PairPresence``): a pair enters at distance ``≤ r0`` and
    stays while within ``r1``."""
    def __new__(
        cls, r0: float, exclude_self: bool, *, r1: float | None = None
    ) -> Self: ...
    def compute(
        self, *, frames: Sequence[Neighbors | NeighborList]
    ) -> Presence | None: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class Pca:
    """Principal component analysis of the rows of a feature matrix
    (``molrs::compute::Pca``)."""
    def __new__(cls, n_components: int) -> Self: ...
    def compute(self, *, features: FeatureMatrix) -> PcaResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PcaResult:
    """Principal components (``molrs::compute::PcaResult``): the
    ``(n_rows, n_components)`` projected ``coords`` and the ``variance``
    of each component. A feature matrix."""
    @property
    def coords(self) -> ArrayF: ...
    @property
    def variance(self) -> ArrayF: ...

@final
class Plateau:
    """The mean of one channel of a correlation over a window of its lags
    (``molrs::compute::Plateau``) — e.g. the plateau of a Green–Kubo integral."""
    def __new__(cls, start: float, end: float) -> Self: ...
    def compute(
        self, *, correlation: TimeCorrelation, channel: int
    ) -> PlateauResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PlateauResult:
    """The mean ``value`` (and its ``std``) of a curve over a window of
    ``n_samples`` lags (``molrs::compute::PlateauResult``)."""
    @property
    def n_samples(self) -> int: ...
    @property
    def std(self) -> float: ...
    @property
    def value(self) -> float: ...

@final
class PmftR12:
    """PMFT in ``(r, θ1, θ2)`` of 2-D pairs (``molrs::compute::Pmft<R12>``)."""
    def __new__(cls, r_max: float, n_r: int, n_t1: int, n_t2: int) -> Self: ...
    def compute(
        self, *, pairs: Neighbors, orientations: ArrayF | None = None
    ) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class PmftXy:
    """PMFT in the ``(x, y)`` frame of a 2-D particle
    (``molrs::compute::Pmft<Xy>``)."""
    def __new__(cls, x_max: float, y_max: float, n_x: int, n_y: int) -> Self: ...
    def compute(
        self, *, pairs: Neighbors, orientations: ArrayF | None = None
    ) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PmftXyt:
    """PMFT in ``(x, y, θ)`` of 2-D pairs (``molrs::compute::Pmft<Xyt>``)."""
    def __new__(
        cls, x_max: float, y_max: float, n_x: int, n_y: int, n_t: int
    ) -> Self: ...
    def compute(
        self, *, pairs: Neighbors, orientations: ArrayF | None = None
    ) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PmftXyz:
    """PMFT in the ``(x, y, z)`` frame of a 3-D particle
    (``molrs::compute::Pmft<Xyz>``)."""
    def __new__(
        cls, x_max: float, y_max: float, z_max: float, n_x: int, n_y: int, n_z: int
    ) -> Self: ...
    def compute(
        self, *, pairs: Neighbors, orientations: ArrayF | None = None
    ) -> PmftResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PolarizabilityParts:
    """Split a polarizability series into its Raman invariants
    (``molrs::compute::PolarizabilityParts``)."""
    def __init__(self) -> None: ...
    def compute(self, *, alpha: ArrayF) -> PolarizabilityPartsResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class PolarizabilityPartsResult:
    """The rotational invariants of a polarizability series
    (``molrs::compute::PolarizabilityPartsResult``): ``isotropic`` ``(T, 1)``
    and ``anisotropic`` ``(T, 6)``."""
    @property
    def anisotropic(self) -> ArrayF: ...
    @property
    def isotropic(self) -> ArrayF: ...

@final
class Presence:
    """A presence series (``molrs::compute::Presence``): ``values[t, k]`` is
    1 when pair ``pairs[k]`` is present at frame ``t``. Pass ``values`` to
    ``Survival`` as its ``presence``."""
    @property
    def pairs(self) -> npt.NDArray[np.uint32]: ...
    @property
    def values(self) -> ArrayF: ...

@final
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

@final
class RadicalVoronoi:
    """The radical (power) Voronoi tessellation of one snapshot
    (``molrs::compute::RadicalVoronoi``)."""
    def __init__(self) -> None: ...
    def compute(
        self, *, positions: ArrayF, simbox: Box, radii: ArrayF
    ) -> VoronoiCells: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class RamanSpectrum:
    """Raman spectrum from the isotropic and anisotropic polarizability
    correlations (``molrs::compute::RamanSpectrum``)."""
    def __new__(
        cls,
        *,
        window: str | None = None,
        pad: int | None = None,
        incident_frequency: float | None = None,
        averaged: bool | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        isotropic: TimeCorrelation,
        anisotropic: TimeCorrelation,
        temperature: float | None = None,
    ) -> RamanSpectrumResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class Rdf:
    """Radial distribution function g(r) (``molrs::compute::Rdf``) — a stream
    only: push one pair source per frame into :meth:`accumulator`."""
    def __new__(
        cls,
        n_bins: int,
        r_max: float,
        r_min: float,
        *,
        dimensionality: int | None = None,
    ) -> Self: ...
    def accumulator(self) -> RdfAccumulator: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class RdfAccumulator:
    """The streaming g(r) accumulator (``molrs::compute::RdfAccumulator``)."""
    def push(self, pairs: Neighbors | NeighborList) -> None:
        """Fold one frame's pairs (box and ``dist_sq`` required); a rejected
        frame leaves the state unchanged."""
    def finish(self) -> RdfResult:
        """The g(r) over every frame pushed so far (not consuming)."""
    @property
    def n_frames(self) -> int: ...

@final
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

@final
class RegularGrid:
    """A regular 3-D voxel grid (``molrs::compute::RegularGrid``): ``origin``,
    the three cell ``basis`` vectors (rows), ``dims`` voxels per axis and the
    periodicity ``pbc`` of each axis."""
    def __new__(
        cls,
        origin: Sequence[float],
        basis: Sequence[Sequence[float]],
        dims: Sequence[int],
        pbc: Sequence[bool],
    ) -> Self: ...
    @property
    def basis(self) -> ArrayF: ...
    @staticmethod
    def centered(extent: Sequence[float], dims: Sequence[int]) -> RegularGrid: ...
    @property
    def dims(self) -> list[int]: ...
    @property
    def origin(self) -> list[float]: ...
    @staticmethod
    def over_box(box: Box, dims: Sequence[int]) -> RegularGrid: ...
    @property
    def pbc(self) -> list[bool]: ...
    @property
    def voxel_volume(self) -> float: ...

@final
class ReorientationAcf:
    """Legendre reorientation correlation ``⟨P_l(u(0)·u(t))⟩`` of ``(T, N, D)``
    vectors, each normalised (``molrs::compute::ReorientationAcf``) — the
    orientational autocorrelation (OACF); ``l <= 2`` runs by FFT."""
    def __new__(
        cls,
        l: int,
        max_lag: int,
        *,
        reduction: str | None = None,
        stride: int | None = None,
    ) -> Self: ...
    def compute(self, *, vectors: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class RotationalAutocorrelation:
    """Rotational autocorrelation of ``(T, N, 4)`` unit quaternions
    (``molrs::compute::RotationalAutocorrelation``)."""
    def __new__(cls, l: int, max_lag: int, *, reduction: str | None = None) -> Self: ...
    def compute(self, *, orientations: ArrayF, dt: float) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class RouteAgreement:
    """Agreement of several ε(ω) routes on one system
    (``molrs::compute::RouteAgreement``)."""
    def __new__(
        cls, *, part: str | None = None, tolerance: float | None = None
    ) -> Self: ...
    def compute(
        self, *, spectra: Sequence[DielectricSpectrumResult]
    ) -> RouteAgreementCheck: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class SolidLiquid:
    """Solid-liquid classification by bond coherence
    (``molrs::compute::SolidLiquid``)."""
    def __new__(
        cls,
        *,
        q_threshold: float | None = None,
        n_threshold: int | None = None,
        normalize_q: bool | None = None,
    ) -> Self: ...
    def compute(self, *, qlm: QlmTable, neighbors: Neighbors) -> SolidLiquidResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class SolidLiquidResult:
    """Solid-like bonds and particles (``molrs::compute::SolidLiquidResult``)."""
    @property
    def is_solid(self) -> ArrayBool: ...
    @property
    def l(self) -> int: ...
    @property
    def n_solid_bonds(self) -> npt.NDArray[Any]: ...

@final
class SpatialDistribution:
    """Spatial distribution of target atoms in the body frame of a rigid
    reference, over frames (``molrs::compute::SpatialDistribution``)."""
    def __new__(
        cls, grid: RegularGrid, *, bulk_density: float | None = None
    ) -> Self: ...
    def compute(
        self,
        *,
        positions: Sequence[ArrayF],
        simbox: Sequence[Box | None] | None = None,
        reference: Sequence[Sequence[int]],
        template: Sequence[ArrayF],
        target: Sequence[Sequence[int]],
        orientation: Sequence[Sequence[Sequence[int]] | None] | None = None,
    ) -> SpatialDistributionResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class Spectrum:
    """The one-sided Fourier transform of a real signal
    (``molrs::compute::Spectrum``)."""
    def __new__(cls, window: str, pad: int) -> Self: ...
    def compute(self, *, signal: ArrayF, dt: float) -> OneSidedSpectrum: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class SphereVoxelization:
    """Voxels covered by spheres of radius ``r_max`` around the points
    (``molrs::compute::SphereVoxelization``)."""
    def __new__(cls, dims: Sequence[int], r_max: float) -> Self: ...
    def compute(
        self, *, positions: ArrayF, simbox: Box
    ) -> SphereVoxelizationResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class SphereVoxelizationResult:
    """Occupied voxels of spheres on a grid
    (``molrs::compute::SphereVoxelizationResult``)."""
    @property
    def grid(self) -> RegularGrid: ...
    @property
    def raw_counts(self) -> npt.NDArray[Any]: ...
    @property
    def voxels(self) -> npt.NDArray[Any]: ...

@final
class StaticDielectric:
    """Static permittivity from the fluctuation of the total dipole
    (``molrs::compute::StaticDielectric``)."""
    def __new__(
        cls, units: str | UnitPreset, *, epsilon_inf: float | None = None
    ) -> Self: ...
    def compute(
        self, *, dipoles: ArrayF, volume: float, temperature: float, boundary: str
    ) -> StaticDielectricResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class StaticStructureFactorDebye:
    """Static structure factor by the Debye sum over all pairs
    (``molrs::compute::StaticStructureFactorDebye``)."""
    def __new__(
        cls, k_values: Sequence[float], *, sinc_tolerance: float | None = None
    ) -> Self: ...
    def compute(self, *, positions: ArrayF) -> StaticStructureFactorDebyeResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class StaticStructureFactorDebyeResult:
    """S(k) by the Debye sum (``molrs::compute::StaticStructureFactorDebyeResult``)."""
    @property
    def k_values(self) -> ArrayF: ...
    @property
    def n_particles(self) -> int: ...
    @property
    def sk(self) -> ArrayF: ...

@final
class StaticStructureFactorDirect:
    """Static structure factor from the density's Fourier components
    (``molrs::compute::StaticStructureFactorDirect``)."""
    def __new__(
        cls,
        *,
        k_vecs: Sequence[Sequence[float]] | None = None,
        k_max: float | None = None,
        n_bins: int | None = None,
    ) -> Self: ...
    def compute(
        self, *, positions: ArrayF, simbox: Box | None = None
    ) -> StaticStructureFactorDirectResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class StaticStructureFactorDirectResult:
    """S(k) on explicit or isotropically binned k-vectors
    (``molrs::compute::StaticStructureFactorDirectResult``)."""
    @property
    def k_magnitudes(self) -> ArrayF: ...
    @property
    def n_particles(self) -> int: ...
    @property
    def sk(self) -> ArrayF: ...

@final
class Steinhardt:
    """Steinhardt bond-orientational order (``molrs::compute::Steinhardt``)."""
    def __new__(
        cls,
        l: Sequence[int],
        *,
        average: bool | None = None,
        wl: bool | None = None,
        wl_normalize: bool | None = None,
    ) -> Self: ...
    def compute(self, *, neighbors: Neighbors) -> SteinhardtResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
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

@final
class Survival:
    """Survival correlation of a ``(T, C)`` presence series
    (``molrs::compute::Survival``), e.g. ``PairPresence``'s ``values``."""
    def __new__(
        cls,
        kind: str,
        max_lag: int,
        *,
        normalization: str | None = None,
        origin: str | None = None,
    ) -> Self: ...
    def compute(
        self,
        *,
        presence: ArrayF,
        dt: float,
        inner: ArrayF | None = None,
        n_reference: int | None = None,
    ) -> TimeCorrelation: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class TimeCorrelation:
    """A correlation curve (``molrs::compute::TimeCorrelation``):
    ``values[lag, channel]`` at ``lag_times[lag]``."""
    def __new__(cls, lag_times: ArrayF, values: ArrayF) -> Self: ...
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

@final
class Unwrap:
    """Unwrap a wrapped ``(T, N, 3)`` trajectory (``molrs::compute::Unwrap``),
    from int32 image flags when given, else by minimum image between frames."""
    def __init__(self) -> None: ...
    def compute(
        self, *, positions: ArrayF, images: ArrayI32 | None = None, boxes: Sequence[Box]
    ) -> ArrayF: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class Vacf:
    """Velocity autocorrelation (``molrs::compute::Vacf``) — a stream only:
    push each frame's velocities as an ``(n_dof, 1)`` column into
    :meth:`accumulator`."""
    def __new__(
        cls,
        max_lag: int,
        *,
        normalization: str | None = None,
        stride: int | None = None,
        mean: str | None = None,
    ) -> Self: ...
    def accumulator(self) -> CorrelatorAccumulator: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class VanHove:
    """The van Hove correlation G(r, t) over a trajectory
    (``molrs::compute::VanHove``)."""
    def __new__(
        cls,
        axis: Axis,
        lags: Sequence[int],
        *,
        origins: str | None = None,
        stride: int | None = None,
        upper_edge: str | None = None,
    ) -> Self: ...
    def compute(self, *, positions: ArrayF, boxes: Sequence[Box]) -> VanHoveResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class VoronoiCells:
    """A tessellation (``molrs::compute::VoronoiCells``): the ``volumes`` of
    the cells and the ``faces`` of each."""
    @property
    def faces(self) -> list[list[VoronoiFace]]: ...
    def neighbors(self, i: int) -> list[int]: ...
    def total_volume(self) -> float: ...
    @property
    def volumes(self) -> ArrayF: ...

@final
class VoronoiDomainAnalysis:
    """Connected domains of the cells a partition selects
    (``molrs::compute::VoronoiDomainAnalysis``)."""
    def __init__(self) -> None: ...
    def compute(
        self, *, cells: VoronoiCells, partition: Partition
    ) -> VoronoiDomainResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class VoronoiFace:
    """One face of a Voronoi cell (``molrs::compute::VoronoiFace``): its
    ``area`` and the ``neighbor`` across it (negative for a boundary)."""
    @property
    def area(self) -> float: ...
    @property
    def neighbor(self) -> int: ...

@final
class VoronoiIntegration:
    """Integrate a density over the radical Voronoi cells of the atoms and sum
    the cells of each group (``molrs::compute::VoronoiIntegration``)."""
    def __new__(cls, unit: str) -> Self: ...
    def compute(
        self,
        *,
        positions: ArrayF,
        radii: ArrayF,
        nuclear_charges: ArrayF,
        partition: Partition,
        simbox: Box,
        grid: RegularGrid,
        density: ArrayF,
    ) -> MolecularMoments: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
class VoronoiVoidAnalysis:
    """Cavities formed by connected void cells
    (``molrs::compute::VoronoiVoidAnalysis``)."""
    def __init__(self) -> None: ...
    def compute(
        self, *, cells: VoronoiCells, is_void: Sequence[bool], box_volume: float
    ) -> VoronoiVoidResult: ...
    @staticmethod
    def spec() -> ComputeSpec: ...
    def tunables(self) -> list[tuple[str, Any]]: ...

@final
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

@final
class ExplicitTerms:
    """One style's terms over explicit instances
    (``molrs::ff::compile::ExplicitTerms``): ``atoms`` ``(n, arity)``, each
    per-term parameter a number (broadcast) or one value per term, as stored
    (angle values in degrees); style parameters (``cutoff``, ``coulomb``, an
    unregistered style's ``expression``) a number or a string. Works for
    every registered style, built-in or custom. A parameter the style does
    not declare raises ``TypeError``."""

    def __new__(
        cls,
        category: str,
        style: str,
        atoms: Sequence[Sequence[int]] | ArrayI64 | ArrayU32,
        *,
        charges: Sequence[float] | ArrayF | None = None,
        **params: float | str | Sequence[float] | Sequence[str] | ArrayF,
    ) -> Self: ...
    def compile(self) -> Potentials:
        """The terms' kernel, built as ``PotentialCompiler.compile`` builds
        it, one type per term. Refusals raise their ``molrs.ff.ir.IrError``
        subclass."""

class PairLjCut:
    """LAMMPS ``pair_style lj/cut``: the one-type cut Lennard-Jones / Mie
    kernel (``n``/``m`` exponents) a neighbour loop feeds (MD's nonbond
    kernel). A pair list with a row per pair is
    ``ExplicitTerms("pair", "lj/cut", pairs, epsilon=..., sigma=...).compile()``."""

    def __new__(
        cls,
        epsilon: float,
        sigma: float,
        cutoff: float,
        *,
        n: int = 12,
        m: int = 6,
        shifted: bool = True,
        smeared: bool = False,
    ) -> Self: ...
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
    def pair_energy_force(
        self, r2: float, disp: Sequence[float]
    ) -> tuple[float, list[float]] | None:
        """Energy and force of one pair (``PairPotential::pair_energy_force``);
        ``None`` beyond the cutoff."""
    def evaluate_pairs(
        self, pos: ArrayF, neighbors: Neighbors, factor: ArrayF | None = None
    ) -> tuple[float, ArrayF]:
        """``(energy, forces)`` over every pair of ``neighbors`` (built with
        ``disp=True``; ``PairDriven::evaluate_pairs``), forces ``(N, 3)`` for
        the rows of ``pos``; ``factor`` is one weight per pair (``None``: all
        ones)."""

# ---------------------------------------------------------------------------
# molrs.md — in-process MD (mirrors molrs-python/src/md.rs).
# Submodule stubbed as a class namespace, following the `keys` precedent.
# ---------------------------------------------------------------------------

class md:
    """The ``_native.md`` submodule (``molrs.md``): NVE/Langevin integrators.
    MD defines no potential; it integrates an ``PairLjCut``, a ``Potentials``
    collection, or any object with ``evaluate(pos) -> (energy, forces)``.

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
        ``Potentials`` / an object with ``evaluate``) and
        ``neighbors`` (a ``VerletSkin``) are
        moved in; the loop feeds fresh pairs to the potential after each
        rebuild. ``PairLjCut`` requires ``neighbors=``."""

        def __init__(
            self,
            dt: float,
            *,
            potential: PairLjCut | Potentials | WeightedTerms | _PotentialLike,
            neighbors: VerletSkin | None = None,
            mass: ArrayF,
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
            potential: PairLjCut | Potentials | WeightedTerms | _PotentialLike,
            neighbors: VerletSkin | None = None,
            mass: ArrayF,
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
        def velocities(self, pos: ArrayF, mass: ArrayF) -> ArrayF: ...

class ir:
    """The ``_native.ir`` submodule: the force-field IR's vocabulary
    (``molrs::ff::ir``), surfaced as :mod:`molrs.ff.ir` — the category and
    style specs and, for what does not conform, the :class:`IrError`
    subclass named after the Rust variant. The registry that takes new
    categories and styles is :class:`style_registry`."""

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
        def default(self) -> ParamValue | None: ...
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

class style_registry:
    """The ``_native.style_registry`` submodule (``molrs::ff::style_registry``),
    surfaced as :mod:`molrs.ff.style_registry`: register a category or a
    style — by an expression, a vectorised Python kernel, or both — into the
    process-wide registry every compile reads."""

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
    def register_engine_form(
        engine: str, category: str, name: str, form: str
    ) -> None: ...
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
    method reads the result. Atoms are the molecule's own views; an atom
    argument must be an atom of that molecule (``ValueError`` otherwise).

    Not to be confused with :func:`assign_rings`, which *decorates* the graph
    with ring flags, in place. This type *reports*, and never touches the
    molecule.

    Examples
    --------
    >>> rings = molrs.perceive.perceive_rings(molrs.io.smiles.SmilesIr.parse("c1ccccc1").to_atomistic())
    >>> rings.n_rings()
    1
    >>> rings.ring_sizes()
    [6]
    """
    def is_atom_in_ring(self, atom: Atom) -> bool: ...
    def max_ring_system_size(self) -> int:
        """Atom count of the largest fused / bridged ring system
        (naphthalene → 10); ``0`` for an acyclic molecule."""
    def n_atom_rings(self, atom: Atom) -> int: ...
    def n_rings(self) -> int: ...
    def ring_sizes(self) -> list[int]: ...
    def ring_systems(self) -> list[list[Atom]]: ...
    def rings(self) -> list[list[Atom]]: ...
    def smallest_ring_containing_atom(self, atom: Atom) -> int | None: ...

def infer_lammps_atom_style(frame: Frame) -> str: ...

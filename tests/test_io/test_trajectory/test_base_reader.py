"""Contract tests for the reader hierarchy.

These tests encode the reader-hierarchy contract:

  - molpy.io.base.BaseReader
  - molpy.io.trajectory.base.BaseTrajectoryReader (pure, mmap-free)

No external resources are used; everything is in-memory or tmp_path.
"""

from __future__ import annotations

import abc
import collections.abc
import inspect
from pathlib import Path

import pytest

import molrs

import molpy as mp


def _make_frame(n_atoms: int, element: str) -> molrs.Frame:
    """Build a tiny, identifiable Frame with ``n_atoms`` atoms of ``element``."""
    return molrs.Frame(
        blocks={
            "atoms": {
                "element": [element] * n_atoms,
                "x": [float(i) for i in range(n_atoms)],
                "y": [0.0] * n_atoms,
                "z": [0.0] * n_atoms,
            }
        }
    )


def _atom_count(frame: molrs.Frame) -> int:
    return len(frame["atoms"]["element"])


# ---------------------------------------------------------------------------
# ac-001 — BaseReader: ABC, path normalization, existence, context manager
# ---------------------------------------------------------------------------


def test_ac001_base_reader_is_abc() -> None:
    """(ac-001) BaseReader is an abc.ABC subclass."""
    from molpy.io.base import BaseReader

    assert issubclass(BaseReader, abc.ABC)


def test_ac001_base_reader_normalizes_existing_path(tmp_path: Path) -> None:
    """(ac-001) A single existing str|Path normalizes into fpaths=[Path]."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    f = tmp_path / "exists.txt"
    f.write_text("data")

    reader = _Concrete(str(f))
    assert reader.fpaths == [Path(f)]


def test_ac001_base_reader_missing_path_raises(tmp_path: Path) -> None:
    """(ac-001) must_exist (default) raises FileNotFoundError for a missing path."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        _Concrete(missing)


def test_ac001_base_reader_must_exist_false_does_not_raise(tmp_path: Path) -> None:
    """(ac-001) must_exist=False tolerates a missing path."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    missing = tmp_path / "nope.txt"
    reader = _Concrete(missing, must_exist=False)
    assert reader.fpaths == [Path(missing)]


def test_ac001_base_reader_fpath_property(tmp_path: Path) -> None:
    """(ac-001) fpath property returns the first path."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    f = tmp_path / "exists.txt"
    f.write_text("data")
    reader = _Concrete(f)
    assert reader.fpath == Path(f)


def test_ac001_base_reader_fpath_empty_raises() -> None:
    """(ac-001) fpath raises ValueError when there are no paths."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    reader = _Concrete([], must_exist=False)
    with pytest.raises(ValueError):
        _ = reader.fpath


def test_ac001_base_reader_enter_returns_self(tmp_path: Path) -> None:
    """(ac-001) __enter__ returns the reader instance itself."""
    from molpy.io.base import BaseReader

    class _Concrete(BaseReader):
        pass

    f = tmp_path / "exists.txt"
    f.write_text("data")
    reader = _Concrete(f)
    with reader as entered:
        assert entered is reader


def test_ac001_base_reader_exit_calls_close(tmp_path: Path) -> None:
    """(ac-001) __exit__ calls close()."""
    from molpy.io.base import BaseReader

    class _ClosingReader(BaseReader):
        def __init__(self, fpath, *, must_exist: bool = True) -> None:
            super().__init__(fpath, must_exist=must_exist)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    f = tmp_path / "exists.txt"
    f.write_text("data")
    reader = _ClosingReader(f)
    with reader:
        assert reader.closed is False
    assert reader.closed is True


# ---------------------------------------------------------------------------
# ac-003 — Purity: BaseTrajectoryReader is an abstract, mmap-free Iterable[Frame]
# ---------------------------------------------------------------------------


def test_ac003_base_trajectory_reader_subclasses_base_reader() -> None:
    """(ac-003) BaseTrajectoryReader is a BaseReader."""
    import molpy.io.base
    from molpy.io.trajectory.base import BaseTrajectoryReader

    assert issubclass(BaseTrajectoryReader, molpy.io.base.BaseReader)


def test_ac003_base_trajectory_reader_is_iterable_subclass() -> None:
    """(ac-003) BaseTrajectoryReader is a collections.abc.Iterable subclass."""
    from molpy.io.trajectory.base import BaseTrajectoryReader

    assert issubclass(BaseTrajectoryReader, collections.abc.Iterable)


def test_ac003_base_trajectory_reader_is_abstract() -> None:
    """(ac-003) BaseTrajectoryReader is abstract (read_frame / n_frames)."""
    from molpy.io.trajectory.base import BaseTrajectoryReader

    assert inspect.isabstract(BaseTrajectoryReader)


def test_ac003_base_trajectory_reader_source_is_mmap_free() -> None:
    """(ac-003) The pure class body contains no mmap machinery."""
    from molpy.io.trajectory.base import BaseTrajectoryReader

    source = inspect.getsource(BaseTrajectoryReader)
    forbidden = (
        "mmap",
        "MappedFile",
        "FrameLocation",
        "_open_files",
        "_load_indexes",
        "_save_index",
    )
    for token in forbidden:
        assert token not in source, f"pure class body must not reference {token!r}"


# ---------------------------------------------------------------------------
# ac-004 — Decoupling proof: concrete iteration in pure terms (no mmap, no files)
# ---------------------------------------------------------------------------


@pytest.fixture
def list_reader_cls():
    from molpy.io.trajectory.base import BaseTrajectoryReader

    class ListTrajectoryReader(BaseTrajectoryReader):
        """In-memory reader: pure read_frame/n_frames, no super().__init__, no files."""

        def __init__(self, frames: list) -> None:
            self._frames = list(frames)

        def read_frame(self, index: int):
            return self._frames[index]

        @property
        def n_frames(self) -> int:
            return len(self._frames)

    return ListTrajectoryReader


@pytest.fixture
def frames() -> list:
    # Five distinct, identifiable frames (distinguishing atom count + element).
    return [
        _make_frame(1, "H"),
        _make_frame(2, "C"),
        _make_frame(3, "N"),
        _make_frame(4, "O"),
        _make_frame(5, "S"),
    ]


@pytest.fixture
def reader(list_reader_cls, frames):
    return list_reader_cls(frames)


def test_ac004_len(reader) -> None:
    """(ac-004) __len__ == n_frames."""
    assert len(reader) == 5


def test_ac004_iter_order(reader, frames) -> None:
    """(ac-004) iteration yields all frames in order."""
    got = list(iter(reader))
    assert [_atom_count(f) for f in got] == [_atom_count(f) for f in frames]


def test_ac004_getitem_positive_index(reader, frames) -> None:
    """(ac-004) reader[0] is the first frame."""
    assert _atom_count(reader[0]) == _atom_count(frames[0])


def test_ac004_getitem_negative_index(reader, frames) -> None:
    """(ac-004) reader[-1] is the last frame."""
    assert _atom_count(reader[-1]) == _atom_count(frames[-1])


def test_ac004_getitem_slice(reader, frames) -> None:
    """(ac-004) reader[1:4] returns the right 3 frames as a list."""
    result = reader[1:4]
    assert isinstance(result, list)
    assert [_atom_count(f) for f in result] == [_atom_count(f) for f in frames[1:4]]


def test_ac004_getitem_strided_slice(reader, frames) -> None:
    """(ac-004) reader[::2] returns the right strided list."""
    result = reader[::2]
    assert [_atom_count(f) for f in result] == [_atom_count(f) for f in frames[::2]]


def test_ac004_read_all(reader, frames) -> None:
    """(ac-004) read_all() returns all frames."""
    result = reader.read_all()
    assert [_atom_count(f) for f in result] == [_atom_count(f) for f in frames]


def test_ac004_read_range(reader, frames) -> None:
    """(ac-004) read_range(1, 4) returns frames 1..3."""
    result = reader.read_range(1, 4)
    assert [_atom_count(f) for f in result] == [_atom_count(f) for f in frames[1:4]]


def test_ac004_read_frames(reader, frames) -> None:
    """(ac-004) read_frames([0, 2, 4]) returns those frames."""
    result = reader.read_frames([0, 2, 4])
    assert [_atom_count(f) for f in result] == [
        _atom_count(frames[i]) for i in (0, 2, 4)
    ]


def test_ac004_getitem_out_of_range_raises(reader) -> None:
    """(ac-004) out-of-range index raises IndexError."""
    with pytest.raises(IndexError):
        _ = reader[99]


def test_ac004_getitem_bad_type_raises(reader) -> None:
    """(ac-004) non int/slice key raises TypeError."""
    with pytest.raises(TypeError):
        _ = reader[object()]

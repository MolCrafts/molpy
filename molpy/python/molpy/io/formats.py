"""Format table for :func:`molpy.io.convert`.

Each row points at a reader and a writer that already exist. ``convert``
walks the table; it does not parse a file itself. A new frame format is one
:func:`register` call.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any

import molpy._core as _core
from molpy._core import (
    CifReader,
    CifWriter,
    CubeReader,
    DcdReader,
    DcdWriter,
    GroReader,
    GroWriter,
    LammpsDumpReader,
    LammpsDumpWriter,
    Mol2Reader,
    Mol2Writer,
    PdbReader,
    PdbWriter,
    TrrReader,
    TrrWriter,
    XtcReader,
    XtcWriter,
    XyzReader,
    XyzWriter,
    read_cif,
    read_cube,
    read_dcd,
    read_gro,
    read_lammps_data,
    read_lammps_dump,
    read_mol2,
    read_pdb,
    read_poscar,
    read_trr,
    read_xsf,
    read_xtc,
    read_xyz,
    write_cif,
    write_cube,
    write_dcd,
    write_gro,
    write_lammps_data,
    write_lammps_dump,
    write_mol2,
    write_pdb,
    write_poscar,
    write_trr,
    write_xsf,
    write_xtc,
    write_xyz,
)

Read = Callable[[Path], Any]
Write = Callable[[Path, Any], None]
ReadFrames = Callable[[Path], Iterable[Any]]
WriteFrames = Callable[[Path, Iterable[Any]], None]
Sniff = Callable[[Path], bool]


@dataclass(frozen=True, slots=True)
class Format:
    """One file format ``convert`` can read or write.

    ``suffixes`` are matched against the file name. A suffix that starts with
    ``.`` matches the end of the name (``.mrec.zip`` before ``.mrec``). Any
    other suffix matches the whole name (``POSCAR``). A trailing ``.gz`` is
    ignored for the match. Readers decompress it, and ``convert`` compresses
    a destination that ends in it.

    ``reader`` / ``writer`` follow the classes in :mod:`molpy.io`: iteration
    yields frames, ``write(frame)`` appends one, and both are context managers.
    ``read_frames`` / ``write_frames`` cover a format whose doors are not that
    shape. ``sniff`` is called only when one name matches more than one row.

    Args:
        name: The name ``in_format`` and ``out_format`` use.
        suffixes: File-name suffixes this row claims.
        read: Read the first frame.
        write: Write one frame.
        reader: Read every frame.
        writer: Write every frame.
        read_frames: Read every frame when ``reader`` does not fit.
        write_frames: Write every frame when ``writer`` does not fit.
        sniff: Return whether ``path`` is this format.
    """

    name: str
    suffixes: tuple[str, ...]
    read: Read | None = None
    write: Write | None = None
    reader: type[Any] | None = None
    writer: type[Any] | None = None
    read_frames: ReadFrames | None = None
    write_frames: WriteFrames | None = None
    sniff: Sniff | None = None


FORMATS: list[Format] = []


def register(fmt: Format) -> None:
    """Add ``fmt`` to :data:`FORMATS`.

    Raises:
        ValueError: ``fmt.name`` is empty or already registered, ``fmt`` has
            neither a reader nor a writer, or a suffix is already claimed by
            a row that does not sniff.
    """
    if fmt.name == "":
        raise ValueError("a format name is empty")
    if any(item.name == fmt.name for item in FORMATS):
        raise ValueError(f"format {fmt.name!r} is already registered")
    readable = fmt.read or fmt.reader or fmt.read_frames
    writable = fmt.write or fmt.writer or fmt.write_frames
    if readable is None and writable is None:
        raise ValueError(f"format {fmt.name!r} has no reader and no writer")
    for suffix in fmt.suffixes:
        for item in FORMATS:
            if suffix not in item.suffixes:
                continue
            if fmt.sniff is None or item.sniff is None:
                raise ValueError(
                    f"suffix {suffix!r} belongs to {item.name!r}; "
                    f"{fmt.name!r} needs a sniff to share it"
                )
    FORMATS.append(fmt)


def known_formats() -> str:
    """The registered names, in registration order, separated by commas."""
    return ", ".join(fmt.name for fmt in FORMATS)


def resolve(path: Path, explicit: str | None) -> Format:
    """The row for ``path``.

    ``explicit`` is a format name and wins. Otherwise the longest matching
    suffix wins. Several rows at that length are narrowed with ``sniff``.

    Raises:
        ValueError: The name is unknown, the suffix matches nothing, or more
            than one row still matches.
    """
    if explicit is not None:
        return _by_name(explicit)
    matches = _by_suffix(path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(
            f"cannot tell the format of {path}. "
            f"Pass in_format or out_format. Known formats: {known_formats()}"
        )
    sniffed = [item for item in matches if item.sniff is not None and item.sniff(path)]
    if len(sniffed) == 1:
        return sniffed[0]
    names = ", ".join(item.name for item in matches)
    raise ValueError(f"{path} matches {names}. Pass in_format or out_format.")


def convert(
    in_path: str | PathLike[str],
    out_path: str | PathLike[str],
    *,
    in_format: str | None = None,
    out_format: str | None = None,
    frame: int | None = None,
) -> None:
    """Copy the frames of ``in_path`` into ``out_path``.

    One input frame is written with the output row's single-frame writer.
    More than one frame is written with its trajectory writer. An output
    format that stores one frame raises unless ``frame`` selects one, counting
    from 0.

    Args:
        in_path: File to read. A bare string is a path.
        out_path: File to write.
        in_format: Format name. The suffix is used when this is omitted.
        out_format: Format name. The suffix is used when this is omitted.
        frame: The only frame to write, counting from 0.

    A destination whose name ends in ``.gz`` receives the gzip of the file
    the format writer produced. A format that writes a directory (an
    unpacked ``.mrec``) cannot be gzipped.

    Raises:
        FileNotFoundError: ``in_path`` does not exist.
        ValueError: A format name or suffix does not resolve, ``frame`` is
            outside the file, the output format cannot store every frame,
            or a ``.gz`` destination is not a single file.
    """
    source = Path(in_path)
    destination = Path(out_path)
    if not source.exists():
        raise FileNotFoundError(source)
    incoming = resolve(source, in_format)
    outgoing = resolve(destination, out_format)
    frames: Iterable[Any] = _read_frames(incoming, source)

    def write(target: Path) -> None:
        if frame is not None:
            _write_one(outgoing, target, _nth(frames, frame, source))
            return
        _write_all(outgoing, target, frames, source)

    _place(destination, write)


def _by_name(name: str) -> Format:
    for item in FORMATS:
        if item.name == name:
            return item
    raise ValueError(f"unknown format {name!r}. Known formats: {known_formats()}")


def _by_suffix(path: Path) -> list[Format]:
    filename = path.name
    if filename.lower().endswith(".gz"):
        filename = filename[:-3]
    found: list[tuple[int, Format]] = []
    for item in FORMATS:
        for suffix in item.suffixes:
            if _suffix_matches(filename, suffix):
                found.append((len(suffix), item))
    if not found:
        return []
    best = max(length for length, _item in found)
    chosen: list[Format] = []
    seen: set[str] = set()
    for length, item in found:
        if length == best and item.name not in seen:
            chosen.append(item)
            seen.add(item.name)
    return chosen


def _suffix_matches(filename: str, suffix: str) -> bool:
    if suffix.startswith("."):
        return filename.lower().endswith(suffix.lower())
    return filename == suffix or filename.upper() == suffix.upper()


def _read_frames(fmt: Format, path: Path) -> Iterator[Any]:
    if fmt.read_frames is not None:
        yield from fmt.read_frames(path)
        return
    if fmt.reader is not None:
        with fmt.reader(path) as reader:
            yield from reader
        return
    if fmt.read is None:
        raise ValueError(f"format {fmt.name!r} cannot be read")
    yield fmt.read(path)


def _nth(frames: Iterable[Any], index: int, source: Path) -> Any:
    if index < 0:
        raise ValueError(f"frame {index} is negative")
    for at, frame in enumerate(frames):
        if at == index:
            return frame
    raise ValueError(f"frame {index} is past the end of {source}")


def _write_one(fmt: Format, path: Path, frame: Any) -> None:
    if fmt.write is not None:
        fmt.write(path, frame)
        return
    if fmt.writer is not None:
        with fmt.writer(path) as writer:
            writer.write(frame)
        return
    if fmt.write_frames is not None:
        fmt.write_frames(path, (frame,))
        return
    raise ValueError(f"format {fmt.name!r} cannot be written")


def _write_all(fmt: Format, path: Path, frames: Iterable[Any], source: Path) -> None:
    stream = iter(frames)
    try:
        first = next(stream)
    except StopIteration:
        raise ValueError(f"{source} has no frames") from None
    second = next(stream, None)
    if second is None:
        _write_one(fmt, path, first)
        return
    rest = chain((first, second), stream)
    if fmt.write_frames is not None:
        fmt.write_frames(path, rest)
        return
    if fmt.writer is not None:
        with fmt.writer(path) as writer:
            writer.write(first)
            writer.write(second)
            for frame in stream:
                writer.write(frame)
        return
    raise ValueError(
        f"{fmt.name} writes one frame and {source} has more than one. Pass frame=N."
    )


def _read_mrec(path: Path) -> Iterator[Any]:
    from molpy.io.mrec import MrecFile, MrecReader

    record = MrecFile(path)
    if "trajectory" in record.sections():
        with MrecReader(path) as reader:
            yield from reader
        return
    frame = record.frame()
    if frame is None:
        raise ValueError(f"{path} has no frame or trajectory section")
    yield frame


def _write_mrec_frame(path: Path, frame: Any) -> None:
    from molpy._core import write_mrec

    if path.name.lower().endswith(".mrec.zip"):
        packed = bytearray()
        write_mrec(packed, frame=frame)
        path.write_bytes(packed)
        return
    write_mrec(path, frame=frame)


def _place(destination: Path, write: Callable[[Path], None]) -> None:
    """Write to ``destination``, gzipping when its name ends in ``.gz``.

    The format writer sees the name with ``.gz`` removed, in a temporary
    directory, so a suffix-sensitive writer still matches. The bytes of
    that file are what gets compressed. A directory has no single payload.
    """
    name = destination.name
    if not name.lower().endswith(".gz"):
        write(destination)
        return
    inner = name[:-3]
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / inner
        write(staged)
        if not staged.is_file():
            raise ValueError(
                f"{destination} ends in .gz but the {inner} writer did not "
                "produce a single file"
            )
        destination.write_bytes(gzip.compress(staged.read_bytes()))


def _mrec_store(path: Path) -> tuple[Path, Path | None]:
    """The directory store for ``path``, and the archive path when packed.

    ``pack_mrec_zip`` appends ``.zip`` only when the store extension is
    exactly ``mrec``. ``Run.MREC.ZIP`` is staged as ``Run.mrec`` and the
    archive is renamed onto the path the caller asked for.
    """
    name = path.name
    if not name.lower().endswith(".mrec.zip"):
        return path, None
    stem = name[: -len(".zip")]
    store_name = stem[: -len(".mrec")] + ".mrec"
    return path.with_name(store_name), path


def _discard_store(store: Path) -> None:
    if store.is_dir():
        shutil.rmtree(store)
    elif store.exists():
        store.unlink()


def _write_mrec_frames(path: Path, frames: Iterable[Any]) -> None:
    from molpy.io.mrec import MrecWriter, SequenceSchema, pack_mrec_zip

    buffered = list(frames)
    if len(buffered) == 0:
        raise ValueError(f"{path} has no frames")
    if len(buffered) == 1:
        _write_mrec_frame(path, buffered[0])
        return
    store, archive = _mrec_store(path)
    # The pin is the union of the run. A schema minted from the first frame
    # alone declares a column non-nullable, and the writer then refuses a
    # mask that shows up on a later frame.
    schema = SequenceSchema.from_frames(buffered)
    existed = store.exists()
    try:
        with MrecWriter(store, schema) as writer:
            for frame in buffered:
                writer.append(frame)
    except Exception:
        if not existed:
            _discard_store(store)
        raise
    if archive is None:
        return
    produced = Path(pack_mrec_zip(store))
    if produced != archive:
        produced.replace(archive)


def _register_builtin() -> None:
    rows = (
        Format("xyz", (".xyz",), read_xyz, write_xyz, XyzReader, XyzWriter),
        Format("pdb", (".pdb",), read_pdb, write_pdb, PdbReader, PdbWriter),
        Format("gro", (".gro",), read_gro, write_gro, GroReader, GroWriter),
        Format("mol2", (".mol2",), read_mol2, write_mol2, Mol2Reader, Mol2Writer),
        Format("cif", (".cif",), read_cif, write_cif, CifReader, CifWriter),
        Format("xsf", (".xsf",), read_xsf, write_xsf),
        Format("cube", (".cube",), read_cube, write_cube, CubeReader),
        Format(
            "poscar",
            (".poscar", ".vasp", "POSCAR", "CONTCAR"),
            read_poscar,
            write_poscar,
        ),
        Format("lammps-data", (".data",), read_lammps_data, write_lammps_data),
        Format(
            "lammps-dump",
            (".lammpstrj", ".dump"),
            read_lammps_dump,
            write_lammps_dump,
            LammpsDumpReader,
            LammpsDumpWriter,
        ),
        Format("dcd", (".dcd",), read_dcd, write_dcd, DcdReader, DcdWriter),
        Format("trr", (".trr",), read_trr, write_trr, TrrReader, TrrWriter),
        Format("xtc", (".xtc",), read_xtc, write_xtc, XtcReader, XtcWriter),
    )
    for row in rows:
        register(row)
    if not hasattr(_core, "write_mrec"):
        return
    register(
        Format(
            "mrec",
            (".mrec.zip", ".mrec"),
            write=_write_mrec_frame,
            read_frames=_read_mrec,
            write_frames=_write_mrec_frames,
        )
    )


_register_builtin()


__all__ = [
    "FORMATS",
    "Format",
    "convert",
    "known_formats",
    "register",
    "resolve",
]

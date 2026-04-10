"""Trajectory index data structures and serialization.

Provides immutable data structures for tracking frame locations within
trajectory files, with stale-index detection and persistent serialization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import getLogger as get_logger
from pathlib import Path

logger = get_logger(__name__)

_INDEX_VERSION = 1
_HEADER_HASH_BYTES = 65536  # 64 KB for content fingerprint

# Try msgpack for fast binary serialization, fall back to JSON
try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Identifies a specific version of a trajectory file."""

    path_relative: str
    path_absolute: str
    size_bytes: int
    mtime_ns: int
    header_hash: str  # hex-encoded SHA-256 of first 64KB


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """Location of a single frame within a trajectory file."""

    file_idx: int
    byte_offset: int
    byte_length: int
    timestep: int | None


@dataclass(frozen=True, slots=True)
class TrajectoryIndex:
    """Complete trajectory index (immutable)."""

    version: int
    format_id: str
    files: tuple[FileFingerprint, ...]
    frames: tuple[FrameEntry, ...]
    created_utc: str

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def frame_for_timestep(self, timestep: int) -> FrameEntry | None:
        """Find frame by simulation timestep (binary search if sorted)."""
        # Try binary search assuming sorted timesteps
        lo, hi = 0, len(self.frames) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_ts = self.frames[mid].timestep
            if mid_ts is None:
                break  # Fall back to linear scan
            if mid_ts == timestep:
                return self.frames[mid]
            elif mid_ts < timestep:
                lo = mid + 1
            else:
                hi = mid - 1

        # Linear fallback
        for entry in self.frames:
            if entry.timestep == timestep:
                return entry
        return None


def _hash_header(path: Path) -> str:
    """SHA-256 hex digest of the first 64KB of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(_HEADER_HASH_BYTES))
    return h.hexdigest()


def create_fingerprint(path: Path, index_dir: Path | None = None) -> FileFingerprint:
    """Create a fingerprint for a trajectory file.

    Args:
        path: Absolute path to the trajectory file.
        index_dir: Directory where the index will be stored (for relative path).
    """
    stat = path.stat()
    rel = str(path.relative_to(index_dir)) if index_dir else str(path)
    return FileFingerprint(
        path_relative=rel,
        path_absolute=str(path.resolve()),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        header_hash=_hash_header(path),
    )


def is_index_fresh(index: TrajectoryIndex, files: list[Path]) -> bool:
    """Check whether the index is still valid for the given files.

    Three-level check:
    1. File count mismatch -> stale
    2. File size mismatch -> stale (fastest)
    3. mtime changed -> check header hash (handles benign `touch`)
    """
    if len(files) != len(index.files):
        return False
    if index.version != _INDEX_VERSION:
        return False

    for fp, stored in zip(files, index.files):
        try:
            stat = fp.stat()
        except OSError:
            return False
        if stat.st_size != stored.size_bytes:
            return False
        if stat.st_mtime_ns != stored.mtime_ns:
            if _hash_header(fp) != stored.header_hash:
                return False
    return True


# ── Serialization ──────────────────────────────────────────────────


def _index_to_dict(index: TrajectoryIndex) -> dict:
    """Convert index to a plain dict for serialization."""
    return {
        "version": index.version,
        "format_id": index.format_id,
        "created_utc": index.created_utc,
        "files": [
            {
                "path_relative": fp.path_relative,
                "path_absolute": fp.path_absolute,
                "size_bytes": fp.size_bytes,
                "mtime_ns": fp.mtime_ns,
                "header_hash": fp.header_hash,
            }
            for fp in index.files
        ],
        "frames": [
            {
                "file_idx": fe.file_idx,
                "byte_offset": fe.byte_offset,
                "byte_length": fe.byte_length,
                "timestep": fe.timestep,
            }
            for fe in index.frames
        ],
    }


def _dict_to_index(d: dict) -> TrajectoryIndex:
    """Reconstruct index from a plain dict."""
    return TrajectoryIndex(
        version=d["version"],
        format_id=d["format_id"],
        created_utc=d["created_utc"],
        files=tuple(
            FileFingerprint(
                path_relative=f["path_relative"],
                path_absolute=f["path_absolute"],
                size_bytes=f["size_bytes"],
                mtime_ns=f["mtime_ns"],
                header_hash=f["header_hash"],
            )
            for f in d["files"]
        ),
        frames=tuple(
            FrameEntry(
                file_idx=fe["file_idx"],
                byte_offset=fe["byte_offset"],
                byte_length=fe["byte_length"],
                timestep=fe["timestep"],
            )
            for fe in d["frames"]
        ),
    )


def serialize_index(index: TrajectoryIndex) -> bytes:
    """Serialize index to bytes (msgpack preferred, JSON fallback)."""
    d = _index_to_dict(index)
    if _HAS_MSGPACK:
        return msgpack.packb(d, use_bin_type=True)
    return json.dumps(d).encode("utf-8")


def deserialize_index(data: bytes) -> TrajectoryIndex:
    """Deserialize index, auto-detecting format."""
    if _HAS_MSGPACK:
        try:
            d = msgpack.unpackb(data, raw=False)
            return _dict_to_index(d)
        except Exception:
            pass  # Fall through to JSON
    d = json.loads(data.decode("utf-8"))
    return _dict_to_index(d)


def index_file_path(traj_path: Path) -> Path:
    """Canonical index file path for a trajectory file."""
    suffix = ".tridx" if _HAS_MSGPACK else ".tridx.json"
    return traj_path.with_suffix(traj_path.suffix + suffix)


def save_index(index: TrajectoryIndex, path: Path) -> None:
    """Persist index to disk."""
    data = serialize_index(index)
    path.write_bytes(data)
    logger.debug("Saved trajectory index to %s (%d bytes)", path, len(data))


def load_index(path: Path) -> TrajectoryIndex | None:
    """Load index from disk, returning None on failure."""
    try:
        data = path.read_bytes()
        return deserialize_index(data)
    except Exception as e:
        logger.warning("Failed to load index from %s: %s", path, e)
        return None


def build_index(
    format_id: str,
    files: list[Path],
    frames: list[FrameEntry],
    index_dir: Path | None = None,
) -> TrajectoryIndex:
    """Create a new TrajectoryIndex from scan results.

    Args:
        format_id: Format identifier (e.g. "lammps-dump", "xyz").
        files: List of trajectory file paths.
        frames: List of FrameEntry from scanning.
        index_dir: Base directory for relative paths.
    """
    fingerprints = tuple(create_fingerprint(f, index_dir) for f in files)
    return TrajectoryIndex(
        version=_INDEX_VERSION,
        format_id=format_id,
        files=fingerprints,
        frames=tuple(frames),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )

"""The metric-reader contract molpy's format modules implement (private).

A viewer (molplot, through molexp) never parses anything and never imports
molpy: it looks a reader up in the ``molcrafts.metric_readers`` entry-point
group, by format, and asks it for records. The contract is a structural
Protocol — ``format`` / ``sniff`` / ``read`` plus the optional ``patterns``
and ``tailable`` hints — so matching it is a matter of shape, not of
dependency. The readers themselves are their formats' modules'
(:class:`molpy.io.lammps.LammpsLogMetricReader`,
:class:`molpy.io.mlp_jsonl.MlpJsonlMetricReader`,
:class:`molpy.io.mrec.MrecMetricReader`); this module holds what they share.

**The viewer sets the sampling policy.** ``read`` receives a request carrying
``stride`` — how many samples of each series to skip — and each reader pushes
it as far down as its parser allows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

PROBE_BYTES = 64 * 1024


class ReadRequest(Protocol):
    """What the viewer asked for. Supplied by the platform, matched by shape."""

    since: int
    stride: int
    limit: int | None
    metric_type: str | None
    keys: tuple[str, ...]


def wanted(request: ReadRequest | None, record: dict[str, Any]) -> bool:
    if request is None:
        return True
    if request.metric_type is not None and record.get("t") != request.metric_type:
        return False
    return not (request.keys and record.get("k") not in request.keys)


def stride_of(request: ReadRequest | None) -> int:
    return 1 if request is None else max(1, request.stride)


def limit_of(request: ReadRequest | None) -> int | None:
    return None if request is None else request.limit


def since_of(request: ReadRequest | None) -> int:
    return 0 if request is None else max(0, request.since)


def head(path: Path, size: int = PROBE_BYTES) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)

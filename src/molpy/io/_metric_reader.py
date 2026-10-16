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

**The viewer sets the sampling policy; one generator enforces it.** Each
reader turns its file into a *raw stream* — every record it can produce, in a
fixed order, before any filter or sampling — and hands that stream to
:func:`apply_request`, so a request means the same thing whatever the format:

``since``
    The number of records of the raw stream to skip: a raw index, counted
    before the ``metric_type`` / ``keys`` filters and before ``stride``. A
    viewer resumes a source by passing back how many raw records it has
    already consumed. This is the semantics of molab's reference
    ``metrics_ingest.readers.apply``.
``stride``
    Keep every *stride*-th sample **of each series** (``k``) among the records
    that survive ``since`` and the filters. Records interleave by series, so
    striding the flat stream would keep one series and drop the rest.
``limit``
    Stop after this many records are yielded; ``0`` or less yields nothing.

**Every record's ``s`` is the simulation step of its sample** where the
source records one — the ``Step`` thermo column of a LAMMPS log, the ``step``
column of an ``*.mrec`` record. Where it does not, ``s`` is the sample's
ordinal in its source (the thermo row counted across every run; the frame
index). ``*.mlp.jsonl`` records carry the ``s`` their writer gave them.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

PROBE_BYTES = 64 * 1024


class ReadRequest(Protocol):
    """What the viewer asked for. Supplied by the platform, matched by shape."""

    since: int
    stride: int
    limit: int | None
    metric_type: str | None
    keys: tuple[str, ...]


def apply_request(
    request: ReadRequest | None, records: Iterable[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Enforce *request* over a reader's raw record stream.

    Args:
        request: The viewer's policy, or ``None`` for every record.
        records: The raw stream, lazily: records past the ``limit`` are never
            built.

    Yields:
        The records the request selects, in stream order.
    """
    if request is None:
        yield from records
        return
    limit = request.limit
    if limit is not None and limit <= 0:
        return
    stride = max(1, request.stride)
    positions: dict[Any, int] = {}
    kept = 0
    for record in islice(records, max(0, request.since), None):
        if request.metric_type is not None and record.get("t") != request.metric_type:
            continue
        if request.keys and record.get("k") not in request.keys:
            continue
        if stride > 1:
            key = record.get("k")
            position = positions.get(key, 0)
            positions[key] = position + 1
            if position % stride:
                continue
        yield record
        kept += 1
        if limit is not None and kept >= limit:
            return


def head(path: Path, size: int = PROBE_BYTES) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)

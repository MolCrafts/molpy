"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store: ``meta`` plus a snapshot (``frame``),
a topology (``system``), a time-ordered frame sequence (``trajectory``)
and/or a force field (``forcefield``). :class:`~molpy.core.Frame` and
:class:`~molpy.core.Trajectory` are the in-memory objects.

A whole record is written like every other format, by a function at the top
of :mod:`molpy.io`: :func:`~molpy.io.write_mrec` (a path is a directory
store, a ``bytearray`` receives the packed record). This module holds the
rest:

* :class:`MrecFile` — a record opened for its sections: ``sections()``,
  ``meta()``, ``frame()``, ``system()``, ``forcefield()``, ``status()``,
  ``method()``, ``metrics()``, ``observables()``, each decoding only the
  section it names
* :class:`MrecReader` — lazy frame cursor over the ``trajectory`` section
  (``len()``, ``reader[i]``, iteration, ``frame_columns``, ``.step`` /
  ``.time`` labels, ``has_block``, ``to_trajectory()``)
* :class:`SequenceSchema` / :class:`MrecWriter` — pin a schema and write a run
  frame by frame
* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`ForceFieldSection.from_forcefield` /
  :meth:`ForceFieldSection.to_forcefield` map it onto a force field)
* :func:`pack_mrec_zip` — collapse a closed store into one ``*.mrec.zip``

The names are those of ``molrs::io::mrec``.

It also holds :class:`MrecMetricReader`, published in the
``molcrafts.metric_readers`` entry-point group for any molcrafts viewer::

    [project.entry-points."molcrafts.metric_readers"]
    mrec = "molpy.io.mrec:MrecMetricReader"
"""

from __future__ import annotations

from molpy import _core

ForceFieldSection = _core.mrec.ForceFieldSection

MrecFile = _core.mrec.MrecFile

MrecReader = _core.mrec.MrecReader

MrecWriter = _core.mrec.MrecWriter

SequenceSchema = _core.mrec.SequenceSchema

pack_mrec_zip = _core.mrec.pack_mrec_zip

from typing import TYPE_CHECKING
from molpy.io import metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from molpy.io.metric_reader import ReadRequest
_PACKED_SUFFIX = ".mrec.zip"


class MrecMetricReader:
    """``*.mrec`` records — the per-frame series a trajectory carries.

    A record is not a metrics file: what it has that a chart can draw is its
    simulation ``step`` and ``time`` per frame. Those are emitted and nothing
    else is invented. Both forms of a record are read: the ``*.mrec`` store
    directory a run writes, and the ``*.mrec.zip`` archive molrs packs it into
    once closed (``pack_mrec_zip``).
    """

    format = "mrec"
    patterns = ("**/*.mrec", "**/*.mrec/", "**/*.mrec.zip")
    tailable = False

    def sniff(self, path: Path) -> bool:
        """True for a ``*.mrec`` directory or ``*.mrec.zip`` file molrs can open."""
        if path.is_dir():
            if path.suffix != ".mrec":
                return False
        elif not (path.is_file() and path.name.endswith(_PACKED_SUFFIX)):
            return False
        try:
            with MrecReader(path):
                return True
        except (OSError, ValueError):
            return False

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the ``step`` / ``time`` records *request* selects.

        The raw stream is series-major — every frame's ``mrec/step``, then
        every frame's ``mrec/time`` — so ``since`` counts those records. ``s``
        is the frame's simulation step, or its index when the record has no
        ``step`` column (see :mod:`molpy.io.metric_reader`).
        """
        with MrecReader(path) as reader:
            columns = {name: getattr(reader, name) for name in ("step", "time")}
        yield from metric_reader.apply_request(
            request, _series_records(columns, source)
        )


def _series_records(columns: dict[str, Any], source: str) -> Iterator[dict[str, Any]]:
    present = {name: values for name, values in columns.items() if values is not None}
    steps = present.get("step")
    for name, values in present.items():
        for index, value in enumerate(values):
            yield {
                "t": "scalar",
                "k": f"mrec/{name}",
                "s": float(steps[index] if steps is not None else index),
                "v": float(value),
                "tags": {"source": source},
            }


__all__ = [
    "ForceFieldSection",
    "MrecFile",
    "MrecReader",
    "MrecWriter",
    "SequenceSchema",
    "pack_mrec_zip",
    "MrecMetricReader",
]

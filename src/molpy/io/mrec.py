"""``*.mrec`` records — :mod:`molrs.io.mrec`, mirrored by identity — and molpy's metric reader.

Every native name is the molrs object (``mp.io.mrec.MrecReader is
molrs.io.mrec.MrecReader``): the lazy record cursor ``MrecReader`` and its
``MrecWriter``, ``SequenceSchema``, ``ForceFieldSection`` (a force field's
record section: ``ForceFieldSection.from_forcefield(ff)`` /
``section.to_forcefield()``), ``section_names``, ``pack_mrec_zip`` and the
``validation`` checks. A record is read and written whole by ``mp.io.read_mrec_frame`` /
``write_mrec_frame`` and their ``_system`` / ``_trajectory`` / ``_forcefield``
partners.

molpy adds :class:`MrecMetricReader`, published in the
``molcrafts.metric_readers`` entry-point group for any molcrafts viewer::

    [project.entry-points."molcrafts.metric_readers"]
    mrec = "molpy.io.mrec:MrecMetricReader"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molrs.io.mrec import *  # noqa: F403
from molrs.io.mrec import __all__ as _native

from . import _metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from ._metric_reader import ReadRequest


class MrecMetricReader:
    """``*.mrec`` records — the per-frame series a trajectory carries.

    A record is not a metrics file: what it has that a chart can draw is
    the frame index against simulation ``step`` and ``time``. Those are
    emitted and nothing else is invented.
    """

    format = "mrec"
    patterns = ("**/*.mrec", "**/*.mrec/")
    tailable = False

    def sniff(self, path: Path) -> bool:
        if path.suffix != ".mrec":
            return False
        if path.is_dir():
            return True
        return path.is_file()

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        from molrs.io import read_mrec_trajectory

        stride = _metric_reader.stride_of(request)
        limit = _metric_reader.limit_of(request)
        skip = _metric_reader.since_of(request)
        trajectory = read_mrec_trajectory(path)

        series: dict[str, Any] = {}
        for name in ("step", "time"):
            values = getattr(trajectory, name)
            if values is not None and len(values):
                series[name] = values

        emitted = 0
        seen = 0
        for name, values in series.items():
            sampled = values[::stride] if stride > 1 else values
            for index, value in enumerate(sampled):
                record = {
                    "t": "scalar",
                    "k": f"mrec/{name}",
                    "s": float(index * stride),
                    "v": float(value),
                    "tags": {"source": source},
                }
                seen += 1
                if seen <= skip or not _metric_reader.wanted(request, record):
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


__all__ = [*_native, "MrecMetricReader"]

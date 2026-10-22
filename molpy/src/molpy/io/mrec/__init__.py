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

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

# A molrs build without the fs feature has no ``molrs.io.mrec``: importing it
# raises ImportError, so this module is absent too (``molpy.io`` skips it).
from molrs.io.mrec import *  # noqa: F403
from molrs.io.mrec import MrecReader
from molrs.io.mrec import __all__ as _native

from molpy.io import metric_reader

# The star import bound molrs's ``validation`` module; loading molpy's mirror
# rebinds the attribute, so ``molpy.io.mrec.validation`` also imports by name.
_import_module(f"{__name__}.validation")

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
        ``step`` column (see :mod:`._metric_reader`).
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


__all__ = [*_native, "MrecMetricReader"]

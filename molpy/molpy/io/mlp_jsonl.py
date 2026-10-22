"""``*.mlp.jsonl`` metric logs — molpy's own format module.

A ``*.mlp.jsonl`` file is one metric record per line, the format molexp writes
while a run is in flight. molrs has no reader for it, so this module is
molpy's alone: :class:`MlpJsonlMetricReader`, published in the
``molcrafts.metric_readers`` entry-point group for any molcrafts viewer::

    [project.entry-points."molcrafts.metric_readers"]
    mlp_jsonl = "molpy.io.mlp_jsonl:MlpJsonlMetricReader"
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING

from molpy.io import metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import Any

    from molpy.io.metric_reader import ReadRequest


class MlpJsonlMetricReader:
    """``*.mlp.jsonl`` — records that already are metric records.

    It is the format molexp writes while a run is in flight, which makes it
    the one that grows underneath a live chart. That is why it is *tailable*;
    it is not otherwise privileged, and it is read through the same seam as
    every other format.

    **Only a torn tail is forgiven.** A live writer can be caught mid-line, so
    a last line with no terminating newline that does not parse is not a
    record yet: it is skipped, and the next poll reads it whole. Any other
    line that is not a JSON object with a ``k`` key is corruption, and
    :meth:`read` raises :class:`ValueError` naming the file and line rather
    than skip it — a skipped line would silently shift every later record's
    ``since`` index. Blank lines are not records.
    """

    format = "mlp_jsonl"
    patterns = ("**/*.mlp.jsonl",)
    tailable = True

    def sniff(self, path: Path) -> bool:
        return path.is_file() and path.name.endswith(".mlp.jsonl")

    def read(
        self,
        path: Path,
        *,
        source: str = "",
        request: ReadRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the records *request* selects (see :mod:`._metric_reader`).

        The raw stream is the file's records in line order, so ``since``
        counts records (blank lines and a torn tail excluded). Each record
        keeps its own ``s`` and tags; ``source`` is added unless it has one.

        Raises:
            ValueError: A line other than a torn tail is not a metric record.
        """
        with path.open(encoding="utf-8") as handle:
            yield from metric_reader.apply_request(
                request, _line_records(handle, path, source)
            )


def _line_records(
    lines: Iterable[str], path: Path, source: str
) -> Iterator[dict[str, Any]]:
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = _parse(stripped)
        if record is None:
            if not line.endswith("\n"):
                return  # torn tail: the writer has not finished this line
            raise ValueError(
                f"{path}:{number}: not a metric record (a JSON object with "
                f"a 'k' key): {stripped[:80]!r}"
            )
        tags = record.get("tags")
        merged = (
            {"source": source, **tags} if isinstance(tags, dict) else {"source": source}
        )
        yield {**record, "tags": merged}


def _parse(text: str) -> dict[str, Any] | None:
    try:
        record = _json.loads(text)
    except _json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) and "k" in record else None


__all__ = ["MlpJsonlMetricReader"]

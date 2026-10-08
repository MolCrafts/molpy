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

from . import _metric_reader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from ._metric_reader import ReadRequest


class MlpJsonlMetricReader:
    """``*.mlp.jsonl`` — records that already are metric records.

    It is the format molexp writes while a run is in flight, which makes it
    the one that grows underneath a live chart. That is why it is *tailable*;
    it is not otherwise privileged, and it is read through the same seam as
    every other format.
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
        stride = _metric_reader.stride_of(request)
        limit = _metric_reader.limit_of(request)
        skip = _metric_reader.since_of(request)
        # A JSONL file is one sample per line and the series are interleaved,
        # so striding is counted per series — striding the file would land on
        # the same key every time and drop the others.
        seen_per_series: dict[Any, int] = {}
        emitted = 0
        index = 0

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = _json.loads(stripped)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or "k" not in record:
                    continue
                index += 1
                if index <= skip or not _metric_reader.wanted(request, record):
                    continue
                if stride > 1:
                    key = record.get("k")
                    position = seen_per_series.get(key, 0)
                    seen_per_series[key] = position + 1
                    if position % stride:
                        continue
                tags = record.get("tags")
                if isinstance(tags, dict):
                    tags.setdefault("source", source)
                else:
                    record["tags"] = {"source": source}
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


__all__ = ["MlpJsonlMetricReader"]

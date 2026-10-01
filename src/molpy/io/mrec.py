"""Scientific-record store machinery for ``*.mrec`` stores.

The whole-record path doors sit with every other format on :mod:`molpy.io`,
paired like the rest: ``read_mrec`` / ``write_mrec`` (Structure, ``meta`` +
``frame/``), ``read_mrec_system`` / ``write_mrec_system`` (System-def,
``meta`` + ``system/``), ``read_mrec_trajectory`` / ``write_mrec_trajectory``
(Trajectory shape), ``read_mrec_meta`` (the mandatory identity document) and
``mrec_sections``. What only a store has lives here, as identity re-exports of
the native core:

* :class:`TrajectoryReader` — lazy frame cursor (``len``, ``reader[i]``,
  iteration, ``.step`` / ``.time`` labels, ``has_block``)
* :class:`SequenceSchema` / :class:`TrajectoryWriter` — pin a schema and write
  a run frame by frame, without holding it all in memory
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`schema` — runtime check for path suffix and ``meta`` keys
"""

from __future__ import annotations

from molrs.io.mrec import (
    SequenceSchema,
    TrajectoryReader,
    TrajectoryWriter,
    pack,
    schema,
)

__all__ = [
    "SequenceSchema",
    "TrajectoryReader",
    "TrajectoryWriter",
    "pack",
    "schema",
]

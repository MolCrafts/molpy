"""Scientific-record store machinery for ``*.mrec`` stores.

The whole-record path doors sit with every other format on :mod:`molpy.io`,
paired like the rest: ``read_mrec`` / ``write_mrec`` (Structure, ``meta`` +
``frame/``), ``read_mrec_system`` / ``write_mrec_system`` (System-def,
``meta`` + ``system/``), ``read_mrec_trajectory`` / ``write_mrec_trajectory``
(Trajectory shape), ``read_mrec_forcefield`` / ``write_mrec_forcefield``
(force-field package, ``meta`` + ``forcefield/``; ``write_mrec`` and
``write_mrec_system`` also take ``forcefield=``), ``read_mrec_meta`` (the
identity document) and ``mrec_sections``. What only a store has lives here, as
identity re-exports of the native core:

* :class:`TrajectoryReader` — lazy frame cursor (``len``, ``reader[i]``,
  iteration, ``.step`` / ``.time`` labels, ``has_block``)
* :class:`SequenceSchema` / :class:`TrajectoryWriter` — pin a schema and write
  a run frame by frame, without holding it all in memory
* :class:`ForceFieldSection` — the ``forcefield`` section as data (the document
  and one ``Block`` per style table, units as stored);
  ``ForceField.to_section`` / ``ForceField.from_section`` map it onto a
  :class:`~molpy.ForceField`
* :func:`pack` — collapse a closed store into one ``*.mrec.zip``
* :mod:`schema` — runtime check for path suffix and ``meta`` keys
"""

from __future__ import annotations

from molrs.io.mrec import (
    ForceFieldSection,
    SequenceSchema,
    TrajectoryReader,
    TrajectoryWriter,
    pack,
    schema,
)

__all__ = [
    "ForceFieldSection",
    "SequenceSchema",
    "TrajectoryReader",
    "TrajectoryWriter",
    "pack",
    "schema",
]

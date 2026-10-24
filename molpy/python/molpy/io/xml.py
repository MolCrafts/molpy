"""molrs's own force-field XML — ``molrs::io::xml``.

:class:`XmlForcefieldReader` reads the force-field IR as molrs writes it and
:class:`XmlForcefieldWriter` writes it (``reader.read(source)`` /
``writer.write(sink, forcefield)``). OpenMM's ``<ForceField>`` schema is
:mod:`molpy.io.openmm_xml`'s.
"""

from molpy._core import (
    XmlForcefieldReader,
    XmlForcefieldWriter,
)

__all__ = [
    "XmlForcefieldReader",
    "XmlForcefieldWriter",
]

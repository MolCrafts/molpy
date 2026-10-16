"""OpenMM force-field XML — ``molrs::io::openmm_xml``.

:class:`OpenmmXmlReader` reads OpenMM's own ``<ForceField>`` schema into the
force-field IR; :class:`OpenmmXmlWriter` is the inverse. molrs's own
force-field XML is :class:`molrs.io.xml.XmlForcefieldReader`'s.
"""

from .._native import (
    OpenmmXmlReader,
    OpenmmXmlWriter,
)

__all__ = ["OpenmmXmlReader", "OpenmmXmlWriter"]

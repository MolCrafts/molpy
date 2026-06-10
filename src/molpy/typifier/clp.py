"""CL&P ionic-liquid force field typifier.

CL&P (Canongia Lopes & Padua, *J. Phys. Chem. B* 2004, 108, 2038,
DOI 10.1021/jp0362133) is an all-atom fixed-charge force field for ionic
liquids whose functional form is fully OPLS-AA compatible (harmonic bonds and
angles, OPLS cosine-series dihedrals, LJ 12-6 with geometric combining and
0.5/0.5 1-4 scaling). It therefore reuses the entire OPLS typing engine; only
the force-field *data* differs, shipped as the built-in ``clp.xml``.
"""

from molpy.core.forcefield import ForceField

from .atomistic import OplsTypifier


class ClpTypifier(OplsTypifier):
    """Full CL&P typing pipeline: atom -> pair -> bond -> angle -> dihedral.

    Inherits the OPLS-AA SMARTS/overrides matching engine unchanged. When no
    force field is supplied, the built-in ``clp.xml`` is loaded through the
    OPLS-AA reader (CL&P shares OPLS units and combining rules).
    """

    def __init__(self, forcefield: ForceField | None = None, **kwargs) -> None:
        if forcefield is None:
            forcefield = self.load_forcefield()
        super().__init__(forcefield, **kwargs)

    @staticmethod
    def load_forcefield() -> ForceField:
        """Load the built-in CL&P force field (``clp.xml``)."""
        from molpy.data.forcefield import get_forcefield_path
        from molpy.io.forcefield.xml import read_xml_forcefield

        return read_xml_forcefield(get_forcefield_path("clp.xml"))

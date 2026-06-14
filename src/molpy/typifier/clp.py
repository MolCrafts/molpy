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
        """Load the built-in CL&P force field as an OPLS-AA overlay.

        CL&P *extends* OPLS-AA, so the base ``oplsaa.xml`` is read first and
        ``clp.xml`` is layered on top (``layer=1``). CL&P atom types therefore
        override OPLS-AA wherever their SMARTS match (ionic-liquid atoms), while
        OPLS-AA stays the fallback for any atom CL&P does not specifically cover
        (e.g. molecular co-solvents in a mixed electrolyte).
        """
        from molpy.data.forcefield import get_forcefield_path
        from molpy.io.forcefield.xml import read_oplsaa_forcefield, read_xml_forcefield

        ff = read_oplsaa_forcefield("oplsaa.xml")
        return read_xml_forcefield(get_forcefield_path("clp.xml"), ff, layer=1)

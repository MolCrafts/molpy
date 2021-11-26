import molpy.element as element
from molpy.system import System
from scipy.sparse import find

__all__ = ["auto_bonds"]


def auto_bonds(mpObj: System, auto_style="LAMMPS-INTERFACE"):
    if auto_style == "LAMMPS-INTERFACE":
        auto_bonds_LAMMPS_INTEFACE(mpObj)
    else:
        raise NotImplementedError(
            f"The auto bond style of {aut_style} is not implemented!\n"
        )


def auto_bonds_LAMMPS_INTEFACE(mpObj: System, scale_factor: float = 0.9):
    """Automatically build bond connection"""

    metals = element.metals
    alkali = element.alkali

    natoms = mpObj.natoms
    neigh_csc = mpObj._neigh_csc
    atoms = mpObj.atoms
    for iA, iVec in zip(atoms, neigh_csc):
        iSymbol = iA.getSymbol()
        iRadii = iA.getRadii()
        iMol = iA.parent

        _, neighbors, Distance = find(iVec)
        for jA_index, dist in zip(neighbors, Distance):
            jA = atoms[jA_index]
            jSymbol = jA.getSymbol()
            elements = set([iSymbol, jSymbol])
            rad = iRadii + jA.getRadii()
            tempsf = scale_factor
            if (set("F") < elements) and (elements & metals):
                tempsf = 0.8

            if (set("O") < elements) and (elements & metals):
                tempsf = 0.85

            # fix for water particle recognition.
            if set(["O", "H"]) <= elements:
                tempsf = 0.8

            # fix for M-NDISA MOFs
            if set(["O", "C"]) <= elements:
                tempsf = 0.8
            if (set("O") < elements) and (elements & metals):
                tempsf = 0.82

            # very specific fix for Michelle's amine appended MOF
            if set(["N", "H"]) <= elements:
                tempsf = 0.67
            if set(["Mg", "N"]) <= elements:
                tempsf = 0.80
            if set(["C", "H"]) <= elements:
                tempsf = 0.80

            if dist * tempsf < rad and not (alkali & elements):
                iMol.addBond(iA, jA)

    mpObj._bondList = [iBond for iGroup in mpObj.groups for iBond in iGroup.bonds]

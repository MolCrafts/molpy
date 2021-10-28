# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-09-26
# version: 0.0.1

import numpy as np
import warnings
from molpy.atom import Atom
from molpy.group import Group

def _read_atom_line(line):
    
    """
     COLUMNS       DATATYPE      FIELD        DEFINITION
    -------------------------------------------------------------------------------------
    1 -  6         RecordName    "ATOM  "
    7 - 11         Integer       serial       Atom  serial number.
    13 - 16        Atom          name         Atom name.
    17             Character     altLoc       Alternate location indicator.
    18 - 20        Residue name  resName      Residue name.
    22             Character     chainID      Chain identifier.
    23 - 26        Integer       resSeq       Residue sequence number.
    27             AChar         iCode        Code for insertion of residues.
    31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    55 - 60        Real(6.2)     occupancy    Occupancy.
    61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    77 - 78        LString(2)    element      Element symbol, right-justified.
    79 - 80        LString(2)    charge       Charge  on the atom.

    """
    lineInfo = {}    
    lineInfo['RecordName'] = line[0:6]
    lineInfo['serial'] = int(line[7:12].strip())
    lineInfo['name'] = line[12:16].strip()
    lineInfo['altLoc'] = line[16].strip()
    lineInfo['resName'] = line[17:21].strip()
    lineInfo['chainID'] = line[21].strip()
    lineInfo['resSeq'] = int(line[22:26].strip())
    lineInfo['iCode'] = line[26].strip()

    try:
        lineInfo['position'] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
        )
    except ValueError:
        raise ValueError("Invalid or missing coordinate(s)")

    try:
        lineInfo['occupancy'] = float(line[54:60])
    except ValueError:
        lineInfo['occupancy'] = None  # Rather than arbitrary zero or one

    if lineInfo['occupancy'] is not None and lineInfo['occupancy'] < 0:
        warnings.warn("Negative occupancy in one or more atoms")

    try:
        lineInfo['bfactor'] = float(line[60:66])
    except ValueError:
        # The PDB use a default of zero if the data is missing
        lineInfo['bfactor'] = 0.0

    lineInfo['segid'] = line[72:76].strip()
    lineInfo['element'] = line[76:78].strip().upper()
    lineInfo['charge'] = line[79:81].strip()

    return lineInfo


def read_pdb(fileobj, **kwargs):
    atoms = []
    conects = {}
    for line in fileobj.readlines():

        if line.startswith("REMARK"):
            pass

        if line.startswith("CRYST1"):
            cellpar = [
                float(line[6:15]),  # a
                float(line[15:24]),  # b
                float(line[24:33]),  # c
                float(line[33:40]),  # alpha
                float(line[40:47]),  # beta
                float(line[47:54]),
            ]  # gamma

        if line.startswith("ATOM") or line.startswith("HETATM"):
            line_info = _read_atom_line(line)
            atom = Atom(line_info['name'])
            atom.update(line_info)
            atoms.append(atom)

        if line.startswith("CONECT"):
            l = line.split()
            center_atom_serial = int(l[1])
            bonded_atom_serial = map(int, (int(i) for i in l[2:]))
            conects[center_atom_serial] = list(bonded_atom_serial)

        if line.startswith("END"):
            group = Group('pdb')
            group.addAtoms(atoms)
            for c, nbs in conects.items():
                u = group.getAtomBy('serial', c)
                for nb in nbs:
                    v = group.getAtomBy('serial', nb)
                    group.addBond(u, v)
            atoms = []
            conects = {}

    return group
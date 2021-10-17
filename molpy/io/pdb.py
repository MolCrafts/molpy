# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-09-26
# version: 0.0.1

import numpy as np
import warnings

def _read_atom_line(line):
    """
    Read atom line from pdb format
    1-6 7-11 13-16 17 18-20 22 23-26 27 28-30 31-38 39-46 47-54 55-60 61-66 67-72 73-76 77-78 79-80
    HETATM    1  H14 ORTE    0       6.301   0.693   1.919  1.00  0.00        H
    
    ATOM serial name altLoc resName chainID resSeq iCode _ x y z occupancy tempFactor _ segID element charge
    """
    line = line.strip('\n')
    lineType = line[0: 6]

    serial = line[7: 12].strip()
    name = line[12: 16].strip()
    altLoc = line[16].strip()
    resName = line[17: 21].strip()
    chainID = line[21].strip()
    resSeq = line[22: 26].strip()
    iCode = line[26].strip()
    
    try:
        coord = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except ValueError:
        raise ValueError("Invalid or missing coordinate(s)")
    
    try:
        occupancy = float(line[54:60])
    except ValueError:
        occupancy = None  # Rather than arbitrary zero or one

    if occupancy is not None and occupancy < 0:
        warnings.warn("Negative occupancy in one or more atoms")

    try:
        bfactor = float(line[60:66])
    except ValueError:
        # The PDB use a default of zero if the data is missing
        bfactor = 0.0

    segid = line[72:76].strip()
    element = line[76:78].strip().upper()
    charge = line[79:81].strip() 
        
    return (
        lineType,
        serial,
        name,
        altLoc,
        resName,
        chainID,
        resSeq,
        iCode,
        coord,
        occupancy,
        bfactor,
        segid,
        element,
        charge,
    )

def read_pdb(fileobj, **kwargs):
    
    index = kwargs['index']
    
    frames = []
    
    orig = np.identity(3)
    trans = np.zeros(3)
    serials = []
    names = []
    altLocs = []
    resNames = []
    chainIDs = []
    resSeqs = []
    iCodes = []
    positions = []
    occupancies = []
    tempFactors = []
    segId = []
    elements = []
    charges = []
    
    cell = None
    pbc = None
    cellpar = []
    conects = {}
    if index == 0 or index == '0':
        isMultiFrames = False
    else:
        isMultiFrames = True
    
    for line in fileobj.readlines():
        
        if line.startswith('REMARK'):
            pass
        
        if line.startswith('CRYST1'):
            cellpar = [float(line[6:15]),  # a
                       float(line[15:24]),  # b
                       float(line[24:33]),  # c
                       float(line[33:40]),  # alpha
                       float(line[40:47]),  # beta
                       float(line[47:54])]  # gamma
            
        for c in range(3):
            if line.startswith('ORIGX' + '123'[c]):
                orig[c] = [float(line[10:20]),
                           float(line[20:30]),
                           float(line[30:40])]
                trans[c] = float(line[45:55])

        if (
            line.startswith("ATOM")
            or line.startswith("HETATM")
        ):
            line_info = _read_atom_line(line)
            
            serials.append(line_info[1])
            names.append(line_info[2])
            altLocs.append(line_info[3])
            resNames.append(line_info[4])
            chainIDs.append(line_info[5])
            resSeqs.append(line_info[6])
            iCodes.append(line_info[7])
            positions.append(line_info[8])
            occupancies.append(line_info[9])
            tempFactors.append(line_info[10])
            segId.append(line_info[11])
            elements.append(line_info[12])
            charges.append(line_info[13])
            
        if line.startswith("CONECT"):
            l = line.split()
            center_atom_serial = int(l[1])
            bonded_atom_serial = map(int, (int(i) for i in l[2:]))
            conects[center_atom_serial] = list(bonded_atom_serial)
            
        if line.startswith("END"):
            frames.append(
                {
                    'serials': serials,
                    'names': names,
                    'altNames': altLocs,
                    'resNames': resNames,
                    'chainIDs': chainIDs,
                    'resSeqs': resSeqs,
                    'iCodes': iCodes,
                    'positions': positions,
                    'occupancies': occupancies,
                    'tempFactors': tempFactors,
                    'segIds': segId,
                    'elements': elements,
                    'charges': charges,
                    'cellpar': cellpar,
                    'conects': conects
                }
            )
            orig = np.identity(3)
            trans = np.zeros(3)
            serials = []
            names = []
            altLocs = []
            resNames = []
            chainIDs = []
            resSeqs = []
            iCodes = []
            positions = []
            occupancies = []
            tempFactors = []
            segId = []
            elements = []
            charges = []
            if not isMultiFrames:
                return frames
    if index is None:
        return frames
    else:
        return frames[index]

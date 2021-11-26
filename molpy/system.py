# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1
from typing import Iterable, List, Dict

from molpy.base import Item
from molpy.atom import Atom
from molpy.group import Group
from molpy.molecule import Molecule
import numpy as np

class System(Item):
    def __init__(self, name) -> None:
        super().__init__(name)

        self.cell = None
        self.forcefield = None

        self._atomList = []
        self._bondList = []
        self._angleList = []
        self._dihedralList = []
        self._groupList = []
        self._molecules = {}
        self._unitType = None
        self._units = None

    def setCell(self, cell):
        self.cell = cell

    def setForcefield(self, forcefield):
        # self.forcefield[forcefield.name] = forcefield
        self.forcefield = forcefield

    def setUnitType(self, unitTypeName):
        pass

    def setAtomStyle(self, style):
        pass

    def setBondStyle(self, style):
        self.bondStyle = style

    def setAngleStyle(self, style):
        pass

    def setDihedralStyle(self, style):
        pass

    def setPairStyle(self, style):
        pass

    def setPositions(self, positions : Iterable):
        atoms = self.atoms
        assert len(positions) == len(atoms)
        for i, pos in enumerate(positions):
            atoms[i].position = pos

    def setRadii(self, radii : List[float]):
        atoms = self.atoms
        assert len(radii) == len(atoms)
        for iR, iA in zip(radii, atoms):
            iA.setRadii(iR)
    
    def setRadii_by(self, radii : Dict[str, float], attr = "symbol"):
        atoms = self.atoms
        for iA in atoms:
            iAttr = getattr(iA, attr)
            iA.setRadii(radii[iAttr])
    
    def getPositions(self):
        natoms = self.natoms
        atoms = self.atoms
        positions = np.empty((natoms, 3))
        for i, iA in enumerate(atoms):
            positions[i] = iA.position
        return positions
    
    def getSymbols(self):
        natoms = self.natoms
        atoms = self.atoms
        symbols = [None] * natoms
        for i, iA in enumerate(atoms):
            symbols[i] = iA.symbol
        return symbols
    
    def getRadii(self):
        natoms = self.natoms
        atoms = self.atoms
        radii = np.empty((natoms,))
        for i, iA in enumerate(atoms):
            radii[i] = iA.getRadii()
        return radii
    
    def cutoff(self, bin = 0.1):
        atoms = self.atoms
        max_radii = max(iA.getRadii() for iA in atoms)
        return 2.0 * max_radii + bin
    
    def getAttr_set(self, attr : str = "symbol"):
        atoms = self.atoms
        values_set = set()
        for iA in atoms:
            values_set.update(getattr(iA, attr))
        return values_set

    def getAttr(self, attr : str = "symbol"):
        atoms = self.atoms
        values = []
        for iA in atoms:
            values.append(getattr(iA, attr))
        return values

    @property
    def xlo(self):
        return self.cell.xlo

    @property
    def xhi(self):
        return self.cell.xhi
    
    @property
    def ylo(self):
        return self.cell.ylo

    @property
    def yhi(self):
        return self.cell.yhi
    
    @property
    def zlo(self):
        return self.cell.zlo

    @property
    def zhi(self):
        return self.cell.zhi

    @property
    def natoms(self):
        return len(self._atomList)
    
    @property
    def nmolecules(self):
        return len(self._molecules)

    @property
    def nbonds(self):
        return len(self._bondList)

    @property
    def nangles(self):
        return len(self._angleList)
    
    @property
    def ndihedrals(self):
        return len(self._dihedralList)
    
    def promote(self, item):

        m = Molecule(item.name)
        if item.itemType == "Atom":

            g = Group(item.name)
            g.addAtom(item)
            m.addGroup(g)

        elif item.itemType == "Group":
            m.addGroup(item)

        return m

    def _addAtom(self, atom):
        self._atomList.append(atom)

    def _addBond(self, bond):
        self._bondList.append(bond)

    def _addAngle(self, angle):
        self._angleList.append(angle)

    def addMolecule(self, molecule):

        if molecule.itemType == "Group":
            molecule = self.promote(molecule)
        if molecule.name in self._molecules:
            raise KeyError(f'molecule {molecule.name} is already defined in the system.')
        self._molecules[molecule.name] = molecule
        self._atomList.extend(molecule.atoms)
        self._groupList.extend(molecule.groups)
        self._bondList.extend(molecule.bonds)
        self._angleList.extend(molecule.angles)
        self._dihedralList.extend(molecule.dihedrals)


    @property
    def atomTypes(self):
        return self.forcefield.atomTypes
    
    @property
    def bondTypes(self):
        return self.forcefield.bondTypes
    
    @property
    def angleTypes(self):
        return self.forcefield.angleTypes
    
    @property
    def dihedralTypes(self):
        return self.forcefield.dihedralTypes

    @property
    def natomTypes(self):
        return self.forcefield.natomTypes

    @property
    def nbondTypes(self):
        return self.forcefield.nbondTypes

    @property
    def nangleTypes(self):
        return self.forcefield.nangleTypes

    @property
    def ndihedralTypes(self):
        return self.forcefield.ndihedralTypes

    @property
    def atoms(self):
        atoms = self._atomList
        for id, atom in enumerate(atoms, 1):
            atom.id = id
        if not hasattr(atoms[0], 'molid'):
            for id, molecule in enumerate(self._molecules.values(), 1):
                molecule.molid = id
                for atom in molecule.atoms:
                    atom.molid = id
            
        return atoms
    
    @property
    def groups(self):
        return self._groupList
    
    @property
    def molecules(self):
        return self._molecules.values()
    
    @property
    def bonds(self):
        bonds = self._bondList
        for id, bond in enumerate(bonds, 1):
            bond.id = id
            
        return bonds
    
    @property
    def angles(self):
        angles = self._angleList
        for id, angle in enumerate(angles, 1):
            angle.id = id
            
        return angles
    
    @property
    def dihedrals(self):
        dihes = self._dihedralList
        for id, dihe in enumerate(dihes, 1):
            dihe.id = id
        
        return dihes

    def complete(self):

        atomTypes = self.forcefield.atomTypes
        bondTypes = self.forcefield.bondTypes
        angleTypes = self.forcefield.angleTypes
        dihedralTypes = self.forcefield.dihedralTypes

        # template matching, set up topology

        # find angles, dihedrals etc.
        for mol in self._molecules.values():

            mol.searchAngles()
            self._angleList.extend(mol._angleList)
            mol.searchDihedrals()
            self._dihedralList.extend(mol._dihedralList)

        # ff::atomTypes -> atoms

        # ff::bondTypes -> bonds

        # ff::angleTypes -> angles
        # TODO: very dirty
        for angle in self._angleList:
            self.forcefield.matchAngleType(angle)
            assert angle.angleType

        # ff::dihedrals -> dihedrals

        #

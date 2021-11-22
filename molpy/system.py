# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from molpy.base import Item
from molpy.atom import Atom
from molpy.group import Group
from molpy.molecule import Molecule


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

    def complete(self, noAngle=True, noDihedral=True):
        
        self.noAngle = noAngle
        self.noDihedral = noDihedral

        # atomTypes = self.forcefield.atomTypes
        # bondTypes = self.forcefield.bondTypes
        # angleTypes = self.forcefield.angleTypes
        # dihedralTypes = self.forcefield.dihedralTypes

        # template matching, set up topology

        # find angles, dihedrals etc.
        if not noAngle:
            for mol in self._molecules.values():
                self._angleList.extend(mol.searchAngles())
        if not noDihedral:
            for mol in self._molecules.values():
                self._dihedralList.extend(mol.searchDihedrals())

        # ff::atomTypes -> atoms
        # check template first
        for atom in self.atoms:
            self.forcefield.renderAtom(atom)

        # ff::bondTypes -> bonds
        for bond in self.bonds:
            self.forcefield.renderBond(bond)

        # ff::angleTypes -> angles
        # TODO: very dirty
        for angle in self.angles:
            self.forcefield.renderAngle(angle)

        # ff::dihedrals -> dihedrals
        for dihedral in self.dihedrals:
            self.forcefield.renderDihedral(dihedral)

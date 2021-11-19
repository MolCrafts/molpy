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

    def mapping(self, start=1):
        
        # TODO: It is not decide whether to specify the ID during generation or add it uniformly at last
        
        # atom id mapping
        
        for id, atom in enumerate(self._atomList, start=1):
            atom.id = id
            
        for id, group in enumerate(self._groupList, start=1):
            group.id = id
            
        for id, molecule in enumerate(self.molecules, start=1):
            molecule.id = id
            for atom in molecule.atoms:
                atom.molid = id

        # atom type mapping
        atomTypeMap = {}
        atomTypes = self.forcefield.atomTypes
        for i, atomType in enumerate(atomTypes.values(), start):
            atomTypeMap[atomType.name] = i

        for i, molecule in enumerate(self._molecules.values(), start):
            molecule.molid = i

        for i, bond in enumerate(self.bonds, start):
            bond.id = i

        

        # for i, bondType in enumerate(self.bondTypes.values(), start):
        #     bondType.type

        # angleTypeMap = {}
        # angleTypes = self.forcefield.angleTypes
        # for i, angleType in enumerate(angleTypes.values(), start):
        #     angleTypeMap[angleType.name] = i

        # dihedralTypeMap = {}
        # dihedralTypes = self.forcefield.dihedralTypes
        # for i, dihedralType in enumerate(dihedralTypes.values(), start):
        #     dihedralTypeMap[dihedralType.name] = i
            
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

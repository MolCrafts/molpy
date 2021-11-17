# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from molpy.base import Item

class System(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        
        self.cell = None
        self.forcefield = None
        
        self._atoms = []
        self._bonds = []
        self._angles = []
        self._groups = []
        self._molecules = []
        
    def setCell(self, cell):
        self.cell = cell
        
    def setForcefield(self, forcefield):
        # self.forcefield[forcefield.name] = forcefield
        self.forcefield = forcefield
        
    @property
    def xlo(self):
        return self.cell.xlo
    
    @property
    def xhi(self):
        return self.cell.xhi
    
    @property
    def natoms(self):
        return len(self._atoms)
    
    @property
    def nbonds(self):
        return len(self._bonds)
    
    @property
    def nangles(self):
        return len(self._angles)
    
    def promote(self, item):
        
        if item.itemType == 'Atom':
             
            
    
    def addAtom(self, atom):
        self._atoms.append(atom)
        
    def _addBond(self, bond):
        self._bonds.append(bond)
        
    def addGroup(self, group):
        self._groups.append(group)
        
        for atom in group._atomList:
            self.addAtom(atom)
            
        for bond in group._bondList:
            self._addBond(bond)
            
        for angle in group._angleList:
            self._addAngle(angle)
            
    def addMolecule(self, molecule):
        self._molecules.append(molecule)
    
    def mapping(self, start=1):
        
        # atom type mapping
        atomTypeMap = {}
        atomTypes = self.forcefield.atomTypes
        for i, atomType in enumerate(atomTypes, start):
            atomTypeMap[atomType.name] = i
            
        for i, molecule in enumerate(self._molecules, start):
            molecule.molid = i
            
        for i, bond in enumerate(self._bonds, start):
            bond.id = i
            
        bondTypeMap = {}
        bondTypes = self.forcefield.bondTypes
        for i, bondType in enumerate(bondTypes, start):
            bondTypeMap[bondType.name] = i
            
        angleTypeMap = {}
        angleTypes = self.forcefield.angleTypes
        for i, angleType in enumerate(angleTypes, start):
            angleTypeMap[angleType.name] = i
            
        dihedralTypeMap = {}
        dihedralTypes = self.forcefield.dihedralTypes
        for i, dihedralType in enumerate(dihedralTypes, start):
            dihedralTypeMap[dihedralType.name] = i
            
    
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
        atoms = self._atoms
        for id, atom in enumerate(atoms, 1):
            atom.id = id
            atom.molid = 
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from molpy.base import Item

class System(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        
        self.cell = None
        self.forcefield = {}
        
        self._atoms = []
        self._bonds = []
        self._angles = []
        self._groups = []
        self._molecules = []
        
    def setCell(self, cell):
        self.cell = cell
        
    def setForcefield(self, forcefield):
        self.forcefield[forcefield.name] = forcefield
        
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
        
    
    
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1
from typing import Iterable

from molpy.base import Item
from molpy.atom import Atom
from molpy.group import Group
from molpy.molecule import Molecule
import numpy as np

class System(Item):
    def __init__(self, name) -> None:
        super().__init__(name)

        self.box = None
        self.forcefield = None

        self._atomList = []
        self._bondList = []
        self._angleList = []
        self._dihedralList = []
        self._groupList = []
        self._molecules = {}
        self._unitType = None
        self._units = None

    def setbox(self, box):
        self.box = box

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

    @property
    def xlo(self):
        return self.box.xlo

    @property
    def xhi(self):
        return self.box.xhi

    @property
    def ylo(self):
        return self.box.ylo

    @property
    def yhi(self):
        return self.box.yhi

    @property
    def zlo(self):
        return self.box.zlo

    @property
    def zhi(self):
        return self.box.zhi

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

        if molecule.itemType == "Group" or molecule.itemType == "Atom":
            molecule = self.promote(molecule)
        assert molecule.itemType == "Molecule", TypeError(
            f"{molecule} if not Molecule and it can not be promoted to Molecule"
        )
        if molecule.name in self._molecules:
            raise KeyError(
                f"molecule {molecule.name} is already defined in the system."
            )
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

    def complete(self, noAngle=False, noDihedral=False):

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
            if not hasattr(atom, "atomType"):
                self.forcefield.renderAtom(atom)

        # ff::bondTypes -> bonds
        for bond in self.bonds:
            if not hasattr(bond, "bondType"):
                self.forcefield.renderBond(bond)

        # ff::angleTypes -> angles
        # TODO: very dirty
        for angle in self.angles:
            if not hasattr(angle, "angleType"):
                self.forcefield.renderAngle(angle)

        # ff::dihedrals -> dihedrals
        for dihedral in self.dihedrals:
            if not hasattr(dihedral, "dihedralType"):
                self.forcefield.renderDihedral(dihedral)

    @property
    def charge(self):
        charge = 0
        for atom in self.atoms:
            charge += getattr(atom, "charge", 0)
        return charge

    def addSolvent(
        self,
        solute,
        ionicStrength=None,
        number=None,
    ):

        if ionicStrength is None and number is None:
            raise ValueError(f"either specify ionicStrength or number")
        
        # render solute
        solute = self.promote(solute)
        #TODO: refactor forcefield
        for atom in solute.atoms:
            if not hasattr(atom, "atomType"):
                self.forcefield.renderAtom(atom)        

        if ionicStrength is not None:
            number = int((ionicStrength - self.charge) / solute.charge)
            if number <= 0:
                raise ValueError()

            # TODO: use packing module instead
            rng = np.random.default_rng()
            rng.standard_normal(())

        for i in range(number):
            vec = np.hstack(
                (
                    rng.uniform(self.xlo, self.xhi, 1),
                    rng.uniform(self.ylo, self.yhi, 1),
                    rng.uniform(self.zlo, self.zhi, 1),
                )
            )
            self.addMolecule(solute(name=f"{solute.name}-{i}").moveTo(vec))

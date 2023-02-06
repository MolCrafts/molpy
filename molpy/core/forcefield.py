# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-16
# version: 0.0.1

from copy import deepcopy
from functools import partial
from typing import Iterable, Sequence

from .typing import Dict, Any, ArrayLike, Tuple, List
import numpy as np

class ItemType(dict):

    def render(self, entity):
        raise NotImplementedError()

    def __hash__(self):
        "unsafe hash"
        return id(self)

    def __lt__(self, o):
        return id(self) < id(o)

class AtomType(ItemType):

    def render(self, atom):
        return atom.update(self)

    def __lt__(self, o):
        return 

class BondType(ItemType):

    pass

AtomTypeName = str
AtomTypes = Tuple[AtomType, ...]

class Template:

    def __init__(self, name:str):

        self.name = name
        self.atomTypes:Dict[AtomTypeName, AtomType] = {}
        self.bondTypes:Dict[AtomTypes, BondType] = {}

    def def_atomType(self, name:str, **properties):
        at = AtomType(**properties)
        at['name'] = name
        self.atomTypes[name] = at
        return at

    def get_atomType(self, name:str)->AtomType:
        return self.atomTypes[name]

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        bt = BondType(**properties)
        bt['index'] = identity
        self.bondTypes[identity] = bt
        return bt

    def get_bondType(self, atomtype1:AtomType, atomtype2:AtomType)->BondType:
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        return self.bondTypes[identity]

    def __repr__(self):
        return f'<Template: {self.name}>'

class Forcefield:

    def __init__(self):

        self._parameters:Template = Template('_global_')
        self._residues:Dict[str, Template] = {}

    def def_atomType(self, name:str, **properties):
        return self._parameters.def_atomType(name, **properties)

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        return self._parameters.def_bondType(atomtype1, atomtype2, **properties)

    def get_atomType(self, name:str)->AtomType:
        return self._parameters.get_atomType(name)

    def get_atomType_by_class(self, className:str)->List[AtomType]:
        return [at for at in self._parameters.atomTypes.values() if at['class'] == className]

    def get_bondType(self, atomtype1:AtomType, atomtype2:AtomType)->BondType:
        return self._parameters.get_bondType(atomtype1, atomtype2)

    def match_atomTypes(self, names:Iterable)->List[AtomType | None]:
        return list(map(partial(self._parameters.atomTypes.get, AtomType()), names))

    def match_bondTypes(self, atomTypes:Iterable[Tuple[AtomType, AtomType]])->List[BondType | None]:

        sort_fn = lambda x: (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])
        return list(map(partial(self._parameters.bondTypes.get, BondType()), map(sort_fn, atomTypes)))

    def render_atoms(self, atoms):
        atomNames = map(lambda atom: atom['name'], atoms)
        atomTypes = self.match_atomTypes(atomNames)
        return map(lambda at, atom: at.render(atom), atomTypes, atoms)

    def render_bonds(self, bonds):
    
        # bond_index = np.array(self.bond_idx)

        # _atoms = np.array(self.atoms)
        # bond_atom_type = _atoms[bond_index]
        # bondTypes = self.forcefield.match_bondTypes(np.stack(bond_atom_type))
        # map(lambda bond, bondType: bond.render(bondType), self.bonds, bondTypes)

        return bonds

    def def_residue(self, name:str, )->Template:
        residue = Template(name)
        self._residues[name] = residue
        return residue

    def get_residue(self, name:str)->Template:
        return self._residues[name]

    @classmethod
    def from_xml(cls, path):
        import xml.etree.ElementTree as ET
        ff = cls()
        root = ET.parse(path).getroot()  # <ForceField>

        # get atomTypes
        atomTypes = root.find("AtomTypes")
        if atomTypes is None:
            raise ValueError("ForceField XML file must have AtomTypes tag")

        for at in atomTypes:
            at_dict = at.attrib
            ff.def_atomType(at_dict.pop('name'), **at_dict)

        # get residues
        residues = root.find("Residues")

        if residues:
            for residue in residues:
                
                re = ff.def_residue(residue.attrib['name'])

                atomTypes = residue.findall("Atom")
                if atomTypes:
                    for at in atomTypes:
                        at_dict = deepcopy(at.attrib)
                        re.def_atomType(at_dict.pop('name'), **at_dict)

                bondTypes = residue.findall("Bond")
                if bondTypes:
                    for bt in bondTypes:
                        bt_dict = deepcopy(bt.attrib)
                        re.def_bondType(
                            re.get_atomType(bt_dict.pop('atomName1')),
                            re.get_atomType(bt_dict.pop('atomName2')),
                            **bt_dict
                        )

        # get force
        for force in root:
            if force.tag.endswith("Force"):
                for item in force:
                    if item.tag == "Bond":
                        
                        bt_dict = deepcopy(item.attrib)
                        atomTypes1 = []
                        atomTypes2 = []
                        if "class1" in bt_dict:
                            atomClass1 = bt_dict.pop("class1")
                            atomTypes1 = ff.get_atomType_by_class(atomClass1)
                        if "class2" in bt_dict:
                            atomClass2 = bt_dict.pop("class2")
                            atomTypes2 = ff.get_atomType_by_class(atomClass2)
                        if "atomName1" in bt_dict:
                            atomName1 = bt_dict.pop("atomName1")
                            atomTypes1 = [ff.get_atomType(atomName1)]
                        if "atomName2" in bt_dict:
                            atomName2 = bt_dict.pop("atomName2")
                            atomTypes2 = [ff.get_atomType(atomName2)]
                        
                        if atomTypes1 or atomTypes2:
                            for at1, at2 in zip(atomTypes1, atomTypes2):
                                ff.def_bondType(at1, at2, **bt_dict)
                        else:
                            raise ValueError("Bond must have either class1, class2, atomName1 or atomName2")

        return ff
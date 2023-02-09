# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-16
# version: 0.0.1

from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Iterable, Optional, Sequence

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
        return id(self) < id(o)

class AtomClass(ItemType):
    pass


class BondType(ItemType):

    def render(self, bond):
        return bond.update(self)

AtomTypeName = str
AtomClassName = str
AtomTypes = Tuple[AtomType, ...]

class Template:

    def __init__(self, name:str):

        self.name = name
        self.atomTypes:Dict[AtomTypeName, AtomType] = {}
        self.atomClasses:Dict[AtomClassName, AtomClass] = {}
        self.atomTypeMapToClass:Dict[AtomType, AtomClassName] = {}
        self.atomClassContainsAtomType:Dict[AtomClass, List[AtomType]] = defaultdict(list)
        self.bondTypes:Dict[AtomTypes, BondType] = {}

    def def_atomType(self, name:str, className:Optional[str]=None, **properties):
        at = AtomType(**properties)
        ac = self.atomClasses.setdefault(className, AtomClass(name=className))
        at['name'] = name
        at['class'] = ac
        self.atomTypes[name] = at
        self.atomTypeMapToClass[at] = ac
        self.atomClassContainsAtomType[ac].append(at)
        return at

    def get_atomType(self, name:str)->AtomType:
        return self.atomTypes[name]

    def get_atomTypes_by_class(self, className:str)->List[AtomType]:
        return self.atomClassContainsAtomType[className]

    def def_bondType(self, atom1:AtomType|AtomClass, atom2:AtomType|AtomClass, style:Optional[str]=None, **properties):
        if atom1 > atom2:
            atom1, atom2 = atom2, atom1
        identity = (atom1, atom2)
        bt = BondType(name=style, **properties)
        self.bondTypes[identity] = bt
        return bt

    def get_bondType(self, atom1:AtomType|AtomClass, atom2:AtomType|AtomClass)->BondType:

        candidate_atom1 = [atom1, ]
        candidate_atom2 = [atom2, ]
        if isinstance(atom1, AtomType):
            candidate_atom1.append(self.atomTypeMapToClass[atom1])
        elif isinstance(atom1, AtomClass):
            candidate_atom1.extend(self.atomClassContainsAtomType[atom1])
        if isinstance(atom2, AtomType):
            candidate_atom2.append(self.atomTypeMapToClass[atom2])
        elif isinstance(atom2, AtomClass):
            candidate_atom2.extend(self.atomClassContainsAtomType[atom2])

        for c1 in candidate_atom1:
            for c2 in candidate_atom2:
                if c1 > c2:
                    bt = self.bondTypes.get((c2, c1), None)
                else:
                    bt = self.bondTypes.get((c1, c2), None)
                if bt is not None:
                    return bt
        else:
            raise KeyError(f'<Bond:{atom1}-{atom2}> not found in template {self.name}')
            

    def __repr__(self):
        return f'<Template: {self.name}>'

class Forcefield:

    def __init__(self):

        self._parameters:Template = Template('_global_')
        self._residues:Dict[str, Template] = {}

    def def_atomType(self, name:str, className:Optional[str]=None, **properties):
        return self._parameters.def_atomType(name, className, **properties)

    def def_bondType(self, atom1:AtomType|AtomClass, atom12:AtomType|AtomClass, style:Optional[str], **properties):
        return self._parameters.def_bondType(atom1, atom12, style, **properties)

    def get_atomType(self, name:str)->AtomType:
        return self._parameters.get_atomType(name)

    def get_atomType_by_class(self, className:str)->List[AtomType]:
        return self._parameters.get_atomTypes_by_class(className)

    def get_bondType(self, atomtype1:AtomType|AtomClass, atomtype2:AtomType|AtomClass)->BondType:
        return self._parameters.get_bondType(atomtype1, atomtype2)

    def match_atomTypes(self, names:Iterable)->List[AtomType | None]:
        return list(map(partial(self._parameters.atomTypes.get, AtomType()), names))

    def match_bondTypes(self, atomTypes:Iterable[Tuple[AtomType|AtomClass, AtomType|AtomClass]])->List[BondType | None]:

        sort_fn = lambda x: (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])
        return list(map(partial(self._parameters.bondTypes.get, BondType()), map(sort_fn, atomTypes)))

    def render_atoms(self, atoms):

        for atom in atoms:

            if 'residue' in atom:
                residue = atom['residue']
                template = self.get_residue(residue)
                atomType = template.get_atomType(atom['name'])
                atomType.render(atom)


            atomType = self.get_atomType(atom['type'])
            atomType.render(atom)

        return atoms

    def render_bonds(self, bonds):

        return bonds

    def def_residue(self, name:str, )->Template:
        residue = Template(name)
        self._residues[name] = residue
        return residue

    def get_residue(self, name:str)->Template:
        return self._residues[name]

    def render_residue(self, residue):

        name = residue.name
        template = self.get_residue(name)
        for atom in residue.atoms:
            template.get_atomType(atom['name']).render(atom)  # get type name from residue
            self.get_atomType(atom['type']).render(atom)  # get props

        for bond in residue.bonds:
            at1 = self.get_atomType(bond.itom['type'])  # get global atom type
            at2 = self.get_atomType(bond.jtom['type'])
            bondType = self.get_bondType(at1, at2)
            bondType.render(bond)

        return residue

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
            ff.def_atomType(at_dict.pop('name'), className=at_dict.pop('class', None), **at_dict)

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
                force_name = force.tag[:-5]
                for item in force:
                    if item.tag == "Bond":
                        bt_dict = deepcopy(item.attrib)
                        atom1 = bt_dict.pop('class1', 'atomName1')
                        atom2 = bt_dict.pop('class2', 'atomName2')
                        atomClass1 = ff._parameters.atomClasses[atom1]
                        atomClass2 = ff._parameters.atomClasses[atom2]
                        ff.def_bondType(atomClass1, atomClass2, style=force_name[:-4], **bt_dict)

        return ff
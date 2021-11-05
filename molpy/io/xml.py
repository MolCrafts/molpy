# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-24
# version: 0.0.1

from molpy.atom import Atom
from molpy.forcefield import ForceField, Template
import xml.etree.ElementTree as et
from typing import TextIO

def read_xml_forcefield(fileobj: TextIO, create_using:ForceField=None):
    
    tree = et.parse(fileobj)
    if create_using is None:
        ff = ForceField('')
    else:
        ff = create_using
        
    root = tree.getroot()
    if root.tag != 'ForceField':
        raise NotImplementedError(f'XML file not under forcefield format with starts with <ForceField> tag')
    
    atomTypes = tree.find('AtomTypes')
    for atomType in atomTypes.iter('Type'):
        at = atomType.attrib
        atName = at['name']
        del at['name']
        ff.defAtomType(atName, **at)
        
    residues = tree.find('Residues')
    for residue in residues.iter('Residue'):
        resT = Template(residue.attrib['name'])
        for atom in residue.iter('Atom'):
            at = atom.attrib
            atName = at['name']
            del at['name']
            atomT = Atom(atName, **at)
            resT.addAtom(atomT)
            
        for bond in residue.iter('Bond'):
            from_ = int(bond.attrib['from'])
            to_ = int(bond.attrib['to'])
            resT.addBondByIndex(from_, to_)
        ff.defTemplate(resT)
            
    # bond define section
    for element in root:
        if 'BondForce' in element.tag:
            for bond in element.iter('Bond'):
                if 'name' not in bond.attrib:
                    # TODO: generate new name
                    name = 'Bond'
                ff.defBondType(name, **bond.attrib)
    
    return ff
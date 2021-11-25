# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.atom import Atom
from molpy.group import Group
from molpy.io.pdb import read_pdb
from molpy.io.lmp import write_lmp
from molpy.io.ase import read_ASE_atoms
from molpy.io.xml import read_xml_forcefield

def full(groupName, atomNames, addBondByIndex=None, **properties):
    """ build up a group with atoms

    Args:
        names (List[atomName]): list of atomName to init Atoms
        properties (Dict[str:List]): the len of lists should match with names list
    """
    _atoms = []
    for i, name in enumerate(atomNames):
        atom = Atom(name)
        for k, v in properties.items():
            setattr(atom, k, v[i])
        _atoms.append(atom)
            
    group = Group(groupName)
    for atom in _atoms:
        group.add(atom)
        
    if addBondByIndex is not None:
        for i in addBondByIndex:
            group.addBondByIndex(*i)
        
    return group

def fromPDB(fpath, index=None):
    with open(fpath, 'r') as f:
        group = read_pdb(f, index=None)
    return group

def fromASE(ase_atoms) -> Group:
    return read_ASE_atoms(ase_atoms)

def fromLAMMPS():
    pass

def fromXML(fpath, type='forcefield'):
    with open(fpath, 'r') as f:
        if type == 'forcefield':
            ff = read_xml_forcefield(f)
            return ff

def toLAMMPS(fpath, system, **kwargs):
    with open(fpath, 'w') as f:
        write_lmp(f, system, **kwargs)

def fromNetworkXGraph(name, G):
    
    adj = dict(G.adj)
    group = Group(name)
    
    atoms = [Atom(f'{i}') for i in adj]
    group.addAtoms(atoms)
        
    for uname, nbs in adj.items():
        for nb, dd in nbs.items():
            group.addBond(atoms[uname], atoms[nb], **dd)
            
    return group

def toJAXMD(group: Group):
    d = {}
    R = group.getPositions()
    d['positions'] = R
    atomTypes = group.getAtomTypes()
    d['atomTypes'] = atomTypes
    elements = group.getElements()
    d['elements'] = elements
    bonds = group.getAdjacencyList()
    d['bonds'] = bonds
    natoms = group.natoms
    d['natoms'] = natoms
    
    return d
    
def fromJAXMD():
    pass

def toJraph(group, node_features, edge_features, ):
    pass

def trace(item, sites, anchor=None):
    
    if anchor is None:
        if item.itemType == 'Atom':
            itemPos = item.position
        elif item.itemType == 'Group' or item.itemType == 'Molecule':
            itemPos = item.positions[0]
            
    else:
        itemPos = anchor.position
        
    # calculate displacement vector
    disVecs = sites - itemPos
    
    tmp = []
    for i, disVec in enumerate(disVecs):
        icopy = item(name=f'{item.name}-{i}').move(disVec)
        tmp.append(icopy)
        
    return tmp
    
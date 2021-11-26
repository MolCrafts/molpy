from molpy.atom import Atom
from molpy.group import Group
from molpy.system import System
from molpy.cell   import Cell
from ase.io.cif import parse_cif
from ase.neighborlist import neighbor_list
from ase import Atoms as ASE_atoms
from molpy.system import System
from scipy.sparse import csc_matrix

def read_ASE_atoms(aseAtoms : ASE_atoms, **kwargs) -> Group:
    symbols = aseAtoms.get_chemical_symbols()
    positions = aseAtoms.get_positions()
    atoms_num = len(symbols)
    name = aseAtoms.get_chemical_formula()
    g = Group(name);
    for s, pos in zip(symbols, positions):
        a = Atom(s, element=s, position = pos)
        g.addAtom(a)
    return g

def read_ASE_atoms_S(aseAtoms : ASE_atoms, **kwargs) -> System:
    g = read_ASE_atoms(aseAtoms, **kwargs)
    sys = System(aseAtoms.get_chemical_formula())
    cell = aseAtoms.cell
    if not cell.orthorhombic:
        raise NotImplementedError("non-othorhombi box is not supported!")
    lxyz = cell.lengths()
    g_cell = Cell(3, "ppp", lx = lxyz[0], ly = lxyz[1], lz = lxyz[2])
    sys.cell = g_cell
    sys.addMolecule(g)
    return sys

def read_CIF(filecif : str, fromLabel : bool = True, **kwargs):
    cifBlocks = parse_cif(filecif)
    cifBlock  = next(cifBlocks)
    atoms     = cifBlock.get_atoms()
    us_atoms  = cifBlock.get_unsymmetrized_structure()

    g         = read_ASE_atoms(atoms)
    g.spacegroup = {}
    siteNames  = cifBlock.get("_atom_site_label")
    if len(atoms) != len(us_atoms):
        xyz_scaled = us_atoms.get_scaled_positions()
        spacegroup = cifBlock.get_spacegroup(True)
        j = 0
        for i, i_xyz_scaled in enumerate(xyz_scaled):
            _, kinds = spacegroup.equivalent_sites(i_xyz_scaled)
            num_sites = len(kinds)
            name = siteNames[i]
            g.spacegroup[name] = num_sites
            if fromLabel:
                for _ in range(num_sites):
                    g[j].name = name
                    j += 1
    else:
        if fromLabel:
            for i, name in enumerate(siteNames):
                g[i].name = name
    return g

def read_CIF_S(filecif : str, fromLabel : bool = True, **kwargs) -> System:
    cifBlocks = parse_cif(filecif)
    cifBlock  = next(cifBlocks)
    atoms     = cifBlock.get_atoms()
    us_atoms  = cifBlock.get_unsymmetrized_structure()

    g         = read_ASE_atoms(atoms)
    g.spacegroup = {}
    siteNames  = cifBlock.get("_atom_site_label")
    if len(atoms) != len(us_atoms):
        xyz_scaled = us_atoms.get_scaled_positions()
        spacegroup = cifBlock.get_spacegroup(True)
        j = 0
        for i, i_xyz_scaled in enumerate(xyz_scaled):
            _, kinds = spacegroup.equivalent_sites(i_xyz_scaled)
            num_sites = len(kinds)
            name = siteNames[i]
            g.spacegroup[name] = num_sites
            if fromLabel:
                for _ in range(num_sites):
                    g[j].name = name
                    j += 1
    else:
        if fromLabel:
            for i, name in enumerate(siteNames):
                g[i].name = name
    
    if cifBlock.name:
        sys = System(cifBlock.name)
    else:
        sys = System(atoms.get_chemical_formula())
    
    cell = atoms.cell
    if not cell.orthorhombic:
        raise NotImplementedError("non-othorhombi box is not supported!")
    lxyz = cell.lengths()
    sys.cell = Cell(3, "ppp", lx = lxyz[0], ly = lxyz[1], lz = lxyz[2])
    sys.addMolecule(g)
    return sys

def toASE_atoms(mpObj) -> ASE_atoms:
    print(mpObj)
    symbols = mpObj.getSymbols()
    positions = mpObj.getPositions()
    
    if isinstance(mpObj, Group):
        return ASE_atoms(symbols, positions)
    elif isinstance(mpObj, System):
        return ASE_atoms(symbols, positions, cell = mpObj.cell.matrix, pbc = mpObj.cell.pbc)
    

def build_ASE_neigh(sysObj : System, cutoff = None) -> None:
    atoms = toASE_atoms(sysObj)
    natoms = len(atoms)
    if cutoff is None:
        cutoff = sysObj.cutoff()
    I, J, D = neighbor_list("ijd", atoms, cutoff = cutoff, 
        self_interaction=False)
    sysObj._neigh_csc = csc_matrix((D, (J, I)), shape=(natoms, natoms))
    
    
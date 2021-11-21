# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

def write_lmp(fileobj, system, **kwargs):
    f = fileobj
    
    # comment
    f.write(f'{system.name} written by molpy\n\n')
    
    # summary
    f.write(f'\t{system.natoms}\tatoms\n')
    f.write(f'\t{system.nbonds}\tbonds\n')
    f.write(f'\t{system.nangles}\tangles\n')
    if system.dihedrals:
        f.write(f'\t{system.ndihedrals}\tdihedrals\n')
        
    # forcefield
    f.write(f'\t{system.natomTypes}\tatom types\n')
    f.write(f'\t{system.nbondTypes}\tbond types\n')
    f.write(f'\t{system.nangleTypes}\tangle types\n')
    f.write(f'\t{system.ndihedralTypes}\tdihedral types\n')
           
    # cell
    f.write(f'\t{system.xlo}  {system.xhi}  xlo  xhi\n')
    f.write(f'\t{system.ylo}  {system.yhi}  ylo  yhi\n')
    f.write(f'\t{system.zlo}  {system.zhi}  zlo  zhi\n\n')
    
    # mess section
    f.write('Masses\n\n')
    for atomType in system.atomTypes.values():
        f.write(f'\t{atomType.typeID}\t{atomType.mass}\n')
        
    f.write('\n')
        
    f.write('Atoms\n\n')
    if kwargs['atom_style'] == 'full':
        for atom in system.atoms:
            f.write(f'\t{atom.id}\t{atom.molid}\t{atom.typeID}\t{atom.charge:.4f}\t{atom.x:.4f}\t{atom.y:.4f}\t {atom.z:.4f}\n')
            
    f.write('\n')
    
    if system.bonds:            
        f.write('Bonds\n\n')
        for bond in system.bonds:
            f.write(f'\t{bond.id}\t{bond.typeID}\t{bond.atom.id}\t{bond.btom.id}\n')
  
        f.write('\n')
        
    if system.angles:
        f.write('Angles\n\n')
        for angle in system.angles:
            f.write(f'\t{angle.id}\t{angle.typeID}\t{angle.itom.id}\t{angle.jtom.id}\t{angle.ktom.id}\n')
        
        f.write('\n')
        
    if system.dihedrals:
        f.write('Dihedrals\n\n')
        for dihedral in system.dihedrals:
            f.write(f'\t{dihedral.id}\t{dihedral.typeID}\t{dihedral.itom.id}\t{dihedral.jtom.id}\t{dihedral.ktom.id}\t{dihedral.ltom.id}\n')
        
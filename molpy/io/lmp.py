# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

def write_lmp(fileobj, system, **kwargs):
    f = fileobj
    system.mapping()
    
    # comment
    f.write(f'{system.name} written by molpy\n\n')
    
    # summary
    f.write(f'\t{system.natoms}\tatoms\n')
    f.write(f'\t{system.nbonds}\tbonds\n')
    f.write(f'\t{system.nangles}\tangles\n')
    if system.dihedrals:
        f.write(f'\t{system.ndihedrals}\dihedrals\n')
        
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
    f.write('Messes\n\n')
    for atomType in system.atomTypes:
        f.write(f'\t{atomType.id}\t{atomType.mess}\n')
        
    f.write('Atoms\n\n')
    if kwargs['atom_style'] == 'full':
        for atom in system.atoms:
            f.write(f'{atom.id} {atom.molid} {atom.typeid} {atom.charge} {atom.x} {atom.y} {atom.z}\n')
            
    f.write('Bonds\n\n')
    for bond in system.bonds:
        f.write(f'{bond.id} {bond.typeid} {bond.atom.id} {bond.btom.id}\n')
        
    f.write('Angles\n\n')
    for angle in system.angles:
        f.write(f'{angle.id} {angle.typeid} {angle.itom.id} {angle.jtom.id} {angle.ktom.id}\n')
        
    f.write('Dihedrals\n\n')
    for dihedral in system.dihedrals:
        f.write(f'{dihedral.id} {dihedral.typeid} {dihedral.itom.id} {dihedral.jtom.id} {dihedral.ktom.id} {dihedral.ltom.id}\n')
        
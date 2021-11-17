# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

def write_lmp(fileobj, system, **kwargs):
    f = fileobj
    xlo = system.xlo
    xhi = system.xhi
    
    atoms = system.atoms
    bonds = system.bonds
    angles = system.angles
    dihedrals = system.dihedrals
    
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
    for 
        f.write(f'\t')
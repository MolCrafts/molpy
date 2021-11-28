from copy import deepcopy
import ase
import molpy as mp
class A:
    
    def __init__(self) -> None:
        self.data = {'key': 'value'}
        self.stable = 'stable'
        self.myself = self
        
a = A()
aa = deepcopy(a)


Ni = ase.io.read("./tests/samples/Ni.cif")
ase_cell = Ni.cell
angles = ase_cell.angles()
lengths = ase_cell.lengths()

lbox = lengths[0]
box = mp.Box("ppp", lx = lbox, ly = lbox, lz = lbox, alpha = angles[0], beta = angles[1], gamma = angles[2])

basis = box._basis.T
box2 = mp.Box("ppp", a1 = basis[0], a2 = basis[1], a3 = basis[2])
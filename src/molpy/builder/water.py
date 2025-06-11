from molpy.core import Struct, Atom

class TIP3P(Struct):
    """
    TIP3P water model.
    """
    def __init__(self, name='TIP3P'):
        super().__init__(name=name)
        self.def_atom(name='O', element='O', mass=15.999, charge=-0.834, xyz=(0, 0, 0))
        self.def_atom(name='H1', element='H', mass=1.008, charge=0.417, xyz=(0.9572, 0, 0))
        self.def_atom(name='H2', element='H', mass=1.008, charge=0.417, xyz=(-0.239987, 0.926627, 0))
        self.def_bond(0, 1)
        self.def_bond(0, 2)
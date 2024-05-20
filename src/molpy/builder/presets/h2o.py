from molpy.builder.base import Atom, Bond, DynamicStruct
from molpy import Alias


class SPCE(DynamicStruct):

    def __init__(self):

        super().__init__(n_atoms=3)

        o = Atom(**{
            Alias.name: "O",
            Alias.atomtype: "o1",
            Alias.charge: -0.8476,
            Alias.xyz: (0.00000, -0.06461, 0.00000),
            })
        h1 = Atom(**{
            Alias.name: "H",
            Alias.atomtype: "h1",
            Alias.charge: 0.4238,
            Alias.xyz: (0.81649, 0.51275, 0.00000),
            })
        h2 = Atom(**{
            Alias.name: "H",
            Alias.atomtype: "h1",
            Alias.charge: 0.4238,
            Alias.xyz: (-0.81649, 0.51275, 0.00000),
            })

        self.add_atom(o)
        self.add_atom(h1)
        self.add_atom(h2)
        self.add_bond(0, 1, type="O-H")
        self.add_angle(1, 0, 2, type="H-O-H")

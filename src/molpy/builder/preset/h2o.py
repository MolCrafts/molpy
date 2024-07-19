from molpy.builder.base import Atom, Bond, DynamicStruct
import molpy as mp


class SPCE(DynamicStruct):

    def __init__(self):

        super().__init__(n_atoms=3)

        o = Atom(**{
            mp.alias.name: "O",
            mp.alias.atomtype: 0,
            mp.alias.charge: -0.8476,
            mp.alias.xyz: (0.00000, -0.06461, 0.00000),
            })
        h1 = Atom(**{
            mp.alias.name: "H",
            mp.alias.atomtype: 1,
            mp.alias.charge: 0.4238,
            mp.alias.xyz: (0.81649, 0.51275, 0.00000),
            })
        h2 = Atom(**{
            mp.alias.name: "H",
            mp.alias.atomtype: 1,
            mp.alias.charge: 0.4238,
            mp.alias.xyz: (-0.81649, 0.51275, 0.00000),
            })

        self.add_atom(o)
        self.add_atom(h1)
        self.add_atom(h2)
        self.add_bond(0, 1, type="O-H")
        self.add_angle(1, 0, 2, type="H-O-H")
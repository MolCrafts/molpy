from molpy.builder.base import Atom, Bond, DynamicStruct


class SPCE(DynamicStruct):

    def __init__(self):

        super().__init__(n_atoms=3)

        o = Atom(
                name=1,
                type="O",
                charge=-0.8476,
                position=(0.00000, -0.06461, 0.00000),
            ),
        h1 = Atom(
                name=2,
                type="H",
                charge=0.4238,
                position=(0.81649, 0.51275, 0.00000),
            )
        h2 = Atom(
                name=3,
                type="H",
                charge=0.4238,
                position=(-0.81649, 0.51275, 0.00000),
            )

        self.add_atom(o)
        self.add_atom(h1)
        self.add_atom(h2)
        self.add_bond(0, 1, type="O-H")
        self.add_angle(1, 0, 2, type="H-O-H")

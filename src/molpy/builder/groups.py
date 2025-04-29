import molpy as mp

class CH3(mp.Struct):

    def __init__(self):
        super().__init__()
        self.add_atom(
            id=1,
            name="C",
            type="C",
            mass=12.01,
            charge=0.0,
            xyz=[0.0, 0.0, 0.0],
        )
        self.add_atom(
            id=2,
            name="H1",
            type="H",
            mass=1.008,
            charge=0.0,
            xyz=[0.000, 0.000, 1.090],
        )
        self.add_atom(
            id=3,
            name="H2",
            type="H",
            mass=1.008,
            charge=0.0,
            xyz=[1.026, 0.000, -0.363],
        )
        self.add_atom(
            id=4,
            name="H3",
            type="H",
            mass=1.008,
            charge=0.0,
            xyz=[-0.513, 0.889, -0.363],
        )
        self.add_bond(0, 1)
        self.add_bond(0, 2)
        self.add_bond(0, 3)
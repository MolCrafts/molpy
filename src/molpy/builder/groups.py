import molpy as mp

class CH3(mp.Struct):
    def __init__(self):
        super().__init__()

        c = self.add_atom(
            name="C",
            type="C3",
            element="C",
            mass=12.01,
            charge=0.0,
            xyz=[0.0, 0.0, 0.0],
        )

        h1 = self.add_atom(
            name="H1",
            type="HC",
            element="H",
            mass=1.008,
            charge=0.0,
            xyz=[0.000, 0.000, 1.090],
        )
        h2 = self.add_atom(
            name="H2",
            type="HC",
            element="H",
            mass=1.008,
            charge=0.0,
            xyz=[1.026, 0.000, -0.363],
        )
        h3 = self.add_atom(
            name="H3",
            type="HC",
            element="H",
            mass=1.008,
            charge=0.0,
            xyz=[-0.513, 0.889, -0.363],
        )

        # 添加键
        self.add_bond(c, h1)
        self.add_bond(c, h2)
        self.add_bond(c, h3)

    def carbon(self):
        return self["atoms"][0]  # 默认第一个为 C 原子
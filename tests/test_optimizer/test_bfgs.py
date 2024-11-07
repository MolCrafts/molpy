import numpy as np
import molpy as mp


class TestBFGS:

    def test_harmonic(self):

        ff = mp.ForceField("tip3p")
        atomstyle = ff.def_atomstyle("full")
        atomstyle.def_type("O", mass=15.9994, charge=-0.834)
        atomstyle.def_type("H", mass=1.00794, charge=0.417)
        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_type("OH", k=450, r0=0.9572)
        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type("HOH", k=55, theta0=104.52)
        lj126 = ff.def_pairstyle("lj126/cut")
        lj126.def_type("OO", epsilon=0.1521, sigma=3.1507)
        lj126.def_type("OH", epsilon=0.2104, sigma=0.4)
        lj126.def_type("HH", epsilon=0.0460, sigma=1.7753)
        ff.def_pairstyle("coul/cut")

        frame = mp.Frame("atoms", "bonds", "angles", "pairs")
        frame["atoms"]["x"] = [0.0000, 0.75695, -0.75695]
        frame["atoms"]["y"] = [-0.06556, 0.52032, 0.52032]
        frame["atoms"]["z"] = [0.000, 0.00000, 0.00000]
        frame["atoms"]["type"] = [0, 1, 1]
        frame["bonds"]["i"] = [0, 1]
        frame["bonds"]["j"] = [0, 2]
        frame["angles"]["i"] = [1]
        frame["angles"]["j"] = [0]
        frame["angles"]["k"] = [2]
        frame["angles"]["type"] = [0]
        frame["pairs"]["i"] = [0, 0, 1]
        frame["pairs"]["j"] = [1, 2, 2]
        frame["pairs"]["type"] = [1, 1, 2]

        potential = mp.potential.get_potentials(ff)

        optimizer = mp.optimizer.BFGS(potential)
        optimizer.run(frame, step=100)

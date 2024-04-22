import molpy as mp
import numpy as np


def spc_fw():

    struct = mp.Struct(n_atoms=3, **{mp.Alias.name: "spc/fw"})
    struct.add_atom(
        _element="O",
        _xyz=np.array([0.00000, -0.06461, 0.00000]),
        _charge=-0.8476,
        _mass=15.9994,
    )
    struct.add_atom(
        _element="H", _xyz=np.array([0.81649, 0.51275, 0.00000]), _charge=0.4238,
        _mass=1.008,
    )
    struct.add_atom(
        _element="H",
        _xyz=np.array([-0.81649, 0.51275, 0.00000]),
        _charge=0.4238,
        _mass=1.008,
    )
    struct.topology.add_bond(0, 1)
    struct.topology.add_bond(0, 2)

    forcefield = mp.ForceField()
    bondstyle = forcefield.def_bondstyle("harmonic")
    bondstyle.def_bondtype("O-H", 0, 1, k=1059.162, r0=1.012)
    bondstyle.def_bondtype("O-H", 0, 2, k=1059.162, r0=1.012)

    anglestyle = forcefield.def_anglestyle("harmonic")
    anglestyle.def_angletype("H-O-H", 1, 0, 2, k=75.90, theta0=113.24)
    
    pairstyle = forcefield.def_pairstyle("lj/cut/coul/cut", global_cutoff=10.0) 
    pairstyle.def_pairtype("O-O", 0, 0, epsilon=0.1553, sigma=3.1506)
    pairstyle.def_pairtype("O-H", 0, 1, epsilon=0.0, sigma=1.0)
    pairstyle.def_pairtype("H-H", 1, 1, epsilon=0.0, sigma=1.0)

    return {
        "struct": struct,
        "forcefield": forcefield,
    }
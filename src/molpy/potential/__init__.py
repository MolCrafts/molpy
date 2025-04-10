from .base import Potential, PotentialDict
from .angle import *
from .bond import *
from .pair import *
import molpy as mp
from collections import defaultdict

def get_bond_potentials(style):
    from .bond import Harmonic
    pot = {Harmonic.name: Harmonic}
    params = defaultdict(list)
    for type in style.types.values():
        for key in type.keys():
            params[key].append(type[key])
    return pot[style.name](**params)
    
def get_angle_potentials(style):
    from .angle import Harmonic
    pot = {Harmonic.name: Harmonic}
    params = defaultdict(list)
    for type in style.types.values():
        for key in type.keys():
            params[key].append(type[key])
    return pot[style.name](**params)

def get_pair_potentials(style):
    from .pair import LJ126
    from .pair import CoulCut
    pot = {LJ126.name: LJ126, CoulCut.name: CoulCut}
    params = defaultdict(list)
    for type in style.types.values():
        for key in type.keys():
            params[key].append(type[key])
    return pot[style.name](**params)

def get_potentials(forcefield: mp.ForceField):

    potentials = {}
    # for style in forcefield.bondstyles:
    #     potentials[style.name] = get_bond_potentials(style)

    for style in forcefield.anglestyles:
        potentials[style.name] = get_angle_potentials(style)

    # for style in forcefield.pairstyles:
    #     potentials[style.name] = get_pair_potentials(style)

    return PotentialDict(potentials)
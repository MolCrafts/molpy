from typing import Literal

class Minimizer:

    def __init__(self, criteria: float=1e-6, algorithm: Literal['cg', 'sd'] = 'cg', forcefield:str='gaff'):

        self.criteria = criteria
        self.algorithm = algorithm
        self.forcefield = forcefield

    def optimize(self, structure, nsteps:int=2500):
        pass

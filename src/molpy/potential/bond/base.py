from ..base import Potential

class BondPotential(Potential):
    
    name: str
    potentials: dict = {}

    def __new__(cls, name:str, *args, **kwargs):

        if name in cls.potentials:
            return cls.potentials[name]
        raise NotImplementedError(f"Potential {name} is not implemented")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.potentials[cls.name] = cls
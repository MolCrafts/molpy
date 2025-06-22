from .packmol import Packmol
from .nlopt import NloptPacker

def get_packer(name, *args, **kwargs):

    if name == "packmol":
        return Packmol(*args, **kwargs)
    elif name == "nlopt":
        return NloptPacker(*args, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented")
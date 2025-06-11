from .packmol import Packmol
try:
    import nlopt
    from .nlopt import NloptPacker
except ImportError:
    nlopt = None
    NloptPacker = ImportError

def get_packer(name, *args, **kwargs):

    if name == "packmol":
        return Packmol(*args, **kwargs)
    elif name == "nlopt":
        return NloptPacker(*args, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented")
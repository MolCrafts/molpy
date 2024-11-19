import molpy as mp
from functools import wraps

def to_system(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            system = args[0]
            args = args[1:]
        else:
            system = kwargs.pop("system", None)
        
        if isinstance(system, mp.Frame):
            system = mp.System(frame=system)
        elif isinstance(system, (mp.Struct, mp.Segment)):
            system = mp.System(frame=system.to_frame())

        assert isinstance(system, mp.System), f"Expected system to be a molpy System object, got {type(system)}"
        return func(system, *args, **kwargs)

    return wrapper
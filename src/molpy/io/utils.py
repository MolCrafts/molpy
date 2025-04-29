import molpy as mp
from functools import wraps

FrameLike = mp.Frame | mp.System | mp.Struct

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

def to_frame(framelike: FrameLike) -> mp.Frame:

    if isinstance(framelike, mp.System):
        frame = framelike.frame
    elif isinstance(framelike, mp.Frame):
        frame = framelike
    elif isinstance(framelike, mp.Struct):
        frame = framelike.to_frame()

    return frame
    


class ZipReader:

    def __init__(self, *readers, merge: bool = True):

        self.readers = readers
        self.merge = merge

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self.readers:
            reader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        for frames in zip(*self.readers):
            if self.merge:
                yield mp.Frame.From_frames(frames)
            else:
                yield frames
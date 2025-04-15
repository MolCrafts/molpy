from .base import DataWriter

class XYZWriter(DataWriter):
    """
    A class to write XYZ files.
    """

    def __init__(self, path: str, *args, **kwargs):
        super().__init__(path, *args, **kwargs)

    def write(self, frame):
        """
        Write the XYZ file.
        """
        with open(self._path, "w") as f:
            f.write(f"{len(frame)}\n")
            f.write(f"Step={frame.get('step')}\n")
            for atom in frame["atoms"].itertuples(index=False):
                f.write(f"{atom.name} {atom.x:.3f} {atom.y:.3f} {atom.z:.3f}\n")
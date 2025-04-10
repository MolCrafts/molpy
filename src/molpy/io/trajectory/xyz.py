
class XYZTrajectoryWriter:

    def __init__(self, fpath: str):
        
        self.fpath = fpath
        self.fobj = open(fpath, "w")

    def __del__(self):
        if not self.fobj.closed:
            self.fobj.close()

    def write_frame(self, system):

        frame = system.frame
        box = system.box

        atoms = frame["atoms"]
        n_atoms = len(atoms)

        self.fobj.write(f"{n_atoms}\n")
        self.fobj.write(f"Step={frame.get('step')} Lattic\"{box.matrix.tolist()}\"\n")

        for _, atom in atoms.iterrows():

            x = atom["x"]
            y = atom["y"]
            z = atom["z"]
            elem = atom.get("element", "X")

            self.fobj.write(f"{elem} {x} {y} {z}\n")

    def write(self, system):

        traj = system.trajectory

        for frame in traj:
            self.write_frame(frame)

    def close(self):

        self.fobj.close()
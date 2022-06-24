# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

class System:

    def __init__(self):

        self.box = None
        self.forcefield = None
        self.atoms = None

    def load_data(self):
        pass

    def load_traj(self):
        pass

    # ---= forcefield interface =---

    def select_frame(self, ):
        pass

    def def_atomtype(self):
        pass

    def def_bondtype(self):
        pass

    def def_angletype(self):
        pass

    def def_dihedraltype(self):
        pass

    # ---= box interface =---

    def get_box(self):
        pass

    # ---= atoms interface =---
    def add_atoms(self):
        pass

    def add_edges(self):
        pass

    def get_angles(self):
        pass


# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import itertools


class Atom(dict):
    pass

class Bond(dict):

    def __init__(self, atom1:Atom, atom2:Atom, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.atom1 = atom1
        self.atom2 = atom2

    def __eq__(self, b: 'Bond') -> bool:
        return (self.atom1 == b[0] and self.atom2) == b[1] or (self.atom1 == b[1] and self.atom2 == b[0])

class Angle(dict):

    def __init__(self, i:int, j:int, k:int, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._atom_idx = [i, j, k]

    def __getitem__(self, __k):
        if isinstance(__k, int):
            return self._atom_idx[__k]
        return super().__getitem__(__k)

    def __eq__(self, a: 'Angle') -> bool:
        
        if self[1] == a[1]:
            return (self[0] == a[0] and self[2] == a[2]) or (self[0] == a[2] and self[2] == a[0])
        return False

class Dihedral(dict):

    def __init__(self, i:int, j:int, k:int, l:int, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._atom_idx = [i, j, k, l]

    def __getitem__(self, __k):
        if isinstance(__k, int):
            return self._atom_idx[__k]
        return super().__getitem__(__k)

    def __eq__(self, d: 'Dihedral') -> bool:
        
        if self[1] == d[1] and self[2] == d[2] or self[1] == d[2] and self[2] == d[1]:
            return (self[0] == d[0] and self[3] == d[3]) or (self[0] == d[3] and self[3] == d[0])
        return False

    
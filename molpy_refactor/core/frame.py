# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .struct import StructData

class Frame:

    def __init__(self, ):

        self.atoms = StructData()

    @classmethod
    def from_chemfile_frame(cls, chemfile_frame):

        molpy_frame = cls()
        
        positions = chemfile_frame.positions
        molpy_frame.atoms['positions'] = positions


        return molpy_frame
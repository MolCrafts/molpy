# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path

class LAMMPSLog:

    def __init__(self, file:str|Path, ):

        self.file = Path(file)

    def read(self):
        with open(self.file, 'r') as f:
            lines = f.readlines()

        for line in lines:

            pass
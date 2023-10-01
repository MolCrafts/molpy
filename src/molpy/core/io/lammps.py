# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
from ..forcefield import ForceField

class LammpsScriptParser:

    def __init__(self, fpath:str | Path, mode='r'):
        self._fpath = Path(fpath)
        self._ff = {}

        self._file = open(self._fpath, mode)

    def __del__(self):
        self._file.close()

    def parse(self):
        
        for line in self._file.readlines():

            line = line.strip()

            if line.startswith('#'):
                continue

            if line.startswith('pair'):
                self.parse_pair(line)

    def parse_pair(self, line:str):

        pair = self._ff['pair']

        if line.startswith('pair_style'):
            cmd, style, *global_args = line.split()
            assert cmd == 'pair_style'

            pair[style] = {
                'global_args': global_args,
                'pair_coeffs': {},
            }

        elif line.startswith('pair_coeff'):

            cmd, *args = line.split()
            assert cmd == 'pair_coeff'
            if not 'hybrid' in args:
                assert pair.keys() == 1
                style = pair.keys()[0]
                type1, type2 = args[:2]
                pair[style]['pair_coeffs'][type1] = {type2: [float(arg) for arg in args[2:]]}

        elif line.startswith('pair_modift'):

            cmd, *args = line.split()
            assert cmd == 'pair_modify'
            
        else:
            pass

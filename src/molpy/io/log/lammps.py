# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
import re
import numpy as np

class LAMMPSLog:

    def __init__(self, file:str|Path, style='default'):

        self.file = Path(file)
        self.info = {
            'n_stages': 0,
            'stages': [],

        }
        self.style = style

    def read(self):

        with open(self.file, 'r') as f:
            log_str = f.read()

        self.read_version(log_str)
        self.read_thermo(log_str, self.style)

    def read_version(self, text:str):
        index = text.find('\n')
        self['version'] = text[:index]

    def read_thermo(self, text, style):

        if style == 'default':
            pattern = r'Per MPI rank .*?\n(.*?Loop time.*?)\n'
            result = re.search(pattern, text, re.DOTALL)
            for stage in result.groups():
                
                lines = stage.split('\n')
                fields = lines[0].split()
                stage_dict = dict.fromkeys(fields, list())
                for line in lines[2:]:
                    values = line.split()
                    for k, v in zip(fields, values):
                        print(v)
                        stage_dict[k].append(v)
                    
                # for fields in stage_dict:
                #     stage_dict[fields] = np.array(stage_dict[fields])
                self['stages'].append(stage_dict)
            # self['n_stages'] = len(self['stages'])

    def __getitem__(self, key):
        return self.info[key]
    
    def __setitem__(self, key, value):
        self.info[key] = value
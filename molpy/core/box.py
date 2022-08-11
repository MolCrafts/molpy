# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-12
# version: 0.0.1

from freud.box import Box as _Box

class Box(_Box):
    
    def __repr__(self)->str:
        return f'<Box: LX={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, powered by freud>'
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-15
# version: 0.0.1

from typing import NewType, Any

class MolpyNewType(NewType):

    def __getitem__(self, any):
        pass

ArrayLike = MolpyNewType('ArrayLike', Any)
Number = MolpyNewType('Number', Any)
N = NewType('N', int)

    
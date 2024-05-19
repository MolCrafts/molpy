# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1

from .struct import Struct


class Frame(Struct):

    def __init__(self, *args, **kwargs):

        self._box = None


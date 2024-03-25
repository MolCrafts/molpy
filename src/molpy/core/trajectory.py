# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-05
# version: 0.0.1

from pathlib import Path
from typing import Any, Iterable
from molpy.io.loader import TrajLoader
from .frame import Frame

class Trajectory:

    def __init__(self):
        self._frames = []

    def add_frame(self, frame: Frame):
        self._frames.append(frame)
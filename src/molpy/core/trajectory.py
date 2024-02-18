# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-05
# version: 0.0.1

from pathlib import Path
from typing import Any, Iterable
from molpy.io.chflloader import TrajLoader
from .frame import Frame

class Trajectory:

    def __init__(self):
        self._frames = []

    def append(self, frame):
        self._frames.append(frame)

    def join_frame_property(self, key, props):
        for frame, prop in self._frames, props:
            frame[key] = prop

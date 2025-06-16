# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2024-02-05
# version: 0.0.1

from typing import List, Optional, Any, Iterator, Union
from copy import deepcopy
from .frame import Frame

class Trajectory:
    """
    A collection of Frame objects representing a molecular trajectory.
    """
    def __init__(self, frames: Optional[List[Frame]] = None, **meta):
        self.frames: List[Frame] = list(frames) if frames is not None else []
        self.meta = meta

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Frame, 'Trajectory']:
        if isinstance(idx, slice):
            return Trajectory(self.frames[idx], **self.meta)
        return self.frames[idx]

    def append(self, frame: Frame):
        self.frames.append(frame)

    def extend(self, frames: List[Frame]):
        self.frames.extend(frames)

    def iterframes(self) -> Iterator[Frame]:
        return iter(self.frames)

    def copy(self):
        return Trajectory([f.copy() for f in self.frames], **deepcopy(self.meta))

    def __iter__(self):
        return self.iterframes()

    def __repr__(self):
        return f"<Trajectory n_frames={len(self.frames)} meta={self.meta}>"

    # Placeholder for IO
    def to_file(self, filename, fmt=None):
        raise NotImplementedError("Trajectory export not implemented.")

    @classmethod
    def from_file(cls, filename, fmt=None):
        raise NotImplementedError("Trajectory import not implemented.")

    @classmethod
    def concat(cls, trajectories: list["Trajectory"]) -> "Trajectory":
        """Concatenate multiple Trajectory objects into one."""
        if not trajectories:
            return cls()
        # Concatenate all frames from all trajectories
        all_frames = []
        for traj in trajectories:
            all_frames.extend(traj.frames)
        # Optionally merge meta (here, just take the first's meta)
        meta = trajectories[0].meta.copy() if hasattr(trajectories[0], "meta") else {}
        return cls(all_frames, **meta)

    def to_dict(self) -> dict:
        """Convert the trajectory to a list of frame dicts and meta info."""
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "meta": self.meta.copy()
        }

    def get_meta(self, key: str, default=None):
        return self.meta.get(key, default)

    def set_meta(self, key: str, value: Any):
        self.meta[key] = value
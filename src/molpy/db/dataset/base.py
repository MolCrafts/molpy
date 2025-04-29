from abc import ABC, abstractmethod
from pathlib import Path

import molpy as mp
from molpy.core.alias import NameSpace


class Dataset(ABC):

    def __init__(
        self,
        name: str,
        save_dir: Path | None = None
    ):
        self.labels = NameSpace(name)
        
        if save_dir is None:
            self.save_dir = None
        elif isinstance(save_dir, (Path, str)):
            self.save_dir = Path(save_dir) / name
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("save_dir must be a Path or None.")

    @property
    def streaming(self) -> bool:
        """
        Check if the dataset is in streaming mode.
        """
        return self.save_dir is None

    def download(self): ...

    def parse(self): ...


class IterDatasetMixin:
    """
    Mixin class for iterable datasets.
    """

    def __iter__(self) -> mp.Frame: ...

class MapDatasetMixin:
    """
    Mixin class for map datasets.
    """

    def __getitem__(self, index: int) -> mp.Frame: ...

class TrajectoryLikeDatasetMixin:
    """
    Mixin class for trajectory-like datasets.
    """
    def get_trajectory(self): ...


class FrameLikeDatasetMixin:
    """
    Mixin class for frame-like datasets.
    """
    def get_frames(self): ...
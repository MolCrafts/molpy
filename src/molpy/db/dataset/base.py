from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Sequence

import molpy as mp
from molpy.core.alias import NameSpace


class Dataset(ABC):

    def __init__(
        self,
        name: str,
        save_dir: Path | None = None,
        device: str = "cpu"
    ):
        self.labels = NameSpace(name)
        self.device = device

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():  # create save_dir
                self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None


    def prepare(self): ...

    def download(self): ...

    def update(self, state): ...

    def get_frame(self, idx: int) -> mp.Frame: ...
        

class IterStyleDataset(Dataset):

    def __init__(self, frames: Sequence[mp.Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int) -> Any:
        return self.frames[i]


class MapStyleDataset(Dataset):

    def __getitem__(self, idx):
        return self.get_frame(idx)
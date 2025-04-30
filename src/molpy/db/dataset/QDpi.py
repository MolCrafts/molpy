import io
import tarfile
import time
from pathlib import Path

import h5py
import numpy as np
import requests
from .base import Dataset, MapDatasetMixin, IterDatasetMixin, FrameLikeDatasetMixin
import logging
from molpy.core import alias
from tqdm import tqdm
import molpy as mp
from .utils import download
from nesteddict import ArrayDict

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


class QM9(Dataset, MapDatasetMixin, IterDatasetMixin, FrameLikeDatasetMixin):

    def __init__(
        self,
        save_dir: Path | None = None,
        # only download list
        desired: str | list[str] = "all"
    ):
        super().__init__("qm9", save_dir=save_dir)
        root_url = "https://gitlab.com/RutgersLBSR/QDpiDataset/-/blob/main/data/"
        self._list = {
            "charged": [
                "re_charged",
                "remd_charged",
                "spice_charged"
            ],
            "neutral": [
                "ani",
                "comp6.hdf6",
                "freesolvmd",
                "geom",
                "re",
                "remd",
                "spice"
            ]
        }
        all_datasets = {
            **{name: "charged" for name in self._list["charged"]},
            **{name: "neutral" for name in self._list["neutral"]},
        }

        if desired == "all":
            names = list(all_datasets.keys())
        elif desired == "charged":
            names = self._list["charged"]
        elif desired == "neutral":
            names = self._list["neutral"]
        elif isinstance(desired, str):
            if desired not in all_datasets:
                raise ValueError(f"Unknown dataset: {desired}")
            names = [desired]
        elif isinstance(desired, list):
            unknown = [d for d in desired if d not in all_datasets]
            if unknown:
                raise ValueError(f"Unknown dataset(s): {unknown}")
            names = desired
        else:
            raise ValueError(f"Invalid type for desired: {type(desired)}")

        hdf5_list = [f"{root_url}{all_datasets[name]}/{name}.hdf5" for name in names]
        self._hdf5_list = hdf5_list


    def download(self):

        self._contents = []
        for url in self._hdf5_list:
            logger.info(f"Downloading {url}...")
            content = download(url, self.save_dir)
            self._contents.append(content)

        return self
    
    def parse(self):

        self._frames = []
        for content in self._contents:
            with h5py.File(content, "r") as f:
                mol_names = list(f.keys())
                for molecule in (f[mol_name] for mol_name in mol_names):
                    pbc = not bool(molecule["nopbc"])
                    set_000 = molecule["set.000"]
                    xyz = np.array(set_000["coord.npy"]).reshape(-1, 3)
                    energy = np.array(set_000["energy.npy"]).squeeze()  # (1, 1)
                    forces = np.array(set_000["forces.npy"]).reshape(-1, 3)
                    net_charge = np.array(set_000["net_charge.npy"]).squeeze()  # (1, 1)
                    type_map = np.array(molecule["type_map.raw"])
                    type_idx = np.array(molecule["type.raw"])
                    elements = np.array([mp.Element.get_symbols(i) for i in type_map[type_idx]])

                    frame = mp.Frame({
                        "atoms": ArrayDict({
                            "xyz": xyz,
                            "atomic_number": elements
                        }),
                        "pbc": pbc,
                        "props": {
                            "energy": energy,
                            "forces": forces,
                            "net_charge": net_charge
                        }
                    })
                    self._frames.append(frame)

        return self
    
    def __getitem__(self, index: int) -> mp.Frame:
        return self._frames[index]
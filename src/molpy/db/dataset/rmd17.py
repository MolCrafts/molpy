import io
import logging
import tarfile
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import requests

import molpy as mp
from molpy.core import alias

from .base import Dataset, IterDatasetMixin, MapDatasetMixin, FrameLikeDatasetMixin, TrajectoryLikeDatasetMixin

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


class rMD17(Dataset, IterDatasetMixin, MapDatasetMixin, FrameLikeDatasetMixin, TrajectoryLikeDatasetMixin):

    atomrefs = {
        "energy": np.array(
            [
                0.0,
                -313.5150902000774,
                0.0,
                0.0,
                0.0,
                0.0,
                -23622.587180094913,
                -34219.46811826416,
                -47069.30768969713,
            ]
        )
    }

    datasets_dict = dict(
        aspirin="rmd17_aspirin.npz",
        azobenzene="rmd17_azobenzene.npz",
        benzene="rmd17_benzene.npz",
        ethanol="rmd17_ethanol.npz",
        malonaldehyde="rmd17_malonaldehyde.npz",
        naphthalene="rmd17_naphthalene.npz",
        paracetamol="rmd17_paracetamol.npz",
        salicylic_acid="rmd17_salicylic.npz",
        toluene="rmd17_toluene.npz",
        uracil="rmd17_uracil.npz",
    )

    def __init__(
        self,
        save_dir: Path | None,
        molecule: Literal[
            "aspirin",
            "azobenzene",
            "benzene",
            "ethanol",
            "malonaldehyde",
            "naphthalene",
            "paracetamol",
            "salicylic",
            "toluene",
            "uracil",
        ],
    ):
        super().__init__(name="rmd17", save_dir=save_dir)
        self.labels.set("energy", "total energy", float, "kcal/mol", (None, 1))
        self.labels.set("forces", "all forces", float, "kcal/mol/A", (None, 3))

        self.molecule = molecule

        molecules = list(self.datasets_dict.keys())
        if molecule not in molecules:
            raise ValueError(
                f"Invalid molecule. Choose from {molecules}. Got {molecule}"
            )

        if isinstance(self.save_dir, Path):
            self.archive_path = self.save_dir / "rmd17.tar.gz"
            self.npz_path = (
                self.save_dir / self.molecule / self.datasets_dict[self.molecule]
            )  # save_dir/aspirin/rmd17_aspirin.npz
        else:
            raise NotImplementedError("not implemented for streaming mode")

        self._frames = []

    def __len__(self):
        return len(self._frames)

    def get_frame(self, idx):
        return self._frames[idx]

    @property
    def frames(self):
        return self._frames

    def parse(self, total: int | None = 1000):

        if self.save_dir:
            data = np.load(self.npz_path)
            frames = self.parse_data(data, total=total)
            self._frames = frames
        return frames

    # def _fetch_data(self) -> Sequence[mp.Frame]:
    #     logger.info(f"Fetching {self.molecule} data")
    #     rmd17_url = "https://figshare.com/ndownloader/files/23950376"
    #     rmd17_bytes = requests.get(rmd17_url, allow_redirects=True).content
    #     rmd17_fobj = io.BytesIO(rmd17_bytes)
    #     rmd17_fobj.seek(0)
    #     rmd17_tar = tarfile.open(fileobj=rmd17_fobj, mode="r:bz2")
    #     data = np.load(rmd17_tar.extractfile(self.npz_path))
    #     return data

    def download(self) -> Sequence[mp.Frame]:

        url = "https://figshare.com/ndownloader/files/23950376"
        if not self.archive_path.exists():
            logger.info(f"Downloading {self.molecule} data")
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(self.archive_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logger.info("Done.")

        if not self.npz_path.exists():
            logger.info("Extracting data...")
            with tarfile.open(self.archive_path) as tar:
                tar.extract(
                    path=self.npz_path.parent,
                    member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}",
                    filter=lambda member, path: member.replace(
                        name=self.datasets_dict[self.molecule]
                    ),
                )

        logger.info(f"Parsing molecule {self.molecule}")
        return self

    def parse_data(self, data, total: int | None) -> Sequence[mp.Frame]:
        numbers = np.array(data["nuclear_charges"])
        frames = []
        if total is None:
            total = len(data["energies"])
        indices = rng.permutation(total)
        for idx in indices:
            frame = mp.Frame()
            frame[alias.Z] = numbers
            frame[alias.R] = np.array(data["coords"][idx], dtype=np.float32)
            frame["labels", "energy"] = np.array([data["energies"][idx]])
            frame["labels", "forces"] = np.array(data["forces"][idx])
            frame[alias.n_atoms] = np.array([len(numbers)])
            frames.append(frame)
        return frames

    def __iter__(self):
        for frame in self._frames:
            yield frame

    def __getitem__(self, index: int) -> mp.Frame:
        return self._frames[index]

    def get_trajectory(self):
        traj = mp.Trajectory(self._frames)
        return traj
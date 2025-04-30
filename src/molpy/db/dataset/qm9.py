import io
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
from .base import Dataset, MapDatasetMixin, IterDatasetMixin, FrameLikeDatasetMixin
import logging
from molpy.core import alias
from tqdm import tqdm
import molpy as mp

logger = logging.getLogger(__name__)

rng = np.random.default_rng()


class QM9(Dataset, MapDatasetMixin, IterDatasetMixin, FrameLikeDatasetMixin):
    def __init__(
        self,
        save_dir: Path | None = None,
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("qm9", save_dir=save_dir)

        self.remove_uncharacterized = remove_uncharacterized
        self.atom_ref = atom_ref

        self.labels.set("A", "rotational_constant_A", "GHz", float)
        self.labels.set("B", "rotational_constant_B", "GHz", float)
        self.labels.set("C", "rotational_constant_C", "GHz", float)
        self.labels.set("mu", "dipole_moment", "Debye", float)
        self.labels.set("alpha", "isotropic_polarizability", "a0", float)
        self.labels.set("alpha", "isotropic_polarizability", "a0", float)
        self.labels.set("homo", "homo", "hartree", float)
        self.labels.set("lumo", "lumo", "hartree", float)
        self.labels.set("gap", "gap", "hartree", float)
        self.labels.set("r2", "electronic_spatial_extent", "a0", float)
        self.labels.set("zpve", "zpve", "hartree", float)
        self.labels.set("U0", "energy_U0", "hartree", float)
        self.labels.set("U", "energy_U", "hartree", float)
        self.labels.set("H", "enthalpy_H", "hartree", float)
        self.labels.set("G", "free_energy", "hartree", float)
        self.labels.set("Cv", "heat_capacity", "cal/mol/K", float)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def download(self, total: int = None, preprocess=[]):
        logger.info("prepaering QM9 dataset...")

        save_dir = self.save_dir
        if save_dir and (save_dir / "qm9.tar.bz2").exists():
            qm9_bytes = io.BytesIO((save_dir / "qm9.tar.bz2").read_bytes())
            exclude_txt = (save_dir / "exclude.txt").read_text()

        else:
            qm9_url = "https://ndownloader.figshare.com/files/3195389"
            exclude_url = "https://figshare.com/ndownloader/files/3195404"
            logger.info("downloading...")
            qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
            exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
            exclude_txt = exclude_bytes.decode("utf-8")
            if save_dir:
                with open(save_dir / "qm9.tar.bz2", "wb") as f:
                    f.write(qm9_bytes)
                with open(save_dir / "exclude.txt", "w") as f:
                    f.write(exclude_txt)
            # stream
            qm9_bytes = io.BytesIO(qm9_bytes)
            qm9_bytes.seek(0)

        self._qm9_bytes = qm9_bytes
        self._exclude_txt = exclude_txt

        return self

    def parse(self, total: int | None = 1000):

        qm9_bytes = self._qm9_bytes
        exclude_txt = self._exclude_txt
        exclude = set(int(line.split()[0]) for line in exclude_txt.split("\n")[9:-2])

        start_extract_time = time.perf_counter()

        with tarfile.open(fileobj=qm9_bytes, mode="r:bz2") as tar_file:
            names = tar_file.getnames()
            qm9_indices = set(int(name[-10:-4]) for name in names)
            exclude = set(exclude)
            qm9_indices = list(qm9_indices - exclude)
            if total is None:
                total = len(qm9_indices)
            random_indices = rng.permutation(total)
            end_extract_time = time.perf_counter()
            logger.info(
                f"end extract, cost {end_extract_time - start_extract_time:.2f}s, average {(end_extract_time - start_extract_time)/total:.2f} s/file"
            )

            QM9 = self.labels
            props = [
                QM9.A,
                QM9.B,
                QM9.C,
                QM9.mu,
                QM9.alpha,
                QM9.homo,
                QM9.lumo,
                QM9.gap,
                QM9.r2,
                QM9.zpve,
                QM9.U0,
                QM9.U,
                QM9.H,
                QM9.G,
                QM9.Cv,
            ]
            props_ind = {k: i for i, k in enumerate(props)}
            logger.info("parsing...")
            start_time = time.perf_counter()
            frames = []

            tmp_dir = self.save_dir / "qm9_extracted"
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True)
                tar_file.extractall(tmp_dir)
            for idx in tqdm(random_indices):
                with open(tmp_dir / f"dsgdb9nsd_{qm9_indices[idx]:06}.xyz", "r") as f:
                    lines = f.readlines()
                    n_atoms = int(lines[0])
                    Z = [mp.Element(l.split()[0]).number for l in lines[2:-3]]
                    R = [
                        [float(i.replace("*^", "E")) for i in l.split()[1:4]]
                        for l in lines[2:-3]
                    ]
                    frame = mp.Frame()
                    frame["props", "name"] = lines[0].split()[1]  # ?
                    frame[alias.Z] = np.array(Z)
                    frame[alias.R] = np.array(R)
                    frame[alias.n_atoms] = np.array(n_atoms)
                    prop_line = lines[1].split()[2:]
                    # prop_line: tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
                    for p, i in props_ind.items():
                        frame["labels", p[-1]] = np.array([float(prop_line[i])])
                    frames.append(frame)
            ## === on the fly parsing ===
            # files = (
            #     tar_file.extractfile(name).read().decode('utf-8').split('\n') for name in random_seleted_names
            # )  # use a generator to lazy load
            # for lines in tqdm(files):
            #     n_atoms = int(lines[0])
            #     Z = [mpot.Element(l.split()[0]).number for l in lines[2:-4]]
            #     R = [
            #         [float(i.replace("*^", "E")) for i in l.split()[1:4]]
            #         for l in lines[2:-4]
            #     ]
            #     frame = mpot.Frame()
            #     # frame["props", "name"] = lines.stem
            #     frame[alias.Z] = np.array(Z)
            #     frame[alias.R] = np.array(R, dtype=config.ftype)
            #     frame[alias.n_atoms] = np.array(n_atoms, dtype=config.itype)
            #     prop_line = lines[1].split()
            #     for k, v in zip(props, prop_line[1:]):
            #         frame["labels", k[-1]] = np.array(
            #             [float(v)], dtype=config.ftype
            #         )
            #     frames.append(frame)
        logger.info(f"end parse, cost {time.perf_counter() - start_time:.2f}s")

        self._frames = frames
        return frames

    def get_frames(self):
        return self._frames

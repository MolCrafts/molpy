# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from pathlib import Path
import pytest
import subprocess
from molpy import Alias
import numpy as np
from ase.build import bulk, molecule
from ase import Atoms


@pytest.fixture(name="test_data_path", scope="session")
def find_test_data() -> Path:

    data_path = Path(__file__).parent / "chemfile-testcases"
    if not data_path.exists():
        print("Downloading test data...")
        subprocess.run(
            f"git clone https://github.com/molcrafts/chemfile-testcases.git {data_path.parent}/chemfile-testcases",
            shell=True,
            check=True,
        )
    else:
        print("Test data already exists; updating...")
        subprocess.run(f"git pull", shell=True, cwd=data_path, check=True)

    return data_path


@pytest.fixture(name="ase_tests", scope="session")
def ase_tests() -> list[Atoms]:

    # triclinic atomic structure
    CaCrP2O7_mvc_11955_symmetrized = {
        "positions": [
            [3.68954016, 5.03568186, 4.64369552],
            [5.12301681, 2.13482791, 2.66220405],
            [1.99411973, 0.94691001, 1.25068234],
            [6.81843724, 6.22359976, 6.05521724],
            [2.63005662, 4.16863452, 0.86090529],
            [6.18250036, 3.00187525, 6.44499428],
            [2.11497733, 1.98032773, 4.53610884],
            [6.69757964, 5.19018203, 2.76979073],
            [1.39215545, 2.94386142, 5.60917746],
            [7.42040152, 4.22664834, 1.69672212],
            [2.43224207, 5.4571615, 6.70305327],
            [6.3803149, 1.71334827, 0.6028463],
            [1.11265639, 1.50166318, 3.48760997],
            [7.69990058, 5.66884659, 3.8182896],
            [3.56971588, 5.20836551, 1.43673437],
            [5.2428411, 1.96214426, 5.8691652],
            [3.12282634, 2.72812741, 1.05450432],
            [5.68973063, 4.44238236, 6.25139525],
            [3.24868468, 2.83997522, 3.99842386],
            [5.56387229, 4.33053455, 3.30747571],
            [2.60835346, 0.74421609, 5.3236629],
            [6.20420351, 6.42629368, 1.98223667],
        ],
        "cell": [
            [6.19330899, 0.0, 0.0],
            [2.4074486111396207, 6.149627748674982, 0.0],
            [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
        ],
        "numbers": [
            20,
            20,
            24,
            24,
            15,
            15,
            15,
            15,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ],
        "pbc": [True, True, True],
    }

    bulk_metal = [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "diamond", a=6),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "bct", a=6, c=3),
        # test very skewed unit cell
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk("Bi", "rhombohedral", a=6, alpha=10),
        bulk("Bi", "rhombohedral", a=6, alpha=5),
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiFCu", "fluorite", a=6),
        Atoms(**CaCrP2O7_mvc_11955_symmetrized),
    ]

    molecules = [
        molecule("H2O"),
        molecule("OCHCHO"),
        molecule("CH3CH2NH2"),
        molecule("methylenecyclopropane"),
        molecule("C3H9C"),
    ]
    return bulk_metal + molecules


@pytest.fixture(name="batch_ase_tests", scope="session")
def batch_ase_tests() -> list[Atoms]:
    def ase2data(frames):
        n_atoms = [0]
        pos = []
        cell = []
        pbc = []
        for ff in frames:
            n_atoms.append(len(ff))
            pos.append(ff.get_positions())
            cell.append(ff.get_cell().array)
            pbc.append(ff.get_pbc())
        pos = np.concatenate(pos)
        cell = np.concatenate(cell)
        pbc = np.concatenate(pbc)
        stride = np.cumsum(n_atoms)
        batch = np.zeros(pos.shape[0], dtype=int)
        for ii, (st, nd) in enumerate(zip(stride[:-1], stride[1:])):
            batch[st:nd] = ii
        n_atoms = n_atoms[1:]
        return (
            pos,
            cell,
            pbc,
            batch,
            n_atoms,
        )

    return ase2data(ase_tests())

@pytest.fixture(scope="session")
def ase_free_tests(ase_tests):
    return [atoms for atoms in ase_tests if np.all(atoms.cell.array == 0)]

@pytest.fixture(scope="session")
def ase_orth_tests(ase_tests):
    def is_diag(M):
        i, j = np.nonzero(M)
        return np.all(i == j)

    return [
        atoms
        for atoms in ase_tests
        if is_diag(atoms.cell.array) and not np.all(atoms.cell.array == 0)
    ]
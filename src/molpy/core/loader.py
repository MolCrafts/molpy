# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from pathlib import Path
import chemfiles as chfl

from .frame import Frame
from .alias import Alias

__all__ = ["DataLoader", "MemoryLoader"]


class ChflLoader:
    _fileHandler: chfl.Trajectory | chfl.MemoryTrajectory

    def load_frame(self, step: int = 0) -> Frame:
        """Load a frame from the trajectory file."""
        chflFrame = self._fileHandler.read_step(step)
        frame = Frame()

        # get frame properties
        frame[Alias.timestep] = chflFrame.step
        box_matrix = chflFrame.cell.matrix
        frame.get_box().set_matrix(box_matrix)
        frame.atoms[Alias.xyz] = chflFrame.positions
        frame[Alias.natoms] = len(chflFrame.atoms)

        # get atom properties
        INTRINSIC_PROPS = [
            Alias.name,
            Alias.atomic_number,
            Alias.charge,
            Alias.mass,
            Alias.type,
        ]

        first_atom = chflFrame.atoms[0]
        extra_properties = first_atom.list_properties()
        EXTRA_PROPS = [getattr(Alias, prop) for prop in extra_properties]
        for prop in INTRINSIC_PROPS + EXTRA_PROPS:
            if hasattr(first_atom, prop):
                frame.atoms[prop] = [getattr(atom, prop) for atom in chflFrame.atoms]

        # get connectivity
        bonds = chflFrame.topology.bonds
        angles = chflFrame.topology.angles
        dihedrals = chflFrame.topology.dihedrals
        impropers = chflFrame.topology.impropers

        for bond in bonds:
            frame._connectivity.add_bond(*bond)

        for angle in angles:
            frame._connectivity.add_angle(*angle)

        for dihedral in dihedrals:
            frame._connectivity.add_dihedral(*dihedral)

        for improper in impropers:
            frame._connectivity.add_improper(*improper)

        #   residues = chemfile_frame.topology.residues
        #     if residues:
        #         nresidues = len(residues)
        #         molpy_frame._nresidues = nresidues
        #         names = []
        #         ids = []
        #         index = np.empty((nresidues), dtype=object)
        #         props = []
        #         for i, residue in enumerate(residues):
        #             ids.append(residue.id)
        #             names.append(residue.name)
        #             props.append({k:residue[k] for k in residue.list_properties()})
        #             index[i] = np.array(residue.atoms, copy=True)

        #         molpy_frame.residues['id'] = np.array(ids)
        #         molpy_frame.residues['name'] = np.array(names)
        #         molpy_frame.residues['index'] = index
        #         keys = props[0].keys()
        #         for k in keys:
        #             molpy_frame.residues[k] = np.array([p[k] for p in props])

        return frame


class DataLoader(ChflLoader):
    def __init__(self, fpath: str | Path, format: str = "", mode: str = "r"):
        self._fpath = fpath
        self._format = format
        self._mode = mode
        self._fileHandler = chfl.Trajectory(self._fpath, self._mode, self._format)


class MemoryLoader(ChflLoader):
    def __init__(self, data="", format="", mode="r"):
        self._data = data
        self._format = format
        self._mode = mode
        self._fileHandler = chfl.MemoryTrajectory(self._data, self._mode, self._format)

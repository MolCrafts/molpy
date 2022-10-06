# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-03-12
# version: 0.0.1

__all__ = ["DataReader", "DataWriter", "DumpReader"]

from typing import Dict, List

import numpy as np
from molpy.io.base import DataReader, Trajectory
from molpy.io.fileHandler import FileHandler
from molpy.core.frame import StaticFrame, DynamicFrame
from molpy.core.box import Box
from molpy.core.topology import Topology

TYPES = {
    "id": int,
    "molid": int,
    "type": int,
    "q": float,
    "x": float,
    "y": float,
    "z": float,
    "atoms": int,
    "bonds": int,
    "angles": int,
    "dihedrals": int,
    "impropers": int,
    "atom types": int,
    "bond types": int,
    "angle types": int,
    "dihedral types": int,
    "improper types": int,
}

sections = [
    "Atoms",
    "Velocities",
    "Masses",
    "Charges",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Impropers Pair Coeffs",
    "PairIJ Coeffs",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "BondBond Coeffs",
    "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs",
    "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs",
    "AngleAngle Coeffs",
]

header_fields = [
    "atoms",
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "atom types",
    "bond types",
    "angle types",
    "dihedral types",
    "improper types",
    "extra bond per atom",
    "extra angle per atom",
    "extra dihedral per atom",
    "extra improper per atom",
    "extra special per atom",
    "ellipsoids",
    "lines",
    "triangles",
    "bodies",
    "xlo xhi",
    "ylo yhi",
    "zlo zhi",
    "xy xz yz",
]

styles = {
    "full": [
        "id",
        "molid",
        "type",
        "q",
        "x",
        "y",
        "z",
    ],
}

style_dtypes = {
    style: np.dtype([(field, TYPES[field]) for field in fields])
    for style, fields in styles.items()
}

class DumpReader(Trajectory):

    def __init__(self, fpath: str):

        self.filepath = fpath
        self.filehandler = FileHandler(self.filepath)
        self.chunks = self.get_chunks("ITEM: TIMESTEP")
        self.current_nframe: int = 0
        self.current_frame: Dict = None

    @property
    def n_frames(self):
        """total frames of the trajectory"""
        return self.chunks.nchunks

    @property
    def n_frame(self):
        """current frame in the trajectory"""
        return self.current_nframe

    def get_chunks(self, seperator: str):
        chunks = self.filehandler.readchunks(seperator)
        return chunks

    def get_one_frame(self, index:int)->Dict:
        """
        get raw data of a frame

        Args:
            index (int): frame index

        Returns:
            StaticFrame
        """
        self.current_nframe = index

        chunk = self.chunks.getchunk(index)

        self.current_frame = DumpReader.parse(chunk)
        return self.current_frame

    @staticmethod
    def parse(lines: List[str]):
        
        timestep = int(lines[1])
        # n_atoms = int(lines[3])
        box_X = [float(x) for x in lines[5].split()]
        box_Y = [float(y) for y in lines[6].split()]
        box_Z = [float(z) for z in lines[7].split()]
        if len(lines[4]) == 9:
            xlo, xhi, xy = box_X
            ylo, yhi, xz = box_Y
            zlo, zhi, yz = box_Z
        else:
            xlo, xhi = box_Z
            ylo, yhi = box_Y
            zlo, zhi = box_Z
            xy, xz, yz = 0, 0, 0

        box = Box(xhi - xlo, yhi - ylo, zhi - zlo, xy, xz, yz, is2D=False)

        header = lines[8].split()[2:]

        m = list(map(lambda x: tuple(x.split()), lines[9:]))
        atomArr = np.array(
            m,
            dtype={"names": header, "formats": [TYPES.get(k, float) for k in header]},
        )

        frame = StaticFrame(atomArr, box, None, timestep)
        return frame


class DataReader(DataReader):

    def __init__(self, fpath: str, atom_style: str = "full"):

        self.filepath = fpath
        self.atom_style = atom_style

    def __enter__(self):

        self.filehander = FileHandler(self.filepath)

    def get_data(self):

        lines = self.filehander.readlines()
        lines = map(lambda line: DataReader.parse_line(line), lines)
        lines = list(filter(lambda line: line != (), lines))
        frame = DataReader.parse(lines, self.atom_style)
        return frame

    def get_box(self)->Box:
        if self.data is None:
            self.get_data()
        box = self.data["box"]
        return Box(box["xhi"] - box["xlo"], box["yhi"] - box["ylo"], box["zhi"] - box["zlo"], box.get('xy', 0), box.get('xz', 0), box.get('yz', 0), is2D=False)

    @staticmethod
    def parse_line(line: str):

        return tuple(line.partition("#")[0].split())

    @staticmethod
    def parse(lines: List[str], style: str = "full"):

        frame_info = {}

        for i, line in enumerate(lines):

            for field, type in TYPES.items():

                if line[-1] == field:
                    frame_info[field] = type(line[0])
                    break

            if line[-1] == "xhi":
                break

        for line in lines[i : i + 3]:

            if line[-1] == "xhi":
                xlo = float(line[0])
                xhi = float(line[1])

            elif line[-1] == "yhi":
                ylo = float(line[0])
                yhi = float(line[1])

            elif line[-1] == "zhi":
                zlo = float(line[0])
                zhi = float(line[1])

            # elif line[-1] ==

        # TODO: tilt factor
        box = Box(xhi - xlo, yhi - ylo, zhi - zlo, 0, 0, 0, is2D=False)

        section_start_lineno = {}
        # sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
        # header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"
        for lino, line in enumerate(lines):

            if line and line[0] in sections:
                section_start_lineno[line[0]] = lino

        # --- parse atoms ---
        atom_section_starts = section_start_lineno["Atoms"] + 1
        atom_section_ends = atom_section_starts + frame_info["atoms"]

        atomInfo = DataReader.parse_atoms(
            lines[atom_section_starts:atom_section_ends], atom_style=style
        )

        frame = DynamicFrame.from_dict(atomInfo, box, None, 0)
        topo = Topology()
        
        # --- parse bonds ---
        if "Bonds" in section_start_lineno:
            bond_section_starts = section_start_lineno["Bonds"] + 1
            bond_section_ends = bond_section_starts + frame_info["bonds"]

            bondInfo = DataReader.parse_bonds(
                lines[bond_section_starts:bond_section_ends]
            )
            data["Bonds"] = {}
            data["Bonds"]["id"] = bondInfo["id"]
            data["Bonds"]["type"] = bondInfo["type"]
            data["Bonds"]["connect"] = bondInfo[["itom", "jtom"]]

        # # #--- parse angles ---
        # if "Angles" in section_start_lineno:
        #     angles_section_starts = section_start_lineno["Angles"] + 1
        #     angles_section_ends = angles_section_starts + data["angles"]
        #     angleInfo = DataReader.parse_angles(
        #         lines[angles_section_starts:angles_section_ends]
        #     )
        #     data["Angles"] = {}
        #     data["Angles"]["id"] = angleInfo["id"]
        #     data["Angles"]["type"] = angleInfo["type"]
        #     data["Angles"]["connect"] = angleInfo[["itom", "jtom", "ktom"]]

        # # #--- parse dihedrals ---
        # if "Dihedrals" in section_start_lineno:
        #     dihedrals_section_starts = section_start_lineno["Dihedrals"] + 1
        #     dihedrals_section_ends = dihedrals_section_starts + data["dihedrals"]
        #     dihedralInfo = DataReader.parse_dihedrals(
        #         lines[dihedrals_section_starts:dihedrals_section_ends]
        #     )
        #     data["Dihedrals"] = {}
        #     data["Dihedrals"]["id"] = dihedralInfo["id"]
        #     data["Dihedrals"]["type"] = dihedralInfo["type"]
        #     data["Dihedrals"]["connect"] = dihedralInfo[["itom", "jtom", "ktom", "ltom"]]

        # return data
        return frame

    @staticmethod
    def parse_atoms(lines: List[str], atom_style="full"):

        atomInfo = np.array(lines, dtype=style_dtypes[atom_style])

        return atomInfo

    @staticmethod
    def parse_bonds(lines: List[str]):

        return np.array(
            lines, dtype=[("id", int), ("type", int), ("itom", int), ("jtom", int)]
        )

    @staticmethod
    def parse_angles(lines: List[str]):

        return np.array(
            lines,
            dtype=[
                ("id", int),
                ("type", int),
                ("itom", int),
                ("jtom", int),
                ("ktom", int),
            ],
        )

    @staticmethod
    def parse_dihedrals(lines: List[str]):

        return np.array(
            lines,
            dtype=[
                ("id", int),
                ("type", int),
                ("itom", int),
                ("jtom", int),
                ("ktom", int),
                ("ltom", int),
            ],
        )


class DataWriter:
    def __init__(self, fpath: str, atom_style="full"):

        self.filepath = fpath
        self.filehander = FileHandler(fpath, "w")
        self.atom_style = atom_style

    def write(self, system, isBonds=True, isAngles=True, isDihedrals=True):

        write = self.filehander.writeline

        # --- write comment ---
        write("# " + system.comment)
        write("\n\n")

        # --- write profile ---
        write(f"    {system.natoms} atoms\n")
        write(f"    {system.nbonds} bonds\n")
        # write(f'    {system.nangles} angles\n')
        # write(f'    {system.ndihedrals} dihedrals\n')
        # write(f'    {system.nimpropers} impropers\n')
        write(f"    {system.natomTypes} atom types\n")
        write(f"    {system.nbondTypes} bond types\n")
        # write(f'    {system.nangleTypes} angle types\n')
        # write(f'    {system.ndihedralTypes} dihedral types\n')
        # write(f'    {system.nimproperTypes} improper types\n')
        write("\n")

        # --- write box ---
        write(f"    {system.box.xlo} {system.box.xhi} xlo xhi\n")
        write(f"    {system.box.ylo} {system.box.yhi} ylo yhi\n")
        write(f"    {system.box.zlo} {system.box.zhi} zlo zhi\n")
        write("\n")

        # --- write masses section ---
        write("Masses\n\n")
        id = np.arange(system.natomTypes) + 1
        for i, at in enumerate(system.atomTypes):
            write(f"    {id[i]}    {at.mass}  # {at.name}\n")
        write("\n")

        # --- write atoms section ---
        write("Atoms\n\n")
        if "id" not in system.atomManager.atoms:
            id = np.arange(system.natoms) + 1
        else:
            id = system.atomManager.atoms._fields["id"]

        type_map = {}

        for i, at in enumerate(system.atoms):
            if not isinstance(at.type, int):
                type = type_map.get(at.type, len(type_map)) + 1
            write(
                f"    {id[i]}    {at.mol}    {type}    {at.q}    {at.x:.4f}    {at.y:.4f}    {at.z:.4f}\n"
            )
        write("\n")

        # --- write bonds section ---
        if isBonds:
            id = np.arange(system.nbonds) + 1
            write("Bonds\n\n")
            for i, b in enumerate(system.bonds):
                write(f"    {id[i]}    {b.type}    {b[0].id}    {b[1].id}\n")
            write("\n")

        # --- write angle section ---
        if isAngles:
            write("Angles\n\n")
            for a in system.angles:
                write(f"    {a.id}    {a.type}    {a[0]}    {a[1]}    {a[2]}\n")
            write("\n")

        # --- write dihedral section ---
        if isDihedrals:
            write("Dihedrals\n\n")
            for d in system.dihedrals:
                write(
                    f"    {a.id}    {d.type}    {d[0]}    {d[1]}    {d[2]}    {d[3]}\n"
                )
            write("\n")

        self.filehander.close()

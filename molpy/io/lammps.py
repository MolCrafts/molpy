# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-03-12
# version: 0.0.1

__all__ = ["DataReader", "DataWriter", "DumpReader"]

from typing import Dict, List

import numpy as np
from molpy.atoms import Atoms
from molpy.io.base import DataReader, TrajReader
from molpy.io.fileHandler import FileHandler
from molpy.box import Box

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


def data2atoms(data: Dict, out=None):

    if out is None:
        out = Atoms()

    atomData = data["atoms"]
    out.add_atoms(**atomData)

    if "bonds" in data:
        bondData = data["bonds"]
        out.add_bonds(bondData["connect"], id=bondData['id'], type=bondData["type"])

    if "angles" in data:
        angleData = data["angles"]
        out.add_angles(angleData["connect"], id=angleData['id'], type=angleData["type"])
    
    if "dihedrals" in data:
        diheData = data["dihedrals"]
        out.add_dihedrals(diheData["connect"], id=diheData['id'], type=diheData["type"])

    return out


class DumpReader(TrajReader):
    def __init__(self, fpath: str):

        self.filepath = fpath
        self.filehandler = FileHandler(fpath)
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

    def get_frame(self, index:int)->Dict:
        """
        get raw data of a frame

        Args:
            index (int): frame index

        Returns:
            Dict: raw data of a frame
        """
        self.current_nframe = index

        chunk = self.chunks.getchunk(index)

        self.current_frame = DumpReader.parse(chunk)
        return self.current_frame

    def get_atoms(self)->Atoms:

        return data2atoms(self.current_frame)

    def get_box(self)->Box:

        box = self.current_frame["box"]

        # box: {
        #     "Lx": box["xhi"] - box["xlo"],
        #     "Ly": box["yhi"] - box["ylo"],
        #     "Lz": box["zhi"] - box["zlo"],
        #     "xy": box.get("xy", 0),
        #     "xz": box.get("xz", 0),
        #     "yz": box.get("yz", 0),
        #     "is2D": False,
        # }

        return Box(box["xhi"] - box["xlo"], box["yhi"] - box["ylo"], box["zhi"] - box["zlo"], box.get('xy', 0), box.get('xz', 0), box.get('yz', 0), is2D=False)

    @staticmethod
    def parse(lines: List[str]):

        data = {}

        data["timestep"] = int(lines[1])
        data["natoms"] = int(lines[3])
        xlo, xhi = [float(x) for x in lines[5].split()]
        ylo, yhi = [float(x) for x in lines[6].split()]
        zlo, zhi = [float(x) for x in lines[7].split()]
        data["box"] = {
            "xlo": xlo,
            "xhi": xhi,
            "ylo": ylo,
            "yhi": yhi,
            "zlo": zlo,
            "zhi": zhi,
        }

        header = lines[8].split()[2:]

        m = map(lambda x: tuple(x.split()), lines[9:])
        lm = list(m)
        atomArr = np.array(
            lm,
            dtype={"names": header, "formats": [TYPES.get(k, float) for k in header]},
        )

        data["atoms"] = {key: atomArr[key] for key in atomArr.dtype.names}

        return data


class DataReader(DataReader):
    def __init__(self, fpath: str, atom_style: str = "full"):

        self.filepath = fpath
        self.filehander = FileHandler(fpath)
        self.atom_style = atom_style
        self.data: Dict = None

    def get_data(self):
        data = {}
        lines = self.filehander.readlines()
        data["comment"] = lines[0]
        lines = map(lambda line: DataReader.parse_line(line), lines)
        lines = list(filter(lambda line: line != (), lines))
        data.update(DataReader.parse(lines, self.atom_style))
        self.data = data
        return data

    def get_atoms(self)->Atoms:
        if self.data is None:
            self.get_data()
        return data2atoms(self.data)

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

        data = {}

        for i, line in enumerate(lines):

            for field, type in TYPES.items():

                if line[-1] == field:
                    data[field] = type(line[0])
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

        data["box"] = {
            "xhi": xhi,
            "xlo": xlo,
            "yhi": yhi,
            "ylo": ylo,
            "zhi": zhi,
            "zlo": zlo,
            "xy": 0,
            "xz": 0,
            "yz": 0,
            "is2D": False,
        }

        section_start_lineno = {}
        # sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
        # header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"
        for lino, line in enumerate(lines):

            if line and line[0] in sections:
                section_start_lineno[line[0]] = lino

        # --- parse atoms ---
        atom_section_starts = section_start_lineno["Atoms"] + 1
        atom_section_ends = atom_section_starts + data["atoms"]

        atomInfo = DataReader.parse_atoms(
            lines[atom_section_starts:atom_section_ends], atom_style=style
        )

        # TODO: convert unit

        data["atoms"] = {}
        for key in atomInfo.dtype.names:
            data["atoms"][key] = atomInfo[key]

        # --- parse bonds ---
        if "Bonds" in section_start_lineno:
            bond_section_starts = section_start_lineno["Bonds"] + 1
            bond_section_ends = bond_section_starts + data["bonds"]

            bondInfo = DataReader.parse_bonds(
                lines[bond_section_starts:bond_section_ends]
            )
            data["bonds"] = {}
            data["bonds"]["id"] = bondInfo["id"]
            data["bonds"]["type"] = bondInfo["type"]
            data["bonds"]["connect"] = bondInfo[["itom", "jtom"]]

        # #--- parse angles ---
        if "Angles" in section_start_lineno:
            angles_section_starts = section_start_lineno["Angles"] + 1
            angles_section_ends = angles_section_starts + data["angles"]
            angleInfo = DataReader.parse_angles(
                lines[angles_section_starts:angles_section_ends]
            )
            data["angles"] = {}
            data["angles"]["id"] = angleInfo["id"]
            data["angles"]["type"] = angleInfo["type"]
            data["angles"]["connect"] = angleInfo[["itom", "jtom", "ktom"]]

        # #--- parse dihedrals ---
        if "Dihedrals" in section_start_lineno:
            dihedrals_section_starts = section_start_lineno["Dihedrals"] + 1
            dihedrals_section_ends = dihedrals_section_starts + data["dihedrals"]
            dihedralInfo = DataReader.parse_dihedrals(
                lines[dihedrals_section_starts:dihedrals_section_ends]
            )
            data["dihedrals"] = {}
            data["dihedrals"]["id"] = dihedralInfo["id"]
            data["dihedrals"]["type"] = dihedralInfo["type"]
            data["dihedrals"]["connect"] = dihedralInfo[["itom", "jtom", "ktom", "ltom"]]

        return data

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

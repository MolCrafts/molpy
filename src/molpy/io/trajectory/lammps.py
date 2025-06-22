from io import StringIO
from pathlib import Path
from typing import List, Sequence, Union, Optional

import numpy as np
import pandas as pd
import molpy as mp

from .base import TrajectoryReader, TrajectoryWriter


class LammpsTrajectoryReader(TrajectoryReader):
    """Reader for LAMMPS trajectory files, supporting multiple files."""

    def __init__(self, fpaths: Union[str, Path, List[Union[str, Path]]]):
        super().__init__(fpaths)
        self._open_all_files()

    def read_frame(self, index: int) -> mp.Frame:
        if index < 0 or index >= len(self._byte_offsets):
            raise IndexError(f"Frame index {index} out of range ({len(self._byte_offsets)})")
        
        # Check if no frames were found (empty file)
        if len(self._byte_offsets) == 0:
            raise EOFError("No frames found in trajectory file")

        mm, offset = self.get_file_and_offset(index)
        mm.seek(offset)

        file_idx, start_offset = self._byte_offsets[index]

        # 如果有下一个 offset，就用它作为 end
        if (
            index + 1 < len(self._byte_offsets)
            and self._byte_offsets[index + 1][0] == file_idx
        ):
            end_offset = self._byte_offsets[index + 1][1]
        else:
            end_offset = None  # 读到文件尾

        mm = self._fp_list[file_idx]

        if end_offset is None:
            mm.seek(start_offset)
            frame_bytes = mm.read()  # 读到结尾
        else:
            frame_bytes = mm[start_offset:end_offset]  # 一次性读取所需段

        frame_lines = frame_bytes.decode("utf-8").splitlines()

        return self._parse_frame(frame_lines)

    def _parse_trajectories(self):
        self._open_all_files()
        for file_idx, mm in enumerate(self._fp_list):
            if mm is None:  # Empty file
                continue
            mm.seek(0)
            while True:
                pos = mm.tell()
                line = mm.readline()
                if not line:
                    break
                if line.strip().startswith(b"ITEM: TIMESTEP"):
                    self._byte_offsets.append((file_idx, pos))

    def _parse_frame(self, frame_lines: Sequence[str]) -> mp.Frame:
        header = []
        box_bounds = []
        timestep = int(frame_lines[1].strip())
        data_start = 0  # Initialize data_start

        for i, line in enumerate(frame_lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                periodic = line.split()[-3:]
                for j in range(3):
                    box_bounds.append(
                        list(map(float, frame_lines[i + j + 1].strip().split()))
                    )
            elif line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                data_start = i + 1
                break

        # Check if we found atom data
        if not header:
            raise ValueError("No atom data found in trajectory frame")

        df = pd.read_csv(
            StringIO("\n".join(frame_lines[data_start:])),
            sep=r'\s+',
            names=header,
        )
        
        # Convert pandas DataFrame to dictionary format for _dict_to_dataset
        atoms_data = {k: df[k].to_numpy() for k in header}
        
        box_bounds = np.array(box_bounds)

        if box_bounds.shape == (3, 2):
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], 0, 0],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], 0],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        elif box_bounds.shape == (3, 3):
            xy, xz, yz = box_bounds[:, 2]
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], xy, xz],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], yz],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        else:
            raise ValueError(f"Invalid box bounds shape {box_bounds.shape}")

        box = mp.Box(matrix=box_matrix, origin=origin)
        return mp.Frame(
            data={"atoms": atoms_data}, 
            box=box, 
            timestep=timestep
        )


class LammpsTrajectoryWriter(TrajectoryWriter):
    """Writer for LAMMPS trajectory files (dump format)."""

    def __init__(self, fpath: Union[str, Path], atom_style: str = "full"):
        super().__init__(fpath)
        self.atom_style = atom_style
        self._frame_count = 0

    def write_frame(self, frame: mp.Frame, timestep: Optional[int] = None):
        """Write a single frame to the trajectory file.
        
        Args:
            frame: Frame object containing atoms and box information
            timestep: Timestep number (if None, uses auto-increment)
        """
        if timestep is None:
            # Try to get timestep from frame first
            if hasattr(frame, 'timestep') and frame.timestep is not None:
                timestep = frame.timestep
            else:
                timestep = self._frame_count
        
        self._frame_count += 1
        
        # Get atoms data
        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")
        
        atoms = frame["atoms"]
        
        # Get number of atoms
        n_atoms = 0
        if hasattr(atoms, 'sizes'):
            # xarray Dataset - find the main dimension
            sizes = atoms.sizes
            for dim_name in ['index', 'dim_id_0', 'dim_q_0', 'dim_xyz_0']:
                if dim_name in sizes:
                    n_atoms = sizes[dim_name]
                    break
            else:
                if sizes:
                    n_atoms = max(sizes.values())
        else:
            # Dict-like - count entries in first field
            if atoms and isinstance(atoms, dict):
                first_key = next(iter(atoms))
                n_atoms = len(atoms[first_key])
        
        # Write timestep header
        self._fp.write(f"ITEM: TIMESTEP\n{timestep}\n".encode())
        
        # Write number of atoms
        self._fp.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n".encode())
        
        # Write box bounds
        box = getattr(frame, 'box', None)
        if box and hasattr(box, 'matrix'):
            matrix = box.matrix
            origin = getattr(box, 'origin', np.zeros(3))
            
            # For orthogonal box
            if np.allclose(matrix, np.diag(np.diag(matrix))):
                xlo, ylo, zlo = origin
                xhi = xlo + matrix[0, 0]
                yhi = ylo + matrix[1, 1]
                zhi = zlo + matrix[2, 2]
                
                self._fp.write(f"ITEM: BOX BOUNDS pp pp pp\n".encode())
                self._fp.write(f"{xlo:.6f} {xhi:.6f}\n".encode())
                self._fp.write(f"{ylo:.6f} {yhi:.6f}\n".encode())
                self._fp.write(f"{zlo:.6f} {zhi:.6f}\n".encode())
            else:
                # For triclinic box
                xlo, ylo, zlo = origin
                xhi = xlo + matrix[0, 0]
                yhi = ylo + matrix[1, 1]
                zhi = zlo + matrix[2, 2]
                xy = matrix[0, 1]
                xz = matrix[0, 2]
                yz = matrix[1, 2]
                
                self._fp.write(f"ITEM: BOX BOUNDS xy xz yz pp pp pp\n".encode())
                self._fp.write(f"{xlo:.6f} {xhi:.6f} {xy:.6f}\n".encode())
                self._fp.write(f"{ylo:.6f} {yhi:.6f} {xz:.6f}\n".encode())
                self._fp.write(f"{zlo:.6f} {zhi:.6f} {yz:.6f}\n".encode())
        else:
            # Default box if none provided
            self._fp.write(f"ITEM: BOX BOUNDS pp pp pp\n".encode())
            self._fp.write(f"0.000000 10.000000\n".encode())
            self._fp.write(f"0.000000 10.000000\n".encode())
            self._fp.write(f"0.000000 10.000000\n".encode())
        
        # Determine atom columns to write
        if self.atom_style == "full":
            cols = ["id", "mol", "type", "q", "x", "y", "z"]
        else:
            # Default atomic style
            cols = ["id", "type", "x", "y", "z"]
        
        # Map frame fields to dump columns
        field_mapping = {
            "mol": "molid",  # molid in frame -> mol in dump
            "q": "q",        # charge field
            "x": "x", "y": "y", "z": "z"  # coordinates (try individual first)
        }
        
        # Write atom header
        self._fp.write(f"ITEM: ATOMS {' '.join(cols)}\n".encode())
        
        # Prepare atom data
        atom_data = {}
        coords = None
        
        # Handle coordinates - try individual x,y,z first, then xyz array
        if all(coord in atoms for coord in ["x", "y", "z"]):
            atom_data["x"] = atoms["x"].values if hasattr(atoms["x"], 'values') else atoms["x"]
            atom_data["y"] = atoms["y"].values if hasattr(atoms["y"], 'values') else atoms["y"]
            atom_data["z"] = atoms["z"].values if hasattr(atoms["z"], 'values') else atoms["z"]
        elif "xyz" in atoms:
            coords = atoms["xyz"].values if hasattr(atoms["xyz"], 'values') else atoms["xyz"]
            coords = np.asarray(coords)
            if coords.ndim == 2 and coords.shape[1] == 3:
                atom_data["x"] = coords[:, 0]
                atom_data["y"] = coords[:, 1]
                atom_data["z"] = coords[:, 2]
            else:
                raise ValueError("xyz coordinates must be Nx3 array")
        else:
            raise ValueError("No coordinate data found in atoms")
        
        # Handle other fields
        for col in cols:
            if col in ["x", "y", "z"]:
                continue  # Already handled
            
            field_name = field_mapping.get(col, col)
            if field_name in atoms:
                atom_data[col] = atoms[field_name].values if hasattr(atoms[field_name], 'values') else atoms[field_name]
            else:
                # Provide defaults
                if col == "id":
                    atom_data[col] = np.arange(1, n_atoms + 1)
                elif col == "mol":
                    atom_data[col] = np.ones(n_atoms, dtype=int)
                elif col == "type":
                    atom_data[col] = np.ones(n_atoms, dtype=int)
                elif col == "q":
                    atom_data[col] = np.zeros(n_atoms)
        
        # Write atom data
        for i in range(n_atoms):
            line_parts = []
            for col in cols:
                value = atom_data[col][i]
                if col in ["id", "mol"]:
                    line_parts.append(f"{int(value)}")
                elif col == "type":
                    # Handle both string and numeric types
                    if isinstance(value, str):
                        # Create a simple mapping for common atom types
                        type_map = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'P': 6}
                        line_parts.append(f"{type_map.get(value, 1)}")
                    else:
                        line_parts.append(f"{int(value)}")
                else:
                    line_parts.append(f"{float(value):.6f}")
            
            line = " ".join(line_parts) + "\n"
            self._fp.write(line.encode())

"""LAMMPS log file parser.

Parses LAMMPS log files to extract thermodynamic output and simulation metadata.
"""
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

import re
from pathlib import Path

import numpy as np


class LAMMPSLog:
    """
    Parser for LAMMPS log files.
    
    Extracts version info and thermodynamic output from LAMMPS log files.
    
    Args:
        file: Path to LAMMPS log file
        style: Log style (default: "default")
    """
    def __init__(self, file: str | Path, style="default"):
        self.file = Path(file)
        self.info = {
            "n_stages": 0,
            "stages": [],
        }
        self.style = style

    def read(self):
        """Read and parse the log file."""
        with open(self.file) as f:
            log_str = f.read()

        self.read_version(log_str)
        self.read_thermo(log_str, self.style)

    def read_version(self, text: str):
        """Extract LAMMPS version from log text."""
        index = text.find("\n")
        self["version"] = text[:index]

    def read_thermo(self, text, style):
        """Parse thermodynamic output stages from log."""
        if style == "default":
            pattern = r"Per MPI rank .*?\n(.*?Loop time.*?)\n"
            result = re.search(pattern, text, re.DOTALL)
            for stage in result.groups():
                lines = stage.split("\n")
                fields = lines[0].split()

                array = np.loadtxt(
                    lines[1:-1],
                    dtype=np.dtype({"names": fields, "formats": ["f4"] * len(fields)}),
                )

                self["stages"].append(array)
            self["n_stages"] = len(self["stages"])

    def __getitem__(self, key):
        """Get info field by key."""
        return self.info[key]

    def __setitem__(self, key, value):
        """Set info field by key."""
        self.info[key] = value

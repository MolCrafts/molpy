"""LAMMPS log file parser.

Parses LAMMPS log files to extract thermodynamic output and simulation metadata.
"""

# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.2

import re
from pathlib import Path
from typing import Any

import numpy as np

_THERMO_BLOCK = re.compile(
    r"Per MPI rank[^\n]*\n(?P<body>.*?)Loop time",
    re.DOTALL,
)


class LAMMPSLog:
    """
    Parser for LAMMPS log files.

    Extracts version info and thermodynamic output from LAMMPS log files.
    Handles multiple successive ``run`` blocks (each producing its own
    ``Per MPI rank ... Loop time`` section).

    Args:
        file: Path to LAMMPS log file
        style: Log style (default: "default" — matches ``thermo_style custom``)
    """

    def __init__(self, file: str | Path, style: str = "default"):
        self.file = Path(file)
        self.info: dict[str, Any] = {
            "n_stages": 0,
            "stages": [],
        }
        self.style = style

    def read(self) -> "LAMMPSLog":
        """Read and parse the log file. Returns self for chaining."""
        with open(self.file) as fh:
            log_str = fh.read()

        self.read_version(log_str)
        self.read_thermo(log_str, self.style)
        return self

    def read_version(self, text: str) -> None:
        """Extract LAMMPS version from log text (the first line)."""
        index = text.find("\n")
        self["version"] = text[:index] if index >= 0 else text

    def read_thermo(self, text: str, style: str) -> None:
        """Parse thermodynamic output stages from log.

        Captures every ``Per MPI rank ... Loop time`` block. The first line of
        each block holds whitespace-separated column names; subsequent lines
        are numeric data rows.
        """
        if style != "default":
            return

        for match in _THERMO_BLOCK.finditer(text):
            body = match.group("body")
            lines = [line for line in body.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            fields = lines[0].split()
            data_lines = lines[1:]
            try:
                array = np.loadtxt(
                    data_lines,
                    dtype=np.dtype({"names": fields, "formats": ["f8"] * len(fields)}),
                    ndmin=1,
                )
            except ValueError:
                continue
            self["stages"].append(array)
        self["n_stages"] = len(self["stages"])

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization (used by HTTP transports).

        Each stage becomes ``{"columns": [...], "rows": [[...], ...]}``.
        Step (if present) is the first column; downstream consumers may treat
        it as the x-axis.
        """
        stages: list[dict[str, Any]] = []
        for array in self["stages"]:
            columns = list(array.dtype.names) if array.dtype.names else []
            rows = (
                [[float(value) for value in record] for record in array.tolist()]
                if columns
                else []
            )
            stages.append({"columns": columns, "rows": rows})
        return {
            "version": self.info.get("version"),
            "n_stages": self["n_stages"],
            "stages": stages,
        }

    def __getitem__(self, key: str) -> Any:
        """Get info field by key."""
        return self.info[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set info field by key."""
        self.info[key] = value

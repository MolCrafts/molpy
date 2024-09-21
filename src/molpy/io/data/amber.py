from pathlib import Path

import numpy as np
import molpy as mp
import pyarrow as pa

class AmberInpcrdReader:

    def __init__(self, file: str | Path, ):
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line
    
    def read(self, system: mp.System):

        frame = system.frame
        with open(self._file, "r") as f:

            lines = filter(
                lambda line: line,
                map(AmberInpcrdReader.sanitizer, f),
            )

            title = next(lines).strip()

            num_atoms = int(next(lines))

            coordinates = []
            for line in lines:
                values = [float(line[i-12:i].strip()) for i in range(12, len(line), 12)]
                coordinates.extend(values)

            x_coords = coordinates[0::3]
            y_coords = coordinates[1::3]
            z_coords = coordinates[2::3]

            table = pa.table({
                'id': range(1, num_atoms + 1),
                'X': x_coords,
                'Y': y_coords,
                'Z': z_coords
            })

        frame['atoms'] = frame['atoms'].join(table, keys='id')

        frame['props']['name'] = title
        return system
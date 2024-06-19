from pathlib import Path
from abc import ABC
import subprocess

import molpy as mp

class Parameterizer(ABC):

    def __init__(self, name:str, work_dir: str | Path):
        self._name = name
        self.work_dir = Path(work_dir)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"<Parameterizer: {self.name}>"
    
    def parameterzie(self, struct, **kwargs) -> mp.Struct:
        raise NotImplementedError

class AmberTool(Parameterizer):

    def __init__(self, work_dir: str | Path):
        super().__init__("AmberTool", work_dir)
        

    def __repr__(self):
        return f"<AmberTool: {self.name}>"
    
    def parameterize(self, struct, options: dict[str, str] = {}):
        
        name = struct.name
        filename = f"{name}.pdb"
        filepath = self.work_dir / filename
        mp.io.save_struct(filepath, struct)

        args = ' '.join([f'{k} {v}' for k, v in options.items()])

        subprocess.run(f"acpype -i {filename} {args}", shell=True, cwd=self.work_dir, check=True)
        subprocess.run(f'intermol-convert --amb_in {name}_AC.prmtop {name}_AC.inpcrd --lammps', shell=True, cwd=self.work_dir/f"{name}.acpype")

        return self.work_dir/f"{name}.acpype"

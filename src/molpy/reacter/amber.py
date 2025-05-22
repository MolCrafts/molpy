
from pathlib import Path
import shutil
import h_submitor

class AmberToolsReacter:
    """
    A class to handle the Amber force field in molecular simulations.
    """

    def __init__(self, conda_env: str = "AmberTools25"):
        self.conda_env = conda_env

    @h_submitor.local
    def react(self, workdir: Path, struct, init, deletes):
        name = workdir.stem
        workdir.mkdir(parents=True, exist_ok=True)
        if not (workdir/f"{name}.ac").exists():
            shutil.copy(monomer["ac_path"], workdir/f"{name}.ac")
        mainchain = workdir/f"{name}.mc"
        with open(mainchain, "w") as f:
            for link in monomer["links"]:
                this = link.anchor
                delete = link.deletes
                label = link.label
                f.write(f"{label}_NAME {this['name'].item()}\n")
                # f.write(f"PRE_HEAD_TYPE {this['type'].item()}\n")
                # f.write(f"POST_TAIL_TYPE {that['type'].item()}\n")
                for d in delete:
                    f.write(f"OMIT_NAME {d['name'].item()}\n")
        
        yield {
            "job_name": "prepgen",
            "cmd": f"prepgen -i {name}.ac -o {name}.prepi -f prepi -m {name}.mc -rn {name} -rf {name}.res",
            "block": True,
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

        monomer["prepi_path"] = workdir / f"{name}.prepi"
        
        yield {
                "cmd": [f"parmchk2 -i {name}.prepi -f prepi -o {name}.frcmod"],
                "block": True,
                "conda_env": self.conda_env,
                "cwd": workdir,
            }
        
        monomer["frcmod_path"] = workdir / f"{name}.frcmod"

        return monomer
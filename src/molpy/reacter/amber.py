
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
    def react(self, struct, head=None, tail=None, deletes=[], workdir: Path = Path.cwd()):
        name = struct.name
        mainchain = workdir/f"{name}.mc"
        with open(mainchain, "w") as f:
            
            if head:
                f.write(f"HEAD_NAME {head['name'].item()}\n")
            if tail:
                f.write(f"TAIL_NAME {tail['name'].item()}\n")
            # f.write(f"PRE_HEAD_TYPE {this['type'].item()}\n")
            # f.write(f"POST_TAIL_TYPE {that['type'].item()}\n")
            for d in deletes:
                f.write(f"OMIT_NAME {d['name'].item()}\n")
        
        yield {
            "job_name": "prepgen",
            "cmd": f"prepgen -i {name}.ac -o {name}.prepi -f prepi -m {name}.mc -rn {name} -rf {name}.res",
            "block": True,
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

        struct["prepi_path"] = workdir / f"{name}.prepi"
        
        yield {
                "cmd": [f"parmchk2 -i {name}.prepi -f prepi -o {name}.frcmod"],
                "block": True,
                "conda_env": self.conda_env,
                "cwd": workdir,
            }
        
        struct["frcmod_path"] = workdir / f"{name}.frcmod"

        return struct
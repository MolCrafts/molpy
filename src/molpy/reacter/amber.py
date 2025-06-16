from pathlib import Path
import molpy as mp


class AmberToolsReacter:
    """
    A class to handle the Amber force field in molecular simulations.
    """

    def __init__(self, conda_env: str = "AmberTools25"):
        self.conda_env = conda_env

    def react(
        self,
        workdir: Path,
        struct: mp.Struct,
        head: mp.Atom|None=None,
        tail: mp.Atom|None=None,
        omits=[],
    ):
        name = struct["name"]
        mainchain = workdir / f"{name}.mc"
        with open(mainchain, "w") as f:
            if head:
                f.write(f"HEAD_NAME {head['name']}\n")
            if tail:
                f.write(f"TAIL_NAME {tail['name']}\n")
            # f.write(f"PRE_HEAD_TYPE {this['type']}\n")
            # f.write(f"POST_TAIL_TYPE {that['type']}\n")
            for d in omits:
                print(d)
                f.write(f"OMIT_NAME {d['name']}\n")

        yield {
            "job_name": "prepgen",
            "cmd": f"prepgen -i {name}.ac -o {name}.prepi -f prepi -m {name}.mc -rn {name} -rf {name}.res",
            "block": True,
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

        # struct["prepi_path"] = workdir / f"{name}.prepi"

        yield {
            "cmd": [f"parmchk2 -i {name}.prepi -f prepi -o {name}.frcmod"],
            "block": True,
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

        # struct["frcmod_path"] = workdir / f"{name}.frcmod"

        return struct


from pathlib import Path
import shutil
import subprocess
import h_submitor

class AmberToolsReactor:
    """
    A class to handle the Amber force field in molecular simulations.
    """

    def __init__(self, conda_env: str = "AmberTools25"):
        self.conda_env = conda_env

    @h_submitor.local
    def react(self, monomer, workdir: Path):
        name = monomer["name"]
        workdir = workdir/f"{name}"
        workdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(monomer["ac_path"], workdir/f"{name}.ac")
        mainchain = workdir/f"{name}.mc"
        with open(mainchain, "w") as f:
            for port in monomer["ports"]:
                this = port.this
                that = port.that
                f.write(f"TAIL_NAME {that['name'].item()}\n")
                f.write(f"HEAD_NAME {this['name'].item()}\n")
                f.write(f"PRE_HEAD_TYPE {this['type'].item()}\n")
                f.write(f"POST_TAIL_TYPE {that['type'].item()}\n")
                for d in port.delete:
                    f.write(f"OMIT_NAME {d['name'].item()}\n")
        
        yield {
            "job_name": "prepgen",
            "cmd": f"prepgen -i {name}.ac -o {name}.prepi -f prepi -m {name}.mc -rn {name} -rf {name}.res",
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

class AmberToolsPolymerize:

    def __init__(self, conda_env: str = "AmberTools25"):
        self.conda_env = conda_env

    def polymerize(self, polymer, workdir: Path):
        
        seq = [
            monomer["name"]
            for monomer in polymer["atoms"]
        ]

        with open(workdir/f"tleap.in", "w") as f:
            f.write("source leaprc.gaff\n")
            f.write("source leaprc.water.tip3p""\n")
            for mon in set(seq):
                prepi = workdir / f"{mon}.prepi"
                frcmod = workdir / f"{mon}.frcmod"
                f.write(f"loadamberprep {prepi.absolute()}")
                f.write(f"loadamberparams {frcmod.absolute()}")

            f.write(f"chain = sequence {{ {' '.join(seq)} }}")
            f.write(f"savepdb chain {polymer['name']}.pdb\n")
            f.write(f"saveamberparm chain {polymer['name']}.prmtop {polymer['name']}.inpcrd")

        yield {
            "job_name": "tleap",
            "cmd": f"tleap -f tleap.in",
            "conda_env": self.conda_env,
            "cwd": workdir,
        }
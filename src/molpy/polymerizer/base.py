import shutil
import molpy as mp
from pathlib import Path
import h_submitor

class PolymerizerBase:

    def linear(self, seq: str, structs: dict[str, mp.Struct], name=""):

        poly = mp.Polymer(name, )

        for struct in structs.values():
            ports = struct["ports"]
            deletes = [d for p in ports for d in p.delete]
            struct.del_atoms(deletes)

        prev = None
        for mon in (structs[s]() for s in seq):
            poly.add_struct(mon)

            if prev:
                for port in prev["ports"]:
                    this = port.this
                    that = port.that
                    prev.def_bond(
                        prev.atoms[this], mon.atoms[that], 
                    )

            prev = mon
        return poly
    
class Polymerizer(PolymerizerBase): ...
        
class AmberToolsPolymerizer:

    def __init__(self, conda_env: str = "AmberTools25"):
        self.conda_env = conda_env

    @h_submitor.local
    def linear(self, name:str, seq: list[str], structs: dict[str, mp.Struct], workdir: Path):
        
        workdir = Path(workdir) / name
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

        with open(workdir/f"tleap.in", "w") as f:
            f.write("source leaprc.gaff\n")
            f.write("source leaprc.water.tip3p""\n")
            for mon in set(seq):
                prepi = workdir / f"{mon}.prepi"
                frcmod = workdir / f"{mon}.frcmod"
                shutil.copy(structs[mon]["prepi_path"], prepi)
                shutil.copy(structs[mon]["frcmod_path"], frcmod)
                f.write(f"loadamberprep {prepi.absolute()}\n")
                f.write(f"loadamberparams {frcmod.absolute()}\n")

            f.write(f"chain = sequence {{ {' '.join(seq)} }}\n")
            f.write(f"savepdb chain {name}.pdb\n")
            f.write(f"saveamberparm chain {name}.prmtop {name}.inpcrd\n")
            f.write("quit\n")

        yield {
            "job_name": "tleap",
            "cmd": f"tleap -f tleap.in",
            "block": True,
            "conda_env": self.conda_env,
            "cwd": workdir,
        }

        return mp.Polymer.from_frame(mp.io.read_pdb(workdir / f"{name}.pdb"), name=name)
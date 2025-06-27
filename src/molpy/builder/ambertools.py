"""
AmberTools-based polymer builder for molpy.
Uses molq to orchestrate AmberTools workflows for automated polymer construction.
"""

import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Any
from abc import ABC, abstractmethod
import molq

import molpy as mp
from .reacter_lammps import ReactantWrapper
from .polymer import Monomer

logger = logging.getLogger(__name__)


class BuilderStep(ABC):
    """
    Abstract base class for individual AmberTools workflow steps.
    Each step takes a context dictionary and returns an updated context.
    """

    def __init__(self, workdir, conda_env: str):
        self.workdir = Path(workdir)
        self.conda_env = conda_env

    @abstractmethod
    def run(self, context: Dict) -> Dict:
        """
        Execute this step of the workflow.

        Args:
            context: Dictionary containing workflow state and file paths

        Returns:
            Updated context dictionary
        """
        pass


class AntechamberStep(BuilderStep):

    @molq.local
    def run(
        self,
        name,
        monomer: Monomer | ReactantWrapper,
        net_charge: float = 0.0,
        forcefield: str = "gaff",
        charge_type="bcc",
    ) -> Generator[Dict, Any, Path]:
        workdir = Path(self.workdir) / name
        workdir.mkdir(parents=True, exist_ok=True)
        ac_name = f"{name}.ac"
        ac_path = workdir / ac_name

        pdb_name = f"{name}.pdb"
        mp.io.write_pdb(workdir / pdb_name, monomer.to_frame())
        
        if isinstance(monomer, Monomer):


            def get_atom_name(atom_ref) -> str:
                """Convert atom reference to name."""
                return (
                    monomer.atoms[atom_ref]["name"]
                    if isinstance(atom_ref, int)
                    else atom_ref
                )

            with open(workdir / f"{name}.mc", "w") as f:
                # Process head anchor
                if "head" in monomer.anchors:
                    head_anchor = monomer.anchors["head"]
                    head_name = get_atom_name(head_anchor.anchor)
                    f.write(f"HEAD_NAME {head_name}\n")

                    # Write head deletions
                    if hasattr(head_anchor, "deletes"):
                        for delete in head_anchor.deletes:
                            f.write(f"OMIT_NAME {get_atom_name(delete)}\n")

                # Process tail anchor
                if "tail" in monomer.anchors:
                    tail_anchor = monomer.anchors["tail"]
                    # Use anchor.anchor attribute for tail name
                    tail_name = get_atom_name(tail_anchor.anchor)
                    f.write(f"TAIL_NAME {tail_name}\n")

                    # Write tail deletions
                    if hasattr(tail_anchor, "deletes"):
                        for delete in tail_anchor.deletes:
                            f.write(f"OMIT_NAME {get_atom_name(delete)}\n")

        if ac_path.exists():
            return ac_path

        yield {
            "job_name": "antechamber",
            "cmd": f"antechamber -i {pdb_name} -fi pdb -o {ac_name} -fo ac -an y -at {forcefield} -c {charge_type} -nc {net_charge}",
            "conda_env": self.conda_env,
            "cwd": workdir,
            "block": True,
        }

        return ac_path

    typify = run


class PrepgenStep(BuilderStep):

    @molq.local
    def run(
        self, name: str, conda_env: str = "AmberTools25"
    ) -> Generator[Dict, Any, Path]:
        workdir = Path(self.workdir) / name
        cmd = (
            f"prepgen -i {name}.ac -o {name}.prepi -f prepi -rn {name} -rf {name}.res "
        )
        if (workdir / f"{name}.mc").exists():
            cmd += f"-m {name}.mc"
        yield {
            "job_name": "prepgen",
            "cmd": cmd,
            "conda_env": conda_env,
            "cwd": workdir,
            "block": True,
        }
        return workdir / f"{name}.prepi"


class ParmchkStep(BuilderStep):

    @molq.local
    def run(self, name: str) -> Generator[Dict, Any, Path]:
        workdir = Path(self.workdir) / name
        yield {
            "job_name": "parmchk2",
            "cmd": f"parmchk2 -i {name}.ac -f ac -o {name}.frcmod",
            "conda_env": self.conda_env,
            "cwd": workdir,
            "block": True,
        }
        return workdir / f"{name}.frcmod"


class TLeapStep(BuilderStep):

    def __init__(self, workdir, conda_env: str):
        super().__init__(workdir, conda_env)
        self.lines = []

    def source(self, lib: str):
        self.lines.append(f"source {lib}")
        return self
    
    def load_prepi(self, path: Path):
        """
        Load an Amber prepi file into the TLeap environment.
        """
        self.lines.append(f"loadamberprep {path.absolute()}")
        return self
    
    def load_frcmod(self, path: Path):
        """
        Load an Amber frcmod file into the TLeap environment.
        """
        self.lines.append(f"loadamberparams {path.absolute()}")
        return self
    
    def load_ac(self, path: Path):
        """
        Load an Amber AC file into the TLeap environment.
        """
        self.lines.append(f"loadamberac {path.absolute()}")
        return self
    
    def combine(self, name, names: Union[str, List[str]]):
        """
        Combine multiple monomers into a single polymer.
        """
        if isinstance(names, str):
            names = [names]
        joined_names = ' '.join(names)
        self.lines.append(f"{name} = combine {{{joined_names}}}")
        return self

    def define_polymer(self, seq: list[str], var_name="polymer"):
        joined = ' '.join(seq)
        self.lines.append(f"{var_name} = sequence {{ {joined} }}")
        return self

    def add_ions(self, mol_name: str, ion: str, charge=0):
        self.lines.append(f"addIons {mol_name} {ion} {charge}")
        return self

    def save(self, mol_name: str, name: str):
        self.lines.append(f"saveamberparm {mol_name} {name}.prmtop {name}.inpcrd")
        self.lines.append(f"savepdb {mol_name} {name}.pdb")
        return self
    
    def save_amberparm(self, name: str):
        """
        Save the Amber parameters and coordinates for the current polymer.
        """
        self.lines.append(f"saveamberparm {name} {name}.prmtop {name}.inpcrd")
        return self
    
    def save_pdb(self, name: str):
        """
        Save the PDB representation of the current polymer.
        """
        self.lines.append(f"savepdb {name} {name}.pdb")
        return self

    def quit(self):
        self.lines.append("quit")
        return self

    def build(self) -> str:
        return "\n".join(self.lines)
    
    def reset(self):
        self.lines = []        

    @molq.local
    def run(
        self,
        name: str,
    ) -> Generator[Dict, Any, tuple[Path, Path]]:

        with open(self.workdir / name / "tleap.in", "w") as f:
            f.write(self.build())

        yield {
            "job_name": "tleap",
            "cmd": "tleap -f tleap.in",
            "conda_env": self.conda_env,
            "cwd": self.workdir/name,
            "block": True,
        }

        return self.workdir/ name / f"{name}.prmtop", self.workdir/ name / f"{name}.inpcrd"


class AmberToolsBuilder:
    """
    Automated polymer builder using AmberTools workflow via molq.
    """

    def __init__(self, workdir: str|Path, conda_env: str = "AmberTools25"):
        """
        Initialize the AmberTools polymer builder.

        Args:
            steps: List of workflow steps. If None, uses default workflow.
            ambertools_bin: Path to AmberTools binaries. If None, uses system PATH.
            cleanup: Whether to clean up temporary files after completion.
        """

        self.workdir = Path(workdir)
        self.antechamber_step = AntechamberStep(workdir, conda_env)
        self.prepgen_step = PrepgenStep(workdir, conda_env)
        self.parmchk_step = ParmchkStep(workdir, conda_env)
        self.tleap_step = TLeapStep(workdir, conda_env)

    @property
    def typifier(self):
        """
        Get the typifier for this builder.
        """
        return self.antechamber_step

    def build_polymer(
        self,
        name: str,
        monomers: list[mp.AtomicStructure],
        sequence: list[str],
        **kwargs,
    ) -> mp.AtomicStructure:

        workdir = self.workdir / name

        for monomer in monomers:
            m_name = monomer.get("name", None)
            net_charge = monomer.get("net_charge", 0.0)
            self.antechamber_step.run(
                m_name,
                monomer,
                net_charge,
                forcefield="gaff",
                charge_type="bcc",
            )

            self.prepgen_step.run(m_name)

            self.parmchk_step.run(m_name)

        self.tleap_step.source("leaprc.gaff")
        for s in set(sequence):
            base_path = workdir / s / s
            self.tleap_step.load_monomer(s, base_path)

        self.tleap_step.save_amberparm(name)
        self.tleap_step.save_pdb(name)
        self.tleap_step.quit()

        self.tleap_step.run(
            name
        )
        self.tleap_step.reset()
        return mp.AtomicStructure.from_frame(
            mp.io.read_amber(workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd")
        )

    def build_salt(self, name: str, salt: mp.AtomicStructure, ion: str, **kwargs):

        workdir = self.workdir / name

        self.antechamber_step.run(
            name,
            salt,
            net_charge=salt.get("net_charge", 0.0),
            forcefield="gaff",
            charge_type="bcc",
        )
        self.prepgen_step.run(name)
        self.parmchk_step.run(name)
        self.tleap_step.run(name, ["gaff", "water.tip3p"], seq=[name], ion=[ion])

        return mp.AtomicStructure.from_frame(
            mp.io.read_amber(workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd")
        )

    def typify(
        self,
        struct: mp.Struct,
        forcefield: str = "gaff",
        charge_type: str = "bcc",
        net_charge: float = 0.0,
        is_frcmod: bool = False,
        is_prepi: bool = False,
    ):
        name = struct.get("name")
        if name is None:
            raise ValueError("Struct must have a name attribute")
        workdir = Path(self.workdir) / name
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        # pdb_name = f"{name}.pdb"
        ac_name = f"{name}.ac"
        ac_path = workdir / ac_name

        self.antechamber_step.run(
            name,
            struct,
            net_charge=net_charge,
            forcefield=forcefield,
            charge_type=charge_type,
        )
        if is_prepi:
            self.prepgen_step.run(name)
        if is_frcmod:
            self.parmchk_step.run(name)

        frame = mp.io.read_amber_ac(ac_path, frame=mp.Frame())
        atom_types = frame["atoms"]["type"]
        atom_charges = frame["atoms"]["q"]
        for satom, typ, q in zip(struct["atoms"], atom_types, atom_charges):
            satom["type"] = typ.item()
            satom["q"] = q.item()
        bond_types = frame["bonds"]["type"]
        for sbond, typ in zip(struct["bonds"], bond_types):
            sbond["type"] = typ.item()

        return struct

    def parameterize(self, system_name, system: mp.System) -> mp.System:
        """
        Parameterize the system using AmberTools.
        """
        names = set()
        workdir = self.workdir / system_name
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        self.tleap_step.source("leaprc.gaff")
        for struct in system.structs:
            if not struct.get("name"):
                raise ValueError("Struct must have a name attribute")
            struct_name = struct["name"]
            names.add(struct_name)
            self.typify(
                struct=struct,
                forcefield="gaff",
                charge_type="bcc",
                net_charge=0.0,
            )
            self.prepgen_step.run(struct_name)
            self.parmchk_step.run(struct_name)
            self.tleap_step.load_prepi((self.workdir/struct_name/struct_name).with_suffix(".prepi"))
            self.tleap_step.load_frcmod((self.workdir/struct_name/struct_name).with_suffix(".frcmod"))
        self.tleap_step.combine(system_name, list(names))
        self.tleap_step.save_amberparm(system_name)
        self.tleap_step.quit()
        self.tleap_step.run(
            name=system_name
        )
        self.tleap_step.reset()

        frame = mp.io.read_amber(
            workdir / f"{system_name}.prmtop",
            workdir / f"{system_name}.inpcrd",
            frame=mp.Frame(),
        )
        system.set_forcefield(frame.forcefield)

        return system
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
        monomer: Monomer,
        net_charge: float = 0.0,
        forcefield: str = "gaff",
        charge_type="bcc",
    ) -> Generator[Dict, Any, Path]:
        workdir = Path(self.workdir) / name
        workdir.mkdir(parents=True, exist_ok=True)
        pdb_name = f"{name}.pdb"
        ac_name = f"{name}.ac"
        ac_path = workdir / ac_name
        mp.io.write_pdb(workdir / pdb_name, monomer.to_frame())

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

    @molq.local
    def run(
        self,
        name: str,
        params: List[str] = ["gaff"],
        seq: list[str] = [],
        ion: list[str] = [],
    ) -> Generator[Dict, Any, tuple[Path, Path]]:

        workdir = Path(self.workdir) / name
        workdir.mkdir(parents=True, exist_ok=True)
        monomers = set(seq)
        with open(workdir / f"tleap.in", "w") as f:
            for param in params:
                f.write(f"source leaprc.{param}\n")
            for s in seq:
                monomer_path = self.workdir / s / s
                f.write(
                    f"loadamberprep {str(monomer_path.with_suffix('.prepi').absolute())}\n"
                )
                f.write(
                    f"loadamberparams {str(monomer_path.with_suffix('.frcmod').absolute())}\n"
                )
            if len(seq) == 1 and ion:
                f.write(f"addIons {seq[0]} {ion[0]} 0\n")
                f.write(f"saveamberparm {seq[0]} {name}.prmtop {name}.inpcrd\n")
                f.write(f"savepdb {seq[0]} {name}.pdb\n")
            else:
                f.write(f"polymer = sequence {{ {' '.join(seq)} }}\n")
                f.write(f"saveamberparm polymer {name}.prmtop {name}.inpcrd\n")
                f.write(f"savepdb polymer {name}.pdb\n")
            f.write("quit\n")

        yield {
            "job_name": "tleap",
            "cmd": f"tleap -f tleap.in",
            "conda_env": self.conda_env,
            "cwd": workdir,
            "block": True,
        }
        return workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd"


class AmberToolsPolymerBuilder:
    """
    Automated polymer builder using AmberTools workflow via molq.
    """

    def __init__(self, workdir: str, conda_env: str = "AmberTools25"):
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

    def build(
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

        self.tleap_step.run(
            name,
            ["gaff", "water.tip3p"],
            seq=sequence,
        )

        return mp.AtomicStructure.from_frame(
            mp.io.read_amber(workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd")
        )


class AmberToolsSaltBuilder:
    """
    Automated Salt builder using AmberTools workflow via molq.
    """

    def __init__(self, workdir: str, conda_env: str = "AmberTools25"):
        """
        Initialize the AmberTools Salt builder.

        Args:
            steps: List of workflow steps. If None, uses default workflow.
            ambertools_bin: Path to AmberTools binaries. If None, uses system PATH.
            cleanup: Whether to clean up temporary files after completion.
        """

        self.antechamber_step = AntechamberStep(workdir, conda_env)
        self.prepgen_step = PrepgenStep(workdir, conda_env)
        self.parmchk_step = ParmchkStep(workdir, conda_env)
        self.tleap_step = TLeapStep(workdir, conda_env)
        self.workdir = Path(workdir)

    def build(self, name: str, salt: mp.AtomicStructure, ion: str, **kwargs):

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

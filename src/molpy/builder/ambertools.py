"""
AmberTools-based polymer builder for molpy.
Uses molq to orchestrate AmberTools workflows for automated polymer construction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union, Generator, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import molq

if TYPE_CHECKING:
    import molpy as mp
    from molpy.core.system import System
else:
    import molpy as mp

# from molpy.core.frame import _dict_to_dataset
from molpy.core.protocol import Entities
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
        monomer_name,
        monomer: Monomer | ReactantWrapper,
        net_charge: float = 0.0,
        forcefield: str = "gaff",
        charge_type="bcc",
        output_format: str = "ac",
    ) -> Generator[Dict, Any, Path]:
        workdir = Path(self.workdir) / monomer_name
        workdir.mkdir(parents=True, exist_ok=True)
        ac_name = f"{monomer_name}.ac"
        ac_path = workdir / ac_name

        pdb_name = f"{monomer_name}.pdb"
        mp.io.write_pdb(workdir / pdb_name, monomer.to_frame())

        if not ac_path.exists():

            yield {
                "job_name": "antechamber",
                "cmd": f"antechamber -i {pdb_name} -fi pdb -o {ac_name} -fo {output_format} -an y -at {forcefield} -c {charge_type} -nc {net_charge}",
                "conda_env": self.conda_env,
                "cwd": workdir,
                "block": True,
            }

        frame = mp.io.read_amber_ac(ac_path, frame=mp.Frame())
        atom_names = frame["atoms"]["name"]
        atom_types = frame["atoms"]["type"]
        atom_charges = frame["atoms"]["q"]
        for satom, typ, q, name in zip(monomer["atoms"], atom_types, atom_charges, atom_names):
            satom["name"] = name.item()
            satom["type"] = typ.item()
            satom["q"] = q.item()

            
        if isinstance(monomer, Monomer):

            def get_atom_name(atom_ref) -> str:
                """Convert atom reference to name."""
                return (
                    monomer.atoms[atom_ref]["name"]
                    if isinstance(atom_ref, int)
                    else atom_ref
                )

            with open(workdir / f"{monomer_name}.mc", "w") as f:
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
        if not (workdir / f"{name}.prepi").exists():
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
        if not (workdir / f"{name}.frcmod").exists():
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
        joined_names = " ".join(names)
        self.lines.append(f"{name} = combine {{{joined_names}}}")
        return self

    def define_polymer(self, seq: list[str], var_name="polymer"):
        joined = " ".join(seq)
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

        (self.workdir/name).mkdir(parents=True, exist_ok=True)
        with open(self.workdir / name / "tleap.in", "w") as f:
            f.write(self.build())

        yield {
            "job_name": "tleap",
            "cmd": "tleap -f tleap.in",
            "conda_env": self.conda_env,
            "cwd": self.workdir / name,
            "block": True,
        }

        return (
            self.workdir / name / f"{name}.prmtop",
            self.workdir / name / f"{name}.inpcrd",
        )


class AmberToolsBuilder:
    """
    Automated polymer builder using AmberTools workflow via molq.
    """

    def __init__(self, workdir: str | Path, conda_env: str = "AmberTools25"):
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
        monomers: list[Monomer],
        sequence: list[str],
        forcefield: str = "gaff",
        **kwargs,
    ) -> mp.Atomistic:

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
            base_path = self.workdir / s / s
            self.tleap_step.load_prepi(base_path.with_suffix(".prepi"))
            self.tleap_step.load_frcmod(base_path.with_suffix(".frcmod"))
        self.tleap_step.define_polymer(sequence, var_name=name)

        self.tleap_step.save_amberparm(name)
        self.tleap_step.save_pdb(name)
        self.tleap_step.quit()

        self.tleap_step.run(name)
        self.tleap_step.reset()
        return mp.Atomistic.from_frame(
            mp.io.read_amber(workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd")[0]
        )

    def build_salt(self, name: str, salt: mp.Atomistic | Monomer, ion: str, **kwargs):

        workdir = self.workdir / name
        struct_name = salt.get("name", name)
        self.antechamber_step.run(
            struct_name,
            salt,
            net_charge=salt.get("net_charge", 0.0),
            forcefield="gaff",
            charge_type="bcc",
        )
        self.prepgen_step.run(struct_name)
        self.parmchk_step.run(struct_name)
        self.tleap_step.source("leaprc.gaff")
        self.tleap_step.source("leaprc.water.tip3p")
        self.tleap_step.load_prepi(self.workdir / struct_name / f"{struct_name}.prepi")
        self.tleap_step.load_frcmod(self.workdir / struct_name / f"{struct_name}.frcmod")
        self.tleap_step.combine(name, struct_name)
        self.tleap_step.add_ions(name, ion, charge=0)
        self.tleap_step.save_amberparm(name)
        self.tleap_step.save_pdb(name)
        self.tleap_step.quit()
        self.tleap_step.run(name)


        return mp.Atomistic.from_frame(
            mp.io.read_amber(workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd")[0]
        )


class AmberToolsTypifier:
    """
    Automated polymer builder using AmberTools workflow via molq.
    """

    def __init__(self, workdir: str | Path, conda_env: str = "AmberTools25"):
        """
        Initialize the AmberTools polymer builder.

        Args:
            steps: List of workflow steps. If None, uses default workflow.
            ambertools_bin: Path to AmberTools binaries. If None, uses system PATH.
            cleanup: Whether to clean up temporary files after completion.
        """

        self.workdir = Path(workdir)
        self.conda_env = conda_env
        self.antechamber_step = AntechamberStep(workdir, conda_env)
        self.prepgen_step = PrepgenStep(workdir, conda_env)
        self.parmchk_step = ParmchkStep(workdir, conda_env)
        self.tleap_step = TLeapStep(workdir, conda_env)

    def typify(
        self,
        struct: mp.Atomistic,
        forcefield: str = "gaff",
        charge_type: str = "bcc",
        net_charge: float = 0.0,
        is_frcmod: bool = True,
        is_prepi: bool = True,
        is_tleap: bool = True,
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
        logger.info(f"AnteChamber step completed for {name}")
        if is_prepi:
            self.prepgen_step.run(name)
        logger.info(f"Prepgen step completed for {name}")
        if is_frcmod:
            self.parmchk_step.run(name)
        if is_tleap:
            self.tleap_step.source("leaprc.gaff")
            self.tleap_step.load_prepi(workdir / f"{name}.prepi")
            self.tleap_step.load_frcmod(workdir / f"{name}.frcmod")
            self.tleap_step.save_amberparm(name)
            self.tleap_step.save_pdb(name)
            self.tleap_step.quit()
            self.tleap_step.run(name)
            self.tleap_step.reset()
            frame, ff = mp.io.read_amber(
                workdir / f"{name}.prmtop", workdir / f"{name}.inpcrd"
            )
            atoms = []
            atom_ids = frame["atoms"]["id"].tolist()
            atom_names = frame["atoms"]["name"].tolist()
            atom_types = frame["atoms"]["type"].tolist()
            atom_charges = frame["atoms"]["q"].tolist()
            atom_masses = frame["atoms"]["mass"].tolist()
            atom_numbers = frame["atoms"]["atomic_number"].tolist()
            xyz = frame["atoms"]["xyz"].tolist()
            for i in range(len(frame["atoms"]["id"])):
                atoms.append(
                    mp.Atom(
                        id=atom_ids[i],
                        name=atom_names[i],
                        type=atom_types[i],
                        element=mp.Element(atom_numbers[i]).symbol,
                        q=atom_charges[i],
                        mass=atom_masses[i],
                        xyz=xyz[i],
                    )
                )
            struct["atoms"] = Entities(atoms)

            if "bonds" in frame:
                bonds = []
                bond_i = frame["bonds"]["i"].tolist()
                bond_j = frame["bonds"]["j"].tolist()
                bond_types = frame["bonds"]["type"].tolist()
                bond_ids = frame["bonds"]["id"].tolist()
                for i in range(len(bond_ids)):
                    bonds.append(
                        mp.Bond(
                            atoms[bond_i[i]],
                            atoms[bond_j[i]],
                            type=bond_types[i],
                            id=bond_ids[i],
                        )
                    )
                struct["bonds"] = Entities(bonds)

            if "angles" in frame:
                angles = []
                angle_i = frame["angles"]["i"].tolist()
                angle_j = frame["angles"]["j"].tolist()
                angle_k = frame["angles"]["k"].tolist()
                angle_types = frame["angles"]["type"].tolist()
                angle_ids = frame["angles"]["id"].tolist()
                for i in range(len(angle_ids)):
                    angles.append(
                        mp.Angle(
                            atoms[angle_i[i]],
                            atoms[angle_j[i]],
                            atoms[angle_k[i]],
                            type=angle_types[i],
                            id=angle_ids[i],
                        )
                    )
                struct["angles"] = Entities(angles)

            if "dihedrals" in frame:
                dihedrals = []
                dihedral_i = frame["dihedrals"]["i"].tolist()
                dihedral_j = frame["dihedrals"]["j"].tolist()
                dihedral_k = frame["dihedrals"]["k"].tolist()
                dihedral_l = frame["dihedrals"]["l"].tolist()
                dihedral_types = frame["dihedrals"]["type"].tolist()
                dihedral_ids = frame["dihedrals"]["id"].tolist()
                for i in range(len(dihedral_ids)):
                    dihedrals.append(
                        mp.Dihedral(
                            atoms[dihedral_i[i]],
                            atoms[dihedral_j[i]],
                            atoms[dihedral_k[i]],
                            atoms[dihedral_l[i]],
                            type=dihedral_types[i],
                            id=dihedral_ids[i],
                        )
                    )
                struct["dihedrals"] = Entities(dihedrals)

        else:  # if not is_tleap, just read type from ac file
            frame = mp.io.read_amber_ac(ac_path, frame=mp.Frame())
            atom_types = frame["atoms"]["type"]
            atom_charges = frame["atoms"]["q"]
            for satom, typ, q in zip(struct["atoms"], atom_types, atom_charges):
                satom["type"] = typ.item()
                satom["q"] = q.item()

            # Handle bonds - create if not exist
            if "bonds" in frame:
                bond_types = frame["bonds"]["type"]
                if "bonds" not in struct:
                    struct["bonds"] = []
                while len(struct["bonds"]) < len(bond_types):
                    struct["bonds"].append({"type": ""})
                for sbond, typ in zip(struct["bonds"], bond_types):
                    sbond["type"] = typ.item()

        return struct

    def parameterize(self, system_name, system) -> Any:
        """
        Parameterize the system using AmberTools.
        """
        names = set()
        workdir = self.workdir / system_name
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        local_tleap = TLeapStep(self.workdir, self.conda_env)
        local_tleap.source("leaprc.gaff")
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
            local_tleap.load_prepi(
                (self.workdir / struct_name / struct_name).with_suffix(".prepi")
            )
            local_tleap.load_frcmod(
                (self.workdir / struct_name / struct_name).with_suffix(".frcmod")
            )
        local_tleap.combine(system_name, list(names))
        local_tleap.save_amberparm(system_name)
        local_tleap.quit()
        local_tleap.run(name=system_name)
        local_tleap.reset()

        _, ff = mp.io.read_amber(
            workdir / f"{system_name}.prmtop",
            workdir / f"{system_name}.inpcrd",
            frame=mp.Frame(),
        )
        system.set_forcefield(ff)

        return system

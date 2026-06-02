"""
AmberPolymerBuilder: Build polymers using AmberTools backend.

This module provides a polymer builder that uses the AmberTools suite
(antechamber, parmchk2, prepgen, tleap) for construction. The API is
consistent with PolymerBuilder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from molpy.core.atomistic import Atomistic
from molpy.io.readers import read_amber_prmtop
from molpy.parser.smiles import parse_cgsmiles
from molpy.parser.smiles.cgsmiles_ir import CGSmilesGraphIR, CGSmilesIR

from .types import AmberBuildResult


class AmberPolymerBuilder:
    """Build polymers from CGSmiles notation using AmberTools backend.

    This builder parses CGSmiles strings and constructs polymers using
    AmberTools (antechamber, parmchk2, prepgen, tleap).

    Internally, the builder:
    1. Prepares each monomer type (antechamber → parmchk2 → prepgen)
    2. Generates HEAD/CHAIN/TAIL residue variants based on port annotations
    3. Translates CGSmiles to tleap sequence command
    4. Runs tleap to build the polymer
    5. Returns Frame + ForceField from prmtop/inpcrd

    Example:
        >>> builder = AmberPolymerBuilder(
        ...     library={"EO": eo_monomer},
        ...     force_field="gaff2",
        ...     env="AmberTools25",
        ...     env_manager="conda",
        ... )
        >>> result = builder.build("{[#EO]|10}")
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        *,
        force_field: Literal["gaff", "gaff2"] = "gaff2",
        charge_method: str = "bcc",
        work_dir: Path | str | None = None,
        keep_intermediates: bool = True,
        env: str | Path | None = None,
        env_manager: str | None = None,
        reaction_preset: str | None = None,
        net_charges: Mapping[str, int] | None = None,
    ):
        """Initialize the polymer builder.

        Args:
            library: Mapping from CGSmiles labels to Atomistic monomer structures.
                Each Atomistic must have port annotations on atoms (port="<" for
                head, port=">" for tail).
            force_field: Force field to use (gaff or gaff2).
            charge_method: Charge method for antechamber.
            work_dir: Directory for intermediate files.
            keep_intermediates: Whether to keep intermediate files after build.
            env: Conda environment name or path for AmberTools.
            env_manager: Environment manager type ("conda" for conda environments).
            reaction_preset: Named reaction preset for leaving group detection.
            net_charges: Optional mapping from CGSmiles label to the formal net
                charge of that monomer/residue, passed to antechamber's AM1-BCC
                step. Defaults to 0 for any label not present. Required for
                charged residues (e.g. cationic / anionic monomers) so the
                computed partial charges sum to the correct integer.
        """
        self.library = library
        self.force_field = force_field
        self.charge_method = charge_method
        self.net_charges = dict(net_charges or {})
        self.work_dir = (
            Path(work_dir) if work_dir is not None else Path("amber_work")
        ).resolve()
        self.keep_intermediates = keep_intermediates
        self.env = env
        self.env_manager = env_manager
        self.reaction_preset = reaction_preset

        # Internal state
        self._prepared_monomers: dict[str, _PreparedMonomer] = {}

    def build(self, cgsmiles: str) -> AmberBuildResult:
        """Build a polymer from a CGSmiles string.

        Args:
            cgsmiles: CGSmiles notation string (e.g., "{[#EO]|10}")

        Returns:
            AmberBuildResult containing Frame, ForceField, and file paths.

        Raises:
            ValueError: If CGSmiles is invalid or labels not in library.
        """
        # Parse CGSmiles
        ir = parse_cgsmiles(cgsmiles)

        # Validate
        self._validate_ir(ir)

        # Prepare all monomers (antechamber → parmchk2 → prepgen)
        self._prepare_monomers(ir.base_graph)

        # Generate and run tleap
        result = self._build_with_tleap(ir.base_graph, output_prefix="polymer")

        return result

    def _validate_ir(self, ir: CGSmilesIR) -> None:
        """Validate CGSmiles IR."""
        graph = ir.base_graph

        if not graph.nodes:
            raise ValueError("CGSmiles graph is empty")

        # Check all labels exist in library
        missing_labels = set()
        for node in graph.nodes:
            if node.label not in self.library:
                missing_labels.add(node.label)

        if missing_labels:
            available = list(self.library.keys())
            raise ValueError(
                f"Labels {sorted(missing_labels)} not found in library. "
                f"Available labels: {available}"
            )

        # Validate port annotations on each monomer.
        # Interior monomers need both '<' (head) and '>' (tail).
        # End-group / cap monomers need at least one port.
        for label, monomer in self.library.items():
            ports_found = set()
            for atom in monomer.atoms:
                port = atom.get("port")
                if port in ("<", ">"):
                    ports_found.add(port)

            if not ports_found:
                raise ValueError(
                    f"Monomer '{label}' has no port annotations. "
                    "At least one port='<' or port='>' is required."
                )

    def _prepare_monomers(self, graph: CGSmilesGraphIR) -> None:
        """Prepare all monomer types used in the graph.

        For each monomer type:
        1. Write structure to mol2/pdb
        2. Run antechamber for atom typing and charges
        3. Run parmchk2 for missing parameters
        4. Run prepgen to create HEAD/CHAIN/TAIL variants
        """
        from molpy.wrapper.antechamber import AntechamberWrapper
        from molpy.wrapper.prepgen import (
            Parmchk2Wrapper,
            PrepgenWrapper,
            write_prepgen_control_file,
        )

        # Common wrapper kwargs for environment
        wrapper_kwargs: dict = {"workdir": None}  # Will be set per-monomer
        if self.env is not None:
            wrapper_kwargs["env"] = self.env
            wrapper_kwargs["env_manager"] = self.env_manager

        # Get unique labels
        labels = {node.label for node in graph.nodes}

        work_dir = self.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        for label in labels:
            if label in self._prepared_monomers:
                continue  # Already prepared

            monomer = self.library[label]
            net_charge = self.net_charges.get(label, 0)
            monomer_dir = work_dir / "monomers" / label
            monomer_dir.mkdir(parents=True, exist_ok=True)

            # Ensure every atom has a name (PDB / prepgen require it)
            for idx, atom in enumerate(monomer.atoms, start=1):
                if atom.get("name") is None:
                    sym = atom.get("element", "X")
                    atom["name"] = f"{sym}{idx}"

            # Find port atoms
            head_atom_name = None
            tail_atom_name = None
            for atom in monomer.atoms:
                port = atom.get("port")
                if port == "<":
                    head_atom_name = atom["name"]
                elif port == ">":
                    tail_atom_name = atom["name"]

            # Step 1: Write monomer to PDB
            input_pdb = monomer_dir / f"{label}.pdb"
            self._write_atomistic_pdb(monomer, input_pdb)

            # Step 2: Run antechamber
            ac_file = monomer_dir / f"{label}.ac"
            mol2_file = monomer_dir / f"{label}.mol2"

            antechamber = AntechamberWrapper(
                name="antechamber",
                workdir=monomer_dir,
                env=self.env,
                env_manager=self.env_manager,
            )
            r = antechamber.atomtype_assign(
                input_file=input_pdb,
                output_file=mol2_file,
                input_format="pdb",
                output_format="mol2",
                charge_method=self.charge_method,
                atom_type=self.force_field,
                net_charge=net_charge,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"antechamber (mol2) failed for monomer '{label}':\n"
                    f"{r.stderr or r.stdout}"
                )

            # Also generate .ac file for prepgen
            r = antechamber.atomtype_assign(
                input_file=input_pdb,
                output_file=ac_file,
                input_format="pdb",
                output_format="ac",
                charge_method=self.charge_method,
                atom_type=self.force_field,
                net_charge=net_charge,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"antechamber (ac) failed for monomer '{label}':\n"
                    f"{r.stderr or r.stdout}"
                )

            # Step 3: Run parmchk2
            frcmod_file = monomer_dir / f"{label}.frcmod"
            parmchk2 = Parmchk2Wrapper(
                name="parmchk2",
                workdir=monomer_dir,
                env=self.env,
                env_manager=self.env_manager,
            )
            r = parmchk2.generate_parameters(
                input_file=mol2_file,
                output_file=frcmod_file,
                force_field=self.force_field,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"parmchk2 failed for monomer '{label}':\n{r.stderr or r.stdout}"
                )

            # Step 4: Generate prepgen control files and run prepgen.
            # Which variants to generate depends on the ports present:
            #   both < and > : full monomer → HEAD, CHAIN, TAIL
            #   only <       : left cap    → HEAD only (no TAIL connection)
            #   only >       : right cap   → TAIL only (no HEAD connection)
            is_full = head_atom_name is not None and tail_atom_name is not None
            is_left_cap = head_atom_name is not None and tail_atom_name is None
            is_right_cap = head_atom_name is None and tail_atom_name is not None

            prepgen = PrepgenWrapper(
                name="prepgen",
                workdir=monomer_dir,
                env=self.env,
                env_manager=self.env_manager,
            )

            head_prepi = None
            chain_prepi = None
            tail_prepi = None

            # prepgen crashes on long absolute paths (Fortran buffer).
            # Use short filenames — the wrapper runs with cwd=monomer_dir.
            ac_name = f"{label}.ac"

            # Omit rules: each connection point loses 1 H.
            #   HEAD variant (chain start): TAIL connects → omit 1 H from TAIL atom
            #   CHAIN variant (middle):     both connect  → omit 1 H from each
            #   TAIL variant (chain end):   HEAD connects → omit 1 H from HEAD atom
            omit_head = self._get_one_omit_name(monomer, "<")  # 1 H on < port
            omit_tail = self._get_one_omit_name(monomer, ">")  # 1 H on > port

            def _run_prepgen(
                variant: str, output_file: str, control_file: str, resname: str
            ) -> None:
                r = prepgen.generate_residue(
                    input_file=ac_name,
                    output_file=output_file,
                    control_file=control_file,
                    residue_name=resname,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"prepgen ({variant}) failed for monomer '{label}':\n"
                        f"{r.stderr or r.stdout}"
                    )

            if is_full or is_left_cap:
                head_tail_name = tail_atom_name or head_atom_name
                head_ctrl = monomer_dir / f"{label}.head"
                head_prepi = monomer_dir / f"H{label}.prepi"
                write_prepgen_control_file(
                    head_ctrl,
                    variant="head",
                    head_name=None,
                    tail_name=head_tail_name,
                    omit_names=omit_tail if is_full else omit_head,
                    charge=net_charge,
                )
                _run_prepgen(
                    "head", f"H{label}.prepi", f"{label}.head", f"H{label[:2].upper()}"
                )

            if is_full:
                chain_ctrl = monomer_dir / f"{label}.chain"
                chain_prepi = monomer_dir / f"{label}.prepi"
                write_prepgen_control_file(
                    chain_ctrl,
                    variant="chain",
                    head_name=head_atom_name,
                    tail_name=tail_atom_name,
                    omit_names=omit_head + omit_tail,
                    charge=net_charge,
                )
                _run_prepgen(
                    "chain", f"{label}.prepi", f"{label}.chain", label[:3].upper()
                )

            if is_full or is_right_cap:
                tail_head_name = head_atom_name or tail_atom_name
                tail_ctrl = monomer_dir / f"{label}.tail"
                tail_prepi = monomer_dir / f"T{label}.prepi"
                write_prepgen_control_file(
                    tail_ctrl,
                    variant="tail",
                    head_name=tail_head_name,
                    tail_name=None,
                    omit_names=omit_head if is_full else omit_tail,
                    charge=net_charge,
                )
                _run_prepgen(
                    "tail", f"T{label}.prepi", f"{label}.tail", f"T{label[:2].upper()}"
                )

            # Store prepared monomer info
            self._prepared_monomers[label] = _PreparedMonomer(
                label=label,
                frcmod_file=frcmod_file,
                head_prepi=head_prepi,
                chain_prepi=chain_prepi,
                tail_prepi=tail_prepi,
                head_resname=f"H{label[:2].upper()}" if head_prepi else None,
                chain_resname=label[:3].upper() if chain_prepi else None,
                tail_resname=f"T{label[:2].upper()}" if tail_prepi else None,
            )

    def _get_omit_names(self, monomer: Atomistic, port: str) -> list[str]:
        """Return names of ALL hydrogens bonded to the port atom."""
        port_atom = None
        for atom in monomer.atoms:
            if atom.get("port") == port:
                port_atom = atom
                break
        if port_atom is None:
            return []

        names = []
        for bond in monomer.bonds:
            other = None
            if bond.itom is port_atom:
                other = bond.jtom
            elif bond.jtom is port_atom:
                other = bond.itom
            if other is not None and (
                other.get("element") == "H" or other.get("symbol") == "H"
            ):
                names.append(other["name"])
        return names

    def _get_one_omit_name(self, monomer: Atomistic, port: str) -> list[str]:
        """Return the name of ONE hydrogen bonded to the port atom.

        When two residues connect, each side loses one H to free a
        valence for the new bond.  Returning exactly one name is
        correct for standard polymer linkages.
        """
        port_atom = None
        for atom in monomer.atoms:
            if atom.get("port") == port:
                port_atom = atom
                break
        if port_atom is None:
            return []

        for bond in monomer.bonds:
            other = None
            if bond.itom is port_atom:
                other = bond.jtom
            elif bond.jtom is port_atom:
                other = bond.itom
            if other is not None and other.get("element") == "H":
                return [other["name"]]
        return []

    def _write_atomistic_pdb(self, atomistic: Atomistic, path: Path) -> None:
        """Write Atomistic to PDB format."""
        from molpy.io.writers import write_pdb

        frame = atomistic.to_frame()
        write_pdb(path, frame)

    def _build_with_tleap(
        self,
        graph: CGSmilesGraphIR,
        output_prefix: str,
    ) -> AmberBuildResult:
        """Generate tleap script and run tleap to build polymer.

        Uses a two-pass strategy so that *inter-residue* bonded parameters
        (angles / torsions that span a residue junction, e.g. a cap's exotic
        atom type bonded to a backbone carbon) are filled in. parmchk2 run on
        each isolated monomer cannot see across junctions; GAFF base covers the
        plain c3-c3 backbone link but not unusual end-group chemistries. So we
        first assemble the unit and dump it to mol2, run parmchk2 on the whole
        molecule to synthesise any missing junction parameters, then build the
        final topology with that extra frcmod loaded.
        """
        from molpy.wrapper.tleap import TLeapWrapper
        from molpy.wrapper.prepgen import Parmchk2Wrapper

        work_dir = self.work_dir
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        leaprc = f"leaprc.{self.force_field}"

        # Reusable "load every residue's params + templates" block.
        load_lines = [f"source {leaprc}", ""]
        used_labels = {node.label for node in graph.nodes}
        for label in sorted(used_labels):
            prep = self._prepared_monomers[label]
            load_lines.append(f"loadamberparams {prep.frcmod_file}")
            if prep.head_prepi is not None:
                load_lines.append(f"loadamberprep {prep.head_prepi}")
            if prep.chain_prepi is not None:
                load_lines.append(f"loadamberprep {prep.chain_prepi}")
            if prep.tail_prepi is not None:
                load_lines.append(f"loadamberprep {prep.tail_prepi}")
        load_lines.append("")

        sequence = self._build_sequence(graph)
        seq_line = f"mol = sequence {{{sequence}}}"

        prmtop = output_dir / f"{output_prefix}.prmtop"
        inpcrd = output_dir / f"{output_prefix}.inpcrd"
        pdb = output_dir / f"{output_prefix}.pdb"

        tleap = TLeapWrapper(
            name="tleap", workdir=work_dir, env=self.env, env_manager=self.env_manager
        )

        # --- Pass 1: assemble the unit and dump a typed mol2 (no params needed) ---
        full_mol2 = work_dir / f"{output_prefix}_full.mol2"
        pass1 = "\n".join(
            load_lines + [seq_line, "", f"savemol2 mol {full_mol2} 1", "quit"]
        )
        (work_dir / f"{output_prefix}_pass1.in").write_text(pass1)
        r = tleap.run_from_script(pass1)
        interres_frcmod = None
        if full_mol2.exists():
            # --- Inter-residue parmchk2 pass over the whole assembled unit ---
            interres_frcmod = work_dir / f"{output_prefix}_interres.frcmod"
            parmchk2 = Parmchk2Wrapper(
                name="parmchk2",
                workdir=work_dir,
                env=self.env,
                env_manager=self.env_manager,
            )
            cr = parmchk2.generate_parameters(
                input_file=full_mol2,
                output_file=interres_frcmod,
                force_field=self.force_field,
            )
            if cr.returncode != 0 or not interres_frcmod.exists():
                interres_frcmod = None

        # --- Pass 2: build final topology, loading the inter-residue frcmod ---
        final_loads = list(load_lines)
        if interres_frcmod is not None:
            final_loads.insert(-1, f"loadamberparams {interres_frcmod}")
        script_lines = final_loads + [
            seq_line,
            "",
            f"savepdb mol {pdb}",
            f"saveamberparm mol {prmtop} {inpcrd}",
            "quit",
        ]
        script = "\n".join(script_lines)
        script_path = work_dir / f"{output_prefix}.in"
        script_path.write_text(script)

        r = tleap.run_from_script(script)
        if r.returncode != 0 or not prmtop.exists():
            raise RuntimeError(
                f"tleap failed to build polymer (prmtop not created).\n"
                f"Script: {script_path}\n"
                f"Log: {work_dir / 'leap.log'}\n"
                f"{r.stderr or r.stdout}"
            )

        # Load results
        frame, forcefield = read_amber_prmtop(prmtop, inpcrd)

        return AmberBuildResult(
            frame=frame,
            forcefield=forcefield,
            prmtop_path=prmtop,
            inpcrd_path=inpcrd,
            pdb_path=pdb if pdb.exists() else None,
            monomer_count=len(graph.nodes),
            cgsmiles=None,
        )

    def _build_sequence(self, graph: CGSmilesGraphIR) -> str:
        """Build tleap sequence from CGSmiles graph.

        Variant assignment:
        - Cap with only ``<`` → always uses HEAD variant
        - Cap with only ``>`` → always uses TAIL variant
        - Full monomer at first position → HEAD variant
        - Full monomer at last position → TAIL variant
        - Full monomer in middle → CHAIN variant
        """
        nodes = graph.nodes
        n = len(nodes)

        if n == 0:
            raise ValueError("Empty graph")

        residue_names = []
        for i, node in enumerate(nodes):
            prep = self._prepared_monomers[node.label]

            # Cap monomers (only one variant available)
            if prep.chain_prepi is None:
                # This is a cap — use whichever variant was generated
                if prep.head_resname is not None:
                    residue_names.append(prep.head_resname)
                elif prep.tail_resname is not None:
                    residue_names.append(prep.tail_resname)
                else:
                    raise ValueError(
                        f"Cap monomer '{node.label}' has no residue variant"
                    )
                continue

            # Full monomers — position-based variant
            if n == 1:
                residue_names.append(prep.chain_resname)
            elif i == 0:
                residue_names.append(prep.head_resname)
            elif i == n - 1:
                residue_names.append(prep.tail_resname)
            else:
                residue_names.append(prep.chain_resname)

        return " ".join(residue_names)


@dataclass
class _PreparedMonomer:
    """Internal state for a prepared monomer."""

    label: str
    frcmod_file: Path
    head_prepi: Path | None
    chain_prepi: Path | None
    tail_prepi: Path | None
    head_resname: str | None
    chain_resname: str | None
    tail_resname: str | None

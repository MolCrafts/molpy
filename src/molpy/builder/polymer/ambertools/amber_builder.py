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

import molrs

from molpy.builder.assembly import Finalization, PolymerBuilder
from molpy.builder.assembly._topology import TopologySelector
from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.io.readers import read_amber
from molpy.parser.smiles import parse_cgsmiles
from molpy.parser.smiles.cgsmiles_ir import CGSmilesGraphIR, CGSmilesIR

from .types import AmberBuildResult

type _VariantRecipe = tuple[str | None, str | None, tuple[str, ...]]
type _ResidueRecipes = dict[str, dict[str, _VariantRecipe]]


class AmberPolymerBuilder:
    """Build polymers from CGSmiles notation using the AmberTools backend.

    ========================= DESIGN CONTRACT — READ FIRST =========================
    This is the whole reason the class exists; do NOT "optimise" or "fix" around it.

    * Parameterise each MONOMER individually, ONCE. Every unique repeat unit / cap
      is run through antechamber (GAFF atom types + AM1-BCC charges) and prepgen
      (HEAD/CHAIN/TAIL residue templates) on its own small structure. Results are
      cached on disk under ``work_dir/monomers/<label>/`` and reused across every
      chain and every run — AM1-BCC is the expensive step and must never be
      recomputed for a monomer already prepared.

    * Assemble the CHAIN with tleap ``sequence``. tleap stitches the per-monomer
      residue templates head-to-tail; the inter-residue (junction) parameters come
      from GAFF base (``source leaprc.gaff2``), resolved at assembly time.

    * CUT residues on backbone c3-c3 bonds ONLY. Never leave an exotic atom at a
      residue junction — fold such a group INTO an adjacent residue instead. E.g.
      a RAFT dithioester's bridging sulfur must be folded into the first monomer
      so it is an INTERNAL thioether (typed ``ss``); left as a separate cap
      connecting through its S it becomes a junction ``sh`` thiol and needs a
      ``cs-sh-c3`` term GAFF base lacks. With correct segmentation every junction
      is GAFF-covered and per-monomer + tleap alone is sufficient — no extra pass.

    * parmchk2 is a per-MONOMER step (it fills each monomer's missing GAFF
      parameters) and is always run there. NEVER run antechamber or parmchk2 on
      the assembled chain (or any multi-residue fragment) — it is both wrong
      (junction parameters belong to GAFF base) and, on a long flexible chain,
      pathologically slow (100% CPU for tens of minutes). If tleap reports a
      missing junction parameter, the fix is correct segmentation (previous
      bullet), never a parmchk2 pass over the assembly.
    ================================================================================

    Pipeline:
    1. Prepare each monomer type (antechamber → parmchk2 → prepgen), disk-cached.
    2. Compile the MolPy ``fields.SITE`` + ``Reaction`` semantics once and
       translate the resulting residue edits into HEAD/CHAIN/TAIL variants.
    3. Translate CGSmiles to a tleap ``sequence`` command.
    4. Run tleap to build the polymer; read back Frame + ForceField.

    Example:
        >>> builder = AmberPolymerBuilder(
        ...     library={"EO": eo_monomer},
        ...     reaction=ether_reaction,
        ...     force_field="gaff2",
        ...     env="AmberTools25",
        ...     env_manager="conda",
        ... )
        >>> result = builder.build("{[#EO]|10}")
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        reaction: molrs.Reaction,
        *,
        force_field: Literal["gaff", "gaff2"] = "gaff2",
        charge_method: str = "bcc",
        work_dir: Path | str | None = None,
        env: str | Path | None = None,
        env_manager: str | None = None,
        net_charges: Mapping[str, int] | None = None,
    ):
        """Initialize the polymer builder.

        Args:
            library: Mapping from CGSmiles labels to Atomistic monomer structures.
                Reaction sites use the same ``fields.SITE`` annotations as
                :class:`molpy.builder.assembly.PolymerBuilder`.
            reaction: The MolPy reaction that defines connection atoms and
                leaving groups. AmberTools translates its compiled products; it
                does not define a second port/leaving-group language.
            force_field: Force field to use (gaff or gaff2).
            charge_method: Charge method for antechamber.
            work_dir: Directory for intermediate files.
            env: Conda environment name or path for AmberTools.
            env_manager: Environment manager type ("conda" for conda environments).
            net_charges: Optional mapping from CGSmiles label to the formal net
                charge of that monomer/residue, passed to antechamber's AM1-BCC
                step. Defaults to 0 for any label not present. Required for
                charged residues (e.g. cationic / anionic monomers) so the
                computed partial charges sum to the correct integer.
        """
        if not isinstance(reaction, molrs.Reaction):
            raise TypeError("reaction must be a molpy.Reaction instance")
        self.reaction = reaction
        self.library: dict[str, Atomistic] = {}
        for label, template in library.items():
            if not isinstance(template, Atomistic):
                raise TypeError(f"monomer {label!r} must be an Atomistic")
            copied = template.copy()
            names: set[str] = set()
            for index, atom in enumerate(copied.atoms, start=1):
                if atom.get(fields.NAME) is None:
                    element = atom.get(fields.ELEMENT)
                    if not element:
                        raise ValueError(
                            f"monomer {label!r} atom {index} has no element and "
                            "cannot receive a stable Amber atom name"
                        )
                    atom[fields.NAME] = f"{element}{index}"
                name = str(atom.get(fields.NAME))
                if name in names:
                    raise ValueError(
                        f"monomer {label!r} repeats Amber atom name {name!r}"
                    )
                names.add(name)
            self.library[label] = copied
        self.force_field = force_field
        self.charge_method = charge_method
        self.net_charges = dict(net_charges or {})
        self.work_dir = (
            Path(work_dir) if work_dir is not None else Path("amber_work")
        ).resolve()
        self.env = env
        self.env_manager = env_manager

        # Internal state
        self._prepared_monomers: dict[str, _PreparedMonomer] = {}
        self._semantic_cache: dict[str, _ResidueRecipes] = {}

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

        recipes = self._compile_semantics(cgsmiles, ir.base_graph)

        # Prepare all monomers (antechamber → parmchk2 → prepgen)
        self._prepare_monomers(ir.base_graph, recipes)

        # Generate and run tleap
        result = self._build_with_tleap(ir.base_graph, output_prefix="polymer")

        result.cgsmiles = cgsmiles
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

        if len(graph.nodes) < 2:
            raise ValueError(
                "AmberPolymerBuilder requires at least two residues; use "
                "AmberTools.parameterize() for one molecule"
            )
        residue_of = TopologySelector.residue_ids(graph)
        edges = {
            frozenset((residue_of[bond.node_i.id], residue_of[bond.node_j.id]))
            for bond in graph.bonds
        }
        expected = {frozenset((i, i + 1)) for i in range(1, len(graph.nodes))}
        if len(graph.bonds) != len(graph.nodes) - 1 or edges != expected:
            raise ValueError(
                "Amber tleap sequence supports one linear polymer path; "
                "build rings, stars, and networks with PolymerBuilder"
            )

    def _compile_semantics(
        self,
        cgsmiles: str,
        graph: CGSmilesGraphIR,
    ) -> _ResidueRecipes:
        """Translate the standard MolPy reaction product into Amber variants."""
        cached = self._semantic_cache.get(cgsmiles)
        if cached is not None:
            return cached

        product = PolymerBuilder(
            self.library,
            self.reaction,
            finalize=Finalization.ATOMS,
        ).build(cgsmiles)
        atoms_by_residue: dict[int, list] = {}
        for atom in product.atoms:
            residue = atom.get(fields.RES_ID)
            if residue is None:
                raise ValueError(
                    "Amber translation supports bond-forming/deletion reactions, "
                    "not reaction-created atoms without residue identity"
                )
            atoms_by_residue.setdefault(int(residue), []).append(atom)

        connections: dict[int, dict[int, str]] = {}
        for bond in product.bonds:
            left, right = bond.endpoints
            left_residue = int(left.get(fields.RES_ID))
            right_residue = int(right.get(fields.RES_ID))
            if left_residue == right_residue:
                continue
            left_name = left.get(fields.NAME)
            right_name = right.get(fields.NAME)
            if left_name is None or right_name is None:
                raise ValueError("Amber connection atoms require stable atom names")
            left_connections = connections.setdefault(left_residue, {})
            right_connections = connections.setdefault(right_residue, {})
            if right_residue in left_connections or left_residue in right_connections:
                raise ValueError(
                    "Amber tleap sequence supports exactly one bond between "
                    "adjacent residues"
                )
            left_connections[right_residue] = str(left_name)
            right_connections[left_residue] = str(right_name)

        recipes: _ResidueRecipes = {}
        residue_of = TopologySelector.residue_ids(graph)
        count = len(graph.nodes)
        for position, node in enumerate(graph.nodes, start=1):
            residue = residue_of[node.id]
            template_names = {
                str(atom.get(fields.NAME)) for atom in self.library[node.label].atoms
            }
            surviving_names = {
                str(atom.get(fields.NAME)) for atom in atoms_by_residue.get(residue, ())
            }
            created_names = surviving_names - template_names
            if created_names:
                raise ValueError(
                    "Amber translation cannot place reaction-created atoms into a "
                    f"residue: {sorted(created_names)}"
                )
            omitted = tuple(sorted(template_names - surviving_names))
            incoming = connections.get(residue, {}).get(residue - 1)
            outgoing = connections.get(residue, {}).get(residue + 1)
            if position == 1:
                variant = "head"
            elif position == count:
                variant = "tail"
            else:
                variant = "chain"
            recipe = (incoming, outgoing, omitted)
            previous = recipes.setdefault(node.label, {}).get(variant)
            if previous is not None and previous != recipe:
                raise ValueError(
                    f"monomer {node.label!r} needs incompatible {variant} Amber "
                    "variants in one chain"
                )
            recipes[node.label][variant] = recipe

        for label, variants in recipes.items():
            for variant, (head_name, tail_name, _) in variants.items():
                if variant in {"chain", "tail"} and head_name is None:
                    raise ValueError(f"{label!r} {variant} variant has no HEAD atom")
                if variant in {"head", "chain"} and tail_name is None:
                    raise ValueError(f"{label!r} {variant} variant has no TAIL atom")
        self._semantic_cache[cgsmiles] = recipes
        return recipes

    def _prepare_monomers(
        self,
        graph: CGSmilesGraphIR,
        recipes: _ResidueRecipes,
    ) -> None:
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

        # Get unique labels
        labels = {node.label for node in graph.nodes}

        work_dir = self.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        for label in labels:
            monomer = self.library[label]
            net_charge = self.net_charges.get(label, 0)
            monomer_dir = work_dir / "monomers" / label
            monomer_dir.mkdir(parents=True, exist_ok=True)
            variants = recipes[label]

            # Disk cache: a monomer's antechamber (.ac/charges) + prepgen residue
            # templates depend only on the monomer, so reuse them across chains
            # and runs. AM1-BCC is the slow step — never repeat it for a monomer
            # already prepared on disk.
            frcmod_file = monomer_dir / f"{label}.frcmod"
            head_prepi = monomer_dir / f"H{label}.prepi" if "head" in variants else None
            chain_prepi = (
                monomer_dir / f"{label}.prepi" if "chain" in variants else None
            )
            tail_prepi = monomer_dir / f"T{label}.prepi" if "tail" in variants else None
            cached = [
                p for p in (frcmod_file, head_prepi, chain_prepi, tail_prepi) if p
            ]
            if cached and all(p.exists() for p in cached):
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
                continue

            input_pdb = monomer_dir / f"{label}.pdb"
            ac_file = monomer_dir / f"{label}.ac"
            mol2_file = monomer_dir / f"{label}.mol2"
            base_cached = all(
                path.exists() for path in (ac_file, mol2_file, frcmod_file)
            )
            if not base_cached:
                # Steps 1–3 are monomer-only and expensive. A later build may
                # need an extra HEAD/CHAIN/TAIL recipe, but it must reuse these
                # antechamber charges and parmchk2 parameters.
                self._write_atomistic_pdb(monomer, input_pdb)
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
                        f"parmchk2 failed for monomer '{label}':\n"
                        f"{r.stderr or r.stdout}"
                    )

            # Step 4: Generate exactly the variants compiled from MolPy's
            # reaction product. No Amber-only port or leaving-H inference lives
            # below this boundary.
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

            if "head" in variants:
                _, tail_name, omit_names = variants["head"]
                assert tail_name is not None
                head_ctrl = monomer_dir / f"{label}.head"
                head_prepi = monomer_dir / f"H{label}.prepi"
                # A HEAD residue connects forward through ``tail_name``. Pin a
                # heavy neighbour as prepgen's tree root so a terminal connection
                # atom remains the main-chain tail rather than becoming the root.
                head_pin = self._connect_pin_neighbor(monomer, tail_name)
                write_prepgen_control_file(
                    head_ctrl,
                    variant="head",
                    head_name=head_pin,
                    tail_name=tail_name,
                    omit_names=list(omit_names),
                    charge=net_charge,
                )
                _run_prepgen(
                    "head", f"H{label}.prepi", f"{label}.head", f"H{label[:2].upper()}"
                )

            if "chain" in variants:
                head_name, tail_name, omit_names = variants["chain"]
                assert head_name is not None and tail_name is not None
                chain_ctrl = monomer_dir / f"{label}.chain"
                chain_prepi = monomer_dir / f"{label}.prepi"
                write_prepgen_control_file(
                    chain_ctrl,
                    variant="chain",
                    head_name=head_name,
                    tail_name=tail_name,
                    omit_names=list(omit_names),
                    charge=net_charge,
                )
                _run_prepgen(
                    "chain", f"{label}.prepi", f"{label}.chain", label[:3].upper()
                )

            if "tail" in variants:
                head_name, _, omit_names = variants["tail"]
                assert head_name is not None
                tail_ctrl = monomer_dir / f"{label}.tail"
                tail_prepi = monomer_dir / f"T{label}.prepi"
                # Symmetric pinning for a right cap whose connect atom is a
                # graph terminus (see the head-cap note above).
                tail_pin = self._terminus_neighbor_name(monomer, head_name)
                write_prepgen_control_file(
                    tail_ctrl,
                    variant="tail",
                    head_name=head_name,
                    tail_name=tail_pin,
                    omit_names=list(omit_names),
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

    def _terminus_neighbor_name(
        self, monomer: Atomistic, atom_name: str | None
    ) -> str | None:
        """Name of the single heavy neighbour of ``atom_name`` if it is a terminus.

        Returns ``None`` when the atom has zero or more than one heavy neighbour
        (i.e. it is not a graph terminus and needs no main-chain pinning).
        """
        if atom_name is None:
            return None
        atom = next((a for a in monomer.atoms if a.get("name") == atom_name), None)
        if atom is None:
            return None
        heavy: list[str] = []
        for bond in monomer.bonds:
            other = None
            if bond.itom is atom:
                other = bond.jtom
            elif bond.jtom is atom:
                other = bond.itom
            if other is not None and other.get("element") != "H":
                heavy.append(other["name"])
        return heavy[0] if len(heavy) == 1 else None

    def _connect_pin_neighbor(
        self, monomer: Atomistic, atom_name: str | None
    ) -> str | None:
        """A heavy neighbour of ``atom_name`` to pin as the HEAD residue's HEAD_NAME.

        Unlike :meth:`_terminus_neighbor_name` this returns a neighbour for a
        connect atom of *any* degree (not only degree-1): a HEAD cap's forward
        connect atom must always be the main-chain terminus, so it always needs a
        neighbour pinned as the tree root. Returns ``None`` only for an isolated
        atom.
        """
        if atom_name is None:
            return None
        atom = next((a for a in monomer.atoms if a.get("name") == atom_name), None)
        if atom is None:
            return None
        for bond in monomer.bonds:
            other = None
            if bond.itom is atom:
                other = bond.jtom
            elif bond.jtom is atom:
                other = bond.itom
            if other is not None and other.get("element") != "H":
                return other["name"]
        return None

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
        """Assemble the chain with tleap ``sequence`` from the per-monomer templates.

        A single tleap run: load each monomer's cached frcmod + prepgen residue
        templates, ``sequence`` them, and save the prmtop/inpcrd. All junction
        parameters come from GAFF base (``source leaprc.gaff2``); with residues
        cut on backbone c3-c3 bonds this always resolves. See the class
        ``DESIGN CONTRACT`` — do NOT add a parmchk2 pass over the assembly.
        """
        from molpy.wrapper.tleap import TLeapWrapper

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

        # --- Build the topology in one pass: per-monomer params + GAFF base. ---
        script_lines = load_lines + [
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
        frame, forcefield = read_amber(prmtop, inpcrd)

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
        - First residue → HEAD variant
        - Last residue → TAIL variant
        - Interior residue → CHAIN variant

        The available variants were compiled from the MolPy reaction product;
        position selects among them but never re-interprets chemistry.
        """
        nodes = graph.nodes
        n = len(nodes)

        if n == 0:
            raise ValueError("Empty graph")

        residue_names = []
        for i, node in enumerate(nodes):
            prep = self._prepared_monomers[node.label]
            if n == 1:
                name = prep.chain_resname
            elif i == 0:
                name = prep.head_resname
            elif i == n - 1:
                name = prep.tail_resname
            else:
                name = prep.chain_resname
            if name is None:
                raise ValueError(
                    f"monomer {node.label!r} lacks the Amber residue variant "
                    f"required at sequence position {i + 1}"
                )
            residue_names.append(name)

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

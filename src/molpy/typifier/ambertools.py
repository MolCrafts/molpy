"""AmberTools typifiers: GAFF / GAFF2 through the AmberTools executables.

Two :class:`molrs.ff.typifier.Typifier` subclasses drive the
:mod:`molpy.wrapper` shells and read tleap's prmtop back:

* :class:`AntechamberTypifier` types a complete molecule from scratch:
  antechamber (atom types and charges) → parmchk2 (missing parameters) → tleap.
* :class:`TLeapTypifier` parameterises a graph whose atoms already carry AMBER
  types and charges (e.g. a chain assembled from antechamber-typed templates)
  with tleap alone, so a chain never pays for an antechamber / parmchk2 run.

Both return the prmtop's assignment as a :class:`~molrs.ff.typifier.Match`; the
molrs base owns ``typify`` and the accumulated ``forcefield()``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import molrs.ff
from molrs.ff import ForceField
from molrs.ff.typifier import Match, Typifier

from molpy.core import fields
from molrs import Angle, Atomistic, Bond, Dihedral, Improper
from molrs.io import write_mol2

from molpy.io.readers import read_amber
from molpy.wrapper import (
    AntechamberWrapper,
    EnvSpec,
    Parmchk2Wrapper,
    TLeapWrapper,
    Wrapper,
)

# The per-atom formal charge the SMILES reader writes (only on charged bracket
# atoms; an atom without it is neutral). molrs.keys has no constant for it.
_FORMAL_CHARGE = "formal_charge"

# The mol2 writer prints charges with four decimals, so a charge tleap reads
# back from the mol2 it was given differs from the graph's by at most half of
# the last printed digit.
_MOL2_CHARGE_TOLERANCE = 0.5e-4

# The tleap unit name of the molecule being parameterised.
_UNIT = "MOL"


class _Prmtop:
    """A tleap prmtop, read back as the Match it assigns to its source graph.

    antechamber and tleap keep atom order, so prmtop atom *i* is graph atom *i*;
    bonded terms are matched by their endpoint rows.
    """

    def __init__(self, prmtop: Path) -> None:
        self.frame, self.forcefield = read_amber(prmtop)

    def match(self, graph: Atomistic, charges: Sequence[float]) -> Match:
        """Regenerate ``graph``'s terms to the prmtop's set and annotate them.

        Args:
            graph: The graph the prmtop was built from (the typifier's private
                copy; its angles, dihedrals and impropers are rewritten).
            charges: The charge to stamp on each atom, in row order.

        Raises:
            ValueError: A graph term the prmtop lacks, or the reverse; an atom
                count or element that disagrees with the prmtop.
        """
        atoms = self.frame["atoms"]
        types = [str(name) for name in atoms[fields.TYPE]]
        elements = [str(symbol) for symbol in atoms[fields.ELEMENT]]
        graph_atoms = list(graph.atoms)
        if len(graph_atoms) != len(types):
            raise ValueError(
                f"the prmtop has {len(types)} atoms, the graph {len(graph_atoms)}"
            )
        for index, atom in enumerate(graph_atoms):
            if atom.get(fields.ELEMENT) != elements[index]:
                raise ValueError(
                    f"atom {index} is {atom.get(fields.ELEMENT)} in the graph but "
                    f"{elements[index]} in the prmtop"
                )
        row = {atom.handle: index for index, atom in enumerate(graph_atoms)}

        prmtop_terms = self._terms()
        graph.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
        graph.remove_link(*graph.impropers)
        for term in prmtop_terms[Improper]:
            graph.def_improper(*(graph_atoms[i] for i in term))

        links: dict[type, list[dict]] = {}
        for kind in (Bond, Angle, Dihedral, Improper):
            relations = list(graph.links.exact_bucket(kind))
            ends = [tuple(row[a.handle] for a in r.endpoints) for r in relations]
            self._require_same(kind, ends, prmtop_terms[kind])
            lookup = self._lookup(kind.__name__.lower())
            links[kind] = [
                {fields.TYPE: lookup[tuple(types[i] for i in term)]} for term in ends
            ]

        (atom_style,) = self.forcefield.get_styles("atom")
        mass = {t.name: t.params["mass"] for t in atom_style.types}
        nodes = [
            {
                fields.TYPE: (atom_style.name, name, (), {fields.MASS: mass[name]}),
                fields.CHARGE: float(charge),
            }
            for name, charge in zip(types, charges, strict=True)
        ]
        styles = [(s.category, s.name, s.params) for s in self.forcefield.styles]
        pairs = [
            (
                style.name,
                t.name,
                [e.name for e in t.endpoints],
                dict(t.params.items()),
            )
            for style in self.forcefield.get_styles("pair")
            for t in style.types
        ]
        return Match(nodes, links, styles=styles, pairs=pairs)

    def _terms(self) -> dict[type, list[tuple[int, ...]]]:
        """The prmtop's terms per kind, as atom-row tuples in file order.

        The structure reader already gives one proper dihedral per atom quartet
        (a multi-term torsion is one term) and keeps impropers in their own
        block.
        """
        four = (fields.ATOMI, fields.ATOMJ, fields.ATOMK, fields.ATOML)
        columns = {
            Bond: ("bonds", four[:2]),
            Angle: ("angles", four[:3]),
            Dihedral: ("dihedrals", four),
            Improper: ("impropers", four),
        }
        rows: dict[type, list[tuple[int, ...]]] = {}
        for kind, (block, keys) in columns.items():
            rows[kind] = []
            if block in self.frame:
                cols = [self.frame[block][key] for key in keys]
                rows[kind] = [tuple(int(v) for v in term) for term in zip(*cols)]
        return rows

    @staticmethod
    def _require_same(
        kind: type,
        graph_terms: list[tuple[int, ...]],
        prmtop_terms: list[tuple[int, ...]],
    ) -> None:
        """Raise unless the graph and the prmtop carry the same terms of ``kind``."""
        graph_keys = {min(t, t[::-1]) for t in graph_terms}
        prmtop_keys = {min(t, t[::-1]) for t in prmtop_terms}
        for key in sorted(graph_keys - prmtop_keys):
            raise ValueError(f"{kind.__name__.lower()} {key} is not in the prmtop")
        for key in sorted(prmtop_keys - graph_keys):
            raise ValueError(
                f"prmtop {kind.__name__.lower()} {key} is not in the graph"
            )

    def _lookup(self, category: str) -> dict[tuple[str, ...], tuple]:
        """Endpoint atom types (either orientation) → the term's annotation."""
        lookup: dict[tuple[str, ...], tuple] = {}
        for style in self.forcefield.get_styles(category):
            for t in style.types:
                ends = [e.name for e in t.endpoints]
                annotation = (style.name, t.name, ends, dict(t.params.items()))
                lookup[tuple(ends)] = annotation
                lookup[tuple(reversed(ends))] = annotation
        return lookup


class _AmberLibrary(Typifier):
    """The seed both AmberTools typifiers share: AMBER's declared settings.

    A typifier's output force field starts as its ``library()``'s empty
    likeness, so declaring units ``real`` and the AMBER 1-4 scaling here is
    what makes every typed output carry them (tleap's prmtop defaults:
    ``coul_14 = 1 / SCEE``, ``lj_14 = 1 / SCNB``).
    """

    def library(self) -> ForceField:
        """An empty force field declaring AMBER units and 1-4 scaling."""
        ff = ForceField("amber", units="real")
        ff.set_special_bonds(
            [0.0, 0.0, 1.0 / molrs.ff.AMBER_SCNB],
            [0.0, 0.0, 1.0 / molrs.ff.AMBER_SCEE],
        )
        return ff


def _write_mol2(graph: Atomistic, path: Path) -> None:
    """Write ``graph`` as the mol2 an AmberTools program reads.

    Atoms are named element + row (``C1``, ``O2``, ...) so antechamber and tleap
    can tell elements apart. The bond ``type`` column is dropped: MOL2 reads it
    as the SYBYL bond order, which a typed graph's force-field label is not.
    """
    frame = graph.to_frame()
    frame["atoms"][fields.NAME] = [
        f"{symbol}{row}"
        for row, symbol in enumerate(frame["atoms"][fields.ELEMENT], start=1)
    ]
    if "bonds" in frame and fields.TYPE in frame["bonds"]:
        del frame["bonds"][fields.TYPE]
    write_mol2(path, frame)


def _run(
    tool: Wrapper,
    output: Path,
    call: Callable[[], subprocess.CompletedProcess[str]],
) -> None:
    """Run one AmberTools step and require its output file.

    Raises:
        RuntimeError: The executable is missing, exits non-zero, or leaves
            ``output`` unwritten; the message carries the tool's stderr.
    """
    if not tool.is_available():
        raise RuntimeError(
            f"{tool.exe} is not available: not on PATH or in env {tool.env!r}"
        )
    result = call()
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"{tool.exe} failed (exit {result.returncode}, {output.name} "
            f"{'written' if output.is_file() else 'missing'}):\n{result.stderr}"
        )


class AntechamberTypifier(_AmberLibrary):
    """GAFF / GAFF2 atom types, charges and parameters for a complete molecule.

    ``match`` runs antechamber (``-at atom_type -c charge_method``, net charge
    from the atoms' formal charges), parmchk2 and tleap in a per-molecule
    directory under ``work_dir``, and assigns the prmtop's types, charges and
    bonded terms. Ports are allowed; their leaving groups are real atoms.

    tleap never changes types or charges downstream, so a chain's atoms keep
    what antechamber gave them in their monomer. Choose the complete monomer so
    each port's leaving group mimics the neighbour its anchor has in the chain:
    the PEO unit is CH3–O–CH2–CH2–O–CH3 with the terminal CH3 and O–CH3 as
    leaving groups (its O typed ether ``os``), not H-capped HO–CH2–CH3 (whose O
    antechamber types hydroxyl ``oh`` with alcohol charges).

    Args:
        atom_type: The antechamber atom-type set, also the parmchk2 ``-s`` set
            and the leaprc tleap sources.
        charge_method: The antechamber ``-c`` charge method.
        work_dir: Where the per-molecule directories are created.
        env: AmberTools environment (see :class:`~molpy.wrapper.EnvSpec`).
        env_manager: Its manager (``"conda"`` / ``"venv"``).
    """

    def __init__(
        self,
        *,
        atom_type: Literal["gaff", "gaff2"] = "gaff2",
        charge_method: str = "bcc",
        work_dir: str | Path = "amber_work",
        env: str | Path | None = None,
        env_manager: str | None = None,
    ) -> None:
        super().__init__()
        spec = EnvSpec.resolve(env, env_manager)
        self.env = spec.env
        self.env_manager = spec.env_manager
        self.atom_type = atom_type
        self.charge_method = charge_method
        self.work_dir = Path(work_dir).resolve()

    def match(self, graph: Atomistic) -> Match:
        """Type ``graph`` with antechamber and assign tleap's prmtop.

        Raises:
            ValueError: The formal charges do not sum to an integer, or the
                prmtop disagrees with the graph.
            RuntimeError: An AmberTools step failed.
        """
        net = sum(atom.get(_FORMAL_CHARGE) or 0.0 for atom in graph.atoms)
        if net != round(net):
            raise ValueError(f"formal charges sum to {net}, not an integer")
        directory = self.work_dir / f"{graph.structural_hash():016x}"
        directory.mkdir(parents=True, exist_ok=True)

        source = directory / "input.mol2"
        typed = directory / "antechamber.mol2"
        frcmod = directory / "parmchk2.frcmod"
        prmtop = directory / f"{_UNIT}.prmtop"
        inpcrd = directory / f"{_UNIT}.inpcrd"
        _write_mol2(graph, source)

        ante = AntechamberWrapper(
            name="antechamber",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        _run(
            ante,
            typed,
            lambda: ante.atomtype_assign(
                source,
                typed,
                input_format="mol2",
                output_format="mol2",
                charge_method=self.charge_method,
                atom_type=self.atom_type,
                net_charge=int(round(net)),
            ),
        )
        parmchk2 = Parmchk2Wrapper(
            name="parmchk2",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        _run(
            parmchk2,
            frcmod,
            lambda: parmchk2.generate_parameters(
                typed, frcmod, input_format="mol2", force_field=self.atom_type
            ),
        )
        leap = TLeapWrapper(
            name="tleap",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        script = (
            f"source leaprc.{self.atom_type}\n"
            f"{_UNIT} = loadmol2 {typed}\n"
            f"loadamberparams {frcmod}\n"
            f"saveamberparm {_UNIT} {prmtop} {inpcrd}\n"
            "quit\n"
        )
        _run(leap, prmtop, lambda: leap.run_from_script(script))

        result = _Prmtop(prmtop)
        return result.match(graph, result.frame["atoms"][fields.CHARGE])


class TLeapTypifier(_AmberLibrary):
    """Parameterise an already typed graph with tleap alone.

    The graph's atoms must carry AMBER atom types and charges (e.g. a chain
    assembled from :class:`AntechamberTypifier`-typed templates). ``match``
    writes them as mol2, writes ``forcefield`` (the templates' parameters) as a
    frcmod, runs tleap and assigns the prmtop's bonded terms, so junction terms
    come from the leaprc. It never runs antechamber or parmchk2. Types and
    charges are left as they are; tleap changing one raises. Every atom thus
    keeps its monomer's type and charge, so the monomers must be chosen with
    leaving groups that mimic the chain neighbour (see
    :class:`AntechamberTypifier`).

    Args:
        leaprc: The leaprc tleap sources (``source leaprc.<leaprc>``).
        forcefield: Parameters tleap loads on top of the leaprc, written as a
            frcmod; ``None`` loads none.
        work_dir: Where the per-graph directories are created.
        env: AmberTools environment (see :class:`~molpy.wrapper.EnvSpec`).
        env_manager: Its manager (``"conda"`` / ``"venv"``).
    """

    def __init__(
        self,
        *,
        leaprc: Literal["gaff", "gaff2"] = "gaff2",
        forcefield: ForceField | None = None,
        work_dir: str | Path = "amber_work",
        env: str | Path | None = None,
        env_manager: str | None = None,
    ) -> None:
        super().__init__()
        spec = EnvSpec.resolve(env, env_manager)
        self.env = spec.env
        self.env_manager = spec.env_manager
        self.leaprc = leaprc
        self.parameters = forcefield
        self.work_dir = Path(work_dir).resolve()

    def match(self, graph: Atomistic) -> Match:
        """Run tleap over ``graph``'s own types and charges; assign its prmtop.

        Raises:
            ValueError: An atom lacks a type or charge, tleap changed one, or
                the prmtop disagrees with the graph.
            RuntimeError: tleap failed.
        """
        types = [atom.get(fields.TYPE) for atom in graph.atoms]
        charges = [atom.get(fields.CHARGE) for atom in graph.atoms]
        for index, (name, charge) in enumerate(zip(types, charges, strict=True)):
            if not isinstance(name, str) or charge is None:
                raise ValueError(
                    f"atom {index} carries type {name!r} and charge {charge!r}; "
                    "TLeapTypifier needs AMBER types and charges on every atom"
                )
        directory = self.work_dir / f"{graph.structural_hash():016x}"
        directory.mkdir(parents=True, exist_ok=True)

        source = directory / "input.mol2"
        prmtop = directory / f"{_UNIT}.prmtop"
        inpcrd = directory / f"{_UNIT}.inpcrd"
        _write_mol2(graph, source)
        script = f"source leaprc.{self.leaprc}\n"
        if self.parameters is not None:
            frcmod = directory / "forcefield.frcmod"
            molrs.ff.write_amber_frcmod(frcmod, self.parameters)
            script += f"loadamberparams {frcmod}\n"
        script += (
            f"{_UNIT} = loadmol2 {source}\n"
            f"saveamberparm {_UNIT} {prmtop} {inpcrd}\n"
            "quit\n"
        )
        leap = TLeapWrapper(
            name="tleap",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        _run(leap, prmtop, lambda: leap.run_from_script(script))

        result = _Prmtop(prmtop)
        atoms = result.frame["atoms"]
        for index, (name, charge, leap_name, leap_charge) in enumerate(
            zip(types, charges, atoms[fields.TYPE], atoms[fields.CHARGE], strict=True)
        ):
            if name != leap_name or abs(charge - leap_charge) > _MOL2_CHARGE_TOLERANCE:
                raise ValueError(
                    f"tleap changed atom {index}: type {name} -> {leap_name}, "
                    f"charge {charge} -> {leap_charge}"
                )
        return result.match(graph, charges)

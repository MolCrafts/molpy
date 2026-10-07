"""AmberTools typifiers: GAFF / GAFF2 through the AmberTools executables.

Two :class:`molrs.ff.typifier.Typifier` subclasses drive the
:mod:`molpy.wrapper` shells and read tleap's prmtop back:

* :class:`AntechamberTypifier` types a complete molecule from scratch:
  antechamber (atom types and charges) → parmchk2 (missing parameters) → tleap.
* :class:`TLeapTypifier` parameterises a finished graph whose atoms already
  carry AMBER types and charges, with tleap alone. A graph that still has
  ports is refused. A polymer is built with
  :class:`molpy.builder.AmberPolymerBuilder`: antechamber, parmchk2
  and prepgen on one oligomer, then tleap ``sequence``.

Both return the prmtop's assignment as a :class:`~molrs.ff.typifier.Match`; the
molrs base owns ``typify`` and the accumulated ``forcefield()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from molrs.ff.forcefield import (
    ForceField,
    read_amber_prmtop_system,
    write_amber_frcmod,
)
from molrs.ff.params import AMBER_SCEE, AMBER_SCNB
from molrs.ff.typifier import Match, Typifier
from molrs.io import write_mol2
from molrs.store.keys import FORMAL_CHARGE
from molrs.system import Angle, Atomistic, Bond, Dihedral, Improper

from molpy.wrapper import (
    AntechamberWrapper,
    EnvSpec,
    Parmchk2Wrapper,
    TLeapWrapper,
    run_step,
)

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
        self.forcefield, self.frame = read_amber_prmtop_system(prmtop)

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
        types = [str(name) for name in atoms["type"]]
        elements = [str(symbol) for symbol in atoms["element"]]
        graph_atoms = list(graph.atoms)
        if len(graph_atoms) != len(types):
            raise ValueError(
                f"the prmtop has {len(types)} atoms, the graph {len(graph_atoms)}"
            )
        for index, atom in enumerate(graph_atoms):
            if atom.get("element") != elements[index]:
                raise ValueError(
                    f"atom {index} is {atom.get('element')} in the graph but "
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
                {"type": lookup[tuple(types[i] for i in term)]} for term in ends
            ]

        (atom_style,) = self.forcefield.get_styles("atom")
        mass = {t.name: t.params["mass"] for t in atom_style.types}
        nodes = [
            {
                "type": (atom_style.name, name, (), {"mass": mass[name]}),
                "charge": float(charge),
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
        four = ("atomi", "atomj", "atomk", "atoml")
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
            [0.0, 0.0, 1.0 / AMBER_SCNB],
            [0.0, 0.0, 1.0 / AMBER_SCEE],
        )
        return ff


class _PrmtopAssignment(_AmberLibrary):
    """Assign a tleap prmtop to the graph it was written for.

    No program runs: ``match`` regenerates the graph's terms to the prmtop's
    set and stamps the prmtop's types and charges on it, exactly as the
    typifiers below do after tleap. The output force field is the typifiers'
    likeness (AMBER units and 1-4 scaling, types without the prmtop's row
    ids), so it merges with theirs.

    Args:
        prmtop: The prmtop; its atom *i* is graph atom *i*.
    """

    def __init__(self, prmtop: Path) -> None:
        super().__init__()
        self._prmtop = _Prmtop(prmtop)

    def match(self, graph: Atomistic) -> Match:
        """The prmtop's types, charges and terms for ``graph``.

        Raises:
            ValueError: The graph and the prmtop disagree on an atom or term.
        """
        return self._prmtop.match(graph, self._prmtop.frame["atoms"]["charge"])


def net_formal_charge(graph: Atomistic) -> int:
    """The sum of the atoms' formal charges (an atom without one is neutral).

    The SMILES reader writes ``formal_charge`` only on charged bracket atoms,
    and the frame schema declares it an integer, so the sum is one.
    """
    key = FORMAL_CHARGE.key
    return sum(int(atom.get(key) or 0) for atom in graph.atoms)


def write_mol2_input(graph: Atomistic, path: Path, *, rename: bool) -> None:
    """Write ``graph`` as the mol2 an AmberTools program reads.

    mol2 carries the bonds, so antechamber does not perceive them from
    coordinates (AmberTools 26's bondtype crashes on a PDB with CONECT
    records). A ``type`` bond column is a force-field label, not the SYBYL
    bond order mol2 reads there, so it is dropped. With ``rename``, atoms are
    named element + row (``C1``, ``O2``, ...) so antechamber and tleap can
    tell elements apart; otherwise the graph's own names are kept.
    """
    frame = graph.to_frame()
    if rename:
        frame["atoms"]["name"] = [
            f"{symbol}{row}"
            for row, symbol in enumerate(frame["atoms"]["element"], start=1)
        ]
    if "bonds" in frame and "type" in frame["bonds"]:
        del frame["bonds"]["type"]
    write_mol2(path, frame)


class AntechamberTypifier(_AmberLibrary):
    """GAFF / GAFF2 atom types, charges and parameters for a complete molecule.

    ``match`` runs antechamber (``-at atom_type -c charge_method``, net charge
    from the atoms' formal charges), parmchk2 and tleap in a per-molecule
    directory under ``work_dir``, and assigns the prmtop's types, charges and
    bonded terms. Ports are allowed; their leaving groups are real atoms.

    A polymer chain is not typed here: joining typed monomers folds each
    leaving group's charge onto its anchor. Build it with
    :class:`molpy.builder.AmberPolymerBuilder`, which cuts one
    antechamber-typed oligomer with prepgen and sequences it with tleap.

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
            ValueError: The prmtop disagrees with the graph.
            RuntimeError: An AmberTools step failed.
        """
        net = net_formal_charge(graph)
        directory = self.work_dir / f"{graph.structural_hash():016x}"
        directory.mkdir(parents=True, exist_ok=True)

        source = directory / "input.mol2"
        typed = directory / "antechamber.mol2"
        frcmod = directory / "parmchk2.frcmod"
        prmtop = directory / f"{_UNIT}.prmtop"
        inpcrd = directory / f"{_UNIT}.inpcrd"
        write_mol2_input(graph, source, rename=True)

        ante = AntechamberWrapper(
            name="antechamber",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        run_step(
            ante,
            typed,
            lambda: ante.atomtype_assign(
                source,
                typed,
                input_format="mol2",
                output_format="mol2",
                charge_method=self.charge_method,
                atom_type=self.atom_type,
                net_charge=net,
            ),
        )
        parmchk2 = Parmchk2Wrapper(
            name="parmchk2",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        run_step(
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
        run_step(leap, prmtop, lambda: leap.run_from_script(script))

        result = _Prmtop(prmtop)
        return result.match(graph, result.frame["atoms"]["charge"])


class TLeapTypifier(_AmberLibrary):
    """Parameterise an already typed graph with tleap alone.

    The graph's atoms must carry AMBER atom types and charges. ``match``
    writes them as mol2, writes ``forcefield`` (the templates' parameters) as a
    frcmod, runs tleap and assigns the prmtop's bonded terms, so junction terms
    come from the leaprc. It never runs antechamber or parmchk2. Types and
    charges are left as they are; tleap changing one raises. A graph that
    still has ports is refused: joining those ports would fold the
    leaving-group charge onto the anchors. A polymer chain is built with
    :class:`molpy.builder.AmberPolymerBuilder`, whose prepgen step
    spreads that charge instead.

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
        if graph.n_ports:
            raise ValueError(
                "TLeapTypifier parameterises a finished molecule; "
                f"this graph still has {graph.n_ports} ports. "
                "Cut the chain with AmberPolymerBuilder (prepgen), "
                "which is what spreads the omitted charge."
            )
        types = [atom.get("type") for atom in graph.atoms]
        charges = [atom.get("charge") for atom in graph.atoms]
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
        write_mol2_input(graph, source, rename=True)
        script = f"source leaprc.{self.leaprc}\n"
        if self.parameters is not None:
            frcmod = directory / "forcefield.frcmod"
            write_amber_frcmod(frcmod, self.parameters)
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
        run_step(leap, prmtop, lambda: leap.run_from_script(script))

        result = _Prmtop(prmtop)
        atoms = result.frame["atoms"]
        for index, (name, charge, leap_name, leap_charge) in enumerate(
            zip(types, charges, atoms["type"], atoms["charge"], strict=True)
        ):
            if name != leap_name or abs(charge - leap_charge) > _MOL2_CHARGE_TOLERANCE:
                raise ValueError(
                    f"tleap changed atom {index}: type {name} -> {leap_name}, "
                    f"charge {charge} -> {leap_charge}"
                )
        return result.match(graph, charges)

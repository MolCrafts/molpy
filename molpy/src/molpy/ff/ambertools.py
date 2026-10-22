"""AmberTools-backed GAFF / GAFF2 typing and AM1-BCC charges.

GAFF typing is native: :class:`molrs.ff.typifier.AtdTypifier` stamps
antechamber's atom types and :class:`molrs.ff.typifier.GaffTypifier` the
gaff / gaff2 bonded terms (with parmchk2-style estimates), with no
executable. Against AmberTools 26 they assign the types antechamber assigns
and the parameters tleap's prmtop carries for the same ``parameter_set``.

AM1-BCC charges need one program, sqm, for the QM step.
:class:`Am1BccModel` runs it (through antechamber ``-c mul``, the sqm input
``-c bcc`` writes) and corrects its AM1 charges with molrs's
:class:`~molrs.ff.charge.BccModel`, which reproduces antechamber ``-c bcc``
from the same sqm run. That is the default AM1-BCC route: native types and
bonded terms, sqm for the charges alone.

A native force field merges with one read from tleap's prmtop, so a small
molecule mixed with an :class:`molpy.builder.AmberPolymerBuilder` chain is
typed natively too. Two :class:`molrs.ff.typifier.Typifier` subclasses run
the whole AmberTools pipeline (:mod:`molpy.tool.ambertools`) and read tleap's
prmtop back, for what only that pipeline does:

* :class:`AntechamberTypifier` types a complete molecule from scratch:
  antechamber (atom types and charges) → parmchk2 (missing parameters) →
  tleap; for an antechamber charge method other than AM1-BCC / ABCG2.
* :class:`TleapTypifier` parameterises a finished graph whose atoms already
  carry AMBER types and charges, with tleap alone, loading parameters of
  your own on top of the leaprc. A graph that still has ports is refused. A
  polymer is built with :class:`molpy.builder.AmberPolymerBuilder`:
  antechamber, parmchk2 and prepgen on one oligomer, then tleap
  ``sequence``.

Both return the prmtop's assignment as a :class:`~molrs.ff.typifier.TypeAssignment`; the
molrs base owns ``typify`` and the accumulated ``forcefield()``. Every
GAFF entry point names the GAFF generation ``parameter_set``, as molrs does,
and none has a default. Every class here reruns every program on every call
(no caching), in a directory keyed by the input and every setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from molrs.core import Atomistic, Element
from molrs.ff.charge import BccModel
from molrs.ff.forcefield import ForceField
from molrs.ff.typifier import TypeAssignment
from molrs.io import read_ac
from molrs.io.amber import AmberFrcmodWriter

from molpy.tool.ambertools import (
    UNIT,
    AmberSteps,
    AmberTypifier,
    ParameterSet,
    Prmtop,
    input_mol2,
    leap_mol2,
    leap_script,
    net_formal_charge,
    run_directory,
)
from molpy.config import load_config

if TYPE_CHECKING:
    from molcfg import Config

# The mol2 writer prints charges with four decimals, so a charge tleap reads
# back from the mol2 it was given differs from the graph's by at most half of
# the last printed digit.
_MOL2_CHARGE_TOLERANCE = 0.5e-4

#: The ac file the QM step writes.
_AM1_AC = "sqm.ac"


class Am1BccModel:
    """AM1-BCC (or ABCG2) charges: sqm's AM1 charges, corrected natively.

    The QM step is the one AmberTools run: antechamber ``-c mul`` (sqm, with
    the input antechamber ``-c bcc`` writes; net charge from the atoms'
    formal charges) in a directory under ``work_dir`` keyed by the input
    mol2 and the net charge. Its Mulliken charges go to molrs's
    :class:`~molrs.ff.charge.BccModel`, which equivalences and corrects them
    as antechamber does. Given the same sqm run, the result is antechamber
    ``-c bcc`` / ``-c abcg2`` to the ac file's six printed decimals.

    sqm picks its diagonaliser by timing it, and prints Mulliken charges to
    three decimals, so two runs on one input can differ by 0.001 e on an
    atom (antechamber ``-c bcc`` run twice does too).

    Types and bonded terms come from the native typifiers, before or after::

        labelled = AtdTypifier(parameter_set="gaff2").typify(mol)
        charges = Am1BccModel(parameter_set="bcc").assign(labelled)
        for atom, charge in zip(labelled.atoms, charges):
            atom["charge"] = float(charge)
        typed = GaffTypifier(parameter_set="gaff2").typify(labelled)

    Args:
        parameter_set: The correction family, molrs
            :class:`~molrs.ff.charge.BccModel`'s ``parameter_set``:
            ``"bcc"`` (AM1-BCC) or ``"abcg2"``. Required, as in molrs.
        work_dir: Where the per-run directories are created.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[ambertools]``,
            ``[ambertools.<tool>]``); ``None`` loads it.

    Raises:
        ValueError: ``parameter_set`` names no correction family.
    """

    def __init__(
        self,
        *,
        parameter_set: Literal["bcc", "abcg2"],
        work_dir: str | Path = "amber_work",
        config: Config | None = None,
    ) -> None:
        self.correction = BccModel(parameter_set=parameter_set)
        self.config = config if config is not None else load_config()
        self.work_dir = Path(work_dir).resolve()

    @property
    def parameter_set(self) -> str:
        """The correction family: ``"bcc"`` or ``"abcg2"``."""
        return self.correction.parameter_set

    def am1(self, mol: Atomistic) -> np.ndarray:
        """sqm's AM1 Mulliken charges, in graph atom order, unequivalenced.

        ``mol`` needs 3D coordinates; it is left untouched.

        Raises:
            ValueError: antechamber's ac does not list ``mol``'s atoms in
                ``mol``'s order.
            molpy.tool.ToolError: antechamber or sqm failed.
        """
        net = net_formal_charge(mol)
        text = input_mol2(mol, rename=True)
        directory = run_directory(self.work_dir, text, "mul", str(net))
        source = directory / "input.mol2"
        ac = directory / _AM1_AC
        source.write_text(text, encoding="utf-8")
        AmberSteps(directory, self.config).am1(source, ac, net_charge=net)

        atoms = read_ac(ac)["atoms"]
        expected = [
            f"{Element.by_number(atom['atomic_number']).symbol}{row}"
            for row, atom in enumerate(mol.atoms, 1)
        ]
        found = [str(name) for name in atoms["name"]]
        if found != expected:
            raise ValueError(
                f"{ac} lists atoms {found}, not the graph's {expected}: "
                "antechamber reordered or dropped atoms"
            )
        return np.asarray(atoms["charge"], dtype=np.float64)

    def assign(self, mol: Atomistic) -> np.ndarray:
        """The AM1-BCC charges of ``mol``, in graph atom order.

        ``mol`` is left untouched: set ``atoms.charge`` from the result.

        Raises:
            ValueError: The ac disagrees with ``mol`` (see :meth:`am1`), or
                ``mol`` carries an atom or bond the correction family cannot
                type.
            molpy.tool.ToolError: antechamber or sqm failed.
        """
        return self.correction.assign(mol, qm=self.am1(mol))

    def __repr__(self) -> str:
        return f"Am1BccModel(parameter_set={self.parameter_set!r})"


class AntechamberTypifier(AmberTypifier):
    """GAFF / GAFF2 atom types, charges and parameters for a complete molecule.

    ``assign`` runs antechamber (``-at parameter_set -c charge_method``, net charge
    from the atoms' formal charges), parmchk2 and tleap in a directory under
    ``work_dir`` keyed by the input mol2 and every setting, and assigns the
    prmtop's types, charges and bonded terms. Every program reruns, and each
    step's output is deleted before it runs, so nothing an earlier run left
    behind is read. Ports are allowed; their leaving groups are real atoms.

    A polymer chain is not typed here: joining typed monomers folds each
    leaving group's charge onto its anchor. Build it with
    :class:`molpy.builder.AmberPolymerBuilder`, which cuts one
    antechamber-typed oligomer with prepgen and sequences it with tleap.

    The default GAFF route is native: :class:`~molrs.ff.typifier.AtdTypifier`
    and :class:`~molrs.ff.typifier.GaffTypifier` assign the same types and
    parameter values, and :class:`Am1BccModel` the same AM1-BCC charges, with
    sqm the only program; that force field merges with an
    :class:`~molpy.builder.AmberPolymerBuilder` chain's as this one does.
    This typifier is for an antechamber charge method other than AM1-BCC /
    ABCG2. Its prmtop force field also carries no estimate provenance, which
    molrs 0.18.0's LAMMPS and GROMACS writers refuse on a term
    ``GaffTypifier`` estimated.

    Args:
        parameter_set: The GAFF generation, molrs's ``parameter_set``
            (``"gaff"`` or ``"gaff2"``): antechamber's ``-at``, parmchk2's
            ``-s`` and the ``leaprc.<parameter_set>`` tleap sources.
            Required, as in molrs.
        charge_method: The antechamber ``-c`` charge method.
        work_dir: Where the per-run directories are created.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[ambertools]``,
            ``[ambertools.<tool>]``); ``None`` loads it.
    """

    def __init__(
        self,
        *,
        parameter_set: ParameterSet,
        charge_method: str = "bcc",
        work_dir: str | Path = "amber_work",
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else load_config()
        self.parameter_set = parameter_set
        self.charge_method = charge_method
        self.work_dir = Path(work_dir).resolve()

    def assign(self, graph: Atomistic) -> TypeAssignment:
        """Type ``graph`` with antechamber and assign tleap's prmtop.

        Raises:
            ValueError: The prmtop disagrees with the graph.
            molpy.tool.ToolError: An AmberTools step failed.
        """
        net = net_formal_charge(graph)
        text = input_mol2(graph, rename=True)
        directory = run_directory(
            self.work_dir, text, self.parameter_set, self.charge_method, str(net)
        )
        source = directory / "input.mol2"
        typed = directory / "antechamber.mol2"
        frcmod = directory / "parmchk2.frcmod"
        prmtop = directory / f"{UNIT}.prmtop"
        source.write_text(text, encoding="utf-8")

        steps = AmberSteps(directory, self.config)
        steps.antechamber(
            source,
            typed,
            parameter_set=self.parameter_set,
            charge_method=self.charge_method,
            net_charge=net,
        )
        steps.parmchk2(typed, frcmod, parameter_set=self.parameter_set)
        steps.tleap(
            leap_script(
                self.parameter_set,
                leap_mol2(typed),
                prmtop,
                directory / f"{UNIT}.inpcrd",
                frcmods=[frcmod],
            ),
            prmtop,
            directory / f"{UNIT}.inpcrd",
        )

        result = Prmtop(prmtop)
        return result.assignment(graph, result.frame["atoms"]["charge"])


class TleapTypifier(AmberTypifier):
    """Parameterise an already typed graph with tleap alone.

    The graph's atoms must carry AMBER atom types and charges. ``assign``
    writes them as mol2, writes ``forcefield`` (the templates' parameters) as a
    frcmod, runs tleap and assigns the prmtop's bonded terms, so junction terms
    come from the leaprc. The run directory is keyed by the mol2, the
    parameter set and the frcmod, and tleap's outputs are deleted before it runs. It never
    runs antechamber or parmchk2. Types and charges are left as they are;
    tleap changing one raises. A graph that
    still has ports is refused: joining those ports would fold the
    leaving-group charge onto the anchors. A polymer chain is built with
    :class:`molpy.builder.AmberPolymerBuilder`, whose prepgen step
    spreads that charge instead.

    :class:`~molrs.ff.typifier.GaffTypifier` assigns the same bonded terms
    natively, and its force field merges with an
    :class:`~molpy.builder.AmberPolymerBuilder` chain's; this typifier is for
    parameters of your own on top of the leaprc (``forcefield``).

    Args:
        parameter_set: The GAFF generation, molrs's ``parameter_set``
            (``"gaff"`` or ``"gaff2"``): tleap sources
            ``leaprc.<parameter_set>``. Required, as in molrs.
        forcefield: Parameters tleap loads on top of the leaprc, written as a
            frcmod; ``None`` loads none.
        work_dir: Where the per-run directories are created.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[ambertools]``,
            ``[ambertools.<tool>]``); ``None`` loads it.
    """

    def __init__(
        self,
        *,
        parameter_set: ParameterSet,
        forcefield: ForceField | None = None,
        work_dir: str | Path = "amber_work",
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else load_config()
        self.parameter_set = parameter_set
        self.parameters = forcefield
        self.work_dir = Path(work_dir).resolve()

    def assign(self, graph: Atomistic) -> TypeAssignment:
        """Run tleap over ``graph``'s own types and charges; assign its prmtop.

        Raises:
            ValueError: An atom lacks a type or charge, tleap changed one, or
                the prmtop disagrees with the graph.
            molpy.tool.ToolError: tleap failed.
        """
        if graph.n_ports:
            raise ValueError(
                "TleapTypifier parameterises a finished molecule; "
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
                    "TleapTypifier needs AMBER types and charges on every atom"
                )
        text = input_mol2(graph, rename=True)
        buffer = bytearray()
        if self.parameters is not None:
            AmberFrcmodWriter().write(buffer, self.parameters)
        extra = buffer.decode("utf-8")
        directory = run_directory(self.work_dir, text, self.parameter_set, extra)
        source = directory / "input.mol2"
        prmtop = directory / f"{UNIT}.prmtop"
        inpcrd = directory / f"{UNIT}.inpcrd"
        source.write_text(text, encoding="utf-8")
        frcmods = []
        if self.parameters is not None:
            frcmods.append(directory / "forcefield.frcmod")
            frcmods[0].write_text(extra, encoding="utf-8")
        AmberSteps(directory, self.config).tleap(
            leap_script(
                self.parameter_set,
                leap_mol2(source),
                prmtop,
                inpcrd,
                frcmods=frcmods,
            ),
            prmtop,
            inpcrd,
        )

        result = Prmtop(prmtop)
        atoms = result.frame["atoms"]
        for index, (name, charge, leap_name, leap_charge) in enumerate(
            zip(types, charges, atoms["type"], atoms["charge"], strict=True)
        ):
            if name != leap_name or abs(charge - leap_charge) > _MOL2_CHARGE_TOLERANCE:
                raise ValueError(
                    f"tleap changed atom {index}: type {name} -> {leap_name}, "
                    f"charge {charge} -> {leap_charge}"
                )
        return result.assignment(graph, charges)

"""AmberTools typifiers: GAFF / GAFF2 through the AmberTools executables.

GAFF typing is native first: :class:`molrs.ff.typifier.AtdTypifier` stamps
antechamber's atom types and :class:`molrs.ff.typifier.GaffTypifier` the
gaff / gaff2 bonded terms (with parmchk2-style estimates), with no
executable. The AmberTools route is for what that cannot do: AM1-BCC
charges, which need sqm's QM charges, and a molecule that is merged with
an :class:`molpy.builder.AmberPolymerBuilder` chain, whose parameters are
the installed AmberTools' (tleap's prmtop). Every entry point names the GAFF
generation ``parameter_set``, as molrs does, and none has a default.

Two :class:`molrs.ff.typifier.Typifier` subclasses run the shared AmberTools
pipeline (:mod:`molpy._ambertools`) and read tleap's prmtop back:

* :class:`AntechamberTypifier` types a complete molecule from scratch:
  antechamber (atom types and charges) → parmchk2 (missing parameters) → tleap.
* :class:`TleapTypifier` parameterises a finished graph whose atoms already
  carry AMBER types and charges, with tleap alone. A graph that still has
  ports is refused. A polymer is built with
  :class:`molpy.builder.AmberPolymerBuilder`: antechamber, parmchk2
  and prepgen on one oligomer, then tleap ``sequence``.

Both return the prmtop's assignment as a :class:`~molrs.ff.typifier.TypeAssignment`; the
molrs base owns ``typify`` and the accumulated ``forcefield()``. Both rerun
every program on every call (no caching), in a directory keyed by the input
and every setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molrs.core import Atomistic
from molrs.ff.forcefield import ForceField
from molrs.ff.typifier import TypeAssignment
from molrs.io import write_frcmod_str

from molpy._ambertools import (
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

    For GAFF types and bonded terms without AM1-BCC charges, the native
    :class:`~molrs.ff.typifier.AtdTypifier` and
    :class:`~molrs.ff.typifier.GaffTypifier` need no executable.

    Args:
        parameter_set: The GAFF generation, molrs's ``parameter_set``
            (``"gaff"`` or ``"gaff2"``): antechamber's ``-at``, parmchk2's
            ``-s`` and the ``leaprc.<parameter_set>`` tleap sources.
            Required, as in molrs.
        charge_method: The antechamber ``-c`` charge method.
        work_dir: Where the per-run directories are created.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[wrapper]``,
            ``[wrapper.<tool>]``); ``None`` loads it.
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
            molpy.wrapper.ToolError: An AmberTools step failed.
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
    natively; this typifier is for parameters on top of the leaprc
    (``forcefield``) and for a graph merged with an
    :class:`~molpy.builder.AmberPolymerBuilder` chain.

    Args:
        parameter_set: The GAFF generation, molrs's ``parameter_set``
            (``"gaff"`` or ``"gaff2"``): tleap sources
            ``leaprc.<parameter_set>``. Required, as in molrs.
        forcefield: Parameters tleap loads on top of the leaprc, written as a
            frcmod; ``None`` loads none.
        work_dir: Where the per-run directories are created.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[wrapper]``,
            ``[wrapper.<tool>]``); ``None`` loads it.
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
            molpy.wrapper.ToolError: tleap failed.
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
        extra = write_frcmod_str(self.parameters) if self.parameters is not None else ""
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

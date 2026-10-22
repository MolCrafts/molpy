"""Run antechamber, parmchk2, prepgen and tleap on a user-built oligomer.

The oligomer is one molecule: the head, chain and tail monomers already
bonded, so antechamber sees each atom with its chain neighbours. The user
says how prepgen cuts it (:class:`~molpy.builder.AmberCut`). The
site graph only chooses the tleap ``sequence``. This does not call
:class:`molpy.builder.Assembler` and it does not change the charges prepgen wrote.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from molrs.core import Atomistic, Bead, CoarseGrain
from molrs.io import read_ac, read_inpcrd, read_prmtop

from molpy.tool.ambertools import (
    AmberSteps,
    ParameterSet,
    PrmtopTypifier,
    digest,
    input_mol2,
    leap_script,
    leap_sequence,
    net_formal_charge,
    prepgen_control_text,
)
from molpy.builder.polymer.ambertools.cut import AmberCut
from molpy.builder.polymer.ambertools.result import AmberBuildResult
from molpy.config import load_config

if TYPE_CHECKING:
    from molcfg import Config

Variant = Literal["head", "chain", "tail"]
_VARIANTS: tuple[Variant, ...] = ("head", "chain", "tail")

# Written next to an oligomer's antechamber and parmchk2 outputs: the digest
# of what produced them. A directory without it holds files the user put there.
_STAMP = "antechamber.sha1"


@dataclass
class _Prepared:
    frcmod: Path
    prepi: dict[Variant, Path]


class AmberPolymerBuilder:
    """Parameterise one oligomer per bead type and sequence it with tleap.

    tleap's ``sequence`` makes the bonds between residues, so this builder
    has no placer and no orienter. ``library`` maps a bead type to the
    oligomer the user prepared. ``cuts`` maps that bead type to its head,
    chain and tail residues. antechamber and parmchk2 run on the oligomer,
    once. prepgen writes the residues the sequence uses. tleap loads each
    frcmod and prepi and runs ``sequence``. Neither tool sees the assembled
    chain. The box is packed afterwards with molpack.

    Everything is written under ``work_dir``: ``monomers/<label>/`` holds the
    oligomer's input mol2, ac, frcmod, control files and prepi files, and
    ``chains/<digest>/`` one tleap build. antechamber runs once per oligomer
    (``-fo ac``: types and charges, one sqm run); parmchk2 and prepgen read
    that ac. A rerun reuses what is there: antechamber reruns only when the
    oligomer, ``parameter_set``, ``charge_method`` or net charge changed (an ac
    and frcmod placed there by hand are used as they are), prepgen reruns
    when its control file changed, and tleap when the sequence or any file
    it loads changed. A step that runs deletes its outputs first.

    Args:
        library: Bead type → the oligomer (head, chain and tail monomers
            already bonded, with 3D coordinates). Atoms without a ``name``
            are named element + row.
        cuts: Bead type → ``{"head": ..., "chain": ..., "tail": ...}``; only
            the residues a sequence uses are required.
        parameter_set: The GAFF generation, molrs's ``parameter_set``
            (``"gaff"`` or ``"gaff2"``): antechamber's ``-at``, parmchk2's
            ``-s`` and the ``leaprc.<parameter_set>`` tleap sources.
            Required, as in molrs: a chain and the molecules typed for the
            same system name the same set.
        charge_method: The antechamber ``-c`` charge method.
        work_dir: Where the per-oligomer and per-chain directories are made.
        config: molpy's configuration (:func:`molpy.config.load_config`):
            the AmberTools executables and environment (``[ambertools]``,
            ``[ambertools.<tool>]``); ``None`` loads it.
        net_charges: Bead type → oligomer net charge. ``None`` sums each
            oligomer's ``formal_charge``.

    Raises:
        TypeError: A template is not :class:`~molpy.Atomistic`, or a cut is
            not :class:`AmberCut`.
        ValueError: A cut is not head, chain or tail, names an atom the
            oligomer lacks, or gives a head residue a head connection (a
            tail residue a tail connection); two bead types share a tleap
            residue name; a template repeats an atom name.

    Example:
        >>> oligomer, cuts = AmberPieces("COCC", "OCC", "OCCOC").oligomer()
        >>> sites = mp.io.cgsmiles.read_cgsmiles("{[#PEO]|10}").to_coarsegrain()
        >>> config = load_config(
        ...     {"ambertools": {"env": "AmberTools25", "env_manager": "conda"}}
        ... )
        >>> built = AmberPolymerBuilder(
        ...     {"PEO": oligomer}, {"PEO": cuts}, parameter_set="gaff2", config=config
        ... ).assemble(sites)
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        cuts: Mapping[str, Mapping[str, AmberCut]],
        *,
        parameter_set: ParameterSet,
        charge_method: str = "bcc",
        work_dir: Path | str = "amber_work",
        config: Config | None = None,
        net_charges: Mapping[str, int] | None = None,
    ) -> None:
        self.library = {label: _named(template) for label, template in library.items()}
        self.cuts: dict[str, dict[Variant, AmberCut]] = {
            label: _checked_cuts(label, group, self.library.get(label))
            for label, group in cuts.items()
        }
        residues: dict[str, str] = {}
        for label in self.library:
            for variant in _VARIANTS:
                residue = _resname(label, variant)
                if residues.setdefault(residue, label) != label:
                    raise ValueError(
                        f"bead types {residues[residue]!r} and {label!r} both "
                        f"make tleap residue {residue!r}; residue names come "
                        "from the first characters of the bead type"
                    )
        self.parameter_set = parameter_set
        self.charge_method = charge_method
        self.net_charges = dict(net_charges) if net_charges is not None else None
        self.work_dir = Path(work_dir).resolve()
        self.config = config if config is not None else load_config()

    def assemble(self, sites: CoarseGrain) -> AmberBuildResult:
        """Build the chain described by ``sites``.

        The first site uses that label's head cut, the last site its tail
        cut, and every site between them its chain cut.

        Args:
            sites: One linear path of at least two sites; each site's
                ``bead_type`` is a library key.

        Returns:
            The chain tleap wrote, read back with its force field.

        Raises:
            TypeError: ``sites`` is not a coarse-grain graph.
            ValueError: The graph is not one linear path, a bead type has
                no oligomer, or a residue has no cut.
            molpy.tool.ToolError: An AmberTools step failed or is not installed.
        """
        if not isinstance(sites, CoarseGrain):
            raise TypeError(
                "AmberPolymerBuilder.assemble takes a coarse-grain site graph, "
                f"got {type(sites).__name__}"
            )
        labels = [_label(bead) for bead in _linear_path(sites)]
        missing = sorted({label for label in labels if label not in self.library})
        if missing:
            raise ValueError(
                f"labels {missing} are not in the library. "
                f"Available: {sorted(self.library)}"
            )
        needed = _needed(labels)
        absent = sorted(
            f"{label} {variant}"
            for label, variants in needed.items()
            for variant in variants
            if variant not in self.cuts.get(label, {})
        )
        if absent:
            raise ValueError(
                f"no prepgen cut for {absent}. "
                "Each cut is the control file for that residue."
            )
        prepared = {
            label: self._prepare(label, variants) for label, variants in needed.items()
        }
        return self._tleap(labels, prepared)

    def _prepare(self, label: str, variants: set[Variant]) -> _Prepared:
        """antechamber + parmchk2 on the oligomer, prepgen for each residue."""
        directory = self.work_dir / "monomers" / label
        directory.mkdir(parents=True, exist_ok=True)
        steps = AmberSteps(directory, self.config)
        ac = directory / f"{label}.ac"
        frcmod = directory / f"{label}.frcmod"
        oligomer = self.library[label]
        charge = _net_charge(oligomer, label, self.net_charges)
        stale = self._antechamber(steps, label, oligomer, charge)

        frame = read_ac(ac)
        ac_names = [str(name) for name in frame["atoms"]["name"]]
        ac_types = dict(zip(ac_names, map(str, frame["atoms"]["type"]), strict=True))
        renamed = _rename(_atom_names(oligomer), ac_names)
        prepi: dict[Variant, Path] = {}
        for variant in _VARIANTS:
            if variant not in variants:
                continue
            cut = self.cuts[label][variant]
            control = directory / f"{label}.{variant}"
            previous = (
                control.read_text(encoding="utf-8") if control.is_file() else None
            )
            text = prepgen_control_text(
                variant=variant,
                head_name=_mapped(cut.head, renamed),
                tail_name=_mapped(cut.tail, renamed),
                head_type=_connection_type(
                    cut.pre_head, cut.pre_head_type, ac_types, renamed, label
                ),
                tail_type=_connection_type(
                    cut.post_tail, cut.post_tail_type, ac_types, renamed, label
                ),
                omit_names=[renamed[name] for name in cut.omit],
                charge=cut.charge,
            )
            control.write_text(text, encoding="utf-8")
            output = directory / _prepi_name(label, variant)
            prepi[variant] = output
            if stale or not output.is_file() or text != previous:
                steps.prepgen(ac, output, control, _resname(label, variant))
        return _Prepared(frcmod, prepi)

    def _antechamber(
        self, steps: AmberSteps, label: str, oligomer: Atomistic, charge: int
    ) -> bool:
        """Type and charge the oligomer once (``-fo ac``), then parmchk2 on the ac.

        Skipped when the ac and frcmod were made from this input and these
        settings (the stamp holds their digest), or were put there by hand
        (no stamp). Returns whether it ran.
        """
        directory = steps.directory
        ac = directory / f"{label}.ac"
        frcmod = directory / f"{label}.frcmod"
        source = directory / f"{label}.input.mol2"
        text = input_mol2(oligomer, rename=False)
        source.write_text(text, encoding="utf-8")
        stamp = directory / _STAMP
        current = digest(text, self.parameter_set, self.charge_method, str(charge))
        if (
            ac.is_file()
            and frcmod.is_file()
            and (not stamp.is_file() or stamp.read_text(encoding="utf-8") == current)
        ):
            return False
        stamp.unlink(missing_ok=True)
        steps.antechamber(
            source,
            ac,
            parameter_set=self.parameter_set,
            charge_method=self.charge_method,
            net_charge=charge,
        )
        steps.parmchk2(ac, frcmod, parameter_set=self.parameter_set)
        stamp.write_text(current, encoding="utf-8")
        return True

    def _tleap(
        self, labels: list[str], prepared: dict[str, _Prepared]
    ) -> AmberBuildResult:
        residues = [
            _resname(label, _variant(index, len(labels)))
            for index, label in enumerate(labels)
        ]
        frcmods = [prepared[label].frcmod for label in sorted(prepared)]
        preps = [
            path
            for label in sorted(prepared)
            for path in prepared[label].prepi.values()
        ]
        key = digest(
            self.parameter_set,
            *residues,
            *(
                part
                for path in (*frcmods, *preps)
                for part in (str(path), path.read_bytes())
            ),
        )
        directory = self.work_dir / "chains" / key[:16]
        prmtop = directory / "polymer.prmtop"
        inpcrd = directory / "polymer.inpcrd"
        if not (prmtop.is_file() and inpcrd.is_file()):
            directory.mkdir(parents=True, exist_ok=True)
            script = leap_script(
                self.parameter_set,
                leap_sequence(residues),
                prmtop,
                inpcrd,
                frcmods=frcmods,
                preps=preps,
            )
            AmberSteps(directory, self.config).tleap(
                script, prmtop, inpcrd, script_name="polymer.in"
            )
        frame = read_inpcrd(inpcrd, read_prmtop(prmtop))
        assignment = PrmtopTypifier(prmtop)
        chain = assignment.typify(Atomistic.from_frame(frame))
        return AmberBuildResult(
            chain, assignment.forcefield(), prmtop, inpcrd, len(labels)
        )


def _checked_cuts(
    label: str, group: Mapping[str, AmberCut], template: Atomistic | None
) -> dict[Variant, AmberCut]:
    """Validate one bead type's cuts against its residue kind and oligomer."""
    unknown = set(group) - set(_VARIANTS)
    if unknown:
        raise ValueError(
            f"cuts for {label!r} use {sorted(unknown)}; a cut is head, chain or tail"
        )
    names = set(_atom_names(template)) if template is not None else None
    checked: dict[Variant, AmberCut] = {}
    for variant in _VARIANTS:
        if variant not in group:
            continue
        cut = group[variant]
        if not isinstance(cut, AmberCut):
            raise TypeError(
                f"{label!r} {variant} cut must be AmberCut, got {type(cut).__name__}"
            )
        if variant == "head" and (cut.pre_head or cut.pre_head_type):
            raise ValueError(
                f"{label!r} head residue connects forward only; it has no PRE_HEAD_TYPE"
            )
        if variant == "tail" and (cut.post_tail or cut.post_tail_type):
            raise ValueError(
                f"{label!r} tail residue connects backward only; "
                "it has no POST_TAIL_TYPE"
            )
        if variant in ("head", "chain") and cut.tail is None:
            raise ValueError(f"{label!r} {variant} cut needs a tail atom")
        if variant in ("chain", "tail") and cut.head is None:
            raise ValueError(f"{label!r} {variant} cut needs a head atom")
        if names is not None:
            named = [cut.head, cut.tail, cut.pre_head, cut.post_tail, *cut.omit]
            unknown_atoms = [n for n in named if n is not None and n not in names]
            if unknown_atoms:
                raise ValueError(
                    f"{label!r} {variant} cut names {unknown_atoms}, which are "
                    "not atoms of that oligomer"
                )
        checked[variant] = cut
    return checked


def _named(template: Atomistic) -> Atomistic:
    """Copy ``template`` and give every atom a stable Amber name."""
    if not isinstance(template, Atomistic):
        raise TypeError(
            f"library templates must be Atomistic, got {type(template).__name__}"
        )
    from copy import copy
    from molrs.core import Element

    copied = copy(template)
    seen: set[str] = set()
    for index, atom in enumerate(copied.atoms, start=1):
        name = atom.get("name")
        if name is None:
            atomic_number = atom.get("atomic_number")
            if not atomic_number:
                raise ValueError(f"atom {index} has no element and no name")
            name = f"{Element.by_number(int(atomic_number)).symbol}{index}"
        name = str(name)
        if name in seen:
            raise ValueError(f"template repeats Amber atom name {name!r}")
        seen.add(name)
        atom["name"] = name
    return copied


def _label(bead: Bead) -> str:
    name = bead.get("bead_type")
    if not name:
        raise ValueError("a site has no bead type")
    return str(name)


def _linear_path(sites: CoarseGrain) -> list[Bead]:
    beads = list(sites.beads)
    if len(beads) < 2:
        raise ValueError(
            "AmberPolymerBuilder needs at least a head site and a tail site; "
            "type one molecule natively (AtdTypifier, GaffTypifier)"
        )
    bonds = list(sites.bonds)
    adjacent: dict[int, list[Bead]] = defaultdict(list)
    for bond in bonds:
        left, right = bond.endpoints
        adjacent[left.handle].append(right)
        adjacent[right.handle].append(left)
    degrees = [len(adjacent[bead.handle]) for bead in beads]
    if (
        len(bonds) != len(beads) - 1
        or max(degrees) > 2
        or sum(degree == 1 for degree in degrees) != 2
    ):
        raise ValueError("tleap sequence supports one linear path")
    path = [next(bead for bead, d in zip(beads, degrees, strict=True) if d == 1)]
    previous: Bead | None = None
    while len(path) < len(beads):
        nxt = [
            bead
            for bead in adjacent[path[-1].handle]
            if previous is None or bead.handle != previous.handle
        ]
        if len(nxt) != 1:
            raise ValueError("tleap sequence supports one linear path")
        previous = path[-1]
        path.append(nxt[0])
    return path


def _variant(index: int, count: int) -> Variant:
    if index == 0:
        return "head"
    if index == count - 1:
        return "tail"
    return "chain"


def _needed(labels: list[str]) -> dict[str, set[Variant]]:
    found: dict[str, set[Variant]] = defaultdict(set)
    for index, label in enumerate(labels):
        found[label].add(_variant(index, len(labels)))
    return found


def _atom_names(template: Atomistic) -> list[str]:
    return [str(atom.get("name")) for atom in template.atoms]


def _net_charge(
    template: Atomistic, label: str, declared: dict[str, int] | None
) -> int:
    if declared is not None:
        if label not in declared:
            raise KeyError(
                f"net_charges has no entry for {label!r}; "
                "list every label, or pass no net_charges"
            )
        return int(declared[label])
    return net_formal_charge(template)


def _rename(ours: list[str], ac_names: list[str]) -> dict[str, str]:
    """Map oligomer atom names onto the names antechamber wrote."""
    if set(ours) == set(ac_names):
        return {name: name for name in ours}
    if len(ours) != len(ac_names):
        raise RuntimeError(
            f"antechamber wrote {len(ac_names)} atoms for a {len(ours)}-atom oligomer"
        )
    return dict(zip(ours, ac_names, strict=True))


def _mapped(name: str | None, renamed: dict[str, str]) -> str | None:
    return None if name is None else renamed[name]


def _connection_type(
    atom: str | None,
    literal: str | None,
    types: dict[str, str],
    renamed: dict[str, str],
    label: str,
) -> str | None:
    """The type the user wrote, or the GAFF type of the atom they named."""
    if literal:
        return literal
    if atom is None:
        return None
    ac_name = renamed[atom]
    if ac_name not in types:
        raise ValueError(
            f"no atom type for {label!r} atom {atom!r} (PRE_HEAD_TYPE / POST_TAIL_TYPE)"
        )
    return types[ac_name]


def _resname(label: str, variant: Variant) -> str:
    if variant == "chain":
        return label[:3].upper()
    prefix = "H" if variant == "head" else "T"
    return f"{prefix}{label[:2].upper()}"


def _prepi_name(label: str, variant: Variant) -> str:
    if variant == "head":
        return f"H{label}.prepi"
    if variant == "tail":
        return f"T{label}.prepi"
    return f"{label}.prepi"

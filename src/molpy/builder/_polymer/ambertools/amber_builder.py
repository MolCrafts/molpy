"""Run antechamber, parmchk2, prepgen and tleap on a user-built oligomer.

The oligomer is one molecule: the head, chain and tail monomers already
bonded, so antechamber sees each atom with its chain neighbours. The user
says how prepgen cuts it (:class:`~molpy.builder.AmberCut`). The
site graph only chooses the tleap ``sequence``. This does not call
:class:`molpy.builder.Assembler` and it does not change the charges prepgen wrote.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from molrs.io import read_ac, write_mol2
from molrs.system import Atomistic, Bead, CoarseGrain

from molpy.ff._ambertools import _PrmtopAssignment
from molpy.wrapper import (
    AntechamberWrapper,
    EnvSpec,
    Parmchk2Wrapper,
    PrepgenWrapper,
    TLeapWrapper,
    run_step,
    write_prepgen_control_file,
)

from .types import AmberBuildResult, AmberCut

Variant = Literal["head", "chain", "tail"]
_VARIANTS: tuple[Variant, ...] = ("head", "chain", "tail")

# Written next to an oligomer's antechamber outputs: the digest of what
# produced them. A directory without it holds files the user put there.
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
    oligomer's input mol2, ac, mol2, frcmod, control files and prepi files, and
    ``chains/<digest>/`` one tleap build. A rerun reuses what is there:
    antechamber reruns only when the oligomer, ``force_field``,
    ``charge_method`` or net charge changed (an ac, mol2 and frcmod placed
    there by hand are used as they are), prepgen reruns when its control
    file changed, and tleap when the sequence or any file it loads changed.

    Args:
        library: Bead type → the oligomer (head, chain and tail monomers
            already bonded, with 3D coordinates). Atoms without a ``name``
            are named element + row.
        cuts: Bead type → ``{"head": ..., "chain": ..., "tail": ...}``; only
            the residues a sequence uses are required.
        force_field: The antechamber atom-type set, the parmchk2 ``-s`` set
            and the leaprc tleap sources.
        charge_method: The antechamber ``-c`` charge method.
        work_dir: Where the per-oligomer and per-chain directories are made.
        env: AmberTools environment (see :class:`~molpy.wrapper.EnvSpec`).
        env_manager: Its manager (``"conda"`` / ``"venv"``).
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
        >>> sites = mp.io.CGSmilesIR("{[#PEO]|10}").to_coarsegrain()
        >>> built = AmberPolymerBuilder(
        ...     {"PEO": oligomer},
        ...     {"PEO": cuts},
        ...     env="AmberTools25",
        ...     env_manager="conda",
        ... ).assemble(sites)
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        cuts: Mapping[str, Mapping[str, AmberCut]],
        *,
        force_field: Literal["gaff", "gaff2"] = "gaff",
        charge_method: str = "bcc",
        work_dir: Path | str = "amber_work",
        env: str | Path | None = None,
        env_manager: str | None = None,
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
        self.force_field = force_field
        self.charge_method = charge_method
        self.net_charges = dict(net_charges) if net_charges is not None else None
        self.work_dir = Path(work_dir).resolve()
        spec = EnvSpec.resolve(env, env_manager)
        self.env = spec.env
        self.env_manager = spec.env_manager

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
            RuntimeError: An AmberTools step failed or is not installed.
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
        ac = directory / f"{label}.ac"
        mol2 = directory / f"{label}.mol2"
        frcmod = directory / f"{label}.frcmod"
        source = directory / f"{label}.input.mol2"

        oligomer = self.library[label]
        charge = _net_charge(oligomer, label, self.net_charges)
        _write_mol2(oligomer, source)
        digest = _digest(
            source.read_bytes(),
            f"{self.force_field}|{self.charge_method}|{charge}".encode(),
        )
        stamp = directory / _STAMP
        outputs = (ac, mol2, frcmod)
        stale = not all(path.is_file() for path in outputs) or (
            stamp.is_file() and stamp.read_text() != digest
        )
        if stale:
            stamp.unlink(missing_ok=True)
            self._antechamber(directory, source, ac, mol2, frcmod, charge)
            stamp.write_text(digest)

        frame = read_ac(ac)
        ac_names = [str(name) for name in frame["atoms"]["name"]]
        ac_types = dict(zip(ac_names, map(str, frame["atoms"]["type"]), strict=True))
        renamed = _rename(_atom_names(oligomer), ac_names)

        prepgen = PrepgenWrapper(
            name="prepgen",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        prepi: dict[Variant, Path] = {}
        for variant in _VARIANTS:
            if variant not in variants:
                continue
            cut = self.cuts[label][variant]
            control = directory / f"{label}.{variant}"
            previous = control.read_text() if control.is_file() else None
            write_prepgen_control_file(
                control,
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
            output = directory / _prepi_name(label, variant)
            prepi[variant] = output
            if not stale and output.is_file() and control.read_text() == previous:
                continue
            output.unlink(missing_ok=True)
            run_step(
                prepgen,
                output,
                lambda: prepgen.generate_residue(
                    input_file=ac.name,
                    output_file=output.name,
                    control_file=control.name,
                    residue_name=_resname(label, variant),
                ),
            )
        return _Prepared(frcmod, prepi)

    def _antechamber(
        self,
        directory: Path,
        source: Path,
        ac: Path,
        mol2: Path,
        frcmod: Path,
        charge: int,
    ) -> None:
        antechamber = AntechamberWrapper(
            name="antechamber",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        for output, fmt in ((mol2, "mol2"), (ac, "ac")):
            output.unlink(missing_ok=True)
            run_step(
                antechamber,
                output,
                lambda output=output, fmt=fmt: antechamber.atomtype_assign(
                    input_file=source,
                    output_file=output,
                    input_format="mol2",
                    output_format=fmt,
                    charge_method=self.charge_method,  # type: ignore[arg-type]
                    atom_type=self.force_field,
                    net_charge=charge,
                ),
            )
        parmchk2 = Parmchk2Wrapper(
            name="parmchk2",
            workdir=directory,
            env=self.env,
            env_manager=self.env_manager,
        )
        frcmod.unlink(missing_ok=True)
        run_step(
            parmchk2,
            frcmod,
            lambda: parmchk2.generate_parameters(
                input_file=mol2, output_file=frcmod, force_field=self.force_field
            ),
        )

    def _tleap(
        self, labels: list[str], prepared: dict[str, _Prepared]
    ) -> AmberBuildResult:
        from molpy.io._readers import read_amber

        sequence = " ".join(
            _resname(label, _variant(index, len(labels)))
            for index, label in enumerate(labels)
        )
        loads = [f"source leaprc.{self.force_field}"]
        loaded: list[Path] = []
        for label in sorted(prepared):
            loads.append(f"loadamberparams {prepared[label].frcmod}")
            loaded.append(prepared[label].frcmod)
            for path in prepared[label].prepi.values():
                loads.append(f"loadamberprep {path}")
                loaded.append(path)
        digest = _digest(
            "\n".join(loads).encode(),
            sequence.encode(),
            *(path.read_bytes() for path in loaded),
        )
        directory = self.work_dir / "chains" / digest[:16]
        prmtop = directory / "polymer.prmtop"
        inpcrd = directory / "polymer.inpcrd"
        if not (prmtop.is_file() and inpcrd.is_file()):
            directory.mkdir(parents=True, exist_ok=True)
            script = "\n".join(
                [
                    *loads,
                    "",
                    f"mol = sequence {{ {sequence} }}",
                    f"saveamberparm mol {prmtop} {inpcrd}",
                    "quit",
                    "",
                ]
            )
            tleap = TLeapWrapper(
                name="tleap",
                workdir=directory,
                env=self.env,
                env_manager=self.env_manager,
            )
            prmtop.unlink(missing_ok=True)
            run_step(
                tleap,
                prmtop,
                lambda: tleap.run_from_script(script, script_name="polymer.in"),
            )
        frame, _ = read_amber(prmtop, inpcrd)
        assignment = _PrmtopAssignment(prmtop)
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


def _write_mol2(oligomer: Atomistic, path: Path) -> None:
    """Write the oligomer as antechamber's input, atom names kept.

    mol2 carries the bonds, so antechamber does not perceive them from
    coordinates (AmberTools 26's bondtype crashes on a PDB with CONECT
    records). A ``type`` bond column is a force-field label, not the SYBYL
    bond order mol2 reads there, so it is dropped.
    """
    frame = oligomer.to_frame()
    if "bonds" in frame and "type" in frame["bonds"]:
        del frame["bonds"]["type"]
    write_mol2(path, frame)


def _digest(*parts: bytes) -> str:
    sha = hashlib.sha1()
    for part in parts:
        sha.update(len(part).to_bytes(8, "little"))
        sha.update(part)
    return sha.hexdigest()


def _named(template: Atomistic) -> Atomistic:
    """Copy ``template`` and give every atom a stable Amber name."""
    if not isinstance(template, Atomistic):
        raise TypeError(
            f"library templates must be Atomistic, got {type(template).__name__}"
        )
    copied = template.copy()
    seen: set[str] = set()
    for index, atom in enumerate(copied.atoms, start=1):
        name = atom.get("name")
        if name is None:
            element = atom.get("element")
            if not element:
                raise ValueError(f"atom {index} has no element and no name")
            name = f"{element}{index}"
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
            "parameterise one molecule with AntechamberTypifier"
        )
    bonds = list(sites.cgbonds)
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
    # The frame schema declares formal_charge an integer, so the sum is one.
    return sum(int(atom.get("formal_charge") or 0) for atom in template.atoms)


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

"""The ``mp.io`` writers molpy owns.

Every other writer on :mod:`molpy.io` is the native one, re-exported by
identity. The functions here add something the native writer does not do:
fill a PDB ``element`` column from frame meta, or write the whole file set of a
LAMMPS ``fix bond/react`` system.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

import molrs.io
from molrs import Block, Frame
from molrs.ff import ForceField, write_lammps_forcefield

from .data.lammps import write_lammps_data
from .data.lammps_bond_react import (
    TYPE_LABEL_SECTIONS,
    BondReactTemplate,
    LammpsBondReactWriter,
)

PathLike = str | Path


def write_pdb(file: PathLike, frame: Frame) -> None:
    """Write a frame to a PDB file (canonical columns).

    When the atoms carry no ``element`` column, one is built from the
    space-separated ``frame.meta["elements"]`` (padded with ``X``).

    Raises:
        ValueError: If ``frame["atoms"]`` lacks ``x``, ``y`` or ``z``.
    """
    atoms = frame["atoms"]
    for field in ("x", "y", "z"):
        if field not in atoms:
            raise ValueError(f"Required field '{field}' is missing in frame['atoms']")
    elements = frame.meta.get("elements")
    if "element" not in atoms and isinstance(elements, str) and elements.strip():
        n = atoms.nrows
        parts = (elements.split() + ["X"] * n)[:n]
        frame = frame.copy()
        columns = {k: np.asarray(atoms[k]) for k in atoms.keys()}
        columns["element"] = np.asarray(parts, dtype="U8")
        frame["atoms"] = Block(columns)
    molrs.io.write_pdb(file, frame)


def write_lammps_bond_react_system(
    workdir: PathLike,
    frame: Frame,
    forcefield: ForceField,
    templates: dict[str, BondReactTemplate] | Sequence[BondReactTemplate],
) -> None:
    """Write a complete LAMMPS fix bond/react system.

    Produces all files needed for a reactive MD simulation:

    - ``{stem}.data`` — system configuration
    - ``{stem}.ff`` — force field coefficients
    - ``{name}_pre.mol`` / ``{name}_post.mol`` — reaction templates
    - ``{name}.map`` — atom equivalence maps

    Type numbering is unified across the system and all templates so
    that ``fix bond/react`` can match atom types correctly. The data file's
    labelmap declares that unified inventory, so ``{stem}.ff`` carries a
    coefficient for every declared label, including labels only a template
    uses.

    Declared debt (law 5, hide decisions): the native
    ``write_lammps_forcefield`` selects coefficients from a frame's type
    labels only, so this writer passes it a synthetic *label frame* — one
    block per category holding only a ``type`` column with the unified
    labels. Its correctness depends on the native writer reading nothing
    but those ``type`` columns. Removal: a native ``write_lammps_forcefield``
    over an explicit label inventory (native-core ask), after which this
    writer passes the inventory instead of a frame.

    Args:
        workdir: Output directory (created if missing).
        frame: Packed system Frame.
        forcefield: Force field holding every unified label.
        templates: Either a ``{name: BondReactTemplate}`` dict, or a
            sequence of templates (named ``rxn1``, ``rxn2``, …).

    Raises:
        ValueError: A unified label has no type in ``forcefield``.

    Example::

        mp.io.write_lammps_bond_react_system(
            "output", packed_frame, ff,
            templates={"rxn1": template},
        )
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Normalise templates to {name: template} dict
    by_name: dict[str, BondReactTemplate] = (
        templates
        if isinstance(templates, dict)
        else {f"rxn{i + 1}": t for i, t in enumerate(templates)}
    )

    # -- Collect template frames --
    tpl_frames: list[tuple[str, BondReactTemplate, Frame, Frame]] = []
    for name, tpl in by_name.items():
        # Assign 1-based atom IDs before converting to frames
        tpl.assign_atom_ids()
        tpl_frames.append((name, tpl, tpl.pre.to_frame(), tpl.post.to_frame()))

    # -- Build unified type maps from ALL frames --
    all_frames = [frame]
    for _, _, pre_f, post_f in tpl_frames:
        all_frames.extend([pre_f, post_f])

    unified, type_maps = LammpsBondReactWriter.collect_type_maps(all_frames)

    # -- Write system .data + .ff --
    file_stem = workdir_path / workdir_path.stem
    write_lammps_data(file_stem.with_suffix(".data"), frame, type_labels=unified)

    # Label frame: the unified inventory as ``type`` columns (declared debt above).
    label_frame = Frame()
    for label_key, section in TYPE_LABEL_SECTIONS.items():
        if unified[label_key]:
            label_frame[section] = {"type": np.asarray(unified[label_key], dtype=str)}
    write_lammps_forcefield(file_stem.with_suffix(".ff"), forcefield, label_frame)

    # -- Write template files --
    for name, tpl, pre_frame, post_frame in tpl_frames:
        # Convert pre/post string types → unified numeric IDs,
        # dropping rows with None type (boundary topology).
        for tpl_frame in [pre_frame, post_frame]:
            LammpsBondReactWriter.apply_type_maps(
                tpl_frame, type_maps, template_name=name
            )

        molrs.io.write_lammps_molecule(workdir_path / f"{name}_pre.mol", pre_frame)
        molrs.io.write_lammps_molecule(workdir_path / f"{name}_post.mol", post_frame)
        LammpsBondReactWriter(workdir_path / name).write_map(tpl)


def write_bond_react_map(template: BondReactTemplate, base_path: PathLike) -> None:
    """Write the ``.map`` file for a LAMMPS ``fix bond/react`` template.

    Writes ``{base_path}.map``.
    """
    LammpsBondReactWriter(base_path).write_map(template)

"""
Data file writer factory functions.

This module provides convenient factory functions for creating various data file writers.
All functions write Frame or ForceField objects to files.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

PathLike = str | Path


# =============================================================================
# Data File Writers
# =============================================================================


def write_lammps_data(
    file: PathLike,
    frame: Any,
    atom_style: str = "full",
    *,
    type_labels: dict[str, list[str]] | None = None,
    forcefield: Any = None,
) -> None:
    """
    Write a Frame object to a LAMMPS data file.

    Args:
        file: Output file path
        frame: Frame object to write
        atom_style: LAMMPS atom style (default: 'full')
        type_labels: Format-owned label inventory, including unused types.
        forcefield: Optional ForceField whose coefficients belong in this data file.
    """
    from .data.lammps import LammpsDataWriter

    writer = LammpsDataWriter(
        Path(file),
        atom_style=atom_style,
        type_labels=type_labels,
        forcefield=forcefield,
    )
    writer.write(frame)


def write_pdb(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to a PDB file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.pdb import PDBWriter

    writer = PDBWriter(Path(file))
    writer.write(frame)


def write_gro(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to a GROMACS GRO file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.gro import GroWriter

    writer = GroWriter(Path(file))
    writer.write(frame)


def write_xsf(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to an XSF file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.xsf import XsfWriter

    writer = XsfWriter(Path(file))
    writer.write(frame)


def write_amber_frcmod(
    file: PathLike,
    *,
    remark: str = "",
    mass: str = "",
    bond: str = "",
    angle: str = "",
    dihe: str = "",
    improper: str = "",
    nonbon: str = "",
) -> None:
    """
    Write an AMBER FRCMOD file.

    FRCMOD files contain additional force field parameters. This function
    creates a properly formatted file with the provided sections.

    Args:
        file: Output file path
        remark: Optional comment/remark line
        mass: MASS section content
        bond: BOND section content
        angle: ANGLE section content
        dihe: DIHEDRAL section content
        improper: IMPROPER section content
        nonbon: NONBON section content
    """
    from .forcefield.frcmod import write_frcmod

    write_frcmod(
        file,
        remark=remark,
        mass=mass,
        bond=bond,
        angle=angle,
        dihe=dihe,
        improper=improper,
        nonbon=nonbon,
    )


def write_lammps_molecule(
    file: PathLike, frame: Any, format_type: str = "native"
) -> None:
    """
    Write a Frame object to a LAMMPS molecule file.

    Args:
        file: Output file path
        frame: Frame object to write
        format_type: Format type (default: 'native')
    """
    from .data.lammps_molecule import LammpsMoleculeWriter

    writer = LammpsMoleculeWriter(Path(file), format_type)
    writer.write(frame)


# =============================================================================
# Force Field Writers
# =============================================================================


def _frame_used_types(frame: Any) -> dict[str, set | None]:
    """Per-section set of the type names a frame actually uses (``None`` if the
    section/column is absent), for whitelisting FF output to the labelmap.

    A LAMMPS data file's type labels come from the frame's used types, so any
    coeff the FF writer emits for a type *not* in this set references a labelmap
    entry that does not exist and LAMMPS rejects it. Both the system writer and
    the engine writer filter against this so their coeffs match the data file.
    """
    sections = {
        "atom_types": "atoms",
        "bond_types": "bonds",
        "angle_types": "angles",
        "dihedral_types": "dihedrals",
        "improper_types": "impropers",
    }
    used: dict[str, set | None] = {}
    for key, section in sections.items():
        if section in frame and "type" in frame[section]:
            used[key] = set(frame[section]["type"])
        else:
            used[key] = None
    return used


def write_lammps_forcefield(
    file: PathLike,
    forcefield: Any,
    precision: int = 6,
    skip_pair_style: bool = False,
    frame: Any = None,
) -> None:
    """
    Write a ForceField object to a LAMMPS force field file.

    Args:
        file: Output file path
        forcefield: ForceField object to write
        precision: Number of decimal places for floating point values
        skip_pair_style: If True, omit the ``pair_style`` line so the calling
            LAMMPS input script can set it independently (e.g. to switch between
            ``lj/cut/coul/cut`` for minimisation and ``lj/cut/coul/long`` for MD).
        frame: When given, restrict emitted coeffs to the types the frame
            actually uses — so a force field carrying extra types (e.g. cap
            artifacts from region parameterisation) does not emit a coeff for a
            type absent from the data file's labelmap (which LAMMPS rejects).
    """
    from .forcefield.lammps import LAMMPSForceFieldWriter

    writer = LAMMPSForceFieldWriter(Path(file), precision=precision)
    used = _frame_used_types(frame) if frame is not None else {}
    writer.write(forcefield, skip_pair_style=skip_pair_style, **used)


# =============================================================================
# Trajectory Writers
# =============================================================================


def write_lammps_trajectory(
    file: PathLike, frames: list, atom_style: str = "full"
) -> None:
    """
    Write frames to a LAMMPS trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
        atom_style: LAMMPS atom style (default: 'full')
    """
    from .trajectory.lammps import LammpsTrajectoryWriter

    with LammpsTrajectoryWriter(Path(file), atom_style) as writer:
        for i, frame in enumerate(frames):
            timestep = getattr(frame, "step", i)
            writer.write_frame(frame, timestep)


def write_xyz_trajectory(file: PathLike, frames: list) -> None:
    """
    Write frames to an XYZ trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
    """
    from .trajectory.xyz import XYZTrajectoryWriter

    with XYZTrajectoryWriter(file) as writer:
        for frame in frames:
            writer.write_frame(frame)


def write_trr(file: PathLike, frames: list) -> None:
    """Write frames to a GROMACS TRR trajectory (single precision).

    Thin delegation to the native molrs writer. Each frame needs ``x``/``y``/
    ``z`` (nm); optional ``vx``/``vy``/``vz`` and ``fx``/``fy``/``fz`` are
    written when present.

    Args:
        file: Output file path.
        frames: List of Frame objects to write.
    """
    import molrs.io

    molrs.io.write_trr(str(file), list(frames))


def write_xtc(file: PathLike, frames: list) -> None:
    """Write frames to a GROMACS XTC (compressed) trajectory.

    Thin delegation to the native molrs writer. Each frame needs ``x``/``y``/
    ``z`` (nm); quantization precision comes from ``frame.meta['precision']``
    when present, else 1000 (0.001 nm).

    Args:
        file: Output file path.
        frames: List of Frame objects to write.
    """
    import molrs.io

    molrs.io.write_xtc(str(file), list(frames))


def write_lammps_system(
    workdir: PathLike, frame: Any, forcefield: Any
) -> dict[str, Path]:
    """
    Write a complete LAMMPS system (data + forcefield) to a directory.

    Args:
        workdir: Output directory path
        frame: Frame object containing structure
        forcefield: ForceField object containing parameters

    Returns:
        Dict with keys ``"data"`` and ``"ff"`` pointing to the written files.
    """
    workdir_path = Path(workdir)
    if not workdir_path.exists():
        workdir_path.mkdir(parents=True, exist_ok=True)

    # Fixed "system" stem inside the directory, so filenames are predictable
    # (write_lammps_system("out") -> out/system.data + out/system.ff).
    file_stem = workdir_path / "system"
    data_path = file_stem.with_suffix(".data")
    write_lammps_data(data_path, frame)

    # Write forcefield, whitelisted to the types the frame's labelmap actually uses.
    from .forcefield.lammps import LAMMPSForceFieldWriter

    ff_path = file_stem.with_suffix(".ff")
    writer = LAMMPSForceFieldWriter(ff_path)
    writer.write(forcefield, **_frame_used_types(frame))

    return {"data": data_path, "ff": ff_path}


def write_lammps_bond_react_system(
    workdir: PathLike,
    frame: Any,
    forcefield: Any,
    templates: "dict[str, Any] | Sequence[Any]",
) -> None:
    """Write a complete LAMMPS fix bond/react system.

    Produces all files needed for a reactive MD simulation:

    - ``{stem}.data`` — system configuration
    - ``{stem}.ff`` — force field coefficients
    - ``{name}_pre.mol`` / ``{name}_post.mol`` — reaction templates
    - ``{name}.map`` — atom equivalence maps

    Type numbering is unified across the system and all templates so
    that ``fix bond/react`` can match atom types correctly.

    Args:
        workdir: Output directory (created if missing).
        frame: Packed system Frame.
        forcefield: ForceField object.
        templates: Either a ``{name: BondReactTemplate}`` dict, or a
            sequence of templates (named ``rxn1``, ``rxn2``, …).

    Example::

        mp.io.write_lammps_bond_react_system(
            "output", packed_frame, ff,
            templates={"rxn1": template},
        )
    """
    from .data.lammps_bond_react import LammpsBondReactWriter

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Normalise templates to {name: template} dict
    if not isinstance(templates, dict):
        templates = {f"rxn{i + 1}": t for i, t in enumerate(templates)}

    # -- Collect template frames --
    tpl_frames: list[tuple[str, Any, Any, Any]] = []
    for name, tpl in templates.items():
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

    from .forcefield.lammps import LAMMPSForceFieldWriter

    LAMMPSForceFieldWriter(file_stem.with_suffix(".ff")).write(
        forcefield,
        atom_types=set(unified["atom_types"]) or None,
        bond_types=set(unified["bond_types"]) or None,
        angle_types=set(unified["angle_types"]) or None,
        dihedral_types=set(unified["dihedral_types"]) or None,
        improper_types=set(unified["improper_types"]) or None,
    )

    # -- Write template files --
    for name, tpl, pre_frame, post_frame in tpl_frames:
        # Convert pre/post string types → unified numeric IDs,
        # dropping rows with None type (boundary topology).
        for tpl_frame in [pre_frame, post_frame]:
            LammpsBondReactWriter.apply_type_maps(
                tpl_frame, type_maps, template_name=name
            )

        write_lammps_molecule(workdir_path / f"{name}_pre.mol", pre_frame)
        write_lammps_molecule(workdir_path / f"{name}_post.mol", post_frame)
        LammpsBondReactWriter(workdir_path / name).write_map(tpl)


def write_bond_react_map(template: Any, base_path: PathLike) -> None:
    """Write the ``.map`` file for a LAMMPS ``fix bond/react`` template.

    Thin factory over :class:`~molpy.io.data.lammps_bond_react.LammpsBondReactWriter`,
    matching the ``write_*`` convention the rest of this module uses.
    """
    from .data.lammps_bond_react import LammpsBondReactWriter

    LammpsBondReactWriter(base_path).write_map(template)


def write_top(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to a GROMACS topology file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.top import TopWriter

    writer = TopWriter(Path(file))
    writer.write(frame)

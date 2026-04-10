"""
Data file writer factory functions.

This module provides convenient factory functions for creating various data file writers.
All functions write Frame or ForceField objects to files.
"""

from pathlib import Path
from typing import Any

PathLike = str | Path


# =============================================================================
# Data File Writers
# =============================================================================


def write_lammps_data(file: PathLike, frame: Any, atom_style: str = "full") -> None:
    """
    Write a Frame object to a LAMMPS data file.

    Args:
        file: Output file path
        frame: Frame object to write
        atom_style: LAMMPS atom style (default: 'full')
    """
    from .data.lammps import LammpsDataWriter

    writer = LammpsDataWriter(Path(file), atom_style=atom_style)
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


def write_h5(
    file: PathLike,
    frame: Any,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write a Frame object to an HDF5 file.

    Args:
        file: Output file path
        frame: Frame object to write
        compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
            Defaults to 'gzip'.
        compression_opts: Compression level (for gzip: 0-9). Defaults to 4.

    Examples:
        >>> frame = Frame(blocks={"atoms": {"x": [0, 1, 2], "y": [0, 0, 0]}})
        >>> write_h5("structure.h5", frame)
    """
    from .data.h5 import HDF5Writer

    writer = HDF5Writer(
        Path(file), compression=compression, compression_opts=compression_opts
    )
    writer.write(frame)


# =============================================================================
# Force Field Writers
# =============================================================================


def write_lammps_forcefield(
    file: PathLike,
    forcefield: Any,
    precision: int = 6,
    skip_pair_style: bool = False,
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
    """
    from .forcefield.lammps import LAMMPSForceFieldWriter

    writer = LAMMPSForceFieldWriter(Path(file), precision=precision)
    writer.write(forcefield, skip_pair_style=skip_pair_style)


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


def write_h5_trajectory(
    file: PathLike,
    frames: list,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write frames to an HDF5 trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
        compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
            Defaults to 'gzip'.
        compression_opts: Compression level (for gzip: 0-9). Defaults to 4.

    Examples:
        >>> frames = [frame0, frame1, frame2]
        >>> write_h5_trajectory("trajectory.h5", frames)
    """
    from .trajectory.h5 import HDF5TrajectoryWriter

    with HDF5TrajectoryWriter(
        Path(file), compression=compression, compression_opts=compression_opts
    ) as writer:
        for frame in frames:
            writer.write_frame(frame)


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

    # Use directory name as file stem
    file_stem = workdir_path / workdir_path.stem
    data_path = file_stem.with_suffix(".data")
    write_lammps_data(data_path, frame)

    # Extract type names from frame to create whitelist
    atom_types = None
    bond_types = None
    angle_types = None
    dihedral_types = None
    improper_types = None

    if "atoms" in frame and "type" in frame["atoms"]:
        atom_types = set(frame["atoms"]["type"])

    if "bonds" in frame and "type" in frame["bonds"]:
        bond_types = set(frame["bonds"]["type"])

    if "angles" in frame and "type" in frame["angles"]:
        angle_types = set(frame["angles"]["type"])

    if "dihedrals" in frame and "type" in frame["dihedrals"]:
        dihedral_types = set(frame["dihedrals"]["type"])

    if "impropers" in frame and "type" in frame["impropers"]:
        improper_types = set(frame["impropers"]["type"])

    # Write forcefield with whitelist
    from .forcefield.lammps import LAMMPSForceFieldWriter

    ff_path = file_stem.with_suffix(".ff")
    writer = LAMMPSForceFieldWriter(ff_path)
    writer.write(
        forcefield,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
    )

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
    import numpy as np

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Normalise templates to {name: template} dict
    if not isinstance(templates, dict):
        templates = {f"rxn{i + 1}": t for i, t in enumerate(templates)}

    # -- Collect template frames --
    tpl_frames: list[tuple[str, Any, Any, Any]] = []
    for name, tpl in templates.items():
        # Assign 1-based atom IDs before converting to frames
        tpl._assign_atom_ids()
        tpl_frames.append((name, tpl, tpl.pre.to_frame(), tpl.post.to_frame()))

    # -- Build unified type maps from ALL frames --
    all_frames = [frame]
    for _, _, pre_f, post_f in tpl_frames:
        all_frames.extend([pre_f, post_f])

    sections = {
        "atom_types": "atoms",
        "bond_types": "bonds",
        "angle_types": "angles",
        "dihedral_types": "dihedrals",
        "improper_types": "impropers",
    }
    unified: dict[str, list[str]] = {}
    type_maps: dict[str, dict[str, int]] = {}

    for label_key, section in sections.items():
        all_types: set[str] = set()
        for f in all_frames:
            if section in f and f[section].nrows > 0 and "type" in f[section]:
                for t in f[section]["type"]:
                    s = str(t)
                    if s and s != "None" and not s.isdigit():
                        all_types.add(s)
        sorted_types = sorted(all_types)
        unified[label_key] = sorted_types
        type_maps[section] = {name: idx + 1 for idx, name in enumerate(sorted_types)}

    # -- Inject unified type labels so LammpsDataWriter uses them --
    frame.metadata["type_labels"] = unified

    # -- Write system .data + .ff --
    file_stem = workdir_path / workdir_path.stem
    write_lammps_data(file_stem.with_suffix(".data"), frame)

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
            for section, tmap in type_maps.items():
                if section not in tpl_frame or tpl_frame[section].nrows == 0:
                    continue
                block = tpl_frame[section]
                if "type" not in block:
                    continue
                # Keep only rows with a valid type
                keep = [i for i in range(block.nrows) if str(block["type"][i]) in tmap]
                if len(keep) < block.nrows:
                    import warnings

                    dropped = block.nrows - len(keep)
                    warnings.warn(
                        f"Dropped {dropped} {section} entries with "
                        f"unrecognized types from template '{name}'. "
                        f"Ensure all template topology is typed "
                        f"(pass typifier to reacter.run()).",
                        stacklevel=2,
                    )
                    for key in list(block.keys()):
                        block[key] = block[key][keep]
                # Map to numeric IDs
                block["type"] = np.array(
                    [tmap[str(block["type"][i])] for i in range(block.nrows)],
                    dtype=np.int64,
                )

        write_lammps_molecule(workdir_path / f"{name}_pre.mol", pre_frame)
        write_lammps_molecule(workdir_path / f"{name}_post.mol", post_frame)
        tpl.write_map(workdir_path / name)


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

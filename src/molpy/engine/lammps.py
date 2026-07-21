"""LAMMPS molecular dynamics engine.

Wraps the `LAMMPS <https://www.lammps.org>`_ molecular dynamics code.
The engine writes an input script to the working directory and runs::

    [launcher...] lmp -in <input> -log log.lammps -screen none

The ``-screen none`` flag suppresses duplicate stdout output; all
per-timestep data is written exclusively to *log.lammps*.

MPI and scheduler launchers are configured on the :class:`~molpy.engine.base.Engine`
base class::

    engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
    engine = LAMMPSEngine("lmp", launcher=["srun", "--ntasks=16"])

Reference:
    Thompson, A. P. et al. (2022). LAMMPS — A flexible simulation tool for
    particle-based materials modeling. *Comput. Phys. Commun.* **271**, 108171.
    https://doi.org/10.1016/j.cpc.2021.108171
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import Engine

if TYPE_CHECKING:
    from molpy.core.forcefield import ForceField
    from molrs import Frame

# Common LAMMPS binary names, tried in order when no executable is given.
_LAMMPS_CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi")


class LAMMPSEngine(Engine):
    """LAMMPS molecular dynamics engine.

    Runs LAMMPS input scripts.  The engine binary is typically named ``lmp``,
    ``lmp_serial``, or ``lmp_mpi`` depending on the build.

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import LAMMPSEngine
        >>>
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\nrun 0\\n",
        ...     language="other",
        ... )
        >>> engine = LAMMPSEngine(executable="lmp", check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0

        MPI execution::

            engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
            result = engine.run(script, workdir="./calc")
    """

    def __init__(
        self,
        executable: str | None = None,
        *,
        check_executable: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise the LAMMPS engine.

        Differs from :class:`~molpy.engine.base.Engine` only in that
        *executable* is optional: when omitted, the first binary found on
        ``PATH`` among ``lmp``, ``lmp_serial``, ``lmp_mpi`` is used, so
        ``LAMMPSEngine()`` works out of the box on a typical install.

        Args:
            executable: Path or command to the LAMMPS binary.  ``None``
                auto-detects (see above).
            check_executable: Verify the resolved executable is on ``PATH``.
            **kwargs: Forwarded to :class:`~molpy.engine.base.Engine`
                (``workdir``, ``launcher``, ``env_vars``, ``env``,
                ``env_manager``).
        """
        if executable is None:
            executable = next(
                (c for c in _LAMMPS_CANDIDATES if shutil.which(c)),
                _LAMMPS_CANDIDATES[0],
            )
        super().__init__(executable, check_executable=check_executable, **kwargs)

    @property
    def name(self) -> str:
        """Return ``"LAMMPS"``.

        Returns:
            Engine identifier string.
        """
        return "LAMMPS"

    def _get_default_extension(self) -> str:
        """Return ``".lmp"`` — the conventional LAMMPS input extension.

        Returns:
            ``".lmp"``
        """
        return ".lmp"

    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run LAMMPS in *run_dir*.

        Builds the command::

            [launcher...] lmp -in <input_file> -log log.lammps -screen none

        ``-screen none`` prevents LAMMPS from writing timestep data to stdout
        (it still goes to *log.lammps*), avoiding pipe-buffer deadlocks when
        the caller captures output.

        Args:
            run_dir: Directory containing the input files; used as ``cwd``.
            capture_output: Capture stdout/stderr.
            check: Raise :exc:`subprocess.CalledProcessError` on failure.
            timeout: Timeout in seconds.
            **kwargs: Ignored (reserved for future use).

        Returns:
            :class:`subprocess.CompletedProcess`.

        Raises:
            RuntimeError: If no input script has been registered.
            subprocess.CalledProcessError: If *check* is ``True`` and LAMMPS
                exits with a non-zero code.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError("No input script found.  Pass a script to run() first.")

        input_file = self.input_script.path.name
        command = self._build_full_command(
            ["-in", input_file, "-log", "log.lammps", "-screen", "none"]
        )

        return subprocess.run(
            command,
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=self._merged_env(),
        )

    # ------------------------------------------------------------------
    # High-level structure relaxation (frame in -> relaxed frame out)
    # ------------------------------------------------------------------

    def minimize(
        self,
        frame: Frame,
        ff: ForceField,
        *,
        etol: float = 1.0e-4,
        ftol: float = 1.0e-6,
        max_iter: int = 1000,
        max_eval: int = 10000,
        pair_style: str = "lj/cut/coul/cut 10.0",
        atom_style: str = "full",
        units: str = "real",
        workdir: str | Path | None = None,
        capture_output: bool = False,
        timeout: float | None = None,
    ) -> Frame:
        """Energy-minimise *frame* under force field *ff* and return a new frame.

        Writes a LAMMPS data file and coefficient settings from *frame* / *ff*,
        runs ``minimize``, then splices the relaxed coordinates back onto a copy
        of *frame* (topology, types, and box preserved). *frame* is not mutated.

        Typical use is removing residual overlaps after packing::

            eng = LAMMPSEngine()
            relaxed = eng.minimize(pack_result.frame, ff)

        Args:
            frame: Input structure; must carry a periodic box (``frame.box``).
            ff: Typified force field providing pair/bond/angle/... coefficients.
            etol: Energy stopping tolerance (unitless).
            ftol: Force stopping tolerance (force units).
            max_iter: Maximum minimiser iterations.
            max_eval: Maximum force/energy evaluations.
            pair_style: LAMMPS ``pair_style`` line for minimisation.  The default
                ``lj/cut/coul/cut`` avoids a long-range solver; switch to
                ``lj/cut/coul/long`` (with a ``kspace_style``) for production MD.
            atom_style: LAMMPS ``atom_style`` (``full`` by default).
            units: LAMMPS ``units`` (``real`` by default).
            workdir: Directory for input/output files; a temporary directory is
                created when ``None``.
            capture_output: Capture LAMMPS stdout/stderr.
            timeout: Subprocess timeout in seconds.

        Returns:
            A new :class:`~molrs.Frame` with relaxed coordinates.

        Raises:
            ValueError: If *frame* has no box.
            subprocess.CalledProcessError: If LAMMPS exits non-zero.
            RuntimeError: If LAMMPS produces no output structure.
        """
        body = f"minimize {etol:g} {ftol:g} {int(max_iter)} {int(max_eval)}"
        return self._relax(
            frame,
            ff,
            body,
            thermo=max(1, int(max_iter) // 10),
            pair_style=pair_style,
            atom_style=atom_style,
            units=units,
            workdir=workdir,
            capture_output=capture_output,
            timeout=timeout,
        )

    def md(
        self,
        frame: Frame,
        ff: ForceField,
        *,
        ensemble: str = "nve",
        steps: int = 1000,
        temperature: float = 300.0,
        timestep: float = 1.0,
        seed: int = 12345,
        limit: float = 0.1,
        pair_style: str = "lj/cut/coul/cut 10.0",
        atom_style: str = "full",
        units: str = "real",
        workdir: str | Path | None = None,
        capture_output: bool = False,
        timeout: float | None = None,
    ) -> Frame:
        """Run short MD on *frame* under *ff* and return a new frame.

        A thin sibling of :meth:`minimize` for settling a packed box.  Note that
        a freshly packed box carries residual clashes; run :meth:`minimize`
        first, or use ``ensemble="nve/limit"``, to avoid a blow-up under plain
        ``nve``.

        Args:
            frame: Input structure; must carry a periodic box (``frame.box``).
            ff: Typified force field.
            ensemble: One of ``"nve"``, ``"nve/limit"``, ``"nvt"``.
            steps: Number of MD steps.
            temperature: Initial / target temperature (K).
            timestep: Timestep in *units* time (fs for ``real``).
            seed: RNG seed for the initial velocity distribution.
            limit: Per-step displacement cap (Å) for ``ensemble="nve/limit"``.
            pair_style: LAMMPS ``pair_style`` line.
            atom_style: LAMMPS ``atom_style``.
            units: LAMMPS ``units``.
            workdir: Working directory; temporary when ``None``.
            capture_output: Capture LAMMPS stdout/stderr.
            timeout: Subprocess timeout in seconds.

        Returns:
            A new :class:`~molrs.Frame` with the post-MD coordinates.

        Raises:
            ValueError: If *ensemble* is unknown or *frame* has no box.
        """
        fixes = {
            "nve": "fix integ all nve",
            "nve/limit": f"fix integ all nve/limit {limit:g}",
            "nvt": (
                f"fix integ all nvt temp {temperature:g} {temperature:g} "
                f"{100 * timestep:g}"
            ),
        }
        if ensemble not in fixes:
            raise ValueError(
                f"ensemble must be one of {sorted(fixes)}, got {ensemble!r}."
            )
        body = "\n".join(
            [
                f"velocity all create {temperature:g} {int(seed)} loop geom",
                fixes[ensemble],
                f"timestep {timestep:g}",
                f"run {int(steps)}",
                "unfix integ",
            ]
        )
        return self._relax(
            frame,
            ff,
            body,
            thermo=max(1, int(steps) // 10),
            pair_style=pair_style,
            atom_style=atom_style,
            units=units,
            workdir=workdir,
            capture_output=capture_output,
            timeout=timeout,
        )

    def _relax(
        self,
        frame: Frame,
        ff: ForceField,
        body: str,
        *,
        thermo: int,
        pair_style: str,
        atom_style: str,
        units: str,
        workdir: str | Path | None,
        capture_output: bool,
        timeout: float | None,
    ) -> Frame:
        """Shared driver behind :meth:`minimize` / :meth:`md`.

        Writes inputs, runs the *body* of LAMMPS commands wrapped in a standard
        init + ``read_data`` + ``write_data`` scaffold, then reads back and
        splices the relaxed coordinates onto a copy of *frame*.
        """
        from molpy.core.script import Script
        from molpy.io.data.lammps import LammpsDataReader, LammpsDataWriter
        from molpy.io.writers import write_lammps_forcefield

        if frame.box is None:
            raise ValueError(
                "LAMMPS relaxation needs a periodic box on the frame. Set it via "
                "molpack's `with_periodic_box(...)` or assign `frame.box`. "
                "Box-free / shrink-wrap relaxation is not supported yet."
            )

        run_dir = (
            Path(workdir)
            if workdir is not None
            else (self.work_dir or Path(tempfile.mkdtemp()))
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        data_name, settings_name, out_name = (
            "system.data",
            "system.in.settings",
            "relaxed.data",
        )

        LammpsDataWriter(run_dir / data_name, atom_style=atom_style).write(frame)
        # Whitelist coeffs to the frame's used types: the data file's labelmap is
        # built from `frame`, so a coeff for a type the frame lacks (e.g. an `oh`
        # cap artifact left in a merged ff) would reference a missing labelmap
        # entry and LAMMPS would abort.
        write_lammps_forcefield(
            run_dir / settings_name, ff, skip_pair_style=True, frame=frame
        )

        text = _RELAX_TEMPLATE.format(
            units=units,
            atom_style=atom_style,
            pair_style=pair_style,
            styles="\n".join(_style_lines(ff)),
            data=data_name,
            settings=settings_name,
            thermo=int(thermo),
            body=body,
            out=out_name,
        )
        self.run(
            Script.from_text("relax", text, language="other"),
            workdir=run_dir,
            capture_output=capture_output,
            check=True,
            timeout=timeout,
        )

        out_path = run_dir / out_name
        if not out_path.exists():
            raise RuntimeError(
                f"LAMMPS finished but did not write {out_path}; "
                f"inspect {run_dir / 'log.lammps'}."
            )
        relaxed = LammpsDataReader(out_path, atom_style=atom_style).read().frame
        return _splice_coords(frame, relaxed)


# Standard scaffold around a minimise / MD command block.  Styles are emitted
# before ``read_data`` (required for reading topology sections); the settings
# file (included after) supplies the matching coefficients.
_RELAX_TEMPLATE = """\
# molpy-generated LAMMPS relaxation script
units {units}
atom_style {atom_style}
boundary p p p
pair_style {pair_style}
{styles}
read_data {data}
include {settings}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {thermo}
thermo_style custom step temp pe ke etotal press
{body}
write_data {out} nocoeff
"""


def _style_lines(ff: ForceField) -> list[str]:
    """Return ``bond_style``/``angle_style``/... lines derived from *ff*.

    Emitted before ``read_data`` so LAMMPS can allocate the topology arrays.
    The pair style is supplied separately by the caller (it is overridden for
    minimisation), so it is intentionally absent here.
    """
    from molpy.io.emit.lammps import _collect_style_names

    lines: list[str] = []
    for kind, command in (
        ("bond", "bond_style"),
        ("angle", "angle_style"),
        ("dihedral", "dihedral_style"),
        ("improper", "improper_style"),
    ):
        names = _collect_style_names(ff, kind)
        if names:
            lines.append(f"{command} {names[0]}")
    return lines


def _splice_coords(original: Frame, relaxed: Frame) -> Frame:
    """Return a copy of *original* with coordinates taken from *relaxed*.

    Coordinates are matched by atom ``id`` when *original* carries one,
    otherwise by row order (LAMMPS assigns ``id`` sequentially on write). All
    other columns, topology blocks, and the box come from *original*; neither
    input is mutated.
    """
    import molrs
    import numpy as np

    rid = np.asarray(relaxed["atoms"].view("id"))
    rx = np.asarray(relaxed["atoms"].view("x"))
    ry = np.asarray(relaxed["atoms"].view("y"))
    rz = np.asarray(relaxed["atoms"].view("z"))

    data = original.to_dict()
    atoms = data["blocks"]["atoms"]
    n = len(atoms["x"])
    if len(rx) != n:
        raise RuntimeError(
            f"atom count changed during relaxation: {n} in, {len(rx)} out."
        )

    if "id" in atoms:
        row_of = {int(i): k for k, i in enumerate(rid)}
        sel = [row_of[int(i)] for i in np.asarray(atoms["id"])]
    else:
        sel = list(np.argsort(rid, kind="stable"))

    atoms["x"], atoms["y"], atoms["z"] = rx[sel], ry[sel], rz[sel]
    new = molrs.Frame.from_dict(data)
    new.box = original.box
    return new

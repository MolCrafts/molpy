"""LAMMPS molecular dynamics engine.

Wraps the `LAMMPS <https://www.lammps.org>`_ molecular dynamics code.
The engine writes an input script to the working directory and runs::

    [launcher...] lmp -in <input> -log log.lammps -screen none

The ``-screen none`` flag suppresses duplicate stdout output; all
per-timestep data is written exclusively to *log.lammps*.

:meth:`LammpsEngine.generate_inputs` is the one LAMMPS deck writer: data
file, force-field settings, init and input script, for a periodic frame or a
box-free one. :meth:`LammpsEngine.minimize` and :meth:`LammpsEngine.md` run
the same deck with their own command block.

MPI and scheduler launchers are configured on the :class:`~molpy.engine.Engine`
base class::

    engine = LammpsEngine("lmp", launcher=["mpirun", "-np", "16"])
    engine = LammpsEngine("lmp", launcher=["srun", "--ntasks=16"])

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

from molrs.io import read_lammps_data, write_lammps_data, write_lammps_forcefield_str

from ._engine import Engine
from ._script import Script

if TYPE_CHECKING:
    from molrs.ff.forcefield import ForceField
    from molrs.core import Frame

# Common LAMMPS binary names, tried in order when no executable is given.
_LAMMPS_CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi")


class LammpsEngine(Engine):
    """LAMMPS molecular dynamics engine.

    Runs LAMMPS input scripts.  The engine binary is typically named ``lmp``,
    ``lmp_serial``, or ``lmp_mpi`` depending on the build.

    Example:
        >>> from molpy.engine import Script
        >>> from molpy.engine import LammpsEngine
        >>>
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\nrun 0\\n",
        ...     language="other",
        ... )
        >>> engine = LammpsEngine(executable="lmp", check_executable=False)
        >>> result = engine.run(script, workdir="./calc", check=False)
        >>> print(result.returncode)
        0

        MPI execution::

            engine = LammpsEngine("lmp", launcher=["mpirun", "-np", "16"])
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

        Differs from :class:`~molpy.engine.Engine` only in that
        *executable* is optional: when omitted, the first binary found on
        ``PATH`` among ``lmp``, ``lmp_serial``, ``lmp_mpi`` is used, so
        ``LammpsEngine()`` works out of the box on a typical install.

        Args:
            executable: Path or command to the LAMMPS binary.  ``None``
                auto-detects (see above).
            check_executable: Verify the resolved executable is on ``PATH``.
            **kwargs: Forwarded to :class:`~molpy.engine.Engine`
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
            env=self._merged_environment(),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Input generation (the one LAMMPS deck writer)
    # ------------------------------------------------------------------

    def generate_inputs(
        self,
        frame: Frame,
        forcefield: ForceField,
        output_dir: str | Path,
        *,
        prefix: str = "system",
        atom_style: str = "full",
        units: str = "real",
        pair_style: str | None = None,
        body: str | None = None,
    ) -> dict[str, Path]:
        """Write a complete LAMMPS deck for *frame* under *forcefield*.

        Files written (given ``prefix="system"``):

        * ``system.data`` — the structure (molrs's LAMMPS data writer).
        * ``system.in.settings`` — molrs's LAMMPS force-field include: the
          ``*_style`` line of every category the frame uses (built-in or a
          registered force-field IR style; ``hybrid`` when a category spans
          several styles), its coefficients, ``special_bonds`` and
          ``pair_modify``. It is included after ``read_data``, where LAMMPS
          rejects ``units``, so ``units`` is the init's.
        * ``system.in.init`` — ``units``, ``atom_style``, ``boundary``,
          ``neighbor`` and, when *pair_style* is given, that ``pair_style``
          line (the include then leaves its own out).
        * ``system.in`` — the input script: init, ``read_data``, settings,
          then *body* (a starter minimise + NVT run by default).

        A frame with a periodic box gets ``boundary p p p`` and binned
        neighbour lists. A frame without one (``Atomistic.to_frame()``
        carries no box) gets a non-periodic deck: molrs's data writer puts
        the atoms inside the bounds of their coordinates widened by 1 length
        unit on every side, ``boundary s s s`` shrink-wraps that box to the
        atoms, and ``neighbor 2.0 nsq`` searches all pairs, since binning
        needs a box wider than the pair cutoff.

        molpy names no style itself: a style molrs can write is emitted and
        one it cannot is refused by molrs, by name, before any file is
        written.

        Args:
            frame: Typed structure (``type`` columns keyed to *forcefield*).
            forcefield: The force field the settings are written from.
            output_dir: Directory for the deck (created if absent).
            prefix: Stem of the four file names.
            atom_style: LAMMPS ``atom_style``.
            units: LAMMPS ``units`` style; the coefficients are written in it.
            pair_style: A ``pair_style`` line (without the command) that
                replaces the force field's own.
            body: Commands after the settings include; ``None`` writes the
                starter minimise + NVT run.

        Returns:
            ``{"data", "settings", "init", "input"}`` mapped to the written
            paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "data": out / f"{prefix}.data",
            "settings": out / f"{prefix}.in.settings",
            "init": out / f"{prefix}.in.init",
            "input": out / f"{prefix}.in",
        }

        # Settings first: a style molrs cannot write fails before any file is
        # written. They carry the coefficients of the labels `frame` uses, the
        # same labels the data file declares.
        settings = write_lammps_forcefield_str(
            forcefield,
            frame,
            skip_pair_style=pair_style is not None,
            skip_units=True,
            units=units,
        )
        write_lammps_data(paths["data"], frame)
        paths["settings"].write_text(settings, encoding="utf-8")

        periodic = frame.box is not None and not frame.box.is_free
        init = [
            f"# molpy-generated LAMMPS init for {prefix}",
            f"units {units}",
            f"atom_style {atom_style}",
            "boundary p p p" if periodic else "boundary s s s",
            "neighbor 2.0 bin" if periodic else "neighbor 2.0 nsq",
        ]
        if pair_style is not None:
            init.append(f"pair_style {pair_style}")
        paths["init"].write_text("\n".join(init) + "\n", encoding="utf-8")

        paths["input"].write_text(
            _INPUT_TEMPLATE.format(
                prefix=prefix,
                init=paths["init"].name,
                data=paths["data"].name,
                settings=paths["settings"].name,
                body=(_STARTER_BODY if body is None else body).rstrip("\n"),
            ),
            encoding="utf-8",
        )
        return paths

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

            eng = LammpsEngine()
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
            A new :class:`~molpy.Frame` with relaxed coordinates.

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
            A new :class:`~molpy.Frame` with the post-MD coordinates.

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

        Writes the deck with :meth:`generate_inputs` (the script's
        *pair_style*, the force field's ``special_bonds`` and ``pair_modify``
        mix), runs *body* followed by ``write_data``, then reads back and
        splices the relaxed coordinates onto a copy of *frame*.
        """
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
        out_name = "relaxed.data"
        paths = self.generate_inputs(
            frame,
            ff,
            run_dir,
            atom_style=atom_style,
            units=units,
            pair_style=pair_style,
            body="\n".join(
                [
                    f"thermo {int(thermo)}",
                    "thermo_style custom step temp pe ke etotal press",
                    body,
                    f"write_data {out_name} nocoeff",
                ]
            ),
        )
        self.run(
            Script.from_path(paths["input"]),
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
        relaxed = read_lammps_data(out_path, atom_style=atom_style)
        return _splice_coords(frame, relaxed)


# The input script around a command block: the init, the structure, then the
# settings include (molrs's LAMMPS force-field writer: every ``*_style`` line
# with its coefficients, ``special_bonds`` and ``pair_modify``).
_INPUT_TEMPLATE = """\
# molpy-generated LAMMPS input for {prefix}
include {init}
read_data {data}
include {settings}
neigh_modify every 1 delay 0 check yes

{body}
"""

# The default body: minimise, then a short NVT run to edit.
_STARTER_BODY = """\
minimize        1.0e-4 1.0e-6 1000 10000

velocity        all create 300.0 12345 loop geom
fix             1 all nvt temp 300.0 300.0 100.0
timestep        1.0
thermo          100
thermo_style    custom step temp pe ke etotal press
run             1000
unfix           1
"""


def _splice_coords(original: Frame, relaxed: Frame) -> Frame:
    """Return a copy of *original* with coordinates taken from *relaxed*.

    Coordinates are matched by atom ``id`` when *original* carries one,
    otherwise by row order (LAMMPS assigns ``id`` sequentially on write). All
    other columns, topology blocks, and the box come from *original*; neither
    input is mutated.
    """
    import numpy as np

    relaxed_atoms = relaxed["atoms"]
    rid = relaxed_atoms["id"]

    new = original.copy()
    atoms = new["atoms"]
    n = atoms.n_rows
    if relaxed_atoms.n_rows != n:
        raise RuntimeError(
            f"atom count changed during relaxation: {n} in, {relaxed_atoms.n_rows} out."
        )

    if "id" in atoms:
        row_of = {int(i): k for k, i in enumerate(rid)}
        sel = np.array([row_of[int(i)] for i in atoms["id"]], dtype=np.intp)
    else:
        sel = np.argsort(rid, kind="stable")

    atoms["x", "y", "z"] = relaxed_atoms["x", "y", "z"][sel]
    return new

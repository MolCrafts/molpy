"""Run one external program, log it, and raise the one tool failure.

The one place an engine (:mod:`molpy.engine`), an AmberTools program
(:mod:`molpy.tool.ambertools.programs`) or an environment lookup
(:meth:`~molpy.tool.environment.EnvironmentSpec.resolve_executable`) starts a
subprocess. Every record carries ``command`` (the argv list) and ``cwd``,
plus the caller's fields (an engine's ``step``):

* ``process started`` (DEBUG), before the program starts;
* ``process finished`` (INFO, exit 0) or ``process failed`` (ERROR, any other
  exit), with ``returncode`` and ``elapsed_s``;
* ``process timed out`` (ERROR), with ``timeout_s`` and ``elapsed_s``, when
  the time limit kills it;
* ``process not started`` (ERROR), with ``error``, when the OS cannot start
  it (the executable is missing).

A probe (``quiet=True``) logs every record at DEBUG: its non-zero exit is an
answer, not a failure.

Every failure is a :class:`ToolError`: the program could not start, timed
out, or (with ``check``) exited non-zero.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mollog

#: Lines of stderr, stdout or log file a :class:`ToolError` carries.
TAIL_LINES = 20


class ToolError(RuntimeError):
    """An external tool failed.

    It is missing, could not start, timed out, exited non-zero, or left an
    output it must write unwritten.

    Attributes:
        tool: The tool's settings table, ``"<group>.<name>"``
            (``"engine.lammps"``, ``"ambertools.tleap"``).
        reason: What went wrong, in one line.
        returncode: The exit status; ``None`` when the program never exited
            on its own (missing, not started, timed out).
        tail: The last lines of its stderr, its stdout, or its log file
            (whichever is the first non-empty); ``""`` when none was kept.
    """

    def __init__(
        self,
        tool: str,
        reason: str,
        *,
        returncode: int | None = None,
        tail: str = "",
    ) -> None:
        self.tool = tool
        self.reason = reason
        self.returncode = returncode
        self.tail = tail
        message = f"{tool}: {reason}"
        super().__init__(f"{message}\n{tail}" if tail else message)


def tail_of(*sources: str | Path | None, lines: int = TAIL_LINES) -> str:
    """The last *lines* lines of the first non-empty source.

    Args:
        *sources: Captured text, or a log file's path (read when it exists).
        lines: How many trailing lines to keep.
    """
    for source in sources:
        if isinstance(source, Path):
            if not source.is_file():
                continue
            source = source.read_text(encoding="utf-8", errors="replace")
        if source and source.strip():
            return "\n".join(source.rstrip().splitlines()[-lines:])
    return ""


def run_process(
    logger: mollog.Logger,
    command: Sequence[str],
    *,
    tool: str,
    cwd: str | Path | None,
    env: Mapping[str, str] | None,
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_text: str | None = None,
    log: Path | None = None,
    quiet: bool = False,
    **fields: Any,
) -> subprocess.CompletedProcess[str]:
    """:func:`subprocess.run` *command* (text, UTF-8), logging it to *logger*.

    Args:
        logger: The engine's or AmberTools program's logger (``molpy.engine.<name>`` /
            ``molpy.ambertools.<name>``).
        command: The full argv.
        tool: The tool's settings table, named by a :class:`ToolError`.
        cwd: Working directory; ``None`` is the caller's.
        env: The subprocess environment; ``None`` inherits the caller's.
        capture_output: Capture stdout / stderr.
        check: Raise :class:`ToolError` on a non-zero exit.
        timeout: Seconds before the program is killed; ``None`` is no limit.
        input_text: Text sent to stdin.
        log: The program's log file, the error's tail when nothing was
            captured (LAMMPS's ``log.lammps``).
        quiet: Log every record at DEBUG (a probe).
        **fields: Extra structured fields on every record.

    Returns:
        The completed process.

    Raises:
        ToolError: The program could not start, timed out, or (with *check*)
            exited non-zero.
    """
    argv = [str(part) for part in command]
    where = {"command": argv, "cwd": str(cwd) if cwd is not None else None, **fields}
    error = logger.debug if quiet else logger.error
    logger.debug("process started", **where)
    start = time.perf_counter()
    try:
        done = subprocess.run(
            argv,
            cwd=cwd,
            input=input_text,
            capture_output=capture_output,
            text=True,
            check=False,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        error(
            "process timed out",
            timeout_s=timeout,
            elapsed_s=time.perf_counter() - start,
            **where,
        )
        raise ToolError(
            tool,
            f"{argv[0]} timed out after {timeout:g} s",
            tail=tail_of(_text(exc.stderr), _text(exc.stdout), log),
        ) from exc
    except OSError as exc:
        error("process not started", error=str(exc), **where)
        raise ToolError(tool, f"{argv[0]} could not start: {exc}") from exc
    elapsed = time.perf_counter() - start
    if done.returncode == 0:
        (logger.debug if quiet else logger.info)(
            "process finished", returncode=0, elapsed_s=elapsed, **where
        )
        return done
    error("process failed", returncode=done.returncode, elapsed_s=elapsed, **where)
    if check:
        raise ToolError(
            tool,
            f"{argv[0]} exited with status {done.returncode}",
            returncode=done.returncode,
            tail=tail_of(done.stderr, done.stdout, log),
        )
    return done


def _text(output: str | bytes | None) -> str | None:
    """Text of a :exc:`subprocess.TimeoutExpired` stream (bytes on POSIX)."""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output

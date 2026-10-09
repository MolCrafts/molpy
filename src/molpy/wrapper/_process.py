"""Run one external program and log it as structured mollog records.

The one place an engine (:mod:`molpy.engine`) or a wrapper
(:mod:`molpy.wrapper`) starts a subprocess. Every record carries ``command``
(the argv list) and ``cwd``, plus the caller's fields (an engine's ``step``):

* ``process started`` (DEBUG), before the program starts;
* ``process finished`` (INFO, exit 0) or ``process failed`` (ERROR, any other
  exit), with ``returncode`` and ``elapsed_s``;
* ``process timed out`` (ERROR), with ``timeout_s`` and ``elapsed_s``, when
  the time limit kills it;
* ``process not started`` (ERROR), with ``error``, when the OS cannot start
  it (the executable is missing).

The exceptions :func:`subprocess.run` raises propagate unchanged.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mollog


def run_process(
    logger: mollog.Logger,
    command: Sequence[str],
    *,
    cwd: str | Path | None,
    env: Mapping[str, str] | None,
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_text: str | None = None,
    **fields: Any,
) -> subprocess.CompletedProcess[str]:
    """:func:`subprocess.run` *command* (text, UTF-8), logging it to *logger*.

    Args:
        logger: The engine's or wrapper's logger (``molpy.engine.<name>`` /
            ``molpy.wrapper.<name>``).
        command: The full argv.
        cwd: Working directory; ``None`` is the caller's.
        env: The subprocess environment; ``None`` inherits the caller's.
        capture_output: Capture stdout / stderr.
        check: Raise :exc:`subprocess.CalledProcessError` on a non-zero exit.
        timeout: Seconds before the program is killed; ``None`` is no limit.
        input_text: Text sent to stdin.
        **fields: Extra structured fields on every record.

    Returns:
        The completed process.
    """
    argv = [str(part) for part in command]
    where = {"command": argv, "cwd": str(cwd) if cwd is not None else None, **fields}
    logger.debug("process started", **where)
    start = time.perf_counter()
    try:
        done = subprocess.run(
            argv,
            cwd=cwd,
            input=input_text,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "process timed out",
            timeout_s=timeout,
            elapsed_s=time.perf_counter() - start,
            **where,
        )
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(
            "process failed",
            returncode=exc.returncode,
            elapsed_s=time.perf_counter() - start,
            **where,
        )
        raise
    except OSError as exc:
        logger.error("process not started", error=str(exc), **where)
        raise
    elapsed = time.perf_counter() - start
    if done.returncode == 0:
        logger.info("process finished", returncode=0, elapsed_s=elapsed, **where)
    else:
        logger.error(
            "process failed", returncode=done.returncode, elapsed_s=elapsed, **where
        )
    return done

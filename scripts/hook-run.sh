#!/usr/bin/env bash
# Runs ONE heavy hook command: one that compiles, or runs a heavy test suite.
#
#   scripts/hook-run.sh <command...>
#
# When the environment names a runner in MOLCRAFTS_HOOK_RUNNER and this is not
# already inside a Slurm job, the command is handed to that runner. The
# MolCrafts cluster's shared git hooks set it to a script that runs its
# arguments on a compute node, so only these gates wait for Slurm and the cheap
# hooks run in place on the login node. Everywhere else -- another machine,
# CI -- nothing sets it and the command runs right here.
set -euo pipefail
if [ -n "${MOLCRAFTS_HOOK_RUNNER:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    exec "$MOLCRAFTS_HOOK_RUNNER" "$@"
fi
exec "$@"

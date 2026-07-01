#!/usr/bin/env bash
# Guarded release-tag push.
#
# A release tag must be reachable from the canonical master before it is pushed —
# otherwise it is an orphan tag and the release/publish workflow refuses it (this
# is exactly what bit molrs 0.5.1: the tag was pushed while the master push was
# rejected by branch protection, leaving an orphan tag). This script verifies the
# tag is an ancestor of <remote>/master locally, before the push.
#
# Release order (master is branch-protected, so the tag lands AFTER master):
#   1. merge the release commit into master (via PR)
#   2. ./scripts/push-release.sh <remote> <tag>
#
# Usage: ./scripts/push-release.sh main v0.5.1
set -euo pipefail

remote="${1:?usage: push-release.sh <remote> <tag>}"
tag="${2:?usage: push-release.sh <remote> <tag>}"

echo "Fetching ${remote}/master …"
git fetch -q "$remote" master

tag_sha="$(git rev-parse "${tag}^{commit}")"
master_sha="$(git rev-parse "${remote}/master")"

if ! git merge-base --is-ancestor "$tag_sha" "$master_sha"; then
    echo "ERROR: tag ${tag} (${tag_sha}) is NOT reachable from ${remote}/master (${master_sha})." >&2
    echo "       Merge the release commit into master (PR) BEFORE pushing the tag." >&2
    exit 1
fi

echo "OK: ${tag} is reachable from ${remote}/master — pushing tag."
git push "$remote" "$tag"

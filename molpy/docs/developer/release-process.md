# Release Process

molpy is being merged into the molrs repository (`molpy/` there), on the way
to one numpy-shaped project: one `molcrafts-molpy` wheel carrying molrs as its
native core, `molpy._core`. The plan is the molrs repository's
`.claude/specs/molpy-unification.md`.

**Releases are frozen until that merge's API fold (link 03) ships.** No
`molcrafts-molpy` or `molcrafts-molrs` version is tagged in between. The
last standalone molpy release is 0.19.0.

After the fold, a single `v*` tag on the molrs repository publishes the
crate, the npm package and the `molcrafts-molpy` wheel together, from one
version (`[workspace.package].version`). This page will describe that process
once it exists.

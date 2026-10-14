# compute-redesign — brief for each link's implementing agent

## Where to work
- Work only inside the worktree `/Users/roykid/work/molcrafts/molrs/.claude/worktrees/compute-redesign` (branch `feat/compute-redesign`).
- Never touch `/Users/roykid/work/molcrafts/molrs`, which is the main checkout with the user's uncommitted work.
- Do not push. Commit only at the end of your link.

## Read before coding
- `.claude/specs/compute-redesign.md`, all of it: decisions D1–D9, design §1–§8, the "Link 01 audit", the behaviour changes, the chain table (including as-built notes from earlier links) and the testing strategy.
- `.claude/specs/compute-redesign.acceptance.md`.
- `molrs/src/compute/contract.rs`, which holds `Compute`, `Verdict`, `Accumulate`, `Stream`, `Batch`, `Partition` and `FeatureMatrix`.
- `molrs/src/compute/named_enum.rs`. Use `named_enum!` for every user-choice enum.
- `molrs/src/compute/time_series/`. This is link 02's work: `Correlator`, `TimeCorrelation`, `Derivative`, `Unwrap`, `Spectrum`, `Window`, `FrequencyGrid`, and the surfaces `Msd`, `Vacf`, `DipoleAcf`, `CurrentAcf`, …
- `/Users/roykid/work/molcrafts/molrs/AGENTS.md`: OOP, no free functions, inline until the second use, no façades, unit tests next to the code, no third-party oracles.

## Operator rules (non-negotiable)
1. **`new` / `with_*` take only configuration.**
   - Configuration means bins and axes, cutoffs, `max_lag`, kernels, windows, tolerances, seeds and units.
   - **`compute` takes all data:** arrays, `dt`, boxes, neighbor tables, masses, charges, orientations, index lists and upstream results.
   - There are no data-wrapper types the caller must construct.
   - `Compute::Input<'a>` is a per-type plain struct of borrowed fields, written as a struct literal (named arguments). It has no `new`.
2. **No `from_frame` / `from_frames`.**
   - Compute never touches `Frame`, `FrameAccess` or `FrameView`.
   - Compute contains no field-name literals: `"atoms"`, `"x"`, `"mass"`, `"orientations"`, `"bonds"`, …
3. **One implementation per algorithm** (spec §3) and **one entry per quantity** (spec §4).
   - Delete the old copies and aliases; do not keep compatibility shims (molrs is experimental).
   - A chemistry-named surface is allowed only as a thin delegating type over the one engine. Its `compute` is a single line.
4. **No public free functions.** `grep -rnE '^pub fn ' molrs/src/compute` must lose every hit in your scope and gain none.
5. **No silent fallbacks** (spec §5). Each fallback in your scope becomes a specific `ComputeError`, and you add a unit test asserting that variant.
6. **Golden invariance.**
   - Every pre-existing golden numeric literal stays unchanged, except the re-pins the spec assigns to your link under "Behaviour changes".
   - A test moves together with the kernel it tests.
   - Changing a non-golden assertion (e.g. a clamp now erroring) is allowed but must be reported.
7. **The old trait `LegacyCompute`, `Fit`, `Check`, `ComputeResult` and `DescriptorRow` still exist until link 09.** Migrate your scope's types onto the new `Compute`. Code outside your scope may keep using the old traits.

## Verification
The `cargo mrs-*` aliases are mangled by a shell hook, so use these exact commands from the worktree root:
- Tests: `rtk proxy cargo test -p molcrafts-molrs --lib --features full,filesystem,stream -- compute`, then the full run without the filter before finishing.
- Doctests: `rtk proxy cargo test --doc -p molcrafts-molrs --features full,filesystem,stream -- compute`
- Lint: `rtk proxy cargo fmt --all`, then `rtk proxy cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`. Both must end clean.
- Binder crates (`molrs-python`, `molrs-wasm`, `molrs-cxxapi`) may stay broken until link 10. Do not edit them unless your link is 10.
- If a test needs `tests-data/`, run `bash scripts/fetch-test-data.sh` once.

## Finishing your link
1. Update the spec's chain row for your link to read `code-complete <date>` with a one-line as-built note. Record every deviation in the relevant spec section; never diverge silently.
2. Commit with `git add -A && git commit -m "refactor(compute): <summary> (compute-redesign NN)"` and a short body.
3. Your final message is a report of at most 400 words covering:
   - the new and changed public types, with their `new` / `compute` signatures;
   - what you deleted;
   - any assertion literal you changed, and why;
   - spec deviations;
   - test counts and lint status;
   - hand-offs for later links.

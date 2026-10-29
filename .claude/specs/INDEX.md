# Specs

Live specs only. A spec is deleted when its work has landed; git history is the record.

| Spec | State |
|------|-------|
| [molpy-unification](molpy-unification.md) — one project named molpy (numpy-shaped): molrs-python folds into molpy with native `molpy._core`, binders become molwasm/molc/molcxx, molrs-ffi unchanged; repo merge is mechanical | approved 2026-10-22; 01 (molpy 6a2c480), 02 and 03 done locally on molrs `dev`, unpushed; 04 started locally and unpushed (molpack Python host is `molpy`, molvis/molrec/molab/molhub retargeted). molpack's Rust crate and Python extension typecheck against `crates/molrs` 0.19 (`cargo check --all-features --all-targets`, and `python/Cargo.toml`); tests not run. Not force-pushed. npm `@molcrafts/molrs`, Atomiverse's vendored tree, and the `v0.20` tag remain |
| [md-calculator-seam](md-calculator-seam.md) — molpy.md replaces molix.md: CPU integrator in molrs, torch MLIPs from molnex through an ASE-shaped `Calculator` seam (calculator-supplied pinned buffers, one Python call and one device sync per step, zero host allocation); run-level unit preset; Berendsen, Dump/Checkpoint, EnergyDrift | approved; 01–05 implemented 2026-10-29 (uncommitted), see "As built"; next: 06-barostat; open: GH200 transfer mode (09) |
| [compute-redesign](compute-redesign.md) — one `Compute` trait, one entry per quantity, one implementation per algorithm; `signal` folded in; thin derived bindings; molvis/molpy follow | approved — operator rulings 2026-10-14: no from_frame(s); `new` = configuration, `compute` = data (no data-wrapper types); Correlator plus chemistry-named surfaces; next: 01-contract |

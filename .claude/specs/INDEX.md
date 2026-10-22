# Specs

Live specs only. A spec is deleted when its work has landed; git history is the record.

| Spec | State |
|------|-------|
| [molpy-unification](molpy-unification.md) — one project named molpy (numpy-shaped): molrs-python folds into molpy with native `molpy._core`, binders become molwasm/molc/molcxx, molrs-ffi unchanged; repo merge is mechanical | approved 2026-10-22; 01 (molpy 6a2c480), 02 and 03 done locally on molrs `dev`, unpushed, awaiting operator structure review; next: force-push, then 04-downstream |
| [compute-redesign](compute-redesign.md) — one `Compute` trait, one entry per quantity, one implementation per algorithm; `signal` folded in; thin derived bindings; molvis/molpy follow | approved — operator rulings 2026-10-14: no from_frame(s); `new` = configuration, `compute` = data (no data-wrapper types); Correlator plus chemistry-named surfaces; next: 01-contract |

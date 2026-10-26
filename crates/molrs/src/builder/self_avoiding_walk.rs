//! Self-avoiding random walk (SARW) configuration and the multi-chain
//! `generate` driver; the monomer-placement policy it is generic over is
//! [`GrowthStrategy`](super::growth_strategy::GrowthStrategy).

use crate::core::Cell;
use std::fmt;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::growth_strategy::GrowthStrategy;
use super::occupancy::OccupancyGrid;
use crate::core::BoxError;
use crate::core::SimBox;
use crate::core::Trace;
use crate::op::{F, Pbc3};

/// How many attempts a strategy gets to seed the first monomer of a chain
/// before reporting a dead-end for that placement.
pub(crate) const FIRST_POINT_TRIES: usize = 64;

/// How many candidate steps the driver tries per monomer before backtracking.
const STEP_TRIES: usize = 40;

/// Errors returned by [`SelfAvoidingWalk::generate`].
#[derive(Debug, Clone, PartialEq)]
pub enum WalkError {
    /// A configuration field is out of range (non-positive length/density, zero
    /// chain length or chain count, or a box too small for the bond length).
    InvalidConfig(String),
    /// The simulation box could not be constructed.
    BoxError(String),
    /// The walk trapped itself: per-step retries, backtracking, and whole-chain
    /// restarts were all exhausted. `monomer` is the furthest length reached.
    DeadEnd {
        /// Index of the chain that failed to complete.
        chain: usize,
        /// Furthest monomer count reached on that chain across all attempts.
        monomer: usize,
    },
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkError::InvalidConfig(m) => write!(f, "invalid SARW configuration: {m}"),
            WalkError::BoxError(m) => write!(f, "box construction failed: {m}"),
            WalkError::DeadEnd { chain, monomer } => write!(
                f,
                "self-avoiding walk dead-ended on chain {chain} after reaching {monomer} monomers"
            ),
        }
    }
}

impl std::error::Error for WalkError {}

impl From<BoxError> for WalkError {
    fn from(e: BoxError) -> Self {
        WalkError::BoxError(format!("{e:?}"))
    }
}

/// Apply per-axis boundary conditions to a raw candidate grown from `tip`.
///
/// Periodic axes wrap into `[0, a)`; non-periodic axes reflect the step's
/// normal component off the wall (preserving `|candidate - tip|`). The result
/// is always inside the box.
pub(crate) fn apply_boundary(tip: [F; 3], mut cand: [F; 3], a: [F; 3], pbc: Pbc3) -> [F; 3] {
    for ax in 0..3 {
        if pbc[ax] {
            cand[ax] = cand[ax].rem_euclid(a[ax]);
        } else if cand[ax] < 0.0 || cand[ax] >= a[ax] {
            // Elastic reflection: flip the step's normal component about the
            // tip, which keeps the bond length exact.
            cand[ax] = 2.0 * tip[ax] - cand[ax];
            // Guard against a rare double overshoot (bond shorter than the box
            // makes this unreachable in practice).
            cand[ax] = cand[ax].clamp(0.0, a[ax] * (1.0 - 1e-12));
        }
    }
    cand
}

/// Configuration for a periodic/reflective, fixed-bond-length self-avoiding
/// random walk that grows `n_chains` independent chains of `chain_length`
/// monomers each.
///
/// Self-avoidance is decided entirely by an occupancy grid
/// — cell occupancy, never pairwise distance. Boundaries are per-axis: a
/// periodic axis wraps a step to the opposite side; a non-periodic axis
/// reflects the step elastically off the wall (its normal component flips,
/// preserving the bond length). Output coordinates therefore always lie inside
/// the box.
///
/// Construct it as a struct literal and inject a [`GrowthStrategy`] via the
/// `strategy` field, then call [`generate`](SelfAvoidingWalk::generate):
///
/// ```
/// use molrs::builder::{OffLattice, SelfAvoidingWalk};
///
/// let walk = SelfAvoidingWalk {
///     n_chains: 2,
///     chain_length: 20,
///     bond_length: 1.53,
///     target_density: 0.05,
///     pbc: [true, true, true],
///     seed: 9062,
///     strategy: OffLattice { excluded_radius: 1.0 },
/// };
/// let out = walk.generate().unwrap();
/// assert_eq!(out.traces.len(), 2);
/// ```
///
/// `target_density` is in **monomers per unit volume** — mass is out of scope,
/// so the cubic box edge is `a = (n_chains * chain_length / target_density).cbrt()`
/// (a lattice strategy may round it up to stay commensurate).
pub struct SelfAvoidingWalk<S: GrowthStrategy> {
    /// Number of independent chains to grow.
    pub n_chains: usize,
    /// Number of monomers per chain.
    pub chain_length: usize,
    /// Fixed distance between consecutive monomers.
    pub bond_length: F,
    /// Target number density (monomers per unit volume) used to size the box.
    pub target_density: F,
    /// Per-axis boundary flags: `true` = periodic (wrap), `false` = reflective.
    pub pbc: Pbc3,
    /// Seed for the deterministic RNG; equal seeds give identical paths.
    pub seed: u64,
    /// The monomer-placement policy (a struct implementing [`GrowthStrategy`]).
    pub strategy: S,
}

/// The result of [`SelfAvoidingWalk::generate`]: one [`Trace`] per chain plus
/// the box that was used. No topology, chemistry, or IO.
///
/// The walk is the generator. Each chain it grew is a [`Trace`].
pub struct WalkOutput {
    /// One trace per chain, each holding `chain_length` samples, all inside
    /// the box (periodic axes wrapped, reflective axes reflected).
    pub traces: Vec<Trace>,
    /// The cubic periodic/reflective box the traces were grown in.
    pub simbox: SimBox,
}

impl<S: GrowthStrategy> SelfAvoidingWalk<S> {
    /// Grow all chains and return their paths plus the box used.
    ///
    /// Deterministic in `seed`. Returns [`WalkError::InvalidConfig`] for
    /// out-of-range parameters, [`WalkError::BoxError`] if the box cannot be
    /// built, and [`WalkError::DeadEnd`] if a chain cannot be completed within
    /// the retry/backtrack/restart budget.
    pub fn generate(&self) -> Result<WalkOutput, WalkError> {
        if self.bond_length <= 0.0 {
            return Err(WalkError::InvalidConfig("bond_length must be > 0".into()));
        }
        if self.target_density <= 0.0 {
            return Err(WalkError::InvalidConfig(
                "target_density must be > 0".into(),
            ));
        }
        if self.chain_length == 0 {
            return Err(WalkError::InvalidConfig("chain_length must be > 0".into()));
        }
        if self.n_chains == 0 {
            return Err(WalkError::InvalidConfig("n_chains must be > 0".into()));
        }

        let n_total = self.n_chains * self.chain_length;
        let raw_edge = (n_total as F / self.target_density).cbrt();
        let edge = self.strategy.adjust_box_edge(raw_edge, self.bond_length);
        // A bond must be shorter than half the box so wrapping/reflection keeps
        // consecutive monomers exactly `bond_length` apart.
        if edge <= 2.0 * self.bond_length {
            return Err(WalkError::InvalidConfig(
                "box edge too small for bond length; lower the density".into(),
            ));
        }
        let simbox = SimBox::new(Cell::Lengths([edge; 3]), [0.0; 3], self.pbc)?;
        let a = [edge, edge, edge];

        let mode = self.strategy.occupancy_mode(self.bond_length);
        let mut grid = OccupancyGrid::new(mode, &simbox, self.pbc);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut traces: Vec<Trace> = Vec::with_capacity(self.n_chains);

        let max_backtrack = 50 * self.chain_length + 1000;
        const MAX_CHAIN_RESTARTS: usize = 8;

        for c in 0..self.n_chains {
            let mut best_reached = 0usize;
            let mut grown: Option<Vec<[F; 3]>> = None;
            for _ in 0..MAX_CHAIN_RESTARTS {
                if let Some(chain) = self.grow_chain(
                    &simbox,
                    a,
                    &mut grid,
                    &mut rng,
                    max_backtrack,
                    &mut best_reached,
                ) {
                    grown = Some(chain);
                    break;
                }
            }
            let chain = grown.ok_or(WalkError::DeadEnd {
                chain: c,
                monomer: best_reached,
            })?;
            traces.push(Trace::from_points(chain));
        }

        Ok(WalkOutput { traces, simbox })
    }

    /// Grow a single chain with per-step backtracking against the shared grid.
    /// Returns `None` (after un-occupying its own cells) if the backtrack
    /// budget is exhausted, so the caller may restart the chain.
    fn grow_chain(
        &self,
        simbox: &SimBox,
        a: [F; 3],
        grid: &mut OccupancyGrid,
        rng: &mut StdRng,
        max_backtrack: usize,
        best_reached: &mut usize,
    ) -> Option<Vec<[F; 3]>> {
        let mut chain: Vec<[F; 3]> = Vec::with_capacity(self.chain_length);
        let mut backtracks = 0usize;

        while chain.len() < self.chain_length {
            let placed = if let Some(&tip) = chain.last() {
                let mut hit = None;
                for _ in 0..STEP_TRIES {
                    let raw = self.strategy.propose_step(tip, self.bond_length, rng);
                    let cand = apply_boundary(tip, raw, a, self.pbc);
                    if grid.is_free(cand, Some(tip)) {
                        hit = Some(cand);
                        break;
                    }
                }
                hit
            } else {
                let mut hit = None;
                for _ in 0..FIRST_POINT_TRIES {
                    let p = self.strategy.propose_first(simbox, self.bond_length, rng);
                    if grid.is_free(p, None) {
                        hit = Some(p);
                        break;
                    }
                }
                hit
            };

            match placed {
                Some(p) => {
                    grid.insert(p);
                    chain.push(p);
                    if chain.len() > *best_reached {
                        *best_reached = chain.len();
                    }
                }
                None => {
                    if let Some(popped) = chain.pop() {
                        grid.remove(popped);
                    }
                    backtracks += 1;
                    if backtracks > max_backtrack {
                        // Un-occupy everything this attempt placed before giving up.
                        for p in &chain {
                            grid.remove(*p);
                        }
                        return None;
                    }
                }
            }
        }
        Some(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{FccLattice, OffLattice};
    use crate::op::Vec3;

    const B: F = 1.53;

    fn off() -> SelfAvoidingWalk<OffLattice> {
        SelfAvoidingWalk {
            n_chains: 3,
            chain_length: 20,
            bond_length: B,
            target_density: 0.05,
            pbc: [true, true, true],
            seed: 9062,
            strategy: OffLattice {
                excluded_radius: 1.0,
            },
        }
    }

    fn fcc() -> SelfAvoidingWalk<FccLattice> {
        SelfAvoidingWalk {
            n_chains: 3,
            chain_length: 20,
            bond_length: B,
            target_density: 0.05,
            pbc: [true, true, true],
            seed: 9062,
            strategy: FccLattice,
        }
    }

    fn fcc_reflective() -> SelfAvoidingWalk<FccLattice> {
        SelfAvoidingWalk {
            pbc: [false, false, false],
            ..fcc()
        }
    }

    fn off_reflective() -> SelfAvoidingWalk<OffLattice> {
        SelfAvoidingWalk {
            pbc: [false, false, false],
            ..off()
        }
    }

    fn out_off() -> WalkOutput {
        off().generate().unwrap()
    }
    fn out_fcc() -> WalkOutput {
        fcc().generate().unwrap()
    }

    fn pt(v: &Vec3) -> [F; 3] {
        [v[0], v[1], v[2]]
    }

    fn min_image_dist(sb: &SimBox, x: &Vec3, y: &Vec3) -> F {
        let d = sb.mic().apply(crate::op::vec3::sub(pt(y), pt(x)));
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    // ac-005: exact chain count and per-chain length, both strategies.
    #[test]
    fn shape_is_exact() {
        for out in [out_off(), out_fcc()] {
            assert_eq!(out.traces.len(), 3);
            for trace in &out.traces {
                assert_eq!(trace.points().len(), 20usize);
            }
        }
    }

    // ac-001: same seed + config => byte-identical coordinates.
    #[test]
    fn deterministic_under_seed() {
        for (a, b) in [
            (off().generate().unwrap(), off().generate().unwrap()),
            (fcc().generate().unwrap(), fcc().generate().unwrap()),
        ] {
            for (ca, cb) in a.traces.iter().zip(b.traces.iter()) {
                for (pa, pb) in ca.points().iter().zip(cb.points().iter()) {
                    assert_eq!(pt(pa), pt(pb), "coordinates must match exactly");
                }
            }
        }
    }

    // ac-002 / ac-012: consecutive intra-chain monomers are exactly
    // bond_length apart, under periodic wrap and reflective boundaries, for
    // both strategies.
    #[test]
    fn bond_length_invariant() {
        let cases = [
            (out_off(), 1e-9),
            (out_fcc(), 1e-9),
            (off_reflective().generate().unwrap(), 1e-9),
            (fcc_reflective().generate().unwrap(), 1e-9),
        ];
        for (out, tol) in cases {
            for trace in &out.traces {
                for w in trace.points().windows(2) {
                    let d = min_image_dist(&out.simbox, &w[0], &w[1]);
                    assert!((d - B).abs() <= tol, "bond {d} != {B} (tol {tol})");
                }
            }
        }
    }

    // ac-003: OffLattice excluded volume holds under minimum image (the grid
    // BlockClear rule guarantees the geometric separation).
    #[test]
    fn offlattice_excluded_volume() {
        let r = 1.0;
        let out = out_off();
        let all: Vec<&Vec3> = out.traces.iter().flat_map(Trace::points).collect();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let d = min_image_dist(&out.simbox, all[i], all[j]);
                assert!(d >= r - 1e-9, "pair distance {d} < excluded_radius {r}");
            }
        }
    }

    // ac-004: FccLattice never places two monomers closer than the nn spacing,
    // under both periodic and reflective boundaries.
    #[test]
    fn fcc_no_collision() {
        for out in [out_fcc(), fcc_reflective().generate().unwrap()] {
            let all: Vec<&Vec3> = out.traces.iter().flat_map(Trace::points).collect();
            for i in 0..all.len() {
                for j in (i + 1)..all.len() {
                    let d = min_image_dist(&out.simbox, all[i], all[j]);
                    assert!(d >= B - 1e-9, "pair distance {d} < nn spacing {B}");
                }
            }
        }
    }

    // ac-006: OffLattice box volume matches n_total / density exactly; FCC box
    // is the smallest lattice-commensurate box >= that.
    #[test]
    fn density_box_convention() {
        let w = off();
        let n_total = (w.n_chains * w.chain_length) as F;
        let expected = n_total / w.target_density;
        let v = w.generate().unwrap().simbox.volume();
        assert!((v - expected).abs() / expected <= 1e-6, "off volume {v}");

        let fv = fcc().generate().unwrap().simbox.volume();
        assert!(
            fv >= expected - 1e-6,
            "fcc volume {fv} < requested {expected}"
        );
    }

    // ac-012 (boundary): every output coordinate lies inside the box for both
    // periodic and reflective settings, for both strategies.
    #[test]
    fn output_inside_box() {
        for out in [
            out_off(),
            out_fcc(),
            off_reflective().generate().unwrap(),
            fcc_reflective().generate().unwrap(),
        ] {
            let edge = out.simbox.lengths()[0];
            for p in out.traces.iter().flat_map(Trace::points) {
                for c in p {
                    assert!(*c >= 0.0 && *c < edge, "coord {c} out of [0,{edge})");
                }
            }
        }
    }

    // ac-009 (first half): invalid configs return WalkError, no panic.
    #[test]
    fn invalid_config_errors() {
        let bad = |w: SelfAvoidingWalk<OffLattice>| {
            matches!(w.generate(), Err(WalkError::InvalidConfig(_)))
        };
        assert!(bad(SelfAvoidingWalk {
            bond_length: 0.0,
            ..off()
        }));
        assert!(bad(SelfAvoidingWalk {
            target_density: 0.0,
            ..off()
        }));
        assert!(bad(SelfAvoidingWalk {
            chain_length: 0,
            ..off()
        }));
        assert!(bad(SelfAvoidingWalk {
            n_chains: 0,
            ..off()
        }));
    }

    // ac-009 (second half): an over-dense FCC box exhausts retries -> DeadEnd.
    #[test]
    fn exhausted_growth_is_dead_end() {
        let w = SelfAvoidingWalk {
            n_chains: 4,
            chain_length: 50,
            bond_length: B,
            target_density: 6.10,
            pbc: [true, true, true],
            seed: 1,
            strategy: FccLattice,
        };
        assert!(matches!(w.generate(), Err(WalkError::DeadEnd { .. })));
    }

    // ac-007 / ac-008: struct-literal construction with an injected strategy
    // struct; WalkOutput carries only paths + simbox (compile-time contract).
    #[test]
    fn struct_injection_and_output_contract() {
        let out: WalkOutput = SelfAvoidingWalk {
            n_chains: 1,
            chain_length: 5,
            bond_length: B,
            target_density: 0.05,
            pbc: [true, true, true],
            seed: 7,
            strategy: FccLattice,
        }
        .generate()
        .unwrap();
        let _traces: &Vec<Trace> = &out.traces;
        let _box: &SimBox = &out.simbox;
    }
}

//! Van Hove correlation function `G(r, t)` — self and distinct parts.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::core::NeighborQuery;
use molrs::op::F;
use ndarray::{Array1, Array2};

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// Van Hove correlation analyzer.
///
/// Stateless parameter bag: the r-grid (`n_rbins`, `r_max`), the set of integer
/// frame-lag times to evaluate, and the time-origin `stride`.
///
/// The Van Hove function is the space–time generalization of the radial
/// distribution function (L. van Hove, *Phys. Rev.* **1954**, 95, 249; Hansen
/// & McDonald, *Theory of Simple Liquids*, 4th ed., §7):
///
/// ```text
///   G_s(r,t) = (1/N) Σ_i      ⟨ δ(r − |r_i(t) − r_i(0)|) ⟩          (self)
///   G_d(r,t) = (1/N) Σ_i Σ_{j≠i} ⟨ δ(r − |r_j(t) − r_i(0)|) ⟩      (distinct)
/// ```
///
/// with the structural anchor `G_d(r,0) = ρ g(r)` (ρ = N/V the number density,
/// `g(r)` the RDF) and the dynamical anchor `∫ r² G_s(r,t) dr = Msd(t)`. The
/// two thus bridge [`Rdf`](crate::compute::Rdf) (structure) and
/// [`Msd`](crate::compute::Msd) (dynamics).
///
/// # Provenance
///
/// reference implementation (the reference implementation for this `analysis-parity` chain) does
/// **not** ship a dedicated Van Hove analyzer — only its RDF and ACF machinery.
/// Accordingly the **definition** follows van Hove 1954 / Hansen-McDonald, the
/// **distinct-part binning + shell normalization** mirror the RDF pair-binning
/// convention molrs already ports (`CDF::AddToBin`, reference implementation `src/df.cpp`, here
/// a closed-range nearest-bin rule that link 05 of compute-redesign moves onto
/// [`Axis`](crate::compute::Axis)) and
/// the `4π/3 (r_o³−r_i³)` shell volume of [`Rdf`](crate::compute::Rdf)), and the
/// **multi-time-origin averaging** mirrors the ACF origin accumulation in
/// reference implementation `src/reordyn.cpp` / `src/acf.cpp`. Any deviation from a literal
/// reference implementation port is therefore unavoidable (no source to port) and is documented
/// here.
///
/// # Conventions
///
/// - `g_self[t]` is the probability **density of the displacement magnitude**
///   `|Δr|` at lag `t`: `∫ g_self dr = 1`, so its second moment
///   `∫ r² g_self dr` is the MSD. (It is the radial-weighted self-part
///   `4π r² G_s^{3D}`, the directly histogrammable quantity.)
/// - `g_distinct[t]` is the **number density** of other particles at distance
///   `r` from a reference particle, `ρ g(r)` at `t = 0`.
/// - Self displacements use the raw (unwrapped) coordinate difference — exactly
///   as [`Msd`](crate::compute::Msd) does — so the second-moment bridge holds.
///   Distinct distances use the minimum image (matching `rdf`).
#[derive(Debug, Clone)]
pub struct VanHove {
    n_rbins: usize,
    r_max: F,
    lags: Vec<usize>,
    stride: usize,
}

impl VanHove {
    /// New analyzer binning distances in `[0, r_max]` into `n_rbins` bins,
    /// evaluated at the given integer frame `lags` (e.g. `[0, 1, 5, 10]`).
    ///
    /// Lags `≥` the trajectory length are silently dropped at compute time
    /// (you cannot form that displacement); lag `0` is always meaningful.
    pub fn new(n_rbins: usize, r_max: F, lags: Vec<usize>) -> Result<Self, ComputeError> {
        if n_rbins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::n_rbins",
                value: n_rbins.to_string(),
            });
        }
        if !(r_max.is_finite() && r_max > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::r_max",
                value: r_max.to_string(),
            });
        }
        if lags.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::lags",
                value: "empty".to_string(),
            });
        }
        Ok(Self {
            n_rbins,
            r_max,
            lags,
            stride: 1,
        })
    }

    /// Set the time-origin stride (default 1 = use every frame as an origin).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }

    fn shell_volume(&self, r_inner: F, r_outer: F) -> F {
        crate::core::FOUR_THIRDS_PI * (r_outer.powi(3) - r_inner.powi(3))
    }
}

impl LegacyCompute for VanHove {
    type Args<'a> = ();
    type Output = VanHoveResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<VanHoveResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let n_frames = frames.len();

        // Gather positions per frame as three contiguous SoA columns (x/y/z),
        // matching molrs's native Store layout — no interleaved `Array2`. The
        // distinct part hands these straight to `NeighborQuery`'s SoA entry.
        let mut pos: Vec<[Vec<F>; 3]> = Vec::with_capacity(n_frames);
        let mut n_particles: Option<usize> = None;
        for frame in frames {
            let (xp, yp, zp) = get_positions_ref(*frame)?;
            let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
            let m = xs.len();
            match n_particles {
                None => n_particles = Some(m),
                Some(n0) if n0 != m => {
                    return Err(ComputeError::DimensionMismatch {
                        expected: n0,
                        got: m,
                        what: "VanHove particle count",
                    });
                }
                _ => {}
            }
            pos.push([xs.to_vec(), ys.to_vec(), zs.to_vec()]);
        }
        let n = n_particles.unwrap_or(0);

        // Distinct part needs a box volume for the ρ = N/V normalization.
        let simbox = frames[0].simbox_ref();
        let volume = simbox.map(|b| b.volume());

        let dr = self.r_max / self.n_rbins as F;
        let edges = Array1::from_iter((0..=self.n_rbins).map(|i| i as F * dr));
        let centers = Array1::from_iter((0..self.n_rbins).map(|i| (i as F + 0.5) * dr));

        // Keep only realizable lags (< n_frames).
        let used_lags: Vec<usize> = self
            .lags
            .iter()
            .copied()
            .filter(|&t| t < n_frames)
            .collect();
        let n_lags = used_lags.len();

        let mut g_self = Array2::<F>::zeros((n_lags, self.n_rbins));
        let mut g_distinct = Array2::<F>::zeros((n_lags, self.n_rbins));

        let n_rbins = self.n_rbins;
        let r_max = self.r_max;
        // Nearest-bin factor `fac = n_bins / (max − min)`, `min = 0`.
        let fac = n_rbins as F / r_max;

        // Deposit one in-range magnitude with the closed-range nearest-bin
        // rule: with `min = 0` the range test is `0 ≤ d ≤ r_max` and the bin index
        // is `⌊d·fac⌋` clamped to the last bin. Counts are integer `+1.0`
        // increments, so accumulating them across origins/lags in any order (serial
        // or the rayon reduce below) is bit-identical.
        let bin = |counts: &mut [F], base: usize, d: F| {
            if d < 0.0 || d > r_max {
                return;
            }
            let mut ip = (d * fac) as usize;
            if ip >= n_rbins {
                ip = n_rbins - 1;
            }
            counts[base + ip] += 1.0;
        };

        // Per-origin work: build the `NeighborQuery` spatial index over r_i(τ)
        // ONCE, then reuse it for every realizable lag (each lag queries a
        // different r_j(τ+lag) set). The index is built once per origin, not
        // once per (lag, origin), and the pair set is the same either way: the
        // query takes `&self` and never mutates the index.
        //
        // Distinct part uses the same cutoff-`r_max` search as `compute::rdf`
        // (O(N·neighbours), not O(N²)); every pair beyond `r_max` would be dropped
        // by the histogram anyway. The self pair (i == j) is excluded. Self part
        // uses the raw (unwrapped) displacement magnitude — matching `msd`.
        let accumulate =
            |tau: usize, self_counts: &mut [F], dist_counts: &mut [F], n_origins: &mut [usize]| {
                let [axs, ays, azs] = &pos[tau];
                let nq = simbox.map(|sb| NeighborQuery::from_columns(sb, axs, ays, azs, r_max));
                for (li, &lag) in used_lags.iter().enumerate() {
                    if tau + lag >= n_frames {
                        continue;
                    }
                    n_origins[li] += 1;
                    let base = li * n_rbins;
                    let [bxs, bys, bzs] = &pos[tau + lag];
                    for i in 0..n {
                        let dx = bxs[i] - axs[i];
                        let dy = bys[i] - ays[i];
                        let dz = bzs[i] - azs[i];
                        bin(self_counts, base, (dx * dx + dy * dy + dz * dz).sqrt());
                    }
                    if let Some(ref nq) = nq {
                        let nlist = nq.query_columns(bxs, bys, bzs);
                        let ref_i = nlist.point_indices(); // index into a (= r_i(τ))
                        let oth_j = nlist.query_point_indices(); // index into b (= r_j(τ+lag))
                        let d2 = nlist
                            .dist_sq()
                            .expect("NeighborQuery::query_columns stores dist_sq");
                        for k in 0..nlist.n_pairs() {
                            if ref_i[k] == oth_j[k] {
                                continue;
                            }
                            bin(dist_counts, base, d2[k].sqrt());
                        }
                    }
                }
            };

        // Time origins τ = 0, stride, 2·stride, … < n_frames, gathered once and
        // fanned out over lags inside `accumulate` (each lag keeps the origins
        // with `tau + lag < n_frames`).
        let origins: Vec<usize> = (0..n_frames).step_by(self.stride).collect();

        let n_cells = n_lags * self.n_rbins;
        let zero = || {
            (
                vec![0.0 as F; n_cells],
                vec![0.0 as F; n_cells],
                vec![0usize; n_lags],
            )
        };

        #[cfg(feature = "rayon")]
        let (self_counts, dist_counts, origins_per_lag) = {
            use rayon::prelude::*;
            origins
                .par_iter()
                .fold(zero, |(mut sc, mut dc, mut no), &tau| {
                    accumulate(tau, &mut sc, &mut dc, &mut no);
                    (sc, dc, no)
                })
                .reduce(zero, |(mut sca, mut dca, mut noa), (scb, dcb, nob)| {
                    for (x, y) in sca.iter_mut().zip(scb.iter()) {
                        *x += *y;
                    }
                    for (x, y) in dca.iter_mut().zip(dcb.iter()) {
                        *x += *y;
                    }
                    for (x, y) in noa.iter_mut().zip(nob.iter()) {
                        *x += *y;
                    }
                    (sca, dca, noa)
                })
        };
        #[cfg(not(feature = "rayon"))]
        let (self_counts, dist_counts, origins_per_lag) = {
            let (mut sc, mut dc, mut no) = zero();
            for &tau in &origins {
                accumulate(tau, &mut sc, &mut dc, &mut no);
            }
            (sc, dc, no)
        };

        // g_self: probability density of |Δr| (a histogram density:
        // counts / (binned · bin_width), binned = Σ counts, bin_width = dr).
        // g_distinct: number density of others at r → ρ g(r) at lag 0.
        for (li, &n_orig) in origins_per_lag.iter().enumerate() {
            let base = li * self.n_rbins;
            let self_slice = &self_counts[base..base + self.n_rbins];
            let binned: F = self_slice.iter().sum();
            if binned > 0.0 {
                let denom = binned * dr;
                for (k, gk) in g_self.row_mut(li).iter_mut().enumerate() {
                    *gk = self_slice[k] / denom;
                }
            }
            if volume.is_some() {
                let denom = (n as F) * (n_orig.max(1) as F);
                for (k, gk) in g_distinct.row_mut(li).iter_mut().enumerate() {
                    let shell = self.shell_volume(edges[k], edges[k + 1]);
                    if shell > 0.0 {
                        *gk = dist_counts[base + k] / (denom * shell);
                    }
                }
            }
        }

        Ok(VanHoveResult {
            r_edges: edges,
            r_centers: centers,
            lags: used_lags,
            g_self,
            g_distinct,
            dr,
            has_distinct: volume.is_some(),
        })
    }
}

/// Result of a [`VanHove`] computation.
///
/// Rows are lag times (`lags`), columns are r-bins (`r_centers`).
#[derive(Debug, Clone)]
pub struct VanHoveResult {
    /// r-bin edges (`n_rbins + 1`), Å.
    pub r_edges: Array1<F>,
    /// r-bin centers (`n_rbins`), Å.
    pub r_centers: Array1<F>,
    /// Realizable lag times actually evaluated (frame units), aligned with rows.
    pub lags: Vec<usize>,
    /// Self part `g_self[t]`: probability density of `|Δr|`, `∫ g_self dr = 1`.
    pub g_self: Array2<F>,
    /// Distinct part `g_distinct[t]`: number density of others at `r`
    /// (`ρ g(r)` at `t = 0`). All zero when the frame had no `SimBox`.
    pub g_distinct: Array2<F>,
    /// Bin width, Å.
    pub dr: F,
    /// Whether the distinct part was computed (requires a `SimBox`).
    pub has_distinct: bool,
}

impl ComputeResult for VanHoveResult {}

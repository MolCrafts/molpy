//! Legendre reorientational time-correlation functions `C_ℓ(t)`.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Legendre reorientational TCF analyzer.
///
/// Stateless parameter bag: the maximum lag (correlation depth, in frames) and
/// the time-origin stride. The molecular vectors are selected by the `Args`
/// passed to [`compute`](LegacyCompute::compute): one `(a, b)` atom-index pair per
/// molecule, defining `û = normalize(r_b − r_a)` (minimum-image).
///
/// For a molecular **unit vector** `û` (an O–H bond, a dipole axis, …) the
/// reorientational TCFs are
///
/// ```text
///   C_1(t) = ⟨ P_1(û(0)·û(t)) ⟩ = ⟨ cos θ(t) ⟩
///   C_2(t) = ⟨ P_2(û(0)·û(t)) ⟩ = ⟨ (3 cos²θ(t) − 1) / 2 ⟩
/// ```
///
/// averaged over molecules and (multi-)time origins. `C_1` is the dielectric /
/// IR reorientation observable; the `C_2` decay gives the NMR rotational
/// correlation time τ_c (Berne & Pecora, *Dynamic Light Scattering*; standard
/// NMR relaxation theory).
///
/// # Provenance
///
/// Ported from the reference implementation's reorientation-dynamics analyzer `CReorDyn`
/// (`src/reordyn.cpp`): per-molecule unit vectors (cf. `src/order_vector.cpp`),
/// the 1st/2nd-Legendre selection (`m_bLegendre2`), and the multi-time-origin
/// correlation-depth accumulation (`m_iDepth`, the ACF origin sums in
/// `src/acf.cpp`). The reference implementation evaluates the origin average via FFT; molrs uses the
/// algebraically-identical **direct** multi-origin double sum here (exact,
/// O(T²·N)), which the spec permits — the Legendre polynomials are applied to
/// the dot product *before* averaging, so an FFT factorization does not apply to
/// `C_2` anyway.
///
/// # Relation to [`RotationalAutocorrelation`](crate::compute::RotationalAutocorrelation)
///
/// This analyzer is **distinct** from
/// [`RotationalAutocorrelation`](crate::compute::RotationalAutocorrelation):
/// that one is freud's rigid-body **quaternion** autocorrelation (a Wigner-D
/// character of the full orientation), whereas `LegendreReorientation` correlates
/// a single molecular **vector** via Legendre polynomials. They measure different
/// observables and both remain independently usable.
///
/// # Fitting τ_c
///
/// The raw `C_2(t)` curve is `Fit`-ready: feed it to
/// [`DebyeFit`](crate::compute::DebyeFit) (which fits a normalized
/// `Φ(t) → τ` decay) to extract the rotational correlation time — no new fitting
/// code is needed.
#[derive(Debug, Clone)]
pub struct LegendreReorientation {
    max_lag: usize,
    stride: usize,
}

impl LegendreReorientation {
    /// New analyzer evaluating `C_1`/`C_2` at lags `0..=max_lag` (clamped to the
    /// trajectory length at compute time).
    pub fn new(max_lag: usize) -> Self {
        Self { max_lag, stride: 1 }
    }

    /// Set the time-origin stride (default 1).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }
}

fn p1(x: F) -> F {
    x
}
fn p2(x: F) -> F {
    0.5 * (3.0 * x * x - 1.0)
}

impl LegacyCompute for LegendreReorientation {
    /// One `(atom_a, atom_b)` index pair per molecular vector.
    type Args<'a> = &'a [(u32, u32)];
    type Output = LegendreReorientationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        pairs: &'a [(u32, u32)],
    ) -> Result<LegendreReorientationResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if pairs.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "LegendreReorientation::pairs",
                value: "empty".to_string(),
            });
        }
        let n_frames = frames.len();
        let n_vec = pairs.len();

        // Unit vectors in one contiguous, frame-major buffer: `vecs[f * n_vec +
        // m]` is molecule `m`'s unit vector in frame `f`. A flat layout (vs a
        // `Vec<Vec<…>>`) keeps each frame's vectors adjacent so the lag-striding
        // correlation loop below reads them with unit stride.
        let mut vecs: Vec<[F; 3]> = Vec::with_capacity(n_frames * n_vec);
        for frame in frames {
            let (xp, yp, zp) = get_positions_ref(*frame)?;
            let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
            // Resolve the minimum-image state once per frame, not per pair.
            let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
            for &(a, b) in pairs {
                let (a, b) = (a as usize, b as usize);
                if a >= xs.len() || b >= xs.len() {
                    return Err(ComputeError::DimensionMismatch {
                        expected: xs.len(),
                        got: a.max(b) + 1,
                        what: "LegendreReorientation atom index",
                    });
                }
                let d = mic.apply(sub([xs[b], ys[b], zs[b]], [xs[a], ys[a], zs[a]]));
                let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(ComputeError::NonFinite {
                        where_: "LegendreReorientation unit vector (zero-length)",
                        index: 0,
                    });
                }
                // One reciprocal + three multiplies instead of three divides.
                let inv = norm.recip();
                vecs.push([d[0] * inv, d[1] * inv, d[2] * inv]);
            }
        }

        let max_lag = self.max_lag.min(n_frames - 1);
        let stride = self.stride;

        // Multi-time-origin average at a single lag `t`. C1 and C2 share the one
        // dot product; the per-lag summation order is identical to the original
        // serial loop, so parallelizing across lags is bit-for-bit reproducible
        // (each `c1[t]`/`c2[t]` is still a single sequential origin×molecule
        // sum). Kept direct — as the provenance note above records, the P₂
        // nonlinearity means the origin average does not FFT-factorize, so a
        // direct pass (which also yields C1 for free) is both exact and the
        // cheapest route once C2 stays direct.
        let corr_at = |t: usize| -> (F, F) {
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut count: usize = 0;
            let mut tau = 0usize;
            while tau + t < n_frames {
                let base_a = tau * n_vec;
                let base_b = (tau + t) * n_vec;
                for m in 0..n_vec {
                    let a = vecs[base_a + m];
                    let b = vecs[base_b + m];
                    let x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                    s1 += p1(x);
                    s2 += p2(x);
                    count += 1;
                }
                tau += stride;
            }
            let inv = if count > 0 { 1.0 / count as F } else { 0.0 };
            (s1 * inv, s2 * inv)
        };

        // Each lag is an independent reduction over origins → fan out over lags.
        #[cfg(feature = "rayon")]
        let (c1v, c2v): (Vec<F>, Vec<F>) = {
            use rayon::prelude::*;
            (0..=max_lag).into_par_iter().map(corr_at).unzip()
        };
        #[cfg(not(feature = "rayon"))]
        let (c1v, c2v): (Vec<F>, Vec<F>) = (0..=max_lag).map(corr_at).unzip();

        Ok(LegendreReorientationResult {
            lags: (0..=max_lag).collect(),
            c1: Array1::from_vec(c1v),
            c2: Array1::from_vec(c2v),
        })
    }
}

/// Result of a [`LegendreReorientation`] computation.
#[derive(Debug, Clone)]
pub struct LegendreReorientationResult {
    /// Lag times (frame units), aligned with `c1`/`c2`.
    pub lags: Vec<usize>,
    /// `C_1(t) = ⟨cos θ(t)⟩`.
    pub c1: Array1<F>,
    /// `C_2(t) = ⟨(3cos²θ(t) − 1)/2⟩`.
    pub c2: Array1<F>,
}

impl ComputeResult for LegendreReorientationResult {}

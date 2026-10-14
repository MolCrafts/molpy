//! Onsager transport coefficients from collective mean-displacement
//! correlations.

use crate::compute::ComputeResult;
use ndarray::Array1;

use molrs::core::FrameAccess;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

fn validate_series(p: &ndarray::Array2<f64>, name: &'static str) -> Result<usize, ComputeError> {
    let shape = p.shape();
    if shape[1] != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: shape[1],
            what: name,
        });
    }
    let n_frames = shape[0];
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    for (idx, &v) in p.iter().enumerate() {
        if !v.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: name,
                index: idx,
            });
        }
    }
    Ok(n_frames)
}

/// Raw Onsager collective-displacement cross-correlation compute.
///
/// Computes the windowed (all-time-origins) cross-correlation of two species'
/// **collective** displacements — a pure raw observable, with the long-time
/// linear fit `L_ij(τ) ≈ 2·d·D_ij·τ` left to the caller
/// ([`LinearFit`](crate::compute::LinearFit)). The collective-coordinate
/// reduction `P_s = Σ_a r_a` and periodic-image unwrapping are the caller's job.
///
/// For each lag `τ ∈ [0, max_lag]`,
///
/// ```text
///     L_ij(τ) = (1/(N − τ)) Σ_{t=0}^{N−1−τ} Σ_d
///                 (P_i[t+τ, d] − P_i[t, d]) · (P_j[t+τ, d] − P_j[t, d]),
/// ```
///
/// the average over all time origins of the dot product of the two species'
/// collective displacements. When `p_i` and `p_j` are the same array this is the
/// collective mean-square displacement of that species.
///
/// The Onsager phenomenological coefficients `L_ij` describe the coupled
/// transport of species `i` and `j` driven by thermodynamic forces. In the
/// Green–Kubo / Einstein picture they are obtained from the cross-correlation
/// of the **collective** (summed) displacements of each species:
///
/// ```text
///     L_ij(τ) = ⟨ ΔP_i(τ) · ΔP_j(τ) ⟩_t ,
///         with  P_s(t) = Σ_{a ∈ species s} r_a(t)   (unwrapped),
///               ΔP_s(τ) = P_s(t+τ) − P_s(t).
/// ```
///
/// The diagonal term `L_ii` is the collective (distinct-inclusive) mean-square
/// displacement of species `i`; off-diagonal `L_ij` (i ≠ j) captures the
/// cross-correlated drift that distinguishes the Onsager picture from the bare
/// Nernst–Einstein sum. A long-time linear fit `L_ij(τ) ≈ 2·d·D_ij·τ` (done by
/// the caller) yields the transport coefficient.
///
/// This is the molrs port of the `onsager` recipe from the *tame* library
/// (<https://github.com/Roy-Kid/tame>, `tame/recipes/onsager.py`). The
/// collective-coordinate reduction `P_s = Σ_a r_a` and the periodic-image
/// unwrapping are performed by the caller (Python wrapper); this kernel takes
/// the already-assembled per-species collective coordinates and computes the
/// windowed (all-time-origins) cross-correlation.
///
/// # Units
///
/// Unit-agnostic: with positions in Å the correlation is reported in Å². The
/// kernel performs no volume normalization (the *tame* original likewise left
/// the curves un-normalized; the comment there about volume normalization was
/// never applied in code).
#[derive(Debug, Clone, Copy, Default)]
pub struct OnsagerCorrelation;

/// `(p_i, p_j, dt, max_correlation_time)` argument bundle for
/// [`OnsagerCorrelation`].
///
/// * `p_i`, `p_j` — collective coordinates `P_s(t) = Σ_{a ∈ s} r_a(t)`, shape
///   `(n_frames, 3)`. Must already be **unwrapped** (continuous, no
///   periodic-image jumps) and have the same number of frames.
/// * `dt` — frame spacing (> 0); sets the `lag_times` axis.
/// * `max_correlation_time` — longest lag in **frames**, clamped to
///   `n_frames − 1`.
pub type OnsagerCorrelationArgs<'a> = (
    &'a ndarray::Array2<f64>,
    &'a ndarray::Array2<f64>,
    f64,
    usize,
);

fn collective_cross_correlation(
    p_i: &ndarray::Array2<f64>,
    p_j: &ndarray::Array2<f64>,
    dt: f64,
    max_correlation_time: usize,
) -> Result<OnsagerCorrelationResult, ComputeError> {
    let n_i = validate_series(p_i, "onsager p_i (expected (n_frames, 3))")?;
    let n_j = validate_series(p_j, "onsager p_j (expected (n_frames, 3))")?;
    if n_i != n_j {
        return Err(ComputeError::DimensionMismatch {
            expected: n_i,
            got: n_j,
            what: "onsager p_i / p_j frame count",
        });
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }

    let n_frames = n_i;
    let max_lag = max_correlation_time.min(n_frames - 1);

    let mut correlation = Array1::<f64>::zeros(max_lag + 1);
    // τ = 0 is identically zero (Δ = 0); start at 1.
    for tau in 1..=max_lag {
        let count = n_frames - tau;
        let mut acc = 0.0;
        for t in 0..count {
            let mut s = 0.0;
            for d in 0..3 {
                let di = p_i[[t + tau, d]] - p_i[[t, d]];
                let dj = p_j[[t + tau, d]] - p_j[[t, d]];
                s += di * dj;
            }
            acc += s;
        }
        correlation[tau] = acc / count as f64;
    }

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));

    Ok(OnsagerCorrelationResult {
        lag_times,
        correlation,
    })
}

impl LegacyCompute for OnsagerCorrelation {
    /// `(p_i, p_j, dt, max_correlation_time)`. The `frames` slice is unused (the
    /// collective coordinates are pre-assembled by the caller).
    type Args<'a> = OnsagerCorrelationArgs<'a>;
    type Output = OnsagerCorrelationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (p_i, p_j, dt, max_correlation_time) = args;
        collective_cross_correlation(p_i, p_j, dt, max_correlation_time)
    }
}

/// Result of an Onsager collective-displacement cross-correlation.
#[derive(Debug, Clone)]
pub struct OnsagerCorrelationResult {
    /// Lag times τ = i·dt, length `max_lag + 1` (same units as `dt`).
    pub lag_times: Array1<f64>,
    /// Cross-correlation `L_ij(τ) = ⟨ΔP_i(τ)·ΔP_j(τ)⟩_t`, averaged over time
    /// origins, length `max_lag + 1`. For positions in Å this is in Å².
    pub correlation: Array1<f64>,
}

impl ComputeResult for OnsagerCorrelationResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;
    use ndarray::array;

    /// Empty frame slice for the series-based `OnsagerCorrelation` compute.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    #[test]
    fn self_correlation_is_collective_msd() {
        // A single collective coordinate drifting at constant velocity v=(1,0,0):
        // P(t) = (t, 0, 0). ΔP(τ) = (τ, 0, 0). L(τ) = τ² for every origin.
        let p = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let r = OnsagerCorrelation
            .compute(&no_frames(), (&p, &p, 1.0, 4))
            .unwrap();
        for tau in 0..=4 {
            let expected = (tau as f64) * (tau as f64);
            assert!(
                (r.correlation[tau] - expected).abs() < 1e-12,
                "lag {tau}: got {}, expected {expected}",
                r.correlation[tau]
            );
        }
        assert!((r.lag_times[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn anticorrelated_species_give_negative_offdiagonal() {
        // P_i drifts +x, P_j drifts −x. ΔP_i·ΔP_j = (τ)(−τ) = −τ².
        let pi = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let pj = array![[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]];
        let r = OnsagerCorrelation
            .compute(&no_frames(), (&pi, &pj, 0.5, 2))
            .unwrap();
        assert!((r.correlation[1] + 1.0).abs() < 1e-12); // −1²·... origins avg = −1
        assert!((r.correlation[2] + 4.0).abs() < 1e-12);
        assert!((r.lag_times[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn time_origin_average_matches_manual() {
        // Non-uniform motion to exercise the origin average at τ=1.
        let p = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        // τ=1 origins: (t0=0) Δ=(1,0,0) → 1; (t0=1) Δ=(0,2,0) → 4. mean = 2.5
        let r = OnsagerCorrelation
            .compute(&no_frames(), (&p, &p, 1.0, 2))
            .unwrap();
        assert!((r.correlation[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_shape_and_mismatch() {
        let p3 = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let p_short = array![[0.0, 0.0, 0.0]];
        assert!(
            OnsagerCorrelation
                .compute(&no_frames(), (&p3, &p_short, 1.0, 1))
                .is_err()
        );
        let bad = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(
            OnsagerCorrelation
                .compute(&no_frames(), (&bad, &bad, 1.0, 1))
                .is_err()
        );
        assert!(
            OnsagerCorrelation
                .compute(&no_frames(), (&p3, &p3, 0.0, 1))
                .is_err()
        );
    }
}

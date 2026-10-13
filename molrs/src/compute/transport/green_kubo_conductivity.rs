//! Green–Kubo conductivity raw compute — the current-ACF route to σ.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Axis};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::lag_times;
use crate::compute::{TimeCorrelationSpec, autocorrelate};

/// Raw current autocorrelation function, with **no** fitted sigma.
#[derive(Debug, Clone)]
pub struct GreenKuboConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Current ACF `C(τ) = ⟨J(0)·J(τ)⟩` over time origins.
    /// Units: `(e·Å·fs⁻¹)²`.
    pub jacf: Array1<f64>,
}

impl ComputeResult for GreenKuboConductivityResult {}

/// Raw current-ACF compute: the unbiased windowed ACF of the Green–Kubo
/// conductivity, and nothing more (no trapezoid, no σ). The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid) + scale.
///
/// Assemble the collective current series \(J(t)=\sum_a q_a v_a(t)\) yourself
/// (or via an upstream Frame pipeline), then compose:
///
/// 1. [`GreenKuboConductivity`](Self) — raw ACF
/// 2. [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid) — ∫C
/// 3. scale by \(1/(3 V k_B T)\) (SI prefactor in the fit / caller units table)
///
/// There is no separate `Jacf` type. molpy must not invent one either.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboConductivity;

/// `(current, dt, max_correlation_time)` bundle.
pub type GreenKuboConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for GreenKuboConductivity {
    type Args<'a> = GreenKuboConductivityArgs<'a>;
    type Output = GreenKuboConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (current, dt, max_correlation_time) = args;
        let shape = current.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "current (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // ⟨J(0)·J(τ)⟩: Cartesian sum of component ACFs, no mean subtraction
        // (current is already a flux; shared unbiased helper).
        let jacf = autocorrelate(
            current.view().insert_axis(Axis(1)),
            dt,
            &TimeCorrelationSpec::new().max_lag(max_lag),
        )?
        .values;
        Ok(GreenKuboConductivityResult {
            lag_times: lag_times(max_lag, dt),
            jacf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn green_kubo_raw_jacf_matches_direct_acf() {
        // ac-010: GreenKuboConductivity.jacf == the direct unbiased current ACF.
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let current = rng_series(n, 3, 5);
        let max_lag = mct.min(n - 1);
        let mut expected = A1::<f64>::zeros(max_lag + 1);
        for tau in 0..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    s += current[[t, d]] * current[[t + tau, d]];
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        assert_eq!(raw.jacf.len(), expected.len());
        for k in 0..raw.jacf.len() {
            assert!((raw.jacf[k] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn green_kubo_raw_plus_cumulative_trapezoid_matches_manual_trapezoid() {
        // ac-015: CumulativeTrapezoid on GreenKuboConductivity.jacf reproduces a manual
        // trapezoidal integral, and σ = prefactor·∫/(V·k_B·T) is well-defined.
        use crate::compute::CumulativeTrapezoid;
        use crate::compute::Fit;
        use molrs::core::constants::{BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C};
        let angstrom_m = crate::core::unit_factors::ANGSTROM_TO_M.get();
        let femtosecond_s = crate::core::unit_factors::FS_TO_S.get();

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let current = rng_series(n, 3, 19);

        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        let integ = CumulativeTrapezoid.fit((&raw.jacf, dt, None)).unwrap();

        // Manual cumulative trapezoid of the JACF.
        let mut manual = 0.0;
        for tau in 1..raw.jacf.len() {
            manual += 0.5 * (raw.jacf[tau - 1] + raw.jacf[tau]) * dt;
        }
        let last = integ.integral.len() - 1;
        assert!((integ.integral[last] - manual).abs() < 1e-12);

        // Green–Kubo 1/3 prefactor.
        let prefactor = (E_C * E_C * angstrom_m * angstrom_m / femtosecond_s)
            / (3.0 * angstrom_m.powi(3) * K_B_SI);
        let sigma = prefactor * integ.integral[last] / (volume * temperature);
        assert!(sigma.is_finite());
    }
}

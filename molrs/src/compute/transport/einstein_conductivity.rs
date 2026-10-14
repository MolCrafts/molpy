//! Einstein–Helfand conductivity raw compute — the collective-dipole-MSD route
//! to σ.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2, Axis};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{TimeCorrelationSpec, displacement_correlate};

/// Raw collective charge-dipole MSD, with **no** fitted sigma/slope.
#[derive(Debug, Clone)]
pub struct EinsteinConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Collective-dipole MSD ⟨|**M_J**(t+τ) − **M_J**(t)|²⟩ over time origins.
    /// Units: `(e·Å)²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinConductivityResult {}

/// Raw collective charge-dipole MSD compute: the time-origin MSD of the
/// Einstein–Helfand conductivity, and nothing more (no OLS, no σ). The
/// σ = slope/(6·V·k_B·T) step is a downstream
/// [`LinearFit`](crate::compute::LinearFit).
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinConductivity;

/// `(translational_dipole, dt, max_correlation_time)` bundle.
pub type EinsteinConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for EinsteinConductivity {
    type Args<'a> = EinsteinConductivityArgs<'a>;
    type Output = EinsteinConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "translational_dipole (expected (n_frames, 3))",
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
        // Collective-dipole MSD by the FFT route (Kneller / nMoldyn):
        // the (n_frames, 1, 3) displacement correlation over all time origins.
        let corr = displacement_correlate(
            dipole.view().insert_axis(Axis(1)),
            None,
            dt,
            &TimeCorrelationSpec::new().max_lag(max_correlation_time),
        )?;
        let lag_times = corr.lag_times();
        let msd = corr.values;
        Ok(EinsteinConductivityResult { lag_times, msd })
    }
}

#[cfg(test)]
mod tests {
    use super::super::green_kubo_conductivity::GreenKuboConductivity;
    use super::*;
    use molrs::core::Frame;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};

    /// MD→SI conductivity prefactor with the Einstein 1/6 factor.
    fn einstein_helfand_prefactor() -> f64 {
        use molrs::core::constants::{BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C};
        let angstrom_m = crate::core::unit_factors::ANGSTROM_TO_M.get();
        let femtosecond_s = crate::core::unit_factors::FS_TO_S.get();
        (E_C * E_C * angstrom_m * angstrom_m / femtosecond_s) / (6.0 * angstrom_m.powi(3) * K_B_SI)
    }

    /// Empty frame slice for the series-based raw computes.
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
    fn einstein_conductivity_msd_matches_direct_time_origin_average() {
        // ac-009: EinsteinConductivity.msd == the direct time-origin collective-
        // dipole MSD.
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let dipole = rng_series(n, 3, 3);
        let max_lag = mct.min(n - 1);
        let mut expected = A1::<f64>::zeros(max_lag + 1);
        for tau in 1..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    let dx = dipole[[t + tau, d]] - dipole[[t, d]];
                    s += dx * dx;
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        assert_eq!(raw.msd.len(), expected.len());
        for k in 0..raw.msd.len() {
            assert!((raw.msd[k] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn einstein_conductivity_msd_exact_small() {
        // Hand-checkable collective MSD on a 3-frame, purely-x ramp (moved from
        // dielectric.rs): M_J = [0, 1, 3] along x. msd[1] = mean(1², 2²) = 2.5,
        // msd[2] = 3² = 9.
        let dipole = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, 1.0, 2))
            .unwrap();
        assert!((raw.msd[0] - 0.0).abs() < 1e-12);
        assert!((raw.msd[1] - 2.5).abs() < 1e-12);
        assert!((raw.msd[2] - 9.0).abs() < 1e-12);
        assert_eq!(raw.lag_times.len(), 3);

        // Folded MD→SI prefactor (Einstein 1/6) ≈ 3.0988e9 S·m⁻¹ per
        // [(e·Å)²·fs⁻¹·Å⁻³·K⁻¹]. Guards against conversion drift.
        let prefactor = einstein_helfand_prefactor();
        assert!((prefactor - 3.0988e9).abs() / 3.0988e9 < 1e-3);
    }

    #[test]
    fn series_computes_reject_bad_shape() {
        // ac-016: non-(_,3) series -> DimensionMismatch.
        let bad = rng_series(10, 2, 1);
        assert!(matches!(
            EinsteinConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GreenKuboConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn einstein_conductivity_plus_linear_fit_matches_manual_ols() {
        // ac-015: LinearFit slope on EinsteinConductivity.msd reproduces a manual
        // OLS over the same diffusive window, and the σ = slope/(6·V·k_B·T)·prefactor
        // composition is well-defined.
        use crate::compute::Fit;
        use crate::compute::LinearFit;

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let (start_frac, end_frac) = (0.2, 0.8);
        let dipole = rng_series(n, 3, 17);

        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        let fit = LinearFit {
            window: (start_frac, end_frac),
        }
        .fit((&raw.lag_times, &raw.msd))
        .unwrap();

        // Manual OLS over [fit_start, fit_end] on (lag_times, msd).
        let (fs, fe) = (fit.fit_start, fit.fit_end);
        let np = (fe - fs + 1) as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in fs..=fe {
            let x = raw.lag_times[i];
            let y = raw.msd[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let manual_slope = (np * sxy - sx * sy) / (np * sxx - sx * sx);
        assert!((fit.slope - manual_slope).abs() < 1e-12);

        let prefactor = einstein_helfand_prefactor();
        let sigma = prefactor * fit.slope / (volume * temperature);
        assert!(sigma.is_finite());
    }

    #[test]
    fn einstein_conductivity_plus_fit_recovers_nernst_einstein() {
        // ac-008 (scientific regression): N independent
        // ions of charge q on uncorrelated 3-D random walks. For independent
        // carriers EinsteinConductivity + LinearFit must reduce to the
        // Nernst–Einstein value σ = n·q²·D/(k_B·T) within the ≤0.13 ensemble
        // tolerance. M_J(t) is ONE stochastic trajectory, so we ENSEMBLE-AVERAGE
        // σ over many realisations. Seed is fixed → deterministic across CI.
        use crate::compute::Fit;
        use crate::compute::LinearFit;
        use molrs::core::constants::{BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C};
        let angstrom_m = crate::core::unit_factors::ANGSTROM_TO_M.get();
        let femtosecond_s = crate::core::unit_factors::FS_TO_S.get();

        let n_realisations = 48usize;
        let n_ions = 50usize;
        let n_frames = 1500usize;
        let dt = 1.0_f64; // fs
        let q = 1.0_f64; // e
        let volume = 1.0e5_f64; // Å³
        let temperature = 300.0_f64; // K
        let step = 0.5_f64; // Å, uniform per-axis displacement amplitude
        // Nernst–Einstein prefactor (no Einstein 1/6 here: D folds it in).
        let ne_prefactor =
            (E_C * E_C * angstrom_m * angstrom_m / femtosecond_s) / (angstrom_m.powi(3) * K_B_SI);
        let eh_prefactor = einstein_helfand_prefactor();

        let mut rng = rand::rngs::StdRng::seed_from_u64(20260601);
        let mut sigma_eh_sum = 0.0_f64;
        let mut sigma_ne_sum = 0.0_f64;
        for _ in 0..n_realisations {
            let mut pos = vec![[0.0_f64; 3]; n_ions];
            let mut dipole = Array2::<f64>::zeros((n_frames, 3));
            let mut step_sq_sum = 0.0_f64;
            let mut step_count = 0.0_f64;
            for f in 0..n_frames {
                for ion in &mut pos {
                    if f > 0 {
                        for c in ion.iter_mut() {
                            let s = rng.random_range(-step..step);
                            *c += s;
                            step_sq_sum += s * s;
                            step_count += 1.0;
                        }
                    }
                }
                let mut m = [0.0_f64; 3];
                for p in &pos {
                    for d in 0..3 {
                        m[d] += q * p[d];
                    }
                }
                for d in 0..3 {
                    dipole[[f, d]] = m[d];
                }
            }

            let max_corr = n_frames / 5;
            let raw = EinsteinConductivity
                .compute(&no_frames(), (&dipole, dt, max_corr))
                .unwrap();
            let fit = LinearFit { window: (0.1, 0.5) }
                .fit((&raw.lag_times, &raw.msd))
                .unwrap();
            sigma_eh_sum += eh_prefactor * fit.slope / (volume * temperature);

            // Realised per-axis step variance → D = var/(2·dt); analytic
            // Nernst–Einstein σ = n·q²·D/(k_B·T).
            let var_axis = step_sq_sum / step_count; // Å²
            let d_diff = var_axis / (2.0 * dt); // Å²·fs⁻¹
            let number_density = n_ions as f64 / volume; // Å⁻³
            sigma_ne_sum += ne_prefactor * number_density * q * q * d_diff / temperature;
        }

        let sigma_eh = sigma_eh_sum / n_realisations as f64;
        let sigma_ne = sigma_ne_sum / n_realisations as f64;
        let rel_err = (sigma_eh - sigma_ne).abs() / sigma_ne.abs();
        assert!(
            rel_err < 0.13,
            "ensemble EH σ = {sigma_eh} S/m vs Nernst–Einstein {sigma_ne} S/m (rel err {rel_err:.3})"
        );
        assert!(sigma_eh > 0.0);
    }
}

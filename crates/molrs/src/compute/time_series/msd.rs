//! [`Msd`] — the mean squared displacement.

use super::{Correlator, CorrelatorAccumulator, LagKernel, Origins, Reduction};
use crate::compute::{ComputeError, Stream};

/// Mean squared displacement `MSD(t) = ⟨|r_i(τ + t) − r_i(τ)|²⟩`.
///
/// A [`Correlator`] with the [`LagKernel::SquaredDisplacement`] kernel,
/// autocorrelated and normalised by the number of origins. Two choices are
/// left to the caller:
///
/// - [`Origins`]: `All` (default) averages every origin `τ` up to `max_lag`
///   back — the windowed estimator (Kneller et al. 1995); `First` measures
///   from frame 0 only.
/// - [`Reduction`]: `Mean` (default) over items, or `PerItem` for one curve
///   per particle.
///
/// `Msd` is a [`Stream`] only: [`accumulator`](Stream::accumulator) hands
/// out a [`CorrelatorAccumulator`]; push one
/// [`CorrelatorFrame`](super::CorrelatorFrame) per frame (`a` = that frame's
/// `N × D` positions, `b = None`) and [`finish`](crate::compute::Accumulate::finish)
/// reads the curve. Memory is `O(max_lag · N · D)` whatever the trajectory
/// length.
///
/// The positions must be continuous across periodic boundaries. Given each
/// frame's box (`simbox`), the accumulator checks that and returns
/// [`ComputeError::Wrapped`] for a trajectory wrapped into its box; unwrap
/// with [`Unwrap`](super::Unwrap). The Einstein relation `D = slope / (2d)`
/// is a [`LinearFit`](crate::compute::LinearFit) of the curve; applied to
/// the collective charge dipole series it is the Einstein–Helfand
/// conductivity curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Msd {
    correlator: Correlator,
}

impl Msd {
    /// The MSD over lags `0 … max_lag` frames, all origins, averaged over
    /// items.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::SquaredDisplacement, max_lag),
        }
    }

    /// Measure from all origins (`All`) or from frame 0 (`First`).
    pub fn with_origins(self, origins: Origins) -> Self {
        Self {
            correlator: self.correlator.with_origins(origins),
        }
    }

    /// Average over items (`Mean`) or keep one curve per item (`PerItem`).
    pub fn with_reduction(self, reduce: Reduction) -> Self {
        Self {
            correlator: self.correlator.with_reduction(reduce),
        }
    }

    /// Take every `stride`-th frame as a time origin (default
    /// [`Correlator::DEFAULT_STRIDE`]; see [`Correlator::with_stride`]).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0`.
    pub fn with_stride(self, stride: usize) -> Result<Self, ComputeError> {
        Ok(Self {
            correlator: self.correlator.with_stride(stride)?,
        })
    }
}

impl Stream for Msd {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    Msd as "Msd", Frames;
    params |s| {
        max_lag: Int;
        origins: Choice(Origins::NAMES) = Origins::default() => s.correlator.origins();
        reduction: Choice(Reduction::NAMES) = Reduction::default() => s.correlator.reduction();
        stride: Int = Correlator::DEFAULT_STRIDE => s.correlator.stride();
    }
    inputs {
        a: Positions;
        dt: Scalar;
        simbox: SimBox, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::super::TimeCorrelation;
    use super::super::test_support::{rng_series, trajectory};
    use super::*;
    use crate::compute::{Accumulate, Compute, CorrelatorFrame, LinearFit, LinearFitInput};
    use crate::core::Cell;
    use molrs::core::SimBox;
    use molrs::op::F;
    use ndarray::{Array2, Array3, ArrayView3, Axis, array};
    use rand::{RngExt, SeedableRng};

    /// Push every frame of `series` (`T × N × D`) into `acc`, then finish.
    fn fold(
        acc: CorrelatorAccumulator,
        series: ArrayView3<'_, F>,
        dt: F,
    ) -> Result<TimeCorrelation, ComputeError> {
        acc.fold(series.outer_iter().map(|a| CorrelatorFrame {
            a,
            b: None,
            dt,
            simbox: None,
        }))
    }

    fn msd(series: &Array3<F>, max_lag: usize, origins: Origins) -> TimeCorrelation {
        let msd = Msd::new(max_lag).with_origins(origins);
        fold(msd.accumulator(), series.view(), 1.0).unwrap()
    }

    /// `n_frames × n_atoms` random walk, one frame per row of `(x, y, z)`.
    fn walk(n_frames: usize, n_atoms: usize, seed: u64) -> Array3<F> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut series = Array3::<F>::zeros((n_frames, n_atoms, 3));
        for t in 1..n_frames {
            for i in 0..n_atoms {
                for d in 0..3 {
                    series[[t, i, d]] = series[[t - 1, i, d]] + rng.random_range(-0.5..0.5);
                }
            }
        }
        series
    }

    /// The windowed definition, written out.
    fn windowed_reference(series: &Array3<F>) -> Vec<F> {
        let (t, n, d) = series.dim();
        (0..t)
            .map(|lag| {
                let mut acc = 0.0;
                for tau in 0..t - lag {
                    for i in 0..n {
                        for c in 0..d {
                            let dx = series[[tau + lag, i, c]] - series[[tau, i, c]];
                            acc += dx * dx;
                        }
                    }
                }
                acc / ((t - lag) * n) as F
            })
            .collect()
    }

    #[test]
    fn surface_is_its_correlator() {
        let series = walk(20, 4, 3);
        for origins in Origins::ALL.iter().copied() {
            for reduce in [Reduction::Mean, Reduction::PerItem] {
                let surface = Msd::new(9).with_origins(origins).with_reduction(reduce);
                let explicit = Correlator::new(LagKernel::SquaredDisplacement, 9)
                    .with_origins(origins)
                    .with_reduction(reduce);
                assert_eq!(surface.correlator, explicit);
                let got = fold(surface.accumulator(), series.view(), 0.5).unwrap();
                let via = fold(explicit.accumulator(), series.view(), 0.5).unwrap();
                assert_eq!(got, via);
            }
        }
    }

    // ── Single origin (the former `MsdMode::Direct`) ──

    #[test]
    fn reference_frame_msd_is_zero() {
        let series = trajectory(&[
            vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]);
        let curve = msd(&series, 1, Origins::First);
        assert_eq!(curve.n_lags(), 2);
        assert!(curve.values[[0, 0]].abs() < 1e-12);
        assert!((curve.values[[1, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_calls() {
        let series = trajectory(&[
            vec![[0.0; 3]; 3],
            vec![[1.0, 0.0, 0.0]; 3],
            vec![[2.0, 0.0, 0.0]; 3],
        ]);
        let a = msd(&series, 2, Origins::First);
        let b = msd(&series, 2, Origins::First);
        assert_eq!(a.n_lags(), b.n_lags());
        for i in 0..a.n_lags() {
            assert!((a.values[[i, 0]] - b.values[[i, 0]]).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_progression() {
        // Frame i: each particle at (i, 0, 0). MSD(i) = i² per particle.
        let frames: Vec<Vec<[F; 3]>> = (0..4).map(|i| vec![[i as F, 0.0, 0.0]; 3]).collect();
        let curve = msd(&trajectory(&frames), 3, Origins::First);
        for i in 0..4 {
            let expected = (i as F) * (i as F);
            assert!(
                (curve.values[[i, 0]] - expected).abs() < 1e-12,
                "MSD[{i}] = {}, expected {expected}",
                curve.values[[i, 0]]
            );
        }
    }

    #[test]
    fn empty_input_errors() {
        // No frame at all, and a frame of no particles.
        let no_frames = Msd::new(0).accumulator().finish().unwrap_err();
        assert!(matches!(no_frames, ComputeError::EmptyInput));
        let empty = Array2::<F>::zeros((0, 3));
        let err = Msd::new(0)
            .accumulator()
            .push(CorrelatorFrame {
                a: empty.view(),
                b: None,
                dt: 1.0,
                simbox: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_particle_count_errors() {
        let two = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let one = array![[1.0, 0.0, 0.0]]; // one particle only
        let mut acc = Msd::new(1).accumulator();
        let frame = |a| CorrelatorFrame {
            a,
            b: None,
            dt: 1.0,
            simbox: None,
        };
        acc.push(frame(two.view())).unwrap();
        let err = acc.push(frame(one.view())).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    // ── All origins (the former `MsdMode::Window`) ──

    #[test]
    fn windowed_matches_nested_loop_reference() {
        // Two particles drifting along x with a small random component in y.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames: Vec<Vec<[F; 3]>> = (0..6)
            .map(|t| vec![[xs[t], ys[t], 0.0], [xs[t] + 0.5, ys[t] + 0.4, 0.0]])
            .collect();
        let series = trajectory(&frames);
        let reference = windowed_reference(&series);
        let computed = msd(&series, 5, Origins::All);

        assert_eq!(computed.n_lags(), 6);
        for (lag, &ref_val) in reference.iter().enumerate() {
            assert!(
                (computed.values[[lag, 0]] - ref_val).abs() < 1e-9,
                "lag {lag}: window={}, ref={ref_val}",
                computed.values[[lag, 0]],
            );
        }
    }

    #[test]
    fn windowed_lag_zero_is_zero() {
        let series = trajectory(&[
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]);
        assert!(msd(&series, 2, Origins::All).values[[0, 0]].abs() < 1e-10);
    }

    #[test]
    fn windowed_diffusive_signature_linear_in_lag() {
        // Pure ballistic drift along +x: r_i(t) = (t·v_i, 0, 0). Windowed MSD
        // for lag t equals t² v_i² averaged over particles AND time origins,
        // which still gives t² · ⟨v²⟩ since each (τ+t − τ) = t.
        let v = [1.0, 2.0];
        let frames: Vec<Vec<[F; 3]>> = (0..8)
            .map(|t| vec![[t as F * v[0], 0.0, 0.0], [t as F * v[1], 0.0, 0.0]])
            .collect();
        let curve = msd(&trajectory(&frames), 7, Origins::All);
        let mean_v2 = (v[0] * v[0] + v[1] * v[1]) / 2.0;
        for lag in 0..8 {
            let expected = (lag as F) * (lag as F) * mean_v2;
            assert!(
                (curve.values[[lag, 0]] - expected).abs() < 1e-10,
                "lag {lag}: got {}, expected {expected}",
                curve.values[[lag, 0]],
            );
        }
    }

    #[test]
    fn msd_is_dimension_agnostic() {
        // A 1-D series: one component per item.
        let series = Array3::from_shape_vec((3, 1, 1), vec![0.0, 1.0, 3.0]).unwrap();

        let direct = msd(&series, 2, Origins::First);
        let means: Vec<F> = direct.channel(0).unwrap().to_vec();
        assert_eq!(means, vec![0.0, 1.0, 9.0]);

        // and the windowed estimator agrees with the nested loop in 1-D too
        let w = msd(&series, 2, Origins::All).channel(0).unwrap().to_vec();
        assert!((w[1] - ((1.0 + 4.0) / 2.0)).abs() < 1e-12, "{w:?}");
    }

    #[test]
    fn per_item_curves_average_to_the_mean_curve() {
        let series = walk(16, 5, 9);
        let mean = msd(&series, 6, Origins::All);
        let per_item = fold(
            Msd::new(6).with_reduction(Reduction::PerItem).accumulator(),
            series.view(),
            1.0,
        )
        .unwrap();
        assert_eq!(per_item.n_channels(), 5);
        for lag in 0..=6 {
            let avg = per_item.values.row(lag).mean().unwrap();
            assert!((avg - mean.values[[lag, 0]]).abs() < 1e-12, "lag {lag}");
        }
    }

    // ── Streaming (the former `MsdAccumulator`) ──

    fn stream(series: &Array3<F>, msd: Msd) -> CorrelatorAccumulator {
        let mut acc = msd.accumulator();
        for frame in series.outer_iter() {
            acc.push(CorrelatorFrame {
                a: frame,
                b: None,
                dt: 1.0,
                simbox: None,
            })
            .unwrap();
        }
        acc
    }

    /// The first-origin curve, written out: `|r_i(t) − r_i(0)|²` summed over
    /// components, averaged over items.
    #[test]
    fn first_origin_curve_matches_the_reference() {
        let (t, n) = (16, 5);
        let series = walk(t, n, 42);
        let streamed = stream(&series, Msd::new(t - 1).with_origins(Origins::First))
            .finish()
            .unwrap();
        assert_eq!(streamed.n_lags(), t);
        for k in 0..t {
            let mut expected = 0.0;
            for i in 0..n {
                for c in 0..3 {
                    let dx = series[[k, i, c]] - series[[0, i, c]];
                    expected += dx * dx;
                }
            }
            expected /= n as F;
            assert!(
                (streamed.values[[k, 0]] - expected).abs() < 1e-12,
                "first-origin lag {k}"
            );
        }
    }

    /// A window shorter than the trajectory still averages every origin of
    /// each lag it keeps.
    #[test]
    fn windowed_msd_matches_the_reference_up_to_the_window() {
        let (t, n, w) = (24, 4, 10);
        let series = walk(t, n, 7);
        let reference = windowed_reference(&series);
        let streamed = stream(&series, Msd::new(w)).finish().unwrap();
        assert_eq!(streamed.n_lags(), w + 1);
        for (k, expected) in reference.iter().enumerate().take(w + 1) {
            let s = streamed.values[[k, 0]];
            assert!(
                (s - expected).abs() < 1e-9,
                "windowed lag {k}: streamed {s} vs reference {expected}"
            );
        }
    }

    #[test]
    fn window_shorter_than_trajectory_bounds_ring() {
        let (t, n, w) = (50, 3, 4);
        let acc = stream(&walk(t, n, 3), Msd::new(w));
        assert_eq!(acc.finish().unwrap().n_lags(), w + 1);
        assert_eq!(acc.n_frames(), t);
    }

    #[test]
    fn dof_mismatch_is_error() {
        let mut acc = Msd::new(2).accumulator();
        let frame = |a| CorrelatorFrame {
            a,
            b: None,
            dt: 1.0,
            simbox: None,
        };
        let (two, three) = (Array2::<F>::zeros((2, 3)), Array2::<F>::zeros((3, 3)));
        acc.push(frame(two.view())).unwrap();
        assert!(matches!(
            acc.push(frame(three.view())).unwrap_err(),
            ComputeError::DimensionMismatch { .. }
        ));
    }

    // ── Wrapped trajectories ──

    #[test]
    fn a_wrapped_trajectory_is_rejected_given_boxes() {
        // One atom steps +1 Å per frame along x in a 10 Å cube, then is
        // wrapped from 9.5 back to -0.5 between frames 2 and 3.
        let series = trajectory(&[
            vec![[7.5, 1.0, 1.0]],
            vec![[8.5, 1.0, 1.0]],
            vec![[9.5, 1.0, 1.0]],
            vec![[-0.5, 1.0, 1.0]],
        ]);
        let cube = SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
        let boxes = vec![cube; 4];

        // The same trajectory unwrapped passes and gives the ballistic MSD.
        let unwrapped = trajectory(&[
            vec![[7.5, 1.0, 1.0]],
            vec![[8.5, 1.0, 1.0]],
            vec![[9.5, 1.0, 1.0]],
            vec![[10.5, 1.0, 1.0]],
        ]);
        let curve = Msd::new(2)
            .accumulator()
            .fold(
                unwrapped
                    .outer_iter()
                    .zip(&boxes)
                    .map(|(a, simbox)| CorrelatorFrame {
                        a,
                        b: None,
                        dt: 1.0,
                        simbox: Some(simbox),
                    }),
            )
            .unwrap();
        assert!((curve.values[[2, 0]] - 4.0).abs() < 1e-12);

        // The wrapped one is rejected at the frame that jumps.
        let mut acc = Msd::new(2).accumulator();
        let mut outcome = Ok(());
        for (frame, simbox) in series.outer_iter().zip(&boxes) {
            outcome = acc.push(CorrelatorFrame {
                a: frame,
                b: None,
                dt: 1.0,
                simbox: Some(simbox),
            });
            if outcome.is_err() {
                break;
            }
        }
        assert!(matches!(
            outcome,
            Err(ComputeError::Wrapped { frame: 3, item: 0 })
        ));
    }

    // ── Einstein–Helfand conductivity: Msd on the collective dipole ──

    /// MD→SI conductivity prefactor with the Einstein 1/6 factor.
    fn einstein_helfand_prefactor() -> f64 {
        use molrs::core::constants::{BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C};
        let angstrom_m = crate::core::unit_factors::ANGSTROM_TO_M.get();
        let femtosecond_s = crate::core::unit_factors::FS_TO_S.get();
        (E_C * E_C * angstrom_m * angstrom_m / femtosecond_s) / (6.0 * angstrom_m.powi(3) * K_B_SI)
    }

    /// The collective-dipole MSD of a `(T, 3)` dipole series.
    fn dipole_msd(dipole: &Array2<F>, dt: F, max_lag: usize) -> TimeCorrelation {
        let series = dipole.view().insert_axis(Axis(1));
        fold(Msd::new(max_lag).accumulator(), series, dt).unwrap()
    }

    #[test]
    fn einstein_conductivity_msd_matches_direct_time_origin_average() {
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let dipole = rng_series(n, 3, 3);
        let mut expected = ndarray::Array1::<F>::zeros(mct + 1);
        for tau in 1..=mct {
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
        let raw = dipole_msd(&dipole, dt, mct);
        assert_eq!(raw.n_lags(), expected.len());
        for k in 0..raw.n_lags() {
            assert!((raw.values[[k, 0]] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn einstein_conductivity_msd_exact_small() {
        // Hand-checkable collective MSD on a 3-frame, purely-x ramp:
        // M_J = [0, 1, 3] along x. msd[1] = mean(1², 2²) = 2.5, msd[2] = 3² = 9.
        let dipole = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let raw = dipole_msd(&dipole, 1.0, 2);
        assert!((raw.values[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((raw.values[[1, 0]] - 2.5).abs() < 1e-12);
        assert!((raw.values[[2, 0]] - 9.0).abs() < 1e-12);
        assert_eq!(raw.lag_times.len(), 3);

        // Folded MD→SI prefactor (Einstein 1/6) ≈ 3.0988e9 S·m⁻¹ per
        // [(e·Å)²·fs⁻¹·Å⁻³·K⁻¹]. Guards against conversion drift.
        let prefactor = einstein_helfand_prefactor();
        assert!((prefactor - 3.0988e9).abs() / 3.0988e9 < 1e-3);
    }

    #[test]
    fn einstein_conductivity_plus_linear_fit_matches_manual_ols() {
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let (start_frac, end_frac) = (0.2, 0.8);
        let dipole = rng_series(n, 3, 17);

        let raw = dipole_msd(&dipole, dt, mct);
        let msd = raw.channel(0).unwrap().to_owned();
        let fit = LinearFit::new(start_frac, end_frac)
            .unwrap()
            .compute(LinearFitInput {
                correlation: &raw,
                channel: 0,
            })
            .unwrap();

        // Manual OLS over [fit_start, fit_end] on (lag_times, msd).
        let (fs, fe) = (fit.fit_start, fit.fit_end);
        let np = (fe - fs + 1) as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in fs..=fe {
            let x = raw.lag_times[i];
            let y = msd[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let manual_slope = (np * sxy - sx * sy) / (np * sxx - sx * sx);
        assert!((fit.slope - manual_slope).abs() < 1e-12);

        let sigma = einstein_helfand_prefactor() * fit.slope / (volume * temperature);
        assert!(sigma.is_finite());
    }

    #[test]
    fn einstein_conductivity_plus_fit_recovers_nernst_einstein() {
        // N independent ions of charge q on uncorrelated 3-D random walks. For
        // independent carriers Msd + LinearFit on the collective dipole must
        // reduce to the Nernst–Einstein value σ = n·q²·D/(k_B·T) within the
        // ≤0.13 ensemble tolerance. Seed is fixed → deterministic across CI.
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

            let raw = dipole_msd(&dipole, dt, n_frames / 5);
            let fit = LinearFit::new(0.1, 0.5)
                .unwrap()
                .compute(LinearFitInput {
                    correlation: &raw,
                    channel: 0,
                })
                .unwrap();
            sigma_eh_sum += eh_prefactor * fit.slope / (volume * temperature);

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

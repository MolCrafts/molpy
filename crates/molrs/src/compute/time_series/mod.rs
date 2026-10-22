//! Time series: the one correlation estimator, the quantities chemists name
//! after it, and the series operations around it.
//!
//! # One estimator, named surfaces
//!
//! Every lagged correlation in molrs is a configuration of [`Correlator`]:
//! a [`LagKernel`] evaluated between samples `t` apart, averaged over
//! [`Origins`] (all of them, or every `s`-th through
//! [`Correlator::with_stride`]), optionally on fluctuations ([`MeanPolicy`]), scaled by a
//! [`Normalization`] and combined into channels by a [`Reduction`]. Its
//! output, [`TimeCorrelation`], is the input of every fit and spectrum.
//!
//! A quantity with a name of its own is a **surface**: a type that holds a
//! `Correlator` with the configuration that defines the quantity fixed, and
//! nothing else. Its `compute` is one delegating call; its stream is the
//! correlator's accumulator. [`Msd`] and [`Vacf`] are accumulator-only: their
//! one entry is the stream.
//!
//! | Surface | Kernel | Fixed configuration |
//! |---|---|---|
//! | [`Msd`] | squared displacement, auto | origins and reduction tunable |
//! | [`Vacf`] | product, auto | mean-subtracted, mean over degrees of freedom |
//! | [`OnsagerCorrelation`] | squared displacement, cross | `T × 3` collective coordinates |
//! | [`DipoleAcf`] | product, auto | mean-subtracted, summed over Cartesian components |
//! | [`CurrentAcf`] | product, auto | summed over Cartesian components |
//! | [`ReorientationAcf`] | Legendre `P_l`, auto | unit vectors; reduction and stride tunable |
//! | [`RotationalAutocorrelation`] | Wigner-D `χ_l`, auto | first origin |
//! | [`Survival`] | persistence or product | on 0/1 presence |
//!
//! A correlation with no surface — the VCD `μ̇ · ṁ` and ROA `α̇ · Ġ′` cross
//! terms, the Raman isotropic / anisotropic parts, the dipole-rate × dipole
//! cross term `Ṁ · M` — is a direct `Correlator` composition.
//!
//! # Series operations
//!
//! Building the series a correlation reads is a compute of its own:
//! [`Derivative`] (one finite-difference implementation, choice of
//! [`Stencil`]) and [`Unwrap`] (continuous coordinates from image flags or by
//! continuity). The frequency side is [`Spectrum`] — [`Window`], zero
//! padding, one-sided FFT — on the [`FrequencyGrid`].
//!
//! ```
//! use molrs::compute::{Accumulate, CorrelatorFrame, Stream, Vacf};
//! use ndarray::{Array2, Axis};
//!
//! // Two degrees of freedom over four frames, pushed one frame at a time.
//! let v = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap();
//! let mut vacf = Vacf::new(2).accumulator();
//! for row in v.outer_iter() {
//!     vacf.push(CorrelatorFrame { a: row.insert_axis(Axis(1)), b: None, dt: 1.0, simbox: None })
//!         .unwrap();
//! }
//! let acf = vacf.finish().unwrap();
//! assert!((acf.values[[0, 0]] - 0.5).abs() < 1e-12);  // ⟨v_d²⟩
//! assert!((acf.values[[2, 0]] + 0.5).abs() < 1e-12);  // half a period later
//! ```

mod accumulator;
mod correlator;
mod current_acf;
mod derivative;
mod dipole_acf;
pub(crate) mod guards;
mod msd;
mod onsager;
mod options;
mod reorientation_acf;
mod rotational_autocorrelation;
mod spectrum;
mod survival;
mod time_correlation;
mod unwrap;
mod vacf;

pub use accumulator::{CorrelatorAccumulator, CorrelatorFrame};
pub use correlator::{Correlator, CorrelatorInput};
pub use current_acf::{CurrentAcf, CurrentAcfInput};
pub use derivative::{Derivative, DerivativeInput, Stencil};
pub use dipole_acf::{DipoleAcf, DipoleAcfInput};
pub use msd::Msd;
pub use onsager::{OnsagerCorrelation, OnsagerInput};
pub use options::{LagKernel, LagKernelKind, MeanPolicy, Normalization, Origins, Reduction};
pub use reorientation_acf::{ReorientationAcf, ReorientationAcfInput};
pub use rotational_autocorrelation::{RotationalAutocorrelation, RotationalAutocorrelationInput};
pub use spectrum::{FrequencyGrid, OneSidedSpectrum, Spectrum, SpectrumInput, Window};
pub use survival::{
    Survival, SurvivalAccumulator, SurvivalFrame, SurvivalInput, SurvivalKind,
    SurvivalNormalization, SurvivalOrigin,
};
pub use time_correlation::TimeCorrelation;
pub use unwrap::{Unwrap, UnwrapInput};
pub use vacf::Vacf;

/// Series the time-series tests share.
#[cfg(test)]
pub(crate) mod test_support {
    use molrs::op::F;
    use ndarray::{Array2, Array3};
    use rand::{RngExt, SeedableRng};

    /// A seeded `n × cols` series uniform in `[-1, 1)`.
    pub(crate) fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<F> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((n, cols), |_| rng.random_range(-1.0..1.0))
    }

    /// `Σ_τ a(τ) · b(τ + k)` for `k = 0 … max_lag`, summed directly — the
    /// definition every FFT path is checked against.
    pub(crate) fn direct_lagged_sum(a: &[F], b: &[F], max_lag: usize) -> Vec<F> {
        (0..=max_lag)
            .map(|k| (0..a.len() - k).map(|t| a[t] * b[t + k]).sum())
            .collect()
    }

    /// Element-wise `|got − want| ≤ tol · max(1, max |want|)`.
    pub(crate) fn assert_close(got: &[F], want: &[F], tol: F, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * scale, "{what}[{k}]: {g} vs {w}");
        }
    }

    /// A `T × N × 3` trajectory from per-frame lists of points.
    pub(crate) fn trajectory(frames: &[Vec<[F; 3]>]) -> Array3<F> {
        let n = frames[0].len();
        Array3::from_shape_fn((frames.len(), n, 3), |(t, i, d)| frames[t][i][d])
    }
}

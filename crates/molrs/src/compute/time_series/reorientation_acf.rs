//! [`ReorientationAcf`] — Legendre reorientational correlation `C_l(t)`.

use molrs::op::F;
use ndarray::ArrayView3;

use super::{
    Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, Reduction, TimeCorrelation,
};
use crate::compute::{Compute, ComputeError, Stream};

/// Reorientational time-correlation of molecular vectors,
///
/// ```text
/// C_l(t) = ⟨ P_l(û(τ) · û(τ + t)) ⟩_{molecules, τ}
/// ```
///
/// — a [`Correlator`] with the [`LagKernel::Legendre`] kernel over every
/// origin, averaged over vectors. `C_1 = ⟨cos θ⟩` is the dielectric / IR
/// reorientation observable; the decay of `C_2 = ⟨(3 cos²θ − 1)/2⟩` gives the
/// NMR rotational correlation time (Berne & Pecora, *Dynamic Light
/// Scattering*). The vectors (an O–H bond, a dipole axis, …) are the caller's
/// series — minimum-image bond vectors, say — of any length; each is
/// normalised, and a zero-length vector is an error. The Legendre polynomial
/// is applied before the origin average; for `l ≤ 2` it factorises into
/// products of the unit vectors (`P_1`) and of their quadratic components
/// (`P_2`), so the average runs on the correlator's FFT path, `O(T log T)`;
/// higher orders are summed directly. This is also the orientational
/// autocorrelation ("OACF") of unit vectors — the quantity is one, the name
/// is this one.
///
/// Ported from the reference implementation's reorientation-dynamics
/// analyzer `CReorDyn` (`src/reordyn.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReorientationAcf {
    correlator: Correlator,
}

/// The data [`ReorientationAcf`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct ReorientationAcfInput<'a> {
    /// One vector per molecule and frame, `T × N × 3`.
    pub vectors: ArrayView3<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl ReorientationAcf {
    /// `C_l` over lags `0 … max_lag` frames.
    pub fn new(l: u32, max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Legendre { l }, max_lag),
        }
    }

    /// Average over vectors (`Mean`, default) or keep one curve per vector
    /// (`PerItem`); `PerComponent` is refused at `compute` (the Legendre
    /// kernel is not a sum over components).
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

    /// The correlator this correlation is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for ReorientationAcf {
    type Input<'a> = ReorientationAcfInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: ReorientationAcfInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.vectors,
            b: None,
            dt: input.dt,
            boxes: None,
        })
    }
}

impl Stream for ReorientationAcf {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    ReorientationAcf as "ReorientationAcf";
    params |s| {
        l: Int;
        max_lag: Int;
        reduction: Choice(Reduction::NAMES) = Reduction::default() => s.correlator.reduction();
        stride: Int = Correlator::DEFAULT_STRIDE => s.correlator.stride();
    }
    inputs {
        vectors: Trajectory;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::trajectory;
    use super::*;
    use ndarray::Array3;
    use std::f64::consts::FRAC_PI_2;

    /// One vector of length `len` turning by 90° per frame in the xy plane.
    fn quarter_turns(n_frames: usize, len: F) -> Array3<F> {
        let frames: Vec<Vec<[F; 3]>> = (0..n_frames)
            .map(|t| {
                let angle = t as F * FRAC_PI_2;
                vec![[len * angle.cos(), len * angle.sin(), 0.0]]
            })
            .collect();
        trajectory(&frames)
    }

    fn reorientation(l: u32, vectors: &Array3<F>, max_lag: usize) -> TimeCorrelation {
        ReorientationAcf::new(l, max_lag)
            .compute(ReorientationAcfInput {
                vectors: vectors.view(),
                dt: 1.0,
            })
            .unwrap()
    }

    #[test]
    fn surface_is_its_correlator() {
        let v = quarter_turns(6, 1.0);
        let explicit = Correlator::new(LagKernel::Legendre { l: 2 }, 3);
        assert_eq!(*ReorientationAcf::new(2, 3).correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: v.view(),
                b: None,
                dt: 1.0,
                boxes: None,
            })
            .unwrap();
        assert_eq!(reorientation(2, &v, 3), via_correlator);
    }

    /// Quarter turns: `cos θ(t) = cos(tπ/2)` = 1, 0, −1, 0, … at every origin,
    /// so `C_1 = 1, 0, −1` and `C_2 = 1, −½, 1`.
    #[test]
    fn quarter_turns_give_the_legendre_values_of_the_angle() {
        let v = quarter_turns(5, 1.0);
        let c1 = reorientation(1, &v, 2);
        let c2 = reorientation(2, &v, 2);
        for (lag, (want1, want2)) in [(1.0, 1.0), (0.0, -0.5), (-1.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            assert!((c1.values[[lag, 0]] - want1).abs() < 1e-12, "C1({lag})");
            assert!((c2.values[[lag, 0]] - want2).abs() < 1e-12, "C2({lag})");
        }
    }

    /// The tunables reach the correlator: per-vector curves average to the
    /// mean curve, and a strided surface is the strided correlator.
    #[test]
    fn stride_and_reduction_are_the_correlators() {
        let v = Array3::from_shape_fn((12, 3, 3), |(t, i, d)| {
            ((t * 5 + i * 11 + d * 3) as F * 0.41).sin() + 0.3
        });
        let input = ReorientationAcfInput {
            vectors: v.view(),
            dt: 1.0,
        };
        let strided = ReorientationAcf::new(2, 6).with_stride(4).unwrap();
        let explicit = Correlator::new(LagKernel::Legendre { l: 2 }, 6)
            .with_stride(4)
            .unwrap();
        assert_eq!(*strided.correlator(), explicit);
        let mean = strided.compute(input).unwrap();
        let rows = strided
            .with_reduction(Reduction::PerItem)
            .compute(input)
            .unwrap();
        assert_eq!(rows.n_channels(), 3);
        for lag in 0..=6 {
            let avg = rows.values.row(lag).sum() / 3.0;
            assert!((avg - mean.values[[lag, 0]]).abs() < 1e-12, "lag {lag}");
        }
        assert!(ReorientationAcf::new(2, 6).with_stride(0).is_err());
    }

    #[test]
    fn vector_length_does_not_matter() {
        let unit = reorientation(2, &quarter_turns(6, 1.0), 4);
        let long = reorientation(2, &quarter_turns(6, 3.7), 4);
        for lag in 0..=4 {
            assert!((unit.values[[lag, 0]] - long.values[[lag, 0]]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_zero_vector_is_rejected() {
        let mut v = quarter_turns(4, 1.0);
        v.slice_mut(ndarray::s![2, 0, ..]).fill(0.0);
        let err = ReorientationAcf::new(1, 2)
            .compute(ReorientationAcfInput {
                vectors: v.view(),
                dt: 1.0,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { field: "a", .. }));
    }
}

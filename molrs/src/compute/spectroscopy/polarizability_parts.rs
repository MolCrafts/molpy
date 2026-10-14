//! [`PolarizabilityParts`] — the rotational invariants of a polarizability
//! series that Raman scattering reads.

use molrs::op::F;
use ndarray::{Array2, ArrayView2};

use crate::compute::{Compute, ComputeError};

/// `√w` of the anisotropy weights: the three diagonal differences
/// `xx−yy, yy−zz, zz−xx` carry `½`, the off-diagonals `xy, xz, yz` carry `3`.
fn anisotropy_scales() -> [F; 6] {
    let half = (0.5 as F).sqrt();
    let three = (3.0 as F).sqrt();
    [half, half, half, three, three, three]
}

/// Splits a symmetric polarizability-tensor series into the two rotational
/// invariants of Raman scattering.
///
/// The input is `α(t)` in Voigt order `[xx, yy, zz, xy, xz, yz]`, one row
/// per frame. For each frame the output carries
///
/// - the **isotropic** part, the mean polarizability
///   `ᾱ = (α_xx + α_yy + α_zz) / 3` (one column), and
/// - the **anisotropic** part, the six-vector
///   `a = (√½ (xx−yy), √½ (yy−zz), √½ (zz−xx), √3 xy, √3 xz, √3 yz)`, whose
///   squared norm is the anisotropy invariant
///   `γ² = ½[(xx−yy)² + (yy−zz)² + (zz−xx)²] + 3(xy² + xz² + yz²)`.
///
/// Both parts are linear in `α`, so they commute with a
/// [`Derivative`](crate::compute::Derivative), and the Raman correlations are
/// plain lagged products of them: `⟨ᾱ̇(0) ᾱ̇(t)⟩` and `⟨ȧ(0) · ȧ(t)⟩` are the
/// isotropic and anisotropic inputs of
/// [`RamanSpectrum`](crate::compute::RamanSpectrum), each a
/// [`Correlator`](crate::compute::Correlator) with the `Product` kernel. The
/// same parts of a second tensor series give the cross-correlations of Raman
/// optical activity (`α̇ × Ġ′`), and of a resonant (excitation-frequency
/// dependent) polarizability give resonance Raman.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PolarizabilityParts;

/// The polarizability series [`PolarizabilityParts`] splits.
#[derive(Debug, Clone, Copy)]
pub struct PolarizabilityPartsInput<'a> {
    /// `α(t)`, `T × 6` in Voigt order `[xx, yy, zz, xy, xz, yz]`.
    pub alpha: ArrayView2<'a, F>,
}

/// The isotropic and anisotropic parts of a polarizability series.
#[derive(Debug, Clone, PartialEq)]
pub struct PolarizabilityPartsResult {
    /// Mean polarizability per frame, `T × 1`.
    pub isotropic: Array2<F>,
    /// Weighted anisotropy vector per frame, `T × 6` (its squared norm is
    /// `γ²`).
    pub anisotropic: Array2<F>,
}

impl PolarizabilityParts {
    /// The splitter (it has no configuration).
    pub fn new() -> Self {
        Self
    }
}

impl Compute for PolarizabilityParts {
    type Input<'a> = PolarizabilityPartsInput<'a>;
    type Output = PolarizabilityPartsResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] unless `alpha` has six columns;
    /// [`ComputeError::EmptyInput`] for no frames;
    /// [`ComputeError::NonFinite`] for a non-finite component.
    fn compute(
        &self,
        input: PolarizabilityPartsInput<'_>,
    ) -> Result<PolarizabilityPartsResult, ComputeError> {
        let alpha = input.alpha;
        if alpha.ncols() != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: alpha.ncols(),
                what: "polarizability columns (Voigt [xx, yy, zz, xy, xz, yz])",
            });
        }
        if alpha.nrows() == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(index) = alpha.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "alpha",
                index,
            });
        }
        let scale = anisotropy_scales();
        let isotropic = Array2::from_shape_fn((alpha.nrows(), 1), |(t, _)| {
            (alpha[[t, 0]] + alpha[[t, 1]] + alpha[[t, 2]]) / 3.0
        });
        let anisotropic = Array2::from_shape_fn((alpha.nrows(), 6), |(t, c)| {
            let part = match c {
                0 => alpha[[t, 0]] - alpha[[t, 1]],
                1 => alpha[[t, 1]] - alpha[[t, 2]],
                2 => alpha[[t, 2]] - alpha[[t, 0]],
                voigt => alpha[[t, voigt]],
            };
            scale[c] * part
        });
        Ok(PolarizabilityPartsResult {
            isotropic,
            anisotropic,
        })
    }
}

crate::compute::spec::compute_spec! {
    PolarizabilityParts as "PolarizabilityParts";
    params |s| {}
    inputs {
        alpha: Series;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn parts(alpha: &Array2<F>) -> Result<PolarizabilityPartsResult, ComputeError> {
        PolarizabilityParts::new().compute(PolarizabilityPartsInput {
            alpha: alpha.view(),
        })
    }

    /// An isotropic tensor has no anisotropy; the squared anisotropy vector
    /// of a general tensor is the textbook γ².
    #[test]
    fn parts_are_the_rotational_invariants() {
        let iso = parts(&array![[2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]).unwrap();
        assert_eq!(iso.isotropic, array![[2.0]]);
        assert!(iso.anisotropic.iter().all(|v| v.abs() < 1e-15));

        let (xx, yy, zz, xy, xz, yz) = (3.0, 1.0, -0.5, 0.25, -0.75, 0.4);
        let general = parts(&array![[xx, yy, zz, xy, xz, yz]]).unwrap();
        assert!((general.isotropic[[0, 0]] - (xx + yy + zz) / 3.0).abs() < 1e-15);
        let gamma_sq = 0.5 * ((xx - yy).powi(2) + (yy - zz).powi(2) + (zz - xx).powi(2))
            + 3.0 * (xy * xy + xz * xz + yz * yz);
        let norm_sq: F = general.anisotropic.iter().map(|v| v * v).sum();
        assert!((norm_sq - gamma_sq).abs() < 1e-12);
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(matches!(
            parts(&Array2::zeros((4, 5))),
            Err(ComputeError::DimensionMismatch { expected: 6, .. })
        ));
        assert!(matches!(
            parts(&Array2::zeros((0, 6))),
            Err(ComputeError::EmptyInput)
        ));
        let mut nan = Array2::zeros((2, 6));
        nan[[1, 3]] = F::NAN;
        assert!(matches!(
            parts(&nan),
            Err(ComputeError::NonFinite { index: 9, .. })
        ));
    }
}

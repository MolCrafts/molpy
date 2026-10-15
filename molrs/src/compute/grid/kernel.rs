//! [`SpreadKernel`] — how much of a point a grid voxel receives — and its two
//! kernels, [`Gaussian`] and [`Indicator`].

use std::f64::consts::PI;

use molrs::op::F;

use crate::compute::ComputeError;

/// A radially symmetric kernel that spreads a point over the voxels of a
/// [`RegularGrid`](super::RegularGrid) within its [`cutoff`](Self::cutoff).
///
/// The weight a voxel receives at squared distance `r²` from the point is
/// `normalization(dim) · profile(r²)` for `r ≤ cutoff`, and zero beyond.
pub trait SpreadKernel {
    /// Support radius: voxel centres farther than this receive nothing. The
    /// boundary itself is included.
    fn cutoff(&self) -> F;

    /// The prefactor in `dim` dimensions.
    fn normalization(&self, dim: usize) -> F;

    /// The unnormalized profile at squared distance `r_sq ≤ cutoff²`.
    fn profile(&self, r_sq: F) -> F;
}

/// A normalized Gaussian of width `σ`, truncated at `cutoff` (`3σ` unless
/// set): `(2πσ²)^{−d/2} · exp(−r² / 2σ²)` in `d` dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    sigma: F,
    cutoff: F,
}

impl Gaussian {
    /// The truncation radius in units of `σ` unless [`with_cutoff`](Self::with_cutoff)
    /// sets one: `3σ` holds ≳ 99% of a 3-D Gaussian's mass.
    pub const DEFAULT_CUTOFF_SIGMAS: F = 3.0;

    /// A Gaussian of width `sigma`, truncated at `3σ`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite `sigma`.
    pub fn new(sigma: F) -> Result<Self, ComputeError> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Gaussian::sigma",
                value: sigma.to_string(),
            });
        }
        Ok(Self {
            sigma,
            cutoff: Self::DEFAULT_CUTOFF_SIGMAS * sigma,
        })
    }

    /// Truncate at `cutoff` instead of `3σ`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite cutoff.
    pub fn with_cutoff(mut self, cutoff: F) -> Result<Self, ComputeError> {
        if !(cutoff.is_finite() && cutoff > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Gaussian::cutoff",
                value: cutoff.to_string(),
            });
        }
        self.cutoff = cutoff;
        Ok(self)
    }

    /// The width `σ`.
    pub fn sigma(&self) -> F {
        self.sigma
    }

    fn two_sigma_sq(&self) -> F {
        2.0 * self.sigma * self.sigma
    }
}

impl SpreadKernel for Gaussian {
    fn cutoff(&self) -> F {
        self.cutoff
    }

    fn normalization(&self, dim: usize) -> F {
        (self.two_sigma_sq() * PI).powf(-(dim as F) / 2.0)
    }

    fn profile(&self, r_sq: F) -> F {
        (-r_sq / self.two_sigma_sq()).exp()
    }
}

/// The indicator of a ball of radius `radius`: weight 1 inside (boundary
/// included), 0 outside — a hard sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Indicator {
    radius: F,
}

impl Indicator {
    /// A ball of radius `radius`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite radius.
    pub fn new(radius: F) -> Result<Self, ComputeError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Indicator::radius",
                value: radius.to_string(),
            });
        }
        Ok(Self { radius })
    }
}

impl SpreadKernel for Indicator {
    fn cutoff(&self) -> F {
        self.radius
    }

    fn normalization(&self, _dim: usize) -> F {
        1.0
    }

    fn profile(&self, _r_sq: F) -> F {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_is_normalized_in_its_dimension() {
        let g = Gaussian::new(0.5).unwrap();
        assert_eq!(g.cutoff(), 1.5);
        assert!((g.normalization(3) - (0.5 * PI).powf(-1.5)).abs() < 1e-15);
        assert!((g.normalization(2) - 1.0 / (0.5 * PI)).abs() < 1e-15);
        assert_eq!(g.profile(0.0), 1.0);
        assert!((g.profile(0.25) - (-0.5_f64).exp()).abs() < 1e-15);
        assert_eq!(g.with_cutoff(2.0).unwrap().cutoff(), 2.0);
    }

    #[test]
    fn invalid_parameters_are_errors() {
        for bad in [0.0, -1.0, F::NAN, F::INFINITY] {
            assert!(Gaussian::new(bad).is_err());
            assert!(Indicator::new(bad).is_err());
            assert!(Gaussian::new(1.0).unwrap().with_cutoff(bad).is_err());
        }
    }
}

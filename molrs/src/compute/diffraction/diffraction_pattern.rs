//! 2-D FFT diffraction pattern of a projected frame.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;
use ndarray::{Array2, ArrayView2};

use crate::compute::{Compute, ComputeError, Gaussian, RegularGrid, Spectrum};

/// `DiffractionPattern` analyzer.
///
/// Mirrors `freud.diffraction.DiffractionPattern`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/diffraction/DiffractionPattern.cc)).
///
/// Configuration: the image size `n_grid`, the projection axis, and the
/// [`Gaussian`] smearing kernel (width `σ`, truncated at `3σ` unless set).
/// Data: one snapshot's positions and its box. The particles are projected
/// along the axis onto the box face, spread as 2-D Gaussians on an
/// `n_grid × n_grid` [`RegularGrid`] over that face, and the image's 2-D FFT
/// power `|F(k)|²` is returned.
///
/// # Conventions
///
/// - The output image is FFT-shifted so that `k = 0` sits at the centre
///   `(n_grid / 2, n_grid / 2)`, matching freud (and the usual
///   `numpy.fft.fftshift` convention).
/// - Square grid `(n_grid × n_grid)`; rectangular grids are a follow-up.
/// - Orthorhombic boxes only (matches `freud.DiffractionPattern.compute`).
/// - The smearing wraps across an in-plane box face only when that box axis
///   is periodic.
///
/// For many snapshots, wrap in [`Batch`](crate::compute::Batch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffractionPattern {
    n_grid: usize,
    kernel: Gaussian,
    /// Projection axis: 0 = x, 1 = y, 2 = z (default).
    axis: usize,
}

/// The data a [`DiffractionPattern`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct DiffractionPatternInput<'a> {
    /// Positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box whose face the image covers.
    pub simbox: &'a SimBox,
}

impl DiffractionPattern {
    /// The projection axis when not set: 2 (z).
    pub const DEFAULT_AXIS: usize = 2;

    /// New analyzer with `n_grid × n_grid` pixels and Gaussian smearing
    /// `sigma` (in box-length units), projected along `z`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `n_grid == 0` or a non-positive
    /// `sigma`.
    pub fn new(n_grid: usize, sigma: F) -> Result<Self, ComputeError> {
        if n_grid == 0 {
            return Err(ComputeError::OutOfRange {
                field: "DiffractionPattern::n_grid",
                value: "0".into(),
            });
        }
        Ok(Self {
            n_grid,
            kernel: Gaussian::new(sigma)?,
            axis: Self::DEFAULT_AXIS,
        })
    }

    /// Set the projection axis (0/1/2 = x/y/z).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for an axis other than 0, 1 or 2 (it
    /// used to `assert!`).
    pub fn with_axis(mut self, axis: usize) -> Result<Self, ComputeError> {
        if axis >= 3 {
            return Err(ComputeError::OutOfRange {
                field: "DiffractionPattern::axis (0, 1 or 2)",
                value: axis.to_string(),
            });
        }
        self.axis = axis;
        Ok(self)
    }

    /// Truncate the smearing Gaussian at `cutoff` instead of `3σ`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite cutoff.
    pub fn with_cutoff(mut self, cutoff: F) -> Result<Self, ComputeError> {
        self.kernel = self.kernel.with_cutoff(cutoff)?;
        Ok(self)
    }

    /// Pixels per side.
    pub fn n_grid(&self) -> usize {
        self.n_grid
    }

    /// The smearing width `σ`.
    pub fn sigma(&self) -> F {
        self.kernel.sigma()
    }

    /// The projection axis (0/1/2 = x/y/z).
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl Compute for DiffractionPattern {
    type Input<'a> = DiffractionPatternInput<'a>;
    type Output = DiffractionPatternResult;

    /// # Errors
    /// [`ComputeError::OutOfRange`] for a triclinic box;
    /// [`ComputeError::BadShape`] for positions without 3 columns;
    /// [`ComputeError::NonFinite`] for a non-finite position.
    fn compute(
        &self,
        input: DiffractionPatternInput<'_>,
    ) -> Result<DiffractionPatternResult, ComputeError> {
        let simbox = input.simbox;
        let lens = match simbox.kind() {
            BoxKind::Ortho { len, .. } => [len[0], len[1], len[2]],
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "DiffractionPattern::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };
        if input.positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("N × {}", input.positions.ncols()),
            });
        }
        let (a0, a1) = match self.axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let n = self.n_grid;
        let origin = simbox.origin_view();
        let pbc = simbox.pbc();
        let grid = RegularGrid::new(
            [origin[a0], origin[a1]],
            [[lens[a0] / n as F, 0.0], [0.0, lens[a1] / n as F]],
            [n, n],
            [pbc[a0], pbc[a1]],
        )?;

        let mut image = Array2::<F>::zeros((n, n));
        for p in input.positions.outer_iter() {
            grid.spread(&self.kernel, [p[a0], p[a1]], |[i, j], w| {
                image[[i, j]] += w;
            })?;
        }
        let diffraction = Spectrum::power_2d_centered(image.view());
        Ok(DiffractionPatternResult { diffraction, image })
    }
}

/// One snapshot's diffraction pattern.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DiffractionPatternResult {
    /// Power spectrum `|F(k)|²`, shape `(n_grid, n_grid)`. FFT-shifted so
    /// the k=0 component is at the centre.
    pub diffraction: Array2<F>,
    /// Real-space Gaussian-smeared image used as the FFT input
    /// (`(n_grid, n_grid)`).
    pub image: Array2<F>,
}

crate::compute::spec::compute_spec! {
    DiffractionPattern as "DiffractionPattern";
    params |s| {
        n_grid: Int;
        sigma: Float;
        axis: Int = DiffractionPattern::DEFAULT_AXIS => s.axis;
        cutoff: Float
            = crate::compute::ParamDefault::Scaled {
                of: "sigma",
                factor: Gaussian::DEFAULT_CUTOFF_SIGMAS,
            }
            => crate::compute::SpreadKernel::cutoff(&s.kernel);
    }
    inputs {
        positions: Positions;
        simbox: SimBox;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Frame {
        positions: Array2<F>,
        simbox: SimBox,
    }

    impl Frame {
        fn input(&self) -> DiffractionPatternInput<'_> {
            DiffractionPatternInput {
                positions: self.positions.view(),
                simbox: &self.simbox,
            }
        }
    }

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        Frame {
            positions: Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]),
            simbox: SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [true; 3]).unwrap(),
        }
    }

    #[test]
    fn empty_frame_image_is_zero() {
        let frame = frame_with(&[], 10.0);
        let r = DiffractionPattern::new(32, 0.3)
            .unwrap()
            .compute(frame.input())
            .unwrap();
        let s: F = r.image.iter().copied().sum();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn single_particle_dc_component_dominant() {
        // For a single Gaussian at the centre, |F(k=0)|² should be the
        // largest entry in the shifted diffraction image.
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0);
        let r = DiffractionPattern::new(32, 0.4)
            .unwrap()
            .compute(frame.input())
            .unwrap();
        let dc = r.diffraction[[16, 16]];
        let mut max_off = 0.0_f64;
        for ((i, j), &v) in r.diffraction.indexed_iter() {
            if !(i == 16 && j == 16) && v > max_off {
                max_off = v;
            }
        }
        assert!(
            dc > max_off,
            "DC component {dc} must dominate; saw off-DC max {max_off}"
        );
    }

    #[test]
    fn periodic_lattice_has_bragg_peaks() {
        // 4×4 square lattice in xy gives Bragg-like peaks at non-DC k.
        let mut positions = Vec::new();
        let a = 2.5_f64;
        for ix in 0..4 {
            for iy in 0..4 {
                positions.push([ix as F * a, iy as F * a, 5.0]);
            }
        }
        let frame = frame_with(&positions, a * 4.0);
        let r = DiffractionPattern::new(32, 0.2)
            .unwrap()
            .compute(frame.input())
            .unwrap();
        // Some off-DC element must be larger than typical background.
        let dc = r.diffraction[[16, 16]];
        let mut max_off = 0.0_f64;
        for ((i, j), &v) in r.diffraction.indexed_iter() {
            if !(i == 16 && j == 16) && v > max_off {
                max_off = v;
            }
        }
        // Lattices produce strong off-DC peaks comparable to the DC
        // (within a factor of a few in this small system).
        assert!(
            max_off > 0.05 * dc,
            "Bragg peak off-DC max ({max_off}) should be a non-negligible fraction of DC ({dc})"
        );
    }

    #[test]
    fn invalid_args_error() {
        assert!(DiffractionPattern::new(0, 0.3).is_err());
        assert!(DiffractionPattern::new(32, 0.0).is_err());
        assert!(DiffractionPattern::new(32, -1.0).is_err());
        assert!(
            DiffractionPattern::new(32, 0.3)
                .unwrap()
                .with_cutoff(0.0)
                .is_err()
        );
    }

    /// `with_axis` used to `assert!`; an axis past z is an error.
    #[test]
    fn invalid_projection_axis_is_out_of_range() {
        let dp = DiffractionPattern::new(8, 0.3).unwrap();
        assert!(matches!(
            dp.with_axis(3),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert_eq!(dp.with_axis(0).unwrap().axis(), 0);
    }

    /// The smearing wraps across a periodic face only; an open face drops
    /// what falls past it (it used to wrap regardless).
    #[test]
    fn smearing_wraps_only_periodic_faces() {
        let positions = array![[0.05, 5.0, 5.0]];
        let open = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false; 3]).unwrap();
        let periodic = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
        let dp = DiffractionPattern::new(20, 0.3).unwrap();
        let image = |simbox: &SimBox| {
            dp.compute(DiffractionPatternInput {
                positions: positions.view(),
                simbox,
            })
            .unwrap()
            .image
        };
        assert!(image(&periodic)[[19, 10]] > 0.0);
        assert_eq!(image(&open)[[19, 10]], 0.0);
    }
}

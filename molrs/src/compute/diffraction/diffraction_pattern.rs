//! 2-D FFT diffraction pattern of a projected frame.

// 2-D grid kernel reads naturally with double index loops; FFT2 helper
// also needs explicit row/column index ordering.
#![allow(clippy::needless_range_loop)]

use crate::compute::ComputeResult;
use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array2;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex as RfComplex;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// `DiffractionPattern` analyzer.
///
/// Mirrors `freud.diffraction.DiffractionPattern`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/diffraction/DiffractionPattern.cc)).
///
/// Builds a 2-D Gaussian-smeared image of the projected particle positions,
/// takes a 2-D FFT (composed from `rustfft`'s 1-D plans), and returns the
/// power spectrum `|F(k)|²`. Projection axis defaults to `+z`, mapping
/// particles onto the `xy` plane.
///
/// This is the first analyzer in the port to use `rustfft 6` for real 2-D
/// Fourier work; the FFT planner is built once per frame and reused for
/// both row and column passes.
///
/// # Conventions
///
/// - The output image is FFT-shifted so that `k = 0` sits at the centre
///   `(n_grid / 2, n_grid / 2)`, matching freud (and the usual
///   `numpy.fft.fftshift` convention).
/// - Square grid `(n_grid × n_grid)`; rectangular grids are a follow-up.
/// - Orthorhombic boxes only (matches `freud.DiffractionPattern.compute`).
#[derive(Debug, Clone, Copy)]
pub struct DiffractionPattern {
    n_grid: usize,
    sigma: F,
    /// Projection axis: 0 = x, 1 = y, 2 = z (default).
    axis: usize,
}

impl DiffractionPattern {
    /// New analyzer with `n_grid × n_grid` pixels and Gaussian smearing
    /// `sigma` (in box-length units).
    pub fn new(n_grid: usize, sigma: F) -> Result<Self, ComputeError> {
        if n_grid == 0 {
            return Err(ComputeError::OutOfRange {
                field: "DiffractionPattern::n_grid",
                value: "0".into(),
            });
        }
        if sigma.is_nan() || sigma <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "DiffractionPattern::sigma",
                value: sigma.to_string(),
            });
        }
        Ok(Self {
            n_grid,
            sigma,
            axis: 2,
        })
    }

    /// Set the projection axis (0/1/2 = x/y/z).
    pub fn with_axis(mut self, axis: usize) -> Self {
        assert!(axis < 3);
        self.axis = axis;
        self
    }

    pub fn n_grid(&self) -> usize {
        self.n_grid
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
    ) -> Result<DiffractionPatternResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let lens = match simbox.kind() {
            BoxKind::Ortho { len, .. } => [len[0], len[1], len[2]],
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "DiffractionPattern::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };
        let (a0, a1) = match self.axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let l0 = lens[a0];
        let l1 = lens[a1];

        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let coords = [xs_p.slice(), ys_p.slice(), zs_p.slice()];

        let n = self.n_grid;
        let d0 = l0 / n as F;
        let d1 = l1 / n as F;
        let r_max = 3.0 * self.sigma;
        let half_k0 = (r_max / d0).ceil() as isize;
        let half_k1 = (r_max / d1).ceil() as isize;
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;
        let pref = (two_sigma_sq * std::f64::consts::PI).powf(-1.0);
        let r_max_sq = r_max * r_max;

        let mut image = Array2::<F>::zeros((n, n));
        let origin = simbox.origin_view();
        let o0 = origin[a0];
        let o1 = origin[a1];

        for k in 0..coords[0].len() {
            let p0 = coords[a0][k];
            let p1 = coords[a1][k];
            let c0 = ((p0 - o0) / d0).floor() as isize;
            let c1 = ((p1 - o1) / d1).floor() as isize;
            for i0 in (c0 - half_k0)..=(c0 + half_k0) {
                let v0 = o0 + (i0 as F + 0.5) * d0 - p0;
                if v0.abs() > r_max {
                    continue;
                }
                let g0 = i0.rem_euclid(n as isize) as usize;
                for i1 in (c1 - half_k1)..=(c1 + half_k1) {
                    let v1 = o1 + (i1 as F + 0.5) * d1 - p1;
                    let r2 = v0 * v0 + v1 * v1;
                    if r2 > r_max_sq {
                        continue;
                    }
                    let g1 = i1.rem_euclid(n as isize) as usize;
                    image[[g0, g1]] += pref * (-r2 / two_sigma_sq).exp();
                }
            }
        }

        let diffraction = fft2_power_shifted(&image, n);

        Ok(DiffractionPatternResult { diffraction, image })
    }
}

/// 2-D FFT via row+column 1-D FFTs, return `|F|²` shifted so k=0 is at
/// the grid centre.
fn fft2_power_shifted(image: &Array2<F>, n: usize) -> Array2<F> {
    let mut planner = FftPlanner::<F>::new();
    let fft = planner.plan_fft_forward(n);

    // Build a complex buffer.
    let mut buf: Vec<RfComplex<F>> = vec![RfComplex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            buf[i * n + j] = RfComplex::new(image[[i, j]], 0.0);
        }
    }

    // Row pass: FFT every row of length `n`.
    {
        let mut scratch = vec![RfComplex::new(0.0, 0.0); n];
        for i in 0..n {
            scratch.copy_from_slice(&buf[i * n..(i + 1) * n]);
            fft.process(&mut scratch);
            buf[i * n..(i + 1) * n].copy_from_slice(&scratch);
        }
    }
    // Column pass: gather column → FFT → scatter back.
    {
        let mut col = vec![RfComplex::new(0.0, 0.0); n];
        for j in 0..n {
            for i in 0..n {
                col[i] = buf[i * n + j];
            }
            fft.process(&mut col);
            for i in 0..n {
                buf[i * n + j] = col[i];
            }
        }
    }

    // Magnitude squared + FFT-shift (swap upper-half and lower-half along
    // both axes) so DC sits at (n/2, n/2).
    let mut shifted = Array2::<F>::zeros((n, n));
    let half = n / 2;
    for i in 0..n {
        for j in 0..n {
            let si = (i + half) % n;
            let sj = (j + half) % n;
            shifted[[si, sj]] = buf[i * n + j].norm_sqr();
        }
    }
    shifted
}

impl LegacyCompute for DiffractionPattern {
    type Args<'a> = ();
    type Output = Vec<DiffractionPatternResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _: (),
    ) -> Result<Vec<DiffractionPatternResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames.par_iter().map(|f| self.one_frame(*f)).collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            out.push(self.one_frame(*f)?);
        }
        Ok(out)
    }
}

/// Per-frame diffraction-pattern result.
#[derive(Debug, Clone, Default)]
pub struct DiffractionPatternResult {
    /// Power spectrum `|F(k)|²`, shape `(n_grid, n_grid)`. FFT-shifted so
    /// the k=0 component is at the centre.
    pub diffraction: Array2<F>,
    /// Real-space Gaussian-smeared image used as the FFT input
    /// (`(n_grid, n_grid)`).
    pub image: Array2<F>,
}

impl ComputeResult for DiffractionPatternResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [true; 3]).unwrap());
        frame
    }

    #[test]
    fn empty_frame_image_is_zero() {
        let frame = frame_with(&[], 10.0);
        let r = &DiffractionPattern::new(32, 0.3)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
        let s: F = r.image.iter().copied().sum();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn single_particle_dc_component_dominant() {
        // For a single Gaussian at the centre, |F(k=0)|² should be the
        // largest entry in the shifted diffraction image.
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0);
        let r = &DiffractionPattern::new(32, 0.4)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
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
        let r = &DiffractionPattern::new(32, 0.2)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
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
    }
}

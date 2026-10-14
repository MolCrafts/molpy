//! Per-particle spherical-harmonic descriptors of the local neighborhood.

use crate::compute::ComputeResult;
use molrs::core::Complex;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::core::ylm_all;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_disp;

/// `LocalDescriptors` analyzer (Sph-mode).
///
/// Mirrors `freud.environment.LocalDescriptors`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/LocalDescriptors.cc)).
///
/// For each neighbor pair `(i, j)` computes the complex spherical-harmonic
/// coefficients `Y_ℓm(θ_ij, φ_ij)` for every ℓ from 0 to `l_max`. The
/// output is a per-pair table of shape `(n_pairs, n_sphs)` where
/// `n_sphs = Σ_{ℓ=0}^{l_max} (2ℓ + 1) = (l_max + 1)²`. These features are
/// the standard input for ML models that classify local environments
/// (e.g. crystal-vs-liquid classifiers, polymorph identifiers).
///
/// Three modes are supported by freud (`LocalNeighborhood`,
/// `Global`, `ParticleLocal`); the simplest **LocalNeighborhood** mode is
/// implemented here, which expresses the bond direction in the lab frame.
/// `ParticleLocal` (rotate by per-particle quaternion before evaluating
/// `Y_ℓm`) is a follow-up.
#[derive(Debug, Clone, Copy)]
pub struct LocalDescriptors {
    l_max: u32,
}

impl LocalDescriptors {
    /// Spherical-harmonic band limit: descriptors for ℓ = 0…`l_max`.
    pub fn new(l_max: u32) -> Self {
        Self { l_max }
    }

    pub fn l_max(&self) -> u32 {
        self.l_max
    }

    pub fn n_sphs(&self) -> usize {
        ((self.l_max + 1) * (self.l_max + 1)) as usize
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        _frame: &FA,
        nlist: &Neighbors,
    ) -> Result<LocalDescriptorsResult, ComputeError> {
        let n_sphs = self.n_sphs();
        let n_pairs = nlist.n_pairs();
        let mut out = vec![Complex::ZERO; n_pairs * n_sphs];
        // The descriptors are spherical harmonics of the bond direction.
        let disp = require_disp(nlist)?;

        let mut ylm_buf = vec![Complex::ZERO; (2 * self.l_max + 1) as usize];

        for k in 0..n_pairs {
            let dx = disp[[k, 0]];
            let dy = disp[[k, 1]];
            let dz = disp[[k, 2]];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r == 0.0 {
                continue;
            }
            let theta = (dz / r).clamp(-1.0, 1.0).acos();
            let phi = dy.atan2(dx);
            let row_off = k * n_sphs;
            let mut sph_off = 0_usize;
            for l in 0..=self.l_max {
                let band = (2 * l + 1) as usize;
                let slice = &mut ylm_buf[..band];
                ylm_all(l, theta, phi, slice);
                out[row_off + sph_off..row_off + sph_off + band].copy_from_slice(slice);
                sph_off += band;
            }
        }
        Ok(LocalDescriptorsResult {
            l_max: self.l_max,
            descriptors: out,
            n_sphs,
        })
    }
}

impl LegacyCompute for LocalDescriptors {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<LocalDescriptorsResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<LocalDescriptorsResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(nlists.par_iter())
                .map(|(f, nl)| self.one_frame(*f, nl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame descriptors.
#[derive(Debug, Clone, Default)]
pub struct LocalDescriptorsResult {
    /// `l_max` used for this run.
    pub l_max: u32,
    /// Per-pair descriptor row, length `n_pairs · n_sphs`. The descriptor
    /// for pair `k` and ℓ-band component `(ℓ, m)` lives at index
    /// `k · n_sphs + ℓ² + (m + ℓ as i32) as usize`.
    pub descriptors: Vec<Complex>,
    /// `n_sphs = (l_max + 1)²` for convenience.
    pub n_sphs: usize,
}

impl ComputeResult for LocalDescriptorsResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use molrs::core::ylm_complex;
    use molrs::op::F;
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
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    #[test]
    fn descriptor_at_l0_is_constant() {
        // For a single bond pointing in +x, the Y_0^0 component is
        // 1/(2√π) regardless of bond direction.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &LocalDescriptors::new(2).compute(&[&frame], &[nl]).unwrap()[0];
        assert_eq!(r.n_sphs, 9); // (2+1)² = 9
        // Pair 0, ℓ=0, m=0 is at offset 0.
        let y00 = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
        assert!((r.descriptors[0].re - y00).abs() < 1e-12);
        assert!(r.descriptors[0].im.abs() < 1e-12);
    }

    #[test]
    fn descriptor_matches_direct_ylm_call() {
        // Sanity: the descriptor for a single bond equals ylm_complex
        // evaluated directly at the bond's (θ, φ).
        let dx = 0.6;
        let dy = 0.5;
        let dz = 0.7;
        let frame = frame_with(&[[0.0, 0.0, 0.0], [dx, dy, dz]], 10.0);
        let nl = nlist_from_frame(&frame, 2.0);
        let r = &LocalDescriptors::new(3).compute(&[&frame], &[nl]).unwrap()[0];
        let r2 = (dx * dx + dy * dy + dz * dz).sqrt();
        let theta = (dz / r2).acos();
        let phi = dy.atan2(dx);
        // For ℓ=2 the descriptor lives at offsets 4..9 (after ℓ=0 [1] and ℓ=1 [3]).
        let off_l2 = 4_usize;
        for m in -2..=2 {
            let expected = ylm_complex(2, m, theta, phi);
            let got = r.descriptors[off_l2 + (m + 2) as usize];
            assert!(
                (expected.re - got.re).abs() < 1e-12 && (expected.im - got.im).abs() < 1e-12,
                "ℓ=2 m={m} mismatch"
            );
        }
    }

    #[test]
    fn empty_frames_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = LocalDescriptors::new(4)
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn n_sphs_helper_is_correct() {
        assert_eq!(LocalDescriptors::new(0).n_sphs(), 1);
        assert_eq!(LocalDescriptors::new(2).n_sphs(), 9);
        assert_eq!(LocalDescriptors::new(6).n_sphs(), 49);
    }
}

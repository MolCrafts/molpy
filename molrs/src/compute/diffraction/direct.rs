//! Direct k-grid evaluation of the static structure factor.

use crate::compute::ComputeResult;
use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;
use std::f64::consts::TAU;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

#[derive(Debug, Clone)]
enum KMode {
    /// User-supplied k-vectors. `sk[i]` = `S(k_vecs[i])`.
    Explicit { k_vecs: Vec<[F; 3]> },
    /// Spherically averaged S(|k|) on a reciprocal-lattice grid.
    Isotropic { k_max: F, n_bins: usize },
}

/// Direct k-grid static-structure-factor calculator.
///
/// Mirrors `freud.diffraction.StaticStructureFactorDirect`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/diffraction/StaticStructureFactorDirect.cc)).
///
/// Evaluates
///
/// ```text
///   S(k) = (1/N) | Σ_j exp(i k · r_j) |²
/// ```
///
/// on an explicit array of k-vectors. Two convenience constructors are
/// offered:
///
/// - [`StaticStructureFactorDirect::new`] — user-supplied k-vectors
///   (full freedom).
/// - [`StaticStructureFactorDirect::isotropic`] — for orthorhombic boxes,
///   builds a 3-D reciprocal-lattice grid with k ≤ k_max and spherically
///   averages into uniformly spaced k-magnitude bins.
///
/// Unlike [`StaticStructureFactorDebye`](crate::compute::StaticStructureFactorDebye), this analyzer respects the supplied SimBox: the
/// reciprocal-lattice spacing comes from `2π / L_d` along each axis.
#[derive(Debug, Clone)]
pub struct StaticStructureFactorDirect {
    mode: KMode,
}

impl StaticStructureFactorDirect {
    /// Build from an explicit list of k-vectors (Å⁻¹).
    pub fn new(k_vecs: &[[F; 3]]) -> Result<Self, ComputeError> {
        if k_vecs.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "StaticStructureFactorDirect::k_vecs",
                value: "empty".into(),
            });
        }
        Ok(Self {
            mode: KMode::Explicit {
                k_vecs: k_vecs.to_vec(),
            },
        })
    }

    /// Build the spherically averaged form. `n_bins` magnitude bins between
    /// 0 and `k_max`. Reciprocal-lattice points are read from the SimBox
    /// during `compute`.
    pub fn isotropic(k_max: F, n_bins: usize) -> Result<Self, ComputeError> {
        if k_max.is_nan() || k_max <= 0.0 || n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "StaticStructureFactorDirect::isotropic",
                value: format!("k_max={k_max}, n_bins={n_bins}"),
            });
        }
        Ok(Self {
            mode: KMode::Isotropic { k_max, n_bins },
        })
    }

    fn evaluate_explicit<FA: FrameAccess>(
        frame: &FA,
        k_vecs: &[[F; 3]],
    ) -> Result<StaticStructureFactorDirectResult, ComputeError> {
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let n = xs.len();
        let inv_n = if n > 0 { 1.0 / n as F } else { 0.0 };

        let mut sk = Array1::<F>::zeros(k_vecs.len());
        let mut kmags = Array1::<F>::zeros(k_vecs.len());
        for (idx, k) in k_vecs.iter().enumerate() {
            kmags[idx] = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
            // Direct summation Σ_j exp(i k·r_j) for every k: O(N × N_k) by
            // definition of the estimator. The FFT route is `diffraction_pattern`.
            let mut re: F = 0.0;
            let mut im: F = 0.0;
            for j in 0..n {
                let phase = k[0] * xs[j] + k[1] * ys[j] + k[2] * zs[j];
                re += phase.cos();
                im += phase.sin();
            }
            sk[idx] = inv_n * (re * re + im * im);
        }
        Ok(StaticStructureFactorDirectResult {
            k_magnitudes: kmags,
            sk,
            n_particles: n,
        })
    }

    fn evaluate_isotropic<FA: FrameAccess>(
        frame: &FA,
        k_max: F,
        n_bins: usize,
    ) -> Result<StaticStructureFactorDirectResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly, lz) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "StaticStructureFactorDirect::isotropic::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };
        let dkx = TAU / lx;
        let dky = TAU / ly;
        let dkz = TAU / lz;
        let nx = (k_max / dkx).ceil() as i32;
        let ny = (k_max / dky).ceil() as i32;
        let nz = (k_max / dkz).ceil() as i32;

        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let n_atoms = xs.len();
        let inv_n = if n_atoms > 0 { 1.0 / n_atoms as F } else { 0.0 };

        let dk = k_max / n_bins as F;
        let mut sk_sum = vec![0.0_f64; n_bins];
        let mut counts = vec![0_u64; n_bins];

        for ix in -nx..=nx {
            for iy in -ny..=ny {
                for iz in -nz..=nz {
                    if ix == 0 && iy == 0 && iz == 0 {
                        continue;
                    }
                    let kx = ix as F * dkx;
                    let ky = iy as F * dky;
                    let kz = iz as F * dkz;
                    let kmag = (kx * kx + ky * ky + kz * kz).sqrt();
                    if kmag > k_max || kmag <= 0.0 {
                        continue;
                    }
                    let bin = ((kmag / dk) as usize).min(n_bins - 1);
                    // Same direct sum over the reciprocal lattice: O(N × N_k).
                    let mut re: F = 0.0;
                    let mut im: F = 0.0;
                    for j in 0..n_atoms {
                        let phase = kx * xs[j] + ky * ys[j] + kz * zs[j];
                        re += phase.cos();
                        im += phase.sin();
                    }
                    sk_sum[bin] += inv_n * (re * re + im * im);
                    counts[bin] += 1;
                }
            }
        }

        let mut sk = Array1::<F>::zeros(n_bins);
        let mut kmags = Array1::<F>::zeros(n_bins);
        for b in 0..n_bins {
            kmags[b] = (b as F + 0.5) * dk;
            if counts[b] > 0 {
                sk[b] = sk_sum[b] / counts[b] as F;
            }
        }
        Ok(StaticStructureFactorDirectResult {
            k_magnitudes: kmags,
            sk,
            n_particles: n_atoms,
        })
    }
}

impl LegacyCompute for StaticStructureFactorDirect {
    type Args<'a> = ();
    type Output = Vec<StaticStructureFactorDirectResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _: (),
    ) -> Result<Vec<StaticStructureFactorDirectResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .map(|f| match &self.mode {
                    KMode::Explicit { k_vecs } => Self::evaluate_explicit(*f, k_vecs),
                    KMode::Isotropic { k_max, n_bins } => {
                        Self::evaluate_isotropic(*f, *k_max, *n_bins)
                    }
                })
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            let r = match &self.mode {
                KMode::Explicit { k_vecs } => Self::evaluate_explicit(*f, k_vecs)?,
                KMode::Isotropic { k_max, n_bins } => {
                    Self::evaluate_isotropic(*f, *k_max, *n_bins)?
                }
            };
            out.push(r);
        }
        Ok(out)
    }
}

/// Per-frame direct-SSF result.
///
/// When the analyzer was constructed via [`StaticStructureFactorDirect::new`]
/// (explicit k-vectors), `sk` holds `S(k_vec)` for each input k-vector and
/// `k_magnitudes` holds their magnitudes. With
/// [`isotropic`](StaticStructureFactorDirect::isotropic), `sk` is the
/// spherically averaged `S(|k|)` and `k_magnitudes` is the bin-centres
/// array.
#[derive(Debug, Clone, Default)]
pub struct StaticStructureFactorDirectResult {
    pub k_magnitudes: Array1<F>,
    pub sk: Array1<F>,
    pub n_particles: usize,
}

impl ComputeResult for StaticStructureFactorDirectResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
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
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    const TOL: F = 1e-10;

    #[test]
    fn s_at_zero_k_equals_n() {
        // S(k=0) = (1/N) |Σ exp(0)|² = N
        let frame = frame_with(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            10.0,
            [false; 3],
        );
        let r = &StaticStructureFactorDirect::new(&[[0.0, 0.0, 0.0]])
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
        assert!((r.sk[0] - 3.0).abs() < TOL);
    }

    #[test]
    fn two_particle_analytic_explicit() {
        // For two particles separated by d along x:
        // S(k) = (1/2) |1 + exp(i k·d ê_x)|²
        //      = (1/2) (2 + 2 cos(k_x d))
        //      = 1 + cos(k_x d)
        let d = 1.5_f64;
        let frame = frame_with(&[[0.0, 0.0, 0.0], [d, 0.0, 0.0]], 10.0, [false; 3]);
        let kx_vals = [0.5_f64, 1.2, 2.7];
        let k_vecs: Vec<[F; 3]> = kx_vals.iter().map(|&k| [k, 0.0, 0.0]).collect();
        let r = &StaticStructureFactorDirect::new(&k_vecs)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
        for (i, &k) in kx_vals.iter().enumerate() {
            let expected = 1.0 + (k * d).cos();
            assert!(
                (r.sk[i] - expected).abs() < TOL,
                "k={k}: got {}, expected {expected}",
                r.sk[i]
            );
        }
    }

    #[test]
    fn isotropic_has_bragg_peak_for_lattice() {
        // 4 particles on a simple cubic motif inside a 4×4×4 box → strong
        // peak at k = 2π/a with a = 1 (the lattice spacing). The (1,0,0)
        // reciprocal-lattice vector has magnitude 2π/1 = 2π ≈ 6.28.
        let mut positions = Vec::new();
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    positions.push([ix as F, iy as F, iz as F]);
                }
            }
        }
        let frame = frame_with(&positions, 4.0, [true, true, true]);
        let r = &StaticStructureFactorDirect::isotropic(8.0, 16)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
        // Find the largest S(k) in the range [5.5, 7.0]
        let mut max_sk = 0.0_f64;
        let mut max_k = 0.0_f64;
        for b in 0..16 {
            let k = r.k_magnitudes[b];
            if (5.5..=7.0).contains(&k) && r.sk[b] > max_sk {
                max_sk = r.sk[b];
                max_k = k;
            }
        }
        // Expect the peak near k ≈ 2π and S(k) ≫ 1 for a perfect lattice.
        assert!(
            max_sk > 5.0,
            "expected a Bragg peak near k = 2π with S ≫ 1, got S({max_k}) = {max_sk}",
        );
    }

    #[test]
    fn invalid_inputs_error() {
        assert!(StaticStructureFactorDirect::new(&[]).is_err());
        assert!(StaticStructureFactorDirect::isotropic(0.0, 10).is_err());
        assert!(StaticStructureFactorDirect::isotropic(1.0, 0).is_err());
    }

    #[test]
    fn empty_frame_returns_zero_sk() {
        let frame = frame_with(&[], 10.0, [false; 3]);
        let r = &StaticStructureFactorDirect::new(&[[1.0, 0.0, 0.0]])
            .unwrap()
            .compute(&[&frame], ())
            .unwrap()[0];
        assert_eq!(r.n_particles, 0);
        assert_eq!(r.sk[0], 0.0);
    }

    /// The `frames.len() >= PAR_THRESHOLD` rayon branch must match the
    /// serial path exactly, in frame order.
    #[test]
    fn parallel_matches_serial() {
        let frame = frame_with(
            &[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            10.0,
            [false; 3],
        );
        let k_vecs: Vec<[F; 3]> = [0.5_f64, 1.2, 2.7].iter().map(|&k| [k, 0.0, 0.0]).collect();
        let s = StaticStructureFactorDirect::new(&k_vecs).unwrap();
        let solo = s.compute(&[&frame], ()).unwrap();
        let par = s.compute(&[&frame, &frame], ()).unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].sk, solo[0].sk);
        assert_eq!(par[1].sk, solo[0].sk);
    }
}

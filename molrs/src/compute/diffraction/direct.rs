//! Direct k-grid evaluation of the static structure factor.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;
use ndarray::{Array1, ArrayView2};
use std::f64::consts::TAU;

use crate::compute::{Axis, Compute, ComputeError};

#[derive(Debug, Clone, PartialEq)]
enum KMode {
    /// User-supplied k-vectors. `sk[i]` = `S(k_vecs[i])`.
    Explicit { k_vecs: Vec<[F; 3]> },
    /// Spherically averaged S(|k|) on a reciprocal-lattice grid, binned on
    /// `[0, k_max]` (the upper edge closed).
    Isotropic { axis: Axis },
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
/// Unlike [`StaticStructureFactorDebye`](crate::compute::StaticStructureFactorDebye),
/// the isotropic form reads the box given to `compute`: the reciprocal-lattice
/// spacing comes from `2π / L_d` along each axis. Explicit k-vectors need no
/// box.
///
/// For many snapshots, wrap in [`Batch`](crate::compute::Batch).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticStructureFactorDirect {
    mode: KMode,
}

/// The data a [`StaticStructureFactorDirect`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct StaticStructureFactorDirectInput<'a> {
    /// Positions, `N × 3`, Å.
    pub positions: ArrayView2<'a, F>,
    /// The box whose reciprocal lattice the isotropic form samples; unused
    /// with explicit k-vectors.
    pub simbox: Option<&'a SimBox>,
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
            mode: KMode::Isotropic {
                axis: Axis::new(0.0, k_max, n_bins)?,
            },
        })
    }

    fn evaluate_explicit(
        positions: ArrayView2<'_, F>,
        k_vecs: &[[F; 3]],
    ) -> StaticStructureFactorDirectResult {
        let n = positions.nrows();
        let inv_n = if n > 0 { 1.0 / n as F } else { 0.0 };

        let mut sk = Array1::<F>::zeros(k_vecs.len());
        let mut kmags = Array1::<F>::zeros(k_vecs.len());
        for (idx, k) in k_vecs.iter().enumerate() {
            kmags[idx] = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
            // Direct summation Σ_j exp(i k·r_j) for every k: O(N × N_k) by
            // definition of the estimator. The FFT route is `diffraction_pattern`.
            let mut re: F = 0.0;
            let mut im: F = 0.0;
            for r in positions.outer_iter() {
                let phase = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
                re += phase.cos();
                im += phase.sin();
            }
            sk[idx] = inv_n * (re * re + im * im);
        }
        StaticStructureFactorDirectResult {
            k_magnitudes: kmags,
            sk,
            n_particles: n,
        }
    }

    fn evaluate_isotropic(
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
        axis: Axis,
    ) -> Result<StaticStructureFactorDirectResult, ComputeError> {
        let simbox = simbox.ok_or_else(|| ComputeError::BadShape {
            expected: "a SimBox: the isotropic S(k) samples its reciprocal lattice".into(),
            got: "no box".into(),
        })?;
        let k_max = axis.max();
        let n_bins = axis.n_bins();
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

        let n_atoms = positions.nrows();
        let inv_n = if n_atoms > 0 { 1.0 / n_atoms as F } else { 0.0 };

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
                    let Some(bin) = axis.index_closed(kmag)? else {
                        continue;
                    };
                    // Same direct sum over the reciprocal lattice: O(N × N_k).
                    let mut re: F = 0.0;
                    let mut im: F = 0.0;
                    for r in positions.outer_iter() {
                        let phase = kx * r[0] + ky * r[1] + kz * r[2];
                        re += phase.cos();
                        im += phase.sin();
                    }
                    sk_sum[bin] += inv_n * (re * re + im * im);
                    counts[bin] += 1;
                }
            }
        }

        let mut sk = Array1::<F>::zeros(n_bins);
        for b in 0..n_bins {
            if counts[b] > 0 {
                sk[b] = sk_sum[b] / counts[b] as F;
            }
        }
        Ok(StaticStructureFactorDirectResult {
            k_magnitudes: axis.centers(),
            sk,
            n_particles: n_atoms,
        })
    }
}

impl Compute for StaticStructureFactorDirect {
    type Input<'a> = StaticStructureFactorDirectInput<'a>;
    type Output = StaticStructureFactorDirectResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for positions without 3 columns, or the
    /// isotropic form without a box; [`ComputeError::OutOfRange`] for a
    /// triclinic box (isotropic form).
    fn compute(
        &self,
        input: StaticStructureFactorDirectInput<'_>,
    ) -> Result<StaticStructureFactorDirectResult, ComputeError> {
        if input.positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("N × {}", input.positions.ncols()),
            });
        }
        match &self.mode {
            KMode::Explicit { k_vecs } => Ok(Self::evaluate_explicit(input.positions, k_vecs)),
            KMode::Isotropic { axis } => {
                Self::evaluate_isotropic(input.positions, input.simbox, *axis)
            }
        }
    }
}

/// One snapshot's direct structure factor.
///
/// When the analyzer was constructed via [`StaticStructureFactorDirect::new`]
/// (explicit k-vectors), `sk` holds `S(k_vec)` for each input k-vector and
/// `k_magnitudes` holds their magnitudes. With
/// [`isotropic`](StaticStructureFactorDirect::isotropic), `sk` is the
/// spherically averaged `S(|k|)` and `k_magnitudes` is the bin-centres
/// array.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StaticStructureFactorDirectResult {
    pub k_magnitudes: Array1<F>,
    pub sk: Array1<F>,
    pub n_particles: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use ndarray::{Array2, array};

    struct Frame {
        positions: Array2<F>,
        simbox: SimBox,
    }

    impl Frame {
        fn input(&self) -> StaticStructureFactorDirectInput<'_> {
            StaticStructureFactorDirectInput {
                positions: self.positions.view(),
                simbox: Some(&self.simbox),
            }
        }
    }

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
        Frame {
            positions: Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]),
            simbox: SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap(),
        }
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
        let r = StaticStructureFactorDirect::new(&[[0.0, 0.0, 0.0]])
            .unwrap()
            .compute(frame.input())
            .unwrap();
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
        let r = StaticStructureFactorDirect::new(&k_vecs)
            .unwrap()
            .compute(frame.input())
            .unwrap();
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
        let r = StaticStructureFactorDirect::isotropic(8.0, 16)
            .unwrap()
            .compute(frame.input())
            .unwrap();
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
        let r = StaticStructureFactorDirect::new(&[[1.0, 0.0, 0.0]])
            .unwrap()
            .compute(frame.input())
            .unwrap();
        assert_eq!(r.n_particles, 0);
        assert_eq!(r.sk[0], 0.0);
    }

    /// Many snapshots go through `Batch`, in order.
    #[test]
    fn batch_matches_single_snapshots() {
        let frame = frame_with(
            &[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
            10.0,
            [false; 3],
        );
        let k_vecs: Vec<[F; 3]> = [0.5_f64, 1.2, 2.7].iter().map(|&k| [k, 0.0, 0.0]).collect();
        let s = StaticStructureFactorDirect::new(&k_vecs).unwrap();
        let solo = s.compute(frame.input()).unwrap();
        let par = Batch(s)
            .compute(vec![frame.input(), frame.input()])
            .unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].sk, solo.sk);
        assert_eq!(par[1].sk, solo.sk);
    }

    /// The isotropic form samples the box's reciprocal lattice; without a
    /// box it is an error, not a guess.
    #[test]
    fn isotropic_without_a_box_is_an_error() {
        let positions = array![[0.0, 0.0, 0.0]];
        let s = StaticStructureFactorDirect::isotropic(4.0, 4).unwrap();
        assert!(matches!(
            s.compute(StaticStructureFactorDirectInput {
                positions: positions.view(),
                simbox: None,
            }),
            Err(ComputeError::BadShape { .. })
        ));
    }

    /// A reciprocal-lattice point at exactly `k_max` is in the last bin.
    #[test]
    fn isotropic_upper_edge_is_closed() {
        let frame = frame_with(
            &[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            2.0 * std::f64::consts::PI,
            [true; 3],
        );
        // L = 2π → Δk = 1; k_max = 1 holds the six (±1, 0, 0)-type points.
        let r = StaticStructureFactorDirect::isotropic(1.0, 2)
            .unwrap()
            .compute(frame.input())
            .unwrap();
        assert!(r.sk[1] > 0.0);
        assert_eq!(r.sk[0], 0.0);
    }
}

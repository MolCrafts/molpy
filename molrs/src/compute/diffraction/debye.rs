//! Closed-form Debye static structure factor.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// Debye structure-factor calculator.
///
/// Mirrors `freud.diffraction.StaticStructureFactorDebye`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/diffraction/StaticStructureFactorDebye.cc)).
///
/// The spherically-averaged Debye scattering equation reads
///
/// ```text
///   S(k) = (1/N) Σ_{i, j} sin(k r_ij) / (k r_ij)
/// ```
///
/// where the sum runs over **all ordered pairs** including `i = j`
/// (which contributes `1` each, i.e. `N`). The implementation walks every
/// pair once via an `O(N²)` loop — there is no cutoff in Debye's form, so
/// a neighbor list does not help here. For sparse systems the upcoming
/// Phase 9 `StaticStructureFactorDirect` (FFT-based) is preferred.
///
/// # Conventions
///
/// - `k` values are passed in as an explicit array of magnitudes (`Å⁻¹`).
/// - `S(0) = N` exactly (per `lim_{k→0} sin(k r) / (k r) = 1`).
/// - The asymptote `S(k → ∞) → 1` is approached as the off-diagonal sum
///   averages to zero.
/// - Periodic boxes: distances are *not* minimum-imaged here. freud's
///   `Debye` uses the raw inter-particle distance and warns the user that
///   PBC should usually be turned off for a meaningful Debye calculation
///   (the formula assumes an open system).
#[derive(Debug, Clone)]
pub struct StaticStructureFactorDebye {
    k_values: Array1<F>,
}

impl StaticStructureFactorDebye {
    /// Construct from an explicit array of k magnitudes (Å⁻¹). Must be
    /// non-empty.
    pub fn new(k_values: &[F]) -> Result<Self, ComputeError> {
        if k_values.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "StaticStructureFactorDebye::k_values",
                value: "empty".into(),
            });
        }
        Ok(Self {
            k_values: Array1::from_vec(k_values.to_vec()),
        })
    }

    /// Convenience: build a linearly-spaced k grid `[k_min, k_max]` with
    /// `n` points (inclusive of both endpoints).
    pub fn linspace(k_min: F, k_max: F, n: usize) -> Result<Self, ComputeError> {
        if n == 0 {
            return Err(ComputeError::OutOfRange {
                field: "StaticStructureFactorDebye::linspace::n",
                value: "0".into(),
            });
        }
        if k_max < k_min {
            return Err(ComputeError::OutOfRange {
                field: "StaticStructureFactorDebye::linspace::k_max",
                value: format!("k_max={k_max} < k_min={k_min}"),
            });
        }
        let step = if n == 1 {
            0.0
        } else {
            (k_max - k_min) / (n - 1) as F
        };
        let v: Vec<F> = (0..n).map(|i| k_min + i as F * step).collect();
        Self::new(&v)
    }

    pub fn k_values(&self) -> &Array1<F> {
        &self.k_values
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
    ) -> Result<StaticStructureFactorDebyeResult, ComputeError> {
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let n = xs.len();
        let n_k = self.k_values.len();
        let mut sk = Array1::<F>::zeros(n_k);

        if n == 0 {
            return Ok(StaticStructureFactorDebyeResult {
                k_values: self.k_values.clone(),
                sk,
                n_particles: 0,
            });
        }

        // Diagonal: every pair (i, i) contributes 1 → N to every S(k).
        for v in sk.iter_mut() {
            *v = n as F;
        }

        // Off-diagonal: each unordered pair contributes 2 · sin(k r) / (k r)
        // (factor 2 from i↔j symmetry). This O(N²·n_k) pair sum is the
        // dominant cost and Debye is typically run on a single frame, so the
        // outer particle loop is parallelised over thread-local `sk`
        // accumulators (mirroring freud's OpenMP reduction). Cross-thread
        // summation is reassociated, so the parallel result can differ from
        // the serial path in the last ULPs — deterministic for a fixed
        // thread count and far below the physical tolerances on S(k).
        let inv_n = 1.0 / n as F;
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let k_values = &self.k_values;
            let off = (0..n)
                .into_par_iter()
                .fold(
                    || Array1::<F>::zeros(n_k),
                    |mut acc, i| {
                        accumulate_row(&mut acc, xs, ys, zs, k_values, i);
                        acc
                    },
                )
                .reduce(
                    || Array1::<F>::zeros(n_k),
                    |mut a, b| {
                        a += &b;
                        a
                    },
                );
            sk += &off;
        }
        #[cfg(not(feature = "rayon"))]
        for i in 0..n {
            accumulate_row(&mut sk, xs, ys, zs, &self.k_values, i);
        }

        // Normalise by N: S(k) = (1/N) Σ_{i, j} sin(k r_ij) / (k r_ij)
        for v in sk.iter_mut() {
            *v *= inv_n;
        }

        Ok(StaticStructureFactorDebyeResult {
            k_values: self.k_values.clone(),
            sk,
            n_particles: n,
        })
    }
}

/// Accumulate particle `i`'s off-diagonal Debye contributions (all pairs
/// `j > i`) into `acc`, one term per k value. Shared verbatim by the
/// serial and rayon-parallel paths so the pair math lives in exactly one
/// place — the serial path stays bit-for-bit identical to the pre-parallel
/// implementation.
fn accumulate_row(
    acc: &mut Array1<F>,
    xs: &[F],
    ys: &[F],
    zs: &[F],
    k_values: &Array1<F>,
    i: usize,
) {
    let n = xs.len();
    for j in (i + 1)..n {
        let dx = xs[i] - xs[j];
        let dy = ys[i] - ys[j];
        let dz = zs[i] - zs[j];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r == 0.0 {
            // Treat coincident pair as contributing 2 (lim sinx/x = 1).
            for v in acc.iter_mut() {
                *v += 2.0;
            }
            continue;
        }
        for (idx, &k) in k_values.iter().enumerate() {
            let kr = k * r;
            let term = if kr.abs() < 1e-9 { 1.0 } else { kr.sin() / kr };
            acc[idx] += 2.0 * term;
        }
    }
}

impl LegacyCompute for StaticStructureFactorDebye {
    type Args<'a> = ();
    type Output = Vec<StaticStructureFactorDebyeResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _: (),
    ) -> Result<Vec<StaticStructureFactorDebyeResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            out.push(self.one_frame(*f)?);
        }
        Ok(out)
    }
}

/// Per-frame Debye structure-factor result.
#[derive(Debug, Clone, Default)]
pub struct StaticStructureFactorDebyeResult {
    /// k values (Å⁻¹), copied from input.
    pub k_values: Array1<F>,
    /// `S(k)` at each k.
    pub sk: Array1<F>,
    /// Particle count used in the normalisation.
    pub n_particles: usize,
}

impl ComputeResult for StaticStructureFactorDebyeResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        // freud's Debye is open-system; supply a generous non-PBC box so the
        // FrameAccess plumbing is happy.
        frame.simbox =
            Some(SimBox::cube(1000.0, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    const TOL: F = 1e-10;

    #[test]
    fn s_at_k_zero_equals_n() {
        let positions = [[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let frame = frame_with(&positions);
        let s = StaticStructureFactorDebye::new(&[0.0]).unwrap();
        let r = &s.compute(&[&frame], ()).unwrap()[0];
        // S(0) = (1/N) · N² = N
        assert!((r.sk[0] - 3.0).abs() < TOL);
        assert_eq!(r.n_particles, 3);
    }

    #[test]
    fn s_large_k_approaches_one() {
        // For random-ish positions and large k, the oscillating sin(k r)/(k r)
        // off-diagonal terms average to ≈ 0 → S(k) → 1.
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(7);
        let n = 30;
        let positions: Vec<[F; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random::<F>() * 10.0,
                    rng.random::<F>() * 10.0,
                    rng.random::<F>() * 10.0,
                ]
            })
            .collect();
        let frame = frame_with(&positions);
        let s = StaticStructureFactorDebye::linspace(50.0, 60.0, 11).unwrap();
        let r = &s.compute(&[&frame], ()).unwrap()[0];
        let mean: F = r.sk.iter().copied().sum::<F>() / r.sk.len() as F;
        assert!(
            (mean - 1.0).abs() < 0.2,
            "S(k → ∞) should average to ≈ 1; got {mean}"
        );
    }

    #[test]
    fn two_particle_analytic() {
        // For two particles at distance d: S(k) = 1 + sin(k d) / (k d).
        // (Factor 1 from each diagonal; factor 2·sin/(kd)/2 = sin/(kd) from
        // the single unordered pair after dividing by N = 2.)
        let d: F = 1.5;
        let frame = frame_with(&[[0.0, 0.0, 0.0], [d, 0.0, 0.0]]);
        let k_vals = [0.5_f64, 1.0, 2.0, 5.0];
        let s = StaticStructureFactorDebye::new(&k_vals).unwrap();
        let r = &s.compute(&[&frame], ()).unwrap()[0];
        for (i, &k) in k_vals.iter().enumerate() {
            let expected = 1.0 + (k * d).sin() / (k * d);
            assert!(
                (r.sk[i] - expected).abs() < 1e-12,
                "k={k}: got {}, expected {expected}",
                r.sk[i]
            );
        }
    }

    #[test]
    fn coincident_particles_diagonal_only() {
        // Two coincident particles: every off-diagonal term contributes
        // sinc(0) = 1. S(k) = (1/2)(2 + 2·1) = 2 for any k.
        let frame = frame_with(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let s = StaticStructureFactorDebye::new(&[1.0, 5.0, 10.0]).unwrap();
        let r = &s.compute(&[&frame], ()).unwrap()[0];
        for &v in r.sk.iter() {
            assert!((v - 2.0).abs() < TOL);
        }
    }

    #[test]
    fn empty_frame_gives_zero_sk() {
        let frame = frame_with(&[]);
        let s = StaticStructureFactorDebye::new(&[1.0]).unwrap();
        let r = &s.compute(&[&frame], ()).unwrap()[0];
        assert_eq!(r.sk[0], 0.0);
        assert_eq!(r.n_particles, 0);
    }

    #[test]
    fn invalid_k_array_errors() {
        assert!(StaticStructureFactorDebye::new(&[]).is_err());
        assert!(StaticStructureFactorDebye::linspace(2.0, 1.0, 10).is_err());
        assert!(StaticStructureFactorDebye::linspace(0.0, 1.0, 0).is_err());
    }

    #[test]
    fn linspace_n_one_returns_single_k() {
        let s = StaticStructureFactorDebye::linspace(2.5, 7.0, 1).unwrap();
        assert_eq!(s.k_values().len(), 1);
        assert_eq!(s.k_values()[0], 2.5);
    }

    #[test]
    fn multi_frame_returns_one_result_per_frame() {
        let f1 = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let f2 = frame_with(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let s = StaticStructureFactorDebye::new(&[1.0]).unwrap();
        let r = s.compute(&[&f1, &f2], ()).unwrap();
        assert_eq!(r.len(), 2);
        // Different distances → different S(k) values.
        assert!((r[0].sk[0] - r[1].sk[0]).abs() > 1e-3);
    }
}

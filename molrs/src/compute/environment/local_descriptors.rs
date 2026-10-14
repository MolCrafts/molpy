//! Per-pair spherical-harmonic descriptors of the local neighborhood.

use molrs::core::Complex;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::{Array2, ArrayView1};

use crate::compute::order::{BondBasis, SphericalHarmonic};
use crate::compute::{Compute, ComputeError, FeatureMatrix, require_disp};

/// `LocalDescriptors` analyzer (Sph-mode).
///
/// Mirrors `freud.environment.LocalDescriptors`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/LocalDescriptors.cc)).
///
/// Configuration: the band limit `l_max`. Data: one snapshot's neighbor table
/// with the `disp` column. For each neighbor pair `(i, j)` computes the
/// complex spherical-harmonic coefficients `Y_ℓm(θ_ij, φ_ij)` for every ℓ
/// from 0 to `l_max` — the spherical-harmonic bond basis Steinhardt averages.
/// The output is a per-pair table of `n_sphs = Σ_{ℓ=0}^{l_max} (2ℓ + 1) =
/// (l_max + 1)²` coefficients. These features are the standard input for ML
/// models that classify local environments (e.g. crystal-vs-liquid
/// classifiers, polymorph identifiers). A zero-length bond has no direction
/// and gets an all-zero row.
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

/// The data a [`LocalDescriptors`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct LocalDescriptorsInput<'a> {
    /// One snapshot's neighbor table with the `disp` column.
    pub neighbors: &'a Neighbors,
}

impl LocalDescriptors {
    /// Spherical-harmonic band limit: descriptors for ℓ = 0…`l_max`.
    pub fn new(l_max: u32) -> Self {
        Self { l_max }
    }

    /// The band limit.
    pub fn l_max(&self) -> u32 {
        self.l_max
    }

    /// Coefficients per pair, `(l_max + 1)²`.
    pub fn n_sphs(&self) -> usize {
        ((self.l_max + 1) * (self.l_max + 1)) as usize
    }
}

impl Compute for LocalDescriptors {
    type Input<'a> = LocalDescriptorsInput<'a>;
    type Output = LocalDescriptorsResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp`.
    fn compute(
        &self,
        input: LocalDescriptorsInput<'_>,
    ) -> Result<LocalDescriptorsResult, ComputeError> {
        let n_sphs = self.n_sphs();
        // The descriptors are spherical harmonics of the bond direction.
        let disp = require_disp(input.neighbors)?;
        let mut features = Array2::<F>::zeros((disp.nrows(), 2 * n_sphs));
        let mut band = vec![Complex::ZERO; (2 * self.l_max + 1) as usize];
        for (row, mut out) in disp.outer_iter().zip(features.outer_iter_mut()) {
            let bond = [row[0], row[1], row[2]];
            let mut sph = 0_usize;
            for l in 0..=self.l_max {
                let width = (2 * l + 1) as usize;
                let values = &mut band[..width];
                if !(SphericalHarmonic { l }).evaluate(bond, values) {
                    break;
                }
                for v in values.iter() {
                    out[2 * sph] = v.re;
                    out[2 * sph + 1] = v.im;
                    sph += 1;
                }
            }
        }
        Ok(LocalDescriptorsResult {
            l_max: self.l_max,
            n_sphs,
            features,
        })
    }
}

/// One snapshot's descriptors, one row per neighbor pair (table order).
///
/// As a [`FeatureMatrix`] each row holds the pair's `n_sphs` coefficients as
/// `(Re, Im)` column pairs, in `(ℓ, m)` order: `Y_00`, `Y_1,−1`, `Y_10`,
/// `Y_11`, `Y_2,−2`, ….
#[derive(Debug, Clone, Default)]
pub struct LocalDescriptorsResult {
    /// `l_max` used for this run.
    pub l_max: u32,
    /// `n_sphs = (l_max + 1)²`.
    pub n_sphs: usize,
    /// `n_pairs × 2·n_sphs`, `(Re, Im)` interleaved.
    features: Array2<F>,
}

impl LocalDescriptorsResult {
    /// The coefficient `Y_ℓm` of pair `pair`, `ℓ ≤ l_max`, `|m| ≤ ℓ`; `None`
    /// outside those ranges or past the pair count.
    pub fn descriptor(&self, pair: usize, l: u32, m: i32) -> Option<Complex> {
        if l > self.l_max || m.unsigned_abs() > l || pair >= self.features.nrows() {
            return None;
        }
        let sph = (l * l) as usize + (m + l as i32) as usize;
        Some(Complex::new(
            self.features[[pair, 2 * sph]],
            self.features[[pair, 2 * sph + 1]],
        ))
    }
}

impl FeatureMatrix for LocalDescriptorsResult {
    fn n_rows(&self) -> usize {
        self.features.nrows()
    }

    fn n_cols(&self) -> usize {
        self.features.ncols()
    }

    fn row(&self, row: usize) -> ArrayView1<'_, F> {
        self.features.row(row)
    }
}

crate::compute::spec::compute_spec! {
    LocalDescriptors as "LocalDescriptors";
    params |s| {
        l_max: Int;
    }
    inputs {
        neighbors: Neighbors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::SimBox;
    use molrs::core::ylm_complex;
    use ndarray::array;

    /// Descriptors of the self-query table over `positions` in an open cube.
    fn describe(l_max: u32, positions: &[[F; 3]], box_len: F, cutoff: F) -> LocalDescriptorsResult {
        let simbox =
            SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap();
        let nl = neighbors_of(positions, &simbox, cutoff);
        LocalDescriptors::new(l_max)
            .compute(LocalDescriptorsInput { neighbors: &nl })
            .unwrap()
    }

    #[test]
    fn descriptor_at_l0_is_constant() {
        // For a single bond pointing in +x, the Y_0^0 component is
        // 1/(2√π) regardless of bond direction.
        let r = &describe(2, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.5);
        assert_eq!(r.n_sphs, 9); // (2+1)² = 9
        // Pair 0, ℓ=0, m=0.
        let y00 = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
        let d00 = r.descriptor(0, 0, 0).unwrap();
        assert!((d00.re - y00).abs() < 1e-12);
        assert!(d00.im.abs() < 1e-12);
    }

    #[test]
    fn descriptor_matches_direct_ylm_call() {
        // Sanity: the descriptor for a single bond equals ylm_complex
        // evaluated directly at the bond's (θ, φ).
        let dx = 0.6;
        let dy = 0.5;
        let dz = 0.7;
        let r = &describe(3, &[[0.0, 0.0, 0.0], [dx, dy, dz]], 10.0, 2.0);
        let r2 = (dx * dx + dy * dy + dz * dz).sqrt();
        let theta = (dz / r2).acos();
        let phi = dy.atan2(dx);
        // ℓ = 2 sits after ℓ = 0 [1] and ℓ = 1 [3]: features 8..18.
        for m in -2..=2 {
            let expected = ylm_complex(2, m, theta, phi);
            let got = r.descriptor(0, 2, m).unwrap();
            let sph = 4 + (m + 2) as usize;
            assert_eq!(r.row(0)[2 * sph], got.re);
            assert_eq!(r.row(0)[2 * sph + 1], got.im);
            assert!(
                (expected.re - got.re).abs() < 1e-12 && (expected.im - got.im).abs() < 1e-12,
                "ℓ=2 m={m} mismatch"
            );
        }
    }

    /// One row per pair, `2 · n_sphs` real columns.
    #[test]
    fn result_is_a_feature_matrix() {
        let r = describe(
            2,
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            10.0,
            1.2,
        );
        let f: &dyn FeatureMatrix = &r;
        assert_eq!((f.n_rows(), f.n_cols()), (2, 18));
        assert!(r.descriptor(0, 3, 0).is_none(), "ℓ = 3 is past l_max");
    }

    #[test]
    fn table_without_disp_is_bad_shape() {
        use molrs::core::NeighborColumns;
        let simbox = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false; 3]).unwrap();
        let nl = neighbors_of(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &simbox, 1.5)
            .repack(NeighborColumns::INDICES_ONLY);
        let err = LocalDescriptors::new(2)
            .compute(LocalDescriptorsInput { neighbors: &nl })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }

    #[test]
    fn n_sphs_helper_is_correct() {
        assert_eq!(LocalDescriptors::new(0).n_sphs(), 1);
        assert_eq!(LocalDescriptors::new(2).n_sphs(), 9);
        assert_eq!(LocalDescriptors::new(6).n_sphs(), 49);
    }
}

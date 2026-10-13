//! Legendre reorientational time-correlation functions `C_ℓ(t)` of
//! atom-pair vectors — the frame-reading front end of [`Oacf`].

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::{Array1, Array3};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::Oacf;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Legendre reorientational TCF analyzer.
///
/// Stateless parameter bag: the maximum lag (correlation depth, in frames) and
/// the time-origin stride. The molecular vectors are selected by the `Args`
/// passed to [`compute`](Compute::compute): one `(a, b)` atom-index pair per
/// molecule, defining `û = normalize(r_b − r_a)` (minimum-image).
///
/// For a molecular **unit vector** `û` (an O–H bond, a dipole axis, …) the
/// reorientational TCFs are
///
/// ```text
///   C_1(t) = ⟨ P_1(û(0)·û(t)) ⟩ = ⟨ cos θ(t) ⟩
///   C_2(t) = ⟨ P_2(û(0)·û(t)) ⟩ = ⟨ (3 cos²θ(t) − 1) / 2 ⟩
/// ```
///
/// averaged over molecules and (multi-)time origins. `C_1` is the dielectric /
/// IR reorientation observable; the `C_2` decay gives the NMR rotational
/// correlation time τ_c (Berne & Pecora, *Dynamic Light Scattering*; standard
/// NMR relaxation theory).
///
/// This analyzer only builds the `(n_frames, n_pairs, 3)` unit-vector series
/// from the frames; the correlation is [`Oacf`] at `ℓ = 1` and `ℓ = 2`, which
/// is the FFT route for both (`P_2` factorizes through the quadratic
/// components of `û ⊗ û`). Callers with their own vectors — a dipole axis, a
/// ring normal, a bond that needs a different image convention — use
/// [`Oacf`] directly.
///
/// # Provenance
///
/// Ported from the reference implementation's reorientation-dynamics analyzer
/// `CReorDyn` (`src/reordyn.cpp`): per-molecule unit vectors (cf.
/// `src/order_vector.cpp`), the 1st/2nd-Legendre selection (`m_bLegendre2`),
/// and the multi-time-origin correlation-depth accumulation (`m_iDepth`, the
/// ACF origin sums in `src/acf.cpp`), which it likewise evaluates via FFT.
///
/// # Relation to [`RotationalAutocorrelation`](crate::compute::RotationalAutocorrelation)
///
/// This analyzer is **distinct** from
/// [`RotationalAutocorrelation`](crate::compute::RotationalAutocorrelation):
/// that one is freud's rigid-body **quaternion** autocorrelation (a Wigner-D
/// character of the full orientation), whereas `LegendreReorientation` correlates
/// a single molecular **vector** via Legendre polynomials. They measure different
/// observables and both remain independently usable.
///
/// # Fitting τ_c
///
/// The raw `C_2(t)` curve is `Fit`-ready: feed it to
/// [`DebyeFit`](crate::compute::DebyeFit) (which fits a normalized
/// `Φ(t) → τ` decay) to extract the rotational correlation time — no new fitting
/// code is needed.
#[derive(Debug, Clone)]
pub struct LegendreReorientation {
    max_lag: usize,
    stride: usize,
}

impl LegendreReorientation {
    /// New analyzer evaluating `C_1`/`C_2` at lags `0..=max_lag` (clamped to the
    /// trajectory length at compute time).
    pub fn new(max_lag: usize) -> Self {
        Self { max_lag, stride: 1 }
    }

    /// Set the time-origin stride (default 1).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }
}

/// The `(n_frames, n_pairs, 3)` minimum-image unit vectors `r_b − r_a` of
/// `pairs` across `frames`.
pub(crate) fn pair_unit_vectors<'a, FA: FrameAccess + Sync + 'a>(
    frames: &[&'a FA],
    pairs: &[(u32, u32)],
) -> Result<Array3<F>, ComputeError> {
    let n_frames = frames.len();
    let n_vec = pairs.len();
    let mut vecs = Array3::<F>::zeros((n_frames, n_vec, 3));
    for (f, frame) in frames.iter().enumerate() {
        let (xp, yp, zp) = get_positions_ref(*frame)?;
        let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
        // Resolve the minimum-image state once per frame, not per pair.
        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        for (m, &(a, b)) in pairs.iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            if a >= xs.len() || b >= xs.len() {
                return Err(ComputeError::DimensionMismatch {
                    expected: xs.len(),
                    got: a.max(b) + 1,
                    what: "LegendreReorientation atom index",
                });
            }
            let d = mic.apply(sub([xs[b], ys[b], zs[b]], [xs[a], ys[a], zs[a]]));
            let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(ComputeError::NonFinite {
                    where_: "LegendreReorientation unit vector (zero-length)",
                    index: f * n_vec + m,
                });
            }
            // One reciprocal + three multiplies instead of three divides.
            let inv = norm.recip();
            vecs[[f, m, 0]] = d[0] * inv;
            vecs[[f, m, 1]] = d[1] * inv;
            vecs[[f, m, 2]] = d[2] * inv;
        }
    }
    Ok(vecs)
}

impl Compute for LegendreReorientation {
    /// One `(atom_a, atom_b)` index pair per molecular vector.
    type Args<'a> = &'a [(u32, u32)];
    type Output = LegendreReorientationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        pairs: &'a [(u32, u32)],
    ) -> Result<LegendreReorientationResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if pairs.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "LegendreReorientation::pairs",
                value: "empty".to_string(),
            });
        }
        let vecs = pair_unit_vectors(frames, pairs)?;
        if frames.len() == 1 {
            // One frame: lag 0 only, and every vector correlates with itself.
            return Ok(LegendreReorientationResult {
                lags: vec![0],
                c1: Array1::from_elem(1, 1.0),
                c2: Array1::from_elem(1, 1.0),
            });
        }
        let c1 = Oacf::new(1, self.max_lag)
            .with_origin_stride(self.stride)
            .correlate(&vecs, 1.0)?;
        let c2 = Oacf::new(2, self.max_lag)
            .with_origin_stride(self.stride)
            .correlate(&vecs, 1.0)?;
        Ok(LegendreReorientationResult {
            lags: c1.lags.to_vec(),
            c1: c1.values,
            c2: c2.values,
        })
    }
}

/// Result of a [`LegendreReorientation`] computation.
#[derive(Debug, Clone)]
pub struct LegendreReorientationResult {
    /// Lag times (frame units), aligned with `c1`/`c2`.
    pub lags: Vec<usize>,
    /// `C_1(t) = ⟨cos θ(t)⟩`.
    pub c1: Array1<F>,
    /// `C_2(t) = ⟨(3cos²θ(t) − 1)/2⟩`.
    pub c2: Array1<F>,
}

impl ComputeResult for LegendreReorientationResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use ndarray::Array1 as A1;
    use rand::{RngExt, SeedableRng};

    /// Deterministic unit-vector field `(n_frames, n_vec)`.
    fn unit_field(n_frames: usize, n_vec: usize, seed: u64) -> Vec<Vec<[F; 3]>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n_frames)
            .map(|_| {
                (0..n_vec)
                    .map(|_| {
                        let v: [F; 3] = [
                            rng.random_range(-1.0..1.0),
                            rng.random_range(-1.0..1.0),
                            rng.random_range(-1.0..1.0),
                        ];
                        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        [v[0] / n, v[1] / n, v[2] / n]
                    })
                    .collect()
            })
            .collect()
    }

    /// Frames in which atom `2m` sits at a fixed point and atom `2m+1` at that
    /// point plus the field vector, so the pair `(2m, 2m+1)` is the vector. No
    /// box: the minimum image is the identity.
    fn frames_from_field(field: &[Vec<[F; 3]>]) -> (Vec<Frame>, Vec<(u32, u32)>) {
        let n_vec = field[0].len();
        let frames = field
            .iter()
            .map(|vecs| {
                let mut x = Vec::with_capacity(2 * n_vec);
                let mut y = Vec::with_capacity(2 * n_vec);
                let mut z = Vec::with_capacity(2 * n_vec);
                for (m, u) in vecs.iter().enumerate() {
                    let base = [m as F, 0.5 * m as F, 0.25 * m as F];
                    x.push(base[0]);
                    y.push(base[1]);
                    z.push(base[2]);
                    x.push(base[0] + u[0]);
                    y.push(base[1] + u[1]);
                    z.push(base[2] + u[2]);
                }
                let mut block = Block::new();
                block.insert("x", A1::from_vec(x).into_dyn()).unwrap();
                block.insert("y", A1::from_vec(y).into_dyn()).unwrap();
                block.insert("z", A1::from_vec(z).into_dyn()).unwrap();
                let mut frame = Frame::new();
                frame.insert("atoms", block);
                frame
            })
            .collect();
        let pairs = (0..n_vec)
            .map(|m| ((2 * m) as u32, (2 * m + 1) as u32))
            .collect();
        (frames, pairs)
    }

    /// The definition: multi-origin average of `P_ℓ(û(τ)·û(τ+t))` over the
    /// vectors rebuilt from the same positions (`b − a`, normalized).
    fn direct(
        frames: &[Frame],
        pairs: &[(u32, u32)],
        max_lag: usize,
        stride: usize,
    ) -> (Vec<F>, Vec<F>) {
        let n_frames = frames.len();
        let vecs: Vec<Vec<[F; 3]>> = frames
            .iter()
            .map(|f| {
                let (xp, yp, zp) = get_positions_ref(f).unwrap();
                let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
                pairs
                    .iter()
                    .map(|&(a, b)| {
                        let (a, b) = (a as usize, b as usize);
                        let d = [xs[b] - xs[a], ys[b] - ys[a], zs[b] - zs[a]];
                        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        [d[0] / n, d[1] / n, d[2] / n]
                    })
                    .collect()
            })
            .collect();
        let mut c1 = Vec::with_capacity(max_lag + 1);
        let mut c2 = Vec::with_capacity(max_lag + 1);
        for t in 0..=max_lag {
            let (mut s1, mut s2, mut count) = (0.0, 0.0, 0usize);
            let mut tau = 0;
            while tau + t < n_frames {
                for (a, b) in vecs[tau].iter().zip(vecs[tau + t].iter()) {
                    let (a, b) = (*a, *b);
                    let x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                    s1 += x;
                    s2 += 0.5 * (3.0 * x * x - 1.0);
                    count += 1;
                }
                tau += stride;
            }
            c1.push(s1 / count as F);
            c2.push(s2 / count as F);
        }
        (c1, c2)
    }

    #[test]
    fn matches_the_direct_definition_at_stride_1_and_3() {
        let field = unit_field(40, 7, 3);
        let (frames, pairs) = frames_from_field(&field);
        let refs: Vec<&Frame> = frames.iter().collect();
        for stride in [1usize, 3] {
            let got = LegendreReorientation::new(20)
                .with_stride(stride)
                .compute(&refs, &pairs)
                .unwrap();
            let (c1, c2) = direct(&frames, &pairs, 20, stride);
            assert_eq!(got.lags.len(), 21);
            for t in 0..=20 {
                assert!(
                    (got.c1[t] - c1[t]).abs() < 1e-10,
                    "stride {stride} lag {t} C1: {} vs {}",
                    got.c1[t],
                    c1[t]
                );
                assert!(
                    (got.c2[t] - c2[t]).abs() < 1e-10,
                    "stride {stride} lag {t} C2: {} vs {}",
                    got.c2[t],
                    c2[t]
                );
            }
        }
    }

    #[test]
    fn static_vectors_give_unity() {
        let one = unit_field(1, 4, 9).remove(0);
        let field: Vec<Vec<[F; 3]>> = (0..12).map(|_| one.clone()).collect();
        let (frames, pairs) = frames_from_field(&field);
        let refs: Vec<&Frame> = frames.iter().collect();
        let got = LegendreReorientation::new(6)
            .compute(&refs, &pairs)
            .unwrap();
        for t in 0..=6 {
            assert!((got.c1[t] - 1.0).abs() < 1e-12);
            assert!((got.c2[t] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn c2_is_even_under_flipping_alternate_frames() {
        // Reversing û on every odd frame flips the sign of C1 at odd lags and
        // leaves C2 untouched: P2 is even.
        let field = unit_field(30, 5, 5);
        let flipped: Vec<Vec<[F; 3]>> = field
            .iter()
            .enumerate()
            .map(|(k, vecs)| {
                if k % 2 == 1 {
                    vecs.iter().map(|u| [-u[0], -u[1], -u[2]]).collect()
                } else {
                    vecs.clone()
                }
            })
            .collect();
        let (fa, pa) = frames_from_field(&field);
        let (fb, pb) = frames_from_field(&flipped);
        let ra: Vec<&Frame> = fa.iter().collect();
        let rb: Vec<&Frame> = fb.iter().collect();
        let a = LegendreReorientation::new(10).compute(&ra, &pa).unwrap();
        let b = LegendreReorientation::new(10).compute(&rb, &pb).unwrap();
        for t in 0..=10 {
            assert!((a.c2[t] - b.c2[t]).abs() < 1e-12, "lag {t}");
            let sign = if t % 2 == 1 { -1.0 } else { 1.0 };
            assert!((a.c1[t] - sign * b.c1[t]).abs() < 1e-12, "lag {t}");
        }
    }

    #[test]
    fn empty_pairs_and_bad_index_are_errors() {
        let field = unit_field(4, 2, 1);
        let (frames, _) = frames_from_field(&field);
        let refs: Vec<&Frame> = frames.iter().collect();
        assert!(LegendreReorientation::new(2).compute(&refs, &[]).is_err());
        assert!(
            LegendreReorientation::new(2)
                .compute(&refs, &[(0, 99)])
                .is_err()
        );
    }
}

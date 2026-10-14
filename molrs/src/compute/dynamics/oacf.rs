//! [`Oacf`] — orientational (reorientational) autocorrelation of unit
//! vectors, `C_ℓ(t) = ⟨P_ℓ(û(τ) · û(τ+t))⟩`.

use ndarray::{Array2, Array3};

use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::correlation::result::origin_count;
use crate::compute::{CorrelationKind, TimeCorrelation, TimeCorrelationSpec, autocorrelate};
use molrs::core::FrameAccess;

/// Tolerance on `|û|² − 1` for an input vector to count as a unit vector.
const UNIT_TOL: f64 = 1e-6;

/// `(û_x², û_y², û_z², û_x û_y, û_x û_z, û_y û_z)` weights: the dot product of
/// two such tuples is `(û · v̂)²`.
const QUADRATIC_WEIGHTS: [f64; 6] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

/// Orientational autocorrelation function of a unit-vector series.
///
/// For an `(n_frames, n_vectors, 3)` series of **unit** vectors `û` — an O–H
/// bond, a molecular dipole axis, a nematic director — the reorientational
/// correlation of Legendre order `ℓ` is
///
/// ```text
///   C_ℓ(t) = ⟨ P_ℓ(û(τ) · û(τ+t)) ⟩_{τ, vectors}
///   C_1(t) = ⟨ cos θ(t) ⟩                 (dielectric / IR reorientation)
///   C_2(t) = ⟨ (3 cos²θ(t) − 1) / 2 ⟩      (NMR, Raman, fluorescence anisotropy)
/// ```
///
/// averaged over the vectors and every time origin `τ = 0, s, 2s, …`
/// (`s` the origin stride), unbiased. `C_ℓ(0) = 1` by construction and the
/// curve decays to `0` for an isotropic liquid; its integral is the
/// reorientational correlation time `τ_ℓ`.
///
/// The caller builds the unit vectors — which atoms, which minimum-image
/// convention, which molecule — and `Oacf` takes nothing but the series:
/// `|û|² ≠ 1` is an error, not silently renormalized.
/// [`LegendreReorientation`](crate::compute::LegendreReorientation) is the
/// front end that builds O–H-style pair vectors from frames and calls this.
///
/// # Algorithm
///
/// `ℓ = 1` is the vector autocorrelation itself ([`autocorrelate`]). `ℓ = 2`
/// uses the identity `(û · v̂)² = Q(û) · Q(v̂)` for the weighted quadratic
/// tuple `Q = (û_x², û_y², û_z², û_x û_y, û_x û_z, û_y û_z)` with weights
/// `(1, 1, 1, 2, 2, 2)` — the second-rank tensor `û ⊗ û` written in six
/// independent components — so `C_2 = (3 ⟨Q(τ) · Q(τ+t)⟩ − 1) / 2` goes
/// through the same `O(T log T)` FFT route. The Legendre nonlinearity sits
/// *inside* the time average, but for `ℓ ≤ 2` it is a polynomial in the dot
/// product and so factorizes exactly; `ℓ ≥ 3` falls back to the direct
/// origin sum with Bonnet's recurrence (`O(T · lags · N)`).
///
/// # References
///
/// - Berne & Pecora, *Dynamic Light Scattering* (1976), Ch. 7 — `C_ℓ(t)`
///   and which spectroscopy sees which `ℓ`.
/// - Laage & Hynes, *Science* **311** (2006) 832 — water O–H `C_2(t)` and
///   the jump model.
/// - Lipari & Szabo, *J. Am. Chem. Soc.* **104** (1982) 4546 — `C_2` as the
///   NMR relaxation correlation function.
#[derive(Debug, Clone, Copy)]
pub struct Oacf {
    /// Legendre order `ℓ ≥ 1`.
    pub ell: u32,
    /// Longest lag in frames; clamped to `n_frames − 1`.
    pub max_lag: usize,
    /// Time origins every `origin_stride` frames.
    pub origin_stride: usize,
    /// Also return one curve per vector.
    pub per_entity: bool,
}

impl Oacf {
    /// `C_ℓ(t)` for lags `0..=max_lag`, every time origin, no per-vector rows.
    pub fn new(ell: u32, max_lag: usize) -> Self {
        Self {
            ell,
            max_lag,
            origin_stride: 1,
            per_entity: false,
        }
    }

    pub fn with_origin_stride(mut self, stride: usize) -> Self {
        self.origin_stride = stride;
        self
    }

    pub fn with_per_entity(mut self, yes: bool) -> Self {
        self.per_entity = yes;
        self
    }

    fn spec(&self) -> TimeCorrelationSpec<'static> {
        TimeCorrelationSpec::new()
            .max_lag(self.max_lag)
            .origin_stride(self.origin_stride)
            .per_entity(self.per_entity)
    }

    /// Validate the series and compute `C_ℓ(t)`.
    pub fn correlate(
        &self,
        vectors: &Array3<f64>,
        dt: f64,
    ) -> Result<TimeCorrelation, ComputeError> {
        validate_unit_vectors(vectors)?;
        if self.ell == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Oacf::ell",
                value: "0 (P_0 ≡ 1 correlates nothing)".to_string(),
            });
        }
        let kind = CorrelationKind::Legendre { ell: self.ell };
        match self.ell {
            1 => {
                let mut c = autocorrelate(vectors.view(), dt, &self.spec())?;
                c.kind = kind;
                Ok(c)
            }
            2 => {
                let q = quadratic_components(vectors);
                let mut c = autocorrelate(
                    q.view(),
                    dt,
                    &self.spec().component_weights(&QUADRATIC_WEIGHTS),
                )?;
                c.values.mapv_inplace(|v| 1.5 * v - 0.5);
                if let Some(rows) = c.per_entity.as_mut() {
                    rows.mapv_inplace(|v| 1.5 * v - 0.5);
                }
                c.kind = kind;
                Ok(c)
            }
            ell => legendre_direct(vectors, dt, ell, &self.spec()),
        }
    }
}

/// `(unit vectors (n_frames, n_vectors, 3), dt)` for [`Oacf`].
pub type OacfArgs<'a> = (&'a Array3<f64>, f64);

impl Compute for Oacf {
    /// `(unit vectors (n_frames, n_vectors, 3), dt)`. The frames slice is
    /// unused: the caller has already turned its frames into vectors.
    type Args<'a> = OacfArgs<'a>;
    type Output = TimeCorrelation;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        (vectors, dt): Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        self.correlate(vectors, dt)
    }
}

fn validate_unit_vectors(vectors: &Array3<f64>) -> Result<(), ComputeError> {
    let (n_frames, n_vectors, n_components) = vectors.dim();
    if n_components != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: n_components,
            what: "Oacf vectors (expected (n_frames, n_vectors, 3))",
        });
    }
    if n_frames < 2 || n_vectors == 0 {
        return Err(ComputeError::EmptyInput);
    }
    for (index, v) in vectors.iter().enumerate() {
        if !v.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "Oacf vectors",
                index,
            });
        }
    }
    for t in 0..n_frames {
        for m in 0..n_vectors {
            let u = [vectors[[t, m, 0]], vectors[[t, m, 1]], vectors[[t, m, 2]]];
            let norm2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            if (norm2 - 1.0).abs() > UNIT_TOL {
                return Err(ComputeError::OutOfRange {
                    field: "Oacf vectors",
                    value: format!(
                        "|u|² = {norm2} at frame {t}, vector {m}; unit vectors required"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// `(û_x², û_y², û_z², û_x û_y, û_x û_z, û_y û_z)` per vector: the six
/// independent components of `û ⊗ û`.
fn quadratic_components(vectors: &Array3<f64>) -> Array3<f64> {
    let (n_frames, n_vectors, _) = vectors.dim();
    let mut q = Array3::<f64>::zeros((n_frames, n_vectors, 6));
    for t in 0..n_frames {
        for m in 0..n_vectors {
            let (x, y, z) = (vectors[[t, m, 0]], vectors[[t, m, 1]], vectors[[t, m, 2]]);
            q[[t, m, 0]] = x * x;
            q[[t, m, 1]] = y * y;
            q[[t, m, 2]] = z * z;
            q[[t, m, 3]] = x * y;
            q[[t, m, 4]] = x * z;
            q[[t, m, 5]] = y * z;
        }
    }
    q
}

/// `P_ℓ(x)` by Bonnet's recurrence `(n+1) P_{n+1} = (2n+1) x P_n − n P_{n−1}`.
pub(crate) fn legendre(ell: u32, x: f64) -> f64 {
    match ell {
        0 => 1.0,
        1 => x,
        _ => {
            let (mut p_prev, mut p) = (1.0, x);
            for n in 1..ell {
                let n = n as f64;
                let next = ((2.0 * n + 1.0) * x * p - n * p_prev) / (n + 1.0);
                p_prev = p;
                p = next;
            }
            p
        }
    }
}

/// The definition, directly: `C_ℓ(t) = Σ_{τ ∈ origins} Σ_m P_ℓ(û_m(τ) ·
/// û_m(τ+t)) / (N · counts[t])`, one independent reduction per lag (fanned
/// out over lags with rayon). The route for `ℓ ≥ 3`, and the oracle the FFT
/// routes are tested against.
pub(crate) fn legendre_direct(
    vectors: &Array3<f64>,
    dt: f64,
    ell: u32,
    spec: &TimeCorrelationSpec<'_>,
) -> Result<TimeCorrelation, ComputeError> {
    let (n_frames, n_vectors, _) = vectors.dim();
    if spec.origin_stride == 0 {
        return Err(ComputeError::OutOfRange {
            field: "origin_stride",
            value: "0".to_string(),
        });
    }
    if dt <= 0.0 || !dt.is_finite() {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let max_lag = spec.max_lag.min(n_frames - 1);
    let stride = spec.origin_stride;
    let counts: Vec<usize> = (0..=max_lag)
        .map(|t| origin_count(n_frames, t, stride))
        .collect();

    // Per lag: the per-vector origin sums.
    let per_lag = |t: usize| -> Vec<f64> {
        let mut sums = vec![0.0_f64; n_vectors];
        let mut tau = 0usize;
        while tau + t < n_frames {
            for (m, sum) in sums.iter_mut().enumerate() {
                let x = vectors[[tau, m, 0]] * vectors[[tau + t, m, 0]]
                    + vectors[[tau, m, 1]] * vectors[[tau + t, m, 1]]
                    + vectors[[tau, m, 2]] * vectors[[tau + t, m, 2]];
                *sum += legendre(ell, x);
            }
            tau += stride;
        }
        let inv = 1.0 / counts[t] as f64;
        for sum in sums.iter_mut() {
            *sum *= inv;
        }
        sums
    };

    #[cfg(feature = "rayon")]
    let columns: Vec<Vec<f64>> = {
        use rayon::prelude::*;
        (0..=max_lag).into_par_iter().map(per_lag).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let columns: Vec<Vec<f64>> = (0..=max_lag).map(per_lag).collect();

    let mut values = ndarray::Array1::<f64>::zeros(max_lag + 1);
    let mut rows = spec
        .per_entity
        .then(|| Array2::<f64>::zeros((n_vectors, max_lag + 1)));
    for (t, column) in columns.iter().enumerate() {
        let mut total = 0.0;
        for (m, &v) in column.iter().enumerate() {
            total += v;
            if let Some(rows) = rows.as_mut() {
                rows[[m, t]] = v;
            }
        }
        values[t] = total / n_vectors as f64;
    }

    Ok(TimeCorrelation {
        lags: ndarray::Array1::from_iter(0..=max_lag),
        dt,
        values,
        counts: ndarray::Array1::from_vec(counts),
        n_entities: n_vectors,
        normalization: crate::compute::Normalization::Unbiased,
        kind: CorrelationKind::Legendre { ell },
        per_entity: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// Random unit vectors, `(n_frames, n_vectors, 3)`.
    fn unit_vectors(n_frames: usize, n_vectors: usize, seed: u64) -> Array3<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut u = Array3::<f64>::zeros((n_frames, n_vectors, 3));
        for t in 0..n_frames {
            for m in 0..n_vectors {
                let v: [f64; 3] = [
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                ];
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                for c in 0..3 {
                    u[[t, m, c]] = v[c] / n;
                }
            }
        }
        u
    }

    #[test]
    fn fft_routes_match_the_direct_definition_at_stride_1_and_3() {
        let u = unit_vectors(50, 6, 1);
        for stride in [1usize, 3] {
            for ell in [1u32, 2] {
                let fft = Oacf::new(ell, 20)
                    .with_origin_stride(stride)
                    .compute(&no_frames(), (&u, 0.5))
                    .unwrap();
                let direct = legendre_direct(
                    &u,
                    0.5,
                    ell,
                    &TimeCorrelationSpec::new().max_lag(20).origin_stride(stride),
                )
                .unwrap();
                assert_eq!(fft.kind, CorrelationKind::Legendre { ell });
                assert_eq!(fft.counts, direct.counts);
                for t in 0..=20 {
                    assert!(
                        (fft.values[t] - direct.values[t]).abs() < 1e-12,
                        "ell {ell} stride {stride} lag {t}: {} vs {}",
                        fft.values[t],
                        direct.values[t]
                    );
                }
                assert!((fft.values[0] - 1.0).abs() < 1e-12);
                assert!((fft.lag_times()[20] - 10.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn per_entity_rows_match_the_direct_definition_and_average_to_the_curve() {
        let u = unit_vectors(40, 5, 2);
        for ell in [1u32, 2, 3] {
            let got = Oacf::new(ell, 12)
                .with_per_entity(true)
                .compute(&no_frames(), (&u, 1.0))
                .unwrap();
            let rows = got.per_entity.as_ref().unwrap();
            let direct = legendre_direct(
                &u,
                1.0,
                ell,
                &TimeCorrelationSpec::new().max_lag(12).per_entity(true),
            )
            .unwrap();
            let direct_rows = direct.per_entity.as_ref().unwrap();
            assert_eq!(rows.dim(), (5, 13));
            for m in 0..5 {
                for t in 0..=12 {
                    assert!(
                        (rows[[m, t]] - direct_rows[[m, t]]).abs() < 1e-12,
                        "ell {ell} m {m} t {t}"
                    );
                }
            }
            for t in 0..=12 {
                let mean = rows.column(t).sum() / 5.0;
                assert!((mean - got.values[t]).abs() < 1e-12, "ell {ell} t {t}");
            }
        }
    }

    #[test]
    fn ell_3_matches_hand_values_and_bonnet_is_the_textbook_polynomial() {
        for x in [-1.0, -0.6, -0.2, 0.0, 0.3, 0.75, 1.0] {
            let p3 = 0.5 * (5.0 * x * x * x - 3.0 * x);
            assert!((legendre(3, x) - p3).abs() < 1e-14);
            let p4 = (35.0 * x.powi(4) - 30.0 * x * x + 3.0) / 8.0;
            assert!((legendre(4, x) - p4).abs() < 1e-14);
            assert_eq!(legendre(2, x), 0.5 * (3.0 * x * x - 1.0));
        }
        // Three frames of one vector rotating by 60° in the xy-plane.
        let angles = [0.0_f64, 60.0, 120.0];
        let mut u = Array3::<f64>::zeros((3, 1, 3));
        for (t, deg) in angles.iter().enumerate() {
            u[[t, 0, 0]] = deg.to_radians().cos();
            u[[t, 0, 1]] = deg.to_radians().sin();
        }
        let got = Oacf::new(3, 2).compute(&no_frames(), (&u, 1.0)).unwrap();
        let p3 = |deg: f64| legendre(3, deg.to_radians().cos());
        assert!((got.values[0] - 1.0).abs() < 1e-14);
        assert!((got.values[1] - p3(60.0)).abs() < 1e-14); // two origins, same angle
        assert!((got.values[2] - p3(120.0)).abs() < 1e-14);
        assert_eq!(got.counts.to_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn c2_is_exactly_even_under_vector_reversal_and_c1_is_odd() {
        let u = unit_vectors(30, 4, 3);
        let flipped = u.mapv(|v| -v);
        let c2 = Oacf::new(2, 10).compute(&no_frames(), (&u, 1.0)).unwrap();
        let c2f = Oacf::new(2, 10)
            .compute(&no_frames(), (&flipped, 1.0))
            .unwrap();
        assert_eq!(c2.values, c2f.values); // Q(−u) = Q(u) bit for bit
        let c1 = Oacf::new(1, 10).compute(&no_frames(), (&u, 1.0)).unwrap();
        let c1f = Oacf::new(1, 10)
            .compute(&no_frames(), (&flipped, 1.0))
            .unwrap();
        assert_eq!(c1.values, c1f.values); // both vectors flip: the dot product does not
    }

    #[test]
    fn quarter_turns_give_the_textbook_pattern() {
        // û rotates by 90° every frame: cos θ = 1, 0, −1, 0, …
        let n = 16;
        let mut u = Array3::<f64>::zeros((n, 2, 3));
        for t in 0..n {
            let a = (t as f64) * std::f64::consts::FRAC_PI_2;
            for m in 0..2 {
                u[[t, m, 0]] = a.cos();
                u[[t, m, 1]] = a.sin();
            }
        }
        let c1 = Oacf::new(1, 4).compute(&no_frames(), (&u, 1.0)).unwrap();
        let c2 = Oacf::new(2, 4).compute(&no_frames(), (&u, 1.0)).unwrap();
        let want1 = [1.0, 0.0, -1.0, 0.0, 1.0];
        let want2 = [1.0, -0.5, 1.0, -0.5, 1.0];
        for t in 0..=4 {
            assert!(
                (c1.values[t] - want1[t]).abs() < 1e-12,
                "C1 lag {t}: {}",
                c1.values[t]
            );
            assert!(
                (c2.values[t] - want2[t]).abs() < 1e-12,
                "C2 lag {t}: {}",
                c2.values[t]
            );
        }
    }

    #[test]
    fn static_vectors_give_unity_at_every_order() {
        let one = unit_vectors(1, 3, 4);
        let mut u = Array3::<f64>::zeros((10, 3, 3));
        for t in 0..10 {
            for m in 0..3 {
                for c in 0..3 {
                    u[[t, m, c]] = one[[0, m, c]];
                }
            }
        }
        for ell in [1u32, 2, 3, 5] {
            let got = Oacf::new(ell, 6).compute(&no_frames(), (&u, 1.0)).unwrap();
            for t in 0..=6 {
                assert!((got.values[t] - 1.0).abs() < 1e-12, "ell {ell} lag {t}");
            }
        }
    }

    #[test]
    fn rejects_non_unit_non_finite_wrong_shape_and_order_zero() {
        let u = unit_vectors(8, 2, 5);
        assert!(Oacf::new(0, 3).compute(&no_frames(), (&u, 1.0)).is_err());
        let mut long = u.clone();
        long[[2, 1, 0]] *= 1.01;
        assert!(matches!(
            Oacf::new(2, 3).compute(&no_frames(), (&long, 1.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
        let mut nan = u.clone();
        nan[[0, 0, 2]] = f64::NAN;
        assert!(matches!(
            Oacf::new(1, 3).compute(&no_frames(), (&nan, 1.0)),
            Err(ComputeError::NonFinite { .. })
        ));
        let planar = Array3::<f64>::zeros((8, 2, 2));
        assert!(matches!(
            Oacf::new(1, 3).compute(&no_frames(), (&planar, 1.0)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(Oacf::new(1, 3).compute(&no_frames(), (&u, 0.0)).is_err());
        assert!(
            Oacf::new(1, 3)
                .with_origin_stride(0)
                .compute(&no_frames(), (&u, 1.0))
                .is_err()
        );
        let single = unit_vectors(1, 2, 6);
        assert!(matches!(
            Oacf::new(1, 3).compute(&no_frames(), (&single, 1.0)),
            Err(ComputeError::EmptyInput)
        ));
    }
}

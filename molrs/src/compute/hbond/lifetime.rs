//! Hydrogen-bond lifetime correlation functions.

use molrs::op::F;
use ndarray::{Array1, Array3};
use std::collections::HashMap;

use super::detect::HBondsResult;
use crate::compute::ComputeError;
use crate::compute::fitting::running_trapezoid;
use crate::compute::{TimeCorrelationSpec, autocorrelate};

/// Lifetime TCFs for a set of bond presence series.
#[derive(Debug, Clone)]
pub struct HBondLifetimeResult {
    /// Lag times (`tau * dt`), length `max_lag + 1`.
    pub lag_times: Array1<F>,
    /// Continuous `S_HB(t)`, normalized to `S_HB(0) = 1`.
    pub continuous: Array1<F>,
    /// Intermittent `C_HB(t)`, normalized to `C_HB(0) = 1`.
    pub intermittent: Array1<F>,
    /// Continuous lifetime `τ_c = ∫ S_HB(t) dt` (trapezoidal).
    pub tau_continuous: F,
    /// Intermittent lifetime `τ_i = ∫ C_HB(t) dt` (trapezoidal).
    pub tau_intermittent: F,
}

/// Compute continuous / intermittent lifetime TCFs from per-bond binary presence
/// series `present[bond][frame]`.
///
/// All series must share the same frame count. `dt` is the frame spacing
/// (> 0); `max_lag` is clamped to `n_frames − 1`.
///
/// Continuous `S_HB(t)` and intermittent `C_HB(t)` TCFs (Luzar & Chandler,
/// *Phys. Rev. Lett.* **1996**, 76, 928): with `h(t) ∈ {0,1}` the per-bond
/// presence indicator,
///
/// ```text
/// C_HB(t) = ⟨h(0) h(t)⟩ / ⟨h⟩            (intermittent — gaps allowed)
/// S_HB(t) = ⟨h(0) Θ(t) h(t)⟩ / ⟨h⟩       (continuous — bond never broke in [0,t])
/// ```
///
/// The continuous / intermittent accumulation mirrors molrs's
/// [`pair_survival_tcf`](crate::compute::pair_survival_tcf) exactly
/// (`SurvivalMethod::Continuous` walks forward until the first absence;
/// `Intermittent` counts every lag where the bond is present).
///
/// **Documented deviation:** `pair_survival_tcf`'s presence model is a *distance*
/// cutoff and cannot express the D–H···A *angle* criterion, so its API cannot be
/// called directly here. This module applies the identical continuous /
/// intermittent definitions to the geometric presence series produced by
/// [`HBonds`](super::detect::HBonds), rather than re-deriving a different TCF.
pub fn hbond_lifetimes(
    present: &[Vec<bool>],
    dt: F,
    max_lag: usize,
) -> Result<HBondLifetimeResult, ComputeError> {
    if present.is_empty() {
        return Err(ComputeError::EmptyInput);
    }
    let n_frames = present[0].len();
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    for s in present {
        if s.len() != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: s.len(),
                what: "hbond presence series length",
            });
        }
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "hbond_lifetimes::dt",
            value: dt.to_string(),
        });
    }
    let max_lag = max_lag.min(n_frames - 1);
    let n_series = present.len();

    // acc_*[tau] = Σ over (bond, origin t0) of the survival indicator at lag tau.
    //
    // Continuous (`acc_c`) stops at the first absence — a forward walk, not an
    // autocorrelation — so it stays a direct sum. Intermittent (`acc_i`) counts
    // every present lag: `acc_i[tau] = Σ_series Σ_{t0} h[t0]·h[t0+tau]`, i.e. the
    // un-normalized autocorrelation of the binary presence series (Wiener–
    // Khinchin), evaluated in O(n log n) per series with the FFT. Each series is
    // independent, so both accumulators fan out over series and reduce.
    // Continuous: the forward walk from every present origin, summed over
    // bonds (integer counts, so the sum order does not matter).
    let acc_c = accumulate_continuous(present, n_frames, max_lag);
    let acc_c = acc_c.as_slice().expect("contiguous acc_c");

    // Intermittent: the 0/1 presence autocorrelation over all origins, one
    // bond per entity, averaged over bonds and origins by the engine.
    let mut presence = Array3::<F>::zeros((n_frames, n_series, 1));
    for (b, h) in present.iter().enumerate() {
        for (t, &on) in h.iter().enumerate() {
            if on {
                presence[[t, b, 0]] = 1.0;
            }
        }
    }
    let inter_raw = autocorrelate(
        presence.view(),
        dt,
        &TimeCorrelationSpec::new().max_lag(max_lag),
    )?
    .values;

    let mut cont = Array1::<F>::zeros(max_lag + 1);
    let mut inter = Array1::<F>::zeros(max_lag + 1);
    let mean_c = |tau: usize| -> F { acc_c[tau] / (n_series * (n_frames - tau)) as F };
    let c0 = mean_c(0);
    let i0 = inter_raw[0];
    for tau in 0..=max_lag {
        cont[tau] = if c0 > 0.0 { mean_c(tau) / c0 } else { 0.0 };
        inter[tau] = if i0 > 0.0 { inter_raw[tau] / i0 } else { 0.0 };
    }

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as F * dt));
    // τ = ∫ C(t) dt by the trapezoid rule (the running integral's last value).
    let last = |y: &Array1<F>| running_trapezoid(y.as_slice().expect("contiguous"), dt)[max_lag];
    let tau_continuous = last(&cont);
    let tau_intermittent = last(&inter);

    Ok(HBondLifetimeResult {
        lag_times,
        continuous: cont,
        intermittent: inter,
        tau_continuous,
        tau_intermittent,
    })
}

/// Continuous-survival counts `Σ_bonds Σ_{t0: h(t0)} Π_{k≤τ} h(t0+k)` per lag.
fn accumulate_continuous(present: &[Vec<bool>], n_frames: usize, max_lag: usize) -> Array1<f64> {
    let per_series = |h: &Vec<bool>| -> Array1<f64> {
        let mut c = Array1::<f64>::zeros(max_lag + 1);
        for t0 in 0..n_frames {
            if !h[t0] {
                continue;
            }
            let lmax = max_lag.min(n_frames - 1 - t0);
            c[0] += 1.0;
            for tau in 1..=lmax {
                if h[t0 + tau] {
                    c[tau] += 1.0;
                } else {
                    break;
                }
            }
        }
        c
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        present
            .par_iter()
            .map(per_series)
            .reduce(|| Array1::<f64>::zeros(max_lag + 1), |a, b| a + b)
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut acc = Array1::<f64>::zeros(max_lag + 1);
        for h in present {
            acc += &per_series(h);
        }
        acc
    }
}

/// Build per-bond presence series from an [`HBondsResult`], keyed by the ordered
/// `(donor, acceptor)` pair. Returns the bond keys and the
/// `present[bond][frame]` matrix, ready for [`hbond_lifetimes`].
pub fn presence_from_hbonds(res: &HBondsResult) -> (Vec<(u32, u32)>, Vec<Vec<bool>>) {
    let n_frames = res.per_frame.len();
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut keys: Vec<(u32, u32)> = Vec::new();
    for frame in &res.per_frame {
        for b in frame {
            let key = (b.donor, b.acceptor);
            if let std::collections::hash_map::Entry::Vacant(e) = index.entry(key) {
                e.insert(keys.len());
                keys.push(key);
            }
        }
    }
    let mut present = vec![vec![false; n_frames]; keys.len()];
    for (t, frame) in res.per_frame.iter().enumerate() {
        for b in frame {
            let bi = index[&(b.donor, b.acceptor)];
            present[bi][t] = true;
        }
    }
    (keys, present)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    fn presence(n_series: usize, n_frames: usize, seed: u64) -> Vec<Vec<bool>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n_series)
            .map(|_| {
                (0..n_frames)
                    .map(|_| rng.random_range(0.0..1.0) < 0.6)
                    .collect()
            })
            .collect()
    }

    /// The definitions: intermittent `Σ h(t0) h(t0+τ)`, continuous forward
    /// walk, both per-lag normalized and scaled to 1 at lag 0.
    fn direct(present: &[Vec<bool>], max_lag: usize) -> (Vec<F>, Vec<F>) {
        let n_frames = present[0].len();
        let n_series = present.len();
        let mut inter = vec![0.0; max_lag + 1];
        let mut cont = vec![0.0; max_lag + 1];
        for h in present {
            for t0 in 0..n_frames {
                if !h[t0] {
                    continue;
                }
                let mut alive = true;
                for tau in 0..=max_lag.min(n_frames - 1 - t0) {
                    if h[t0 + tau] {
                        inter[tau] += 1.0;
                    } else {
                        alive = false;
                    }
                    if alive {
                        cont[tau] += 1.0;
                    }
                }
            }
        }
        let norm = |acc: &mut Vec<F>| {
            for (tau, v) in acc.iter_mut().enumerate() {
                *v /= (n_series * (n_frames - tau)) as F;
            }
            let c0 = acc[0];
            for v in acc.iter_mut() {
                *v /= c0;
            }
        };
        norm(&mut inter);
        norm(&mut cont);
        (cont, inter)
    }

    #[test]
    fn matches_the_direct_definitions() {
        let present = presence(5, 30, 17);
        let max_lag = 12;
        let dt = 0.5;
        let got = hbond_lifetimes(&present, dt, max_lag).unwrap();
        let (cont, inter) = direct(&present, max_lag);
        for tau in 0..=max_lag {
            assert!(
                (got.intermittent[tau] - inter[tau]).abs() < 1e-12,
                "intermittent lag {tau}: {} vs {}",
                got.intermittent[tau],
                inter[tau]
            );
            assert!(
                (got.continuous[tau] - cont[tau]).abs() < 1e-12,
                "continuous lag {tau}: {} vs {}",
                got.continuous[tau],
                cont[tau]
            );
        }
        assert!((got.intermittent[0] - 1.0).abs() < 1e-12);
        assert!((got.continuous[0] - 1.0).abs() < 1e-12);
        // τ = trapezoid of the normalized curve.
        let mut trap = 0.0;
        for tau in 1..=max_lag {
            trap += 0.5 * (inter[tau - 1] + inter[tau]) * dt;
        }
        assert!((got.tau_intermittent - trap).abs() < 1e-12);
        assert!((got.lag_times[3] - 3.0 * dt).abs() < 1e-12);
    }

    #[test]
    fn always_present_bond_never_decays() {
        let present = vec![vec![true; 10]; 2];
        let got = hbond_lifetimes(&present, 1.0, 5).unwrap();
        for tau in 0..=5 {
            assert!((got.intermittent[tau] - 1.0).abs() < 1e-12);
            assert!((got.continuous[tau] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn bad_input_errors() {
        assert!(hbond_lifetimes(&[], 1.0, 2).is_err());
        assert!(hbond_lifetimes(&[vec![true, false], vec![true]], 1.0, 1).is_err());
        assert!(hbond_lifetimes(&[vec![true, false]], 0.0, 1).is_err());
    }
}

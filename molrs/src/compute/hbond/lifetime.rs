//! Hydrogen-bond lifetime correlation functions.

use molrs::op::F;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::detect::HBondsResult;
use crate::compute::{
    Compute, ComputeError, CumulativeTrapezoid, Survival, SurvivalInput, SurvivalKind,
};

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
/// Both curves are the [`Survival`] correlation of the presence series
/// (`SurvivalKind::Continuous` / `Intermittent`), normalised to 1 at lag 0.
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

    // ⟨h(0) h(t)⟩ (intermittent) and ⟨h(0) Θ(t) h(t)⟩ (continuous) per
    // (bond, origin), then normalised so the curve is 1 at lag 0.
    let presence = Array2::from_shape_fn((n_frames, n_series), |(t, bond)| {
        if present[bond][t] { 1.0 } else { 0.0 }
    });
    let survival = |kind| -> Result<Array1<F>, ComputeError> {
        let mean = Survival::new(kind, max_lag)
            .compute(SurvivalInput {
                presence: presence.view(),
                dt,
            })?
            .channel(0)
            .to_owned();
        let at_zero = mean[0];
        Ok(if at_zero > 0.0 {
            mean / at_zero
        } else {
            Array1::zeros(max_lag + 1)
        })
    };
    let cont = survival(SurvivalKind::Continuous)?;
    let inter = survival(SurvivalKind::Intermittent)?;

    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as F * dt));
    let tau_continuous = CumulativeTrapezoid::total(&cont.to_vec(), dt);
    let tau_intermittent = CumulativeTrapezoid::total(&inter.to_vec(), dt);

    Ok(HBondLifetimeResult {
        lag_times,
        continuous: cont,
        intermittent: inter,
        tau_continuous,
        tau_intermittent,
    })
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

    /// One bond present at frames 0, 1, 3, 4, 5 (absent at 2), dt = 0.5.
    #[test]
    fn lifetimes_by_hand() {
        let present = vec![vec![true, true, false, true, true, true]];
        let res = hbond_lifetimes(&present, 0.5, 3).unwrap();
        // ⟨h⟩ = 5/6. Continuous: 3/5 and 1/4 unbroken, then none; gapped:
        // 3/5, 2/4, 2/3 — each divided by ⟨h⟩.
        let h = 5.0 / 6.0;
        let continuous = [1.0, 0.6 / h, 0.25 / h, 0.0];
        let intermittent = [1.0, 0.6 / h, 0.5 / h, (2.0 / 3.0) / h];
        for lag in 0..=3 {
            assert!(
                (res.continuous[lag] - continuous[lag]).abs() < 1e-12,
                "S({lag})"
            );
            assert!(
                (res.intermittent[lag] - intermittent[lag]).abs() < 1e-12,
                "C({lag})"
            );
        }
        let trapz = |y: &[F]| y.windows(2).map(|w| 0.25 * (w[0] + w[1])).sum::<F>();
        assert!((res.tau_continuous - trapz(&continuous)).abs() < 1e-12);
        assert!((res.tau_intermittent - trapz(&intermittent)).abs() < 1e-12);
        assert_eq!(res.lag_times.to_vec(), vec![0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn a_bond_that_never_forms_gives_zero_curves() {
        let res = hbond_lifetimes(&[vec![false; 4]], 1.0, 2).unwrap();
        assert!(res.continuous.iter().all(|v| *v == 0.0));
        assert!(res.intermittent.iter().all(|v| *v == 0.0));
    }
}

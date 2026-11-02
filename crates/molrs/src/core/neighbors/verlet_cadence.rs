//! When a Verlet list must be searched again: [`VerletCadence`].

use ndarray::ArrayView2;

use crate::core::SimBox;
use crate::op::{F, Fnx3};

use super::verlet_skin::{NeighborPolicy, SkinError};

/// The LAMMPS `neigh_modify every/delay/check` gate, without the search.
///
/// A list built at `r_build = cutoff + skin` stays complete out to `cutoff`
/// while no pair held beyond `r_build` can have come inside `cutoff`. Let `M`
/// be the map from the cell the positions were held in to the current cell,
/// and `Δx̃ᵢ = xᵢ − M xᵢ,hold` each atom's motion beyond that map (minimum
/// image). A pair held at `r > r_build` is now farther than
/// `r_build·(1 − ‖M − I‖) − 2·maxᵢ|Δx̃ᵢ|`, which stays at or beyond `cutoff`
/// while `2·maxᵢ|Δx̃ᵢ| + ‖M − I‖·r_build ≤ skin`; a rebuild is due once that
/// bound exceeds `skin`. With a fixed cell this is the half-skin test of LAMMPS
/// `Neighbor::check_distance`. `‖·‖` is the Frobenius norm, an upper bound of
/// the spectral norm the bound needs.
///
/// The cadence owns no pairs. [`VerletSkin`](super::VerletSkin) pairs it with a
/// search; a force provider that searches on its own device reads only the
/// decision.
///
/// # References
///
/// LAMMPS `Neighbor::decide` / `Neighbor::check_distance`.
#[derive(Clone, Debug)]
pub struct VerletCadence {
    policy: NeighborPolicy,
    r_build: F,
    ago: usize,
    n_rebuilds: usize,
    ndanger: usize,
    danger_ago: usize,
    x_hold: Fnx3,
    /// The cell the positions were held in, with their fractional coordinates.
    /// `None` when they were held without a defined cell.
    held_cell: Option<(SimBox, Fnx3)>,
}

impl VerletCadence {
    /// A cadence for a list built at `r_build`, holding `positions` in
    /// `simbox`. The initial hold is not a rebuild.
    ///
    /// # Errors
    ///
    /// [`SkinError::Invalid`] for an `every` below 1, a `delay` that is not a
    /// multiple of `every`, a negative skin, or positions not of shape `(N, 3)`.
    pub fn new(
        policy: NeighborPolicy,
        r_build: F,
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
    ) -> Result<Self, SkinError> {
        if !(policy.skin >= 0.0 && policy.skin.is_finite()) {
            return Err(SkinError::Invalid(format!(
                "skin must be >= 0 Å, got {} Å",
                policy.skin
            )));
        }
        if policy.every < 1 {
            return Err(SkinError::Invalid(format!(
                "every must be an integer >= 1 step, got {}",
                policy.every
            )));
        }
        if !policy.delay.is_multiple_of(policy.every) {
            return Err(SkinError::Invalid(format!(
                "delay {} steps must be a multiple of every {} steps; LAMMPS Neighbor::init \
                 rejects the same pair, because the danger threshold max(every, delay) would \
                 be an ago the gate never permits, leaving ndanger silently dead.",
                policy.delay, policy.every
            )));
        }
        if positions.ncols() != 3 {
            return Err(SkinError::Invalid(format!(
                "positions must have shape (N, 3), got {:?}",
                positions.shape()
            )));
        }
        let mut cadence = Self {
            policy,
            r_build,
            ago: 0,
            n_rebuilds: 0,
            ndanger: 0,
            danger_ago: policy.every.max(policy.delay),
            x_hold: Fnx3::zeros((positions.nrows(), 3)),
            held_cell: None,
        };
        cadence.hold(positions, simbox);
        Ok(cadence)
    }

    /// The gate's policy (fixed at construction).
    pub fn policy(&self) -> NeighborPolicy {
        self.policy
    }

    /// Steps since the last build (LAMMPS `ago`).
    pub fn ago(&self) -> usize {
        self.ago
    }

    /// Rebuilds since construction; the initial hold is not one of them.
    pub fn n_rebuilds(&self) -> usize {
        self.n_rebuilds
    }

    /// Rebuilds that fired at the first permitted opportunity.
    pub fn ndanger(&self) -> usize {
        self.ndanger
    }

    /// Restore the counters from a trusted persistence snapshot, so a resumed
    /// run makes the same next decision.
    pub fn restore_progress(&mut self, ago: usize, n_rebuilds: usize, ndanger: usize) {
        self.ago = ago;
        self.n_rebuilds = n_rebuilds;
        self.ndanger = ndanger;
    }

    /// Advance one step and say whether the list must be rebuilt at
    /// `positions` in `simbox`. The caller rebuilds and then calls
    /// [`rebuilt`](Self::rebuilt).
    ///
    /// # Errors
    ///
    /// [`SkinError::Invalid`] for positions of the wrong shape.
    /// [`SkinError::Guard`] when a displacement reaches half the smallest
    /// perpendicular width of a periodic cell: the minimum image cannot say
    /// which copy the atom came from, so the number is not a displacement.
    pub fn due(
        &mut self,
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
    ) -> Result<bool, SkinError> {
        self.require_shape(positions)?;
        self.ago += 1;
        let permitted = self.ago >= self.policy.delay && self.ago.is_multiple_of(self.policy.every);
        if !permitted {
            return Ok(false);
        }
        if !self.policy.check {
            return Ok(true);
        }
        let current = simbox.filter(|b| b.is_cell_defined());
        let due = match (&self.held_cell, current) {
            (None, None) => {
                self.bound(positions, self.x_hold.view(), None, 0.0)? > self.policy.skin
            }
            (Some((held, _)), Some(now)) if held.approx_eq(now, 0.0) => {
                self.bound(positions, self.x_hold.view(), Some(now), 0.0)? > self.policy.skin
            }
            (Some((held, frac)), Some(now)) => {
                let predicted = now.to_cart(frac.view());
                let strain = strain(held, now);
                self.bound(positions, predicted.view(), Some(now), strain)? > self.policy.skin
            }
            // A cell appeared or went away: nothing held is comparable.
            _ => true,
        };
        if due && self.ago == self.danger_ago {
            self.ndanger += 1;
        }
        Ok(due)
    }

    /// Record a rebuild at `positions` in `simbox`.
    pub fn rebuilt(&mut self, positions: ArrayView2<'_, F>, simbox: Option<&SimBox>) {
        self.n_rebuilds += 1;
        self.hold(positions, simbox);
    }

    fn hold(&mut self, positions: ArrayView2<'_, F>, simbox: Option<&SimBox>) {
        self.ago = 0;
        self.x_hold.assign(&positions);
        self.held_cell = simbox
            .filter(|b| b.is_cell_defined())
            .map(|b| (b.clone(), b.to_frac(positions)));
    }

    /// `2·max|xᵢ − predictedᵢ| + strain·r_build`, the displacement measured
    /// by the minimum image of `simbox` when it is periodic.
    fn bound(
        &self,
        positions: ArrayView2<'_, F>,
        predicted: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
        strain: F,
    ) -> Result<F, SkinError> {
        let periodic = simbox.filter(|b| !b.is_free());
        let mic = periodic.map(SimBox::mic);
        let mut max_d2: F = 0.0;
        for (x, p) in positions.rows().into_iter().zip(predicted.rows()) {
            let raw = [x[0] - p[0], x[1] - p[1], x[2] - p[2]];
            let d = match &mic {
                Some(mic) => mic.apply(raw),
                None => raw,
            };
            max_d2 = max_d2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        if let Some(cell) = periodic {
            let half_width = 0.5
                * cell
                    .nearest_plane_distance()
                    .iter()
                    .copied()
                    .fold(F::INFINITY, F::min);
            if max_d2 >= half_width * half_width {
                return Err(SkinError::Guard(format!(
                    "displacement {:.3} Å >= half the minimum perpendicular cell \
                     width ({half_width:.3} Å): the minimum image can no longer identify \
                     which copy this atom came from, so the number is not a displacement. \
                     A cell change or a blow-up, not a wrap — wrapped coordinates are \
                     expected and handled.",
                    max_d2.sqrt()
                )));
            }
        }
        Ok(2.0 * max_d2.sqrt() + strain * self.r_build)
    }

    fn require_shape(&self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        let n = self.x_hold.nrows();
        if positions.shape() != [n, 3] {
            return Err(SkinError::Invalid(format!(
                "positions must have shape ({n}, 3) — this cadence was constructed for {n} \
                 atoms — but got {:?}",
                positions.shape()
            )));
        }
        Ok(())
    }
}

/// `‖M − I‖_F` for the linear map `M` taking `held`'s cell onto `now`'s.
fn strain(held: &SimBox, now: &SimBox) -> F {
    let mut basis = Fnx3::zeros((4, 3));
    for k in 0..3 {
        basis[[k + 1, k]] = 1.0;
    }
    let mapped = now.to_cart(held.to_frac(basis.view()).view());
    let mut sum = 0.0;
    for k in 0..3 {
        for axis in 0..3 {
            let identity = if axis == k { 1.0 } else { 0.0 };
            let m = mapped[[k + 1, axis]] - mapped[[0, axis]] - identity;
            sum += m * m;
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    fn cubic(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0; 3], [true; 3]).unwrap()
    }

    fn policy(skin: F) -> NeighborPolicy {
        NeighborPolicy {
            skin,
            ..NeighborPolicy::default()
        }
    }

    /// With a fixed cell the bound is twice the largest displacement: a step
    /// of 0.49·skin holds, 0.51·skin rebuilds.
    #[test]
    fn a_fixed_cell_rebuilds_past_half_the_skin() {
        let skin = 1.0;
        let x0 = array![[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]];
        let bx = cubic(20.0);
        for (step, rebuild) in [(0.49, false), (0.51, true)] {
            let mut cadence = VerletCadence::new(policy(skin), 6.0, x0.view(), Some(&bx)).unwrap();
            let mut x = x0.clone();
            x[[0, 0]] += step * skin;
            assert_eq!(cadence.due(x.view(), Some(&bx)).unwrap(), rebuild, "{step}");
        }
    }

    /// A uniform strain with no motion beyond it: `‖M − I‖_F = √3·ε`, so the
    /// bound is `√3·ε·r_build`. At 0.9·skin it holds, at 1.1·skin it rebuilds.
    #[test]
    fn an_isotropic_strain_is_bounded_by_the_build_radius() {
        let skin = 0.5;
        let r_build = 6.0;
        let l = 20.0;
        let x0 = array![[5.0, 5.0, 5.0], [7.0, 6.0, 5.5]];
        let held = cubic(l);
        for (fraction, rebuild) in [(0.9, false), (1.1, true)] {
            let eps = fraction * skin / (3.0_f64.sqrt() * r_build);
            let now = cubic(l * (1.0 + eps));
            let x = x0.mapv(|v| v * (1.0 + eps));
            let mut cadence =
                VerletCadence::new(policy(skin), r_build, x0.view(), Some(&held)).unwrap();
            assert_eq!(
                cadence.due(x.view(), Some(&now)).unwrap(),
                rebuild,
                "{fraction}"
            );
        }
    }

    /// Wrapped through a face, a small step reads as the small step.
    #[test]
    fn a_wrap_is_a_small_step() {
        let bx = cubic(10.0);
        let x0 = array![[9.9, 5.0, 5.0]];
        let mut cadence = VerletCadence::new(policy(1.0), 3.0, x0.view(), Some(&bx)).unwrap();
        assert!(
            !cadence
                .due(array![[0.1, 5.0, 5.0]].view(), Some(&bx))
                .unwrap()
        );
    }

    /// Without a cell the displacement is the plain difference.
    #[test]
    fn no_cell_measures_the_plain_difference() {
        let x0 = array![[0.0, 0.0, 0.0]];
        let mut cadence = VerletCadence::new(policy(1.0), 3.0, x0.view(), None).unwrap();
        assert!(!cadence.due(array![[0.4, 0.0, 0.0]].view(), None).unwrap());
        assert!(cadence.due(array![[0.6, 0.0, 0.0]].view(), None).unwrap());
        cadence.rebuilt(array![[0.6, 0.0, 0.0]].view(), None);
        assert_eq!(cadence.n_rebuilds(), 1);
        assert_eq!(cadence.ago(), 0);
    }

    #[test]
    fn a_cell_that_appears_rebuilds() {
        let x0 = array![[1.0, 1.0, 1.0]];
        let mut cadence = VerletCadence::new(policy(1.0), 3.0, x0.view(), None).unwrap();
        assert!(cadence.due(x0.view(), Some(&cubic(10.0))).unwrap());
    }

    #[test]
    fn every_and_delay_gate_the_check() {
        let x0 = array![[0.0, 0.0, 0.0]];
        let far = array![[0.9, 0.0, 0.0]];
        let mut cadence = VerletCadence::new(
            NeighborPolicy {
                skin: 1.0,
                every: 2,
                delay: 4,
                check: true,
            },
            3.0,
            x0.view(),
            None,
        )
        .unwrap();
        let decisions: Vec<bool> = (0..4)
            .map(|_| cadence.due(far.view(), None).unwrap())
            .collect();
        assert_eq!(decisions, [false, false, false, true]);
        assert_eq!(cadence.ndanger(), 1);
    }
}

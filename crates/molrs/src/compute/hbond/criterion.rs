//! Geometric hydrogen-bond criterion.

use molrs::op::F;

use crate::compute::ComputeError;

crate::compute::named_enum::named_enum! {
    /// Which interatomic distance the cutoff applies to when pairing candidates.
    pub enum HBondDistanceKind {
        /// Cutoff on the donor–acceptor distance r(D···A) (reference implementation `m_frAD`).
        DonorAcceptor => "donor_acceptor",
        /// Cutoff on the hydrogen–acceptor distance r(H···A) (reference implementation `m_frAH`).
        HydrogenAcceptor => "hydrogen_acceptor",
    }
}

/// Frozen geometric parameters for hydrogen-bond detection.
///
/// `Default` is the Luzar–Chandler water convention: r(D···A) ≤ 3.5 Å,
/// ∠(D–H···A) ≥ 150°.
///
/// Ported from the reference implementation `CHBond` (`src/hbond.cpp`, the `m_frAD` / `m_frAH` /
/// `m_fwinkel` gate around lines 914–961): a D–H···A bond requires the
/// acceptor–donor distance `rAD` below a cutoff, the acceptor–hydrogen distance
/// `rAH` below a cutoff, and an angle test on `cos(winkel)` where `winkel` is the
/// angle **at the donor** between the D→A and D→H vectors (reference implementation default
/// ≤ 30° from collinear).
///
/// **Documented deviation (per the spec's acceptance contract):** molrs gates on
/// the Luzar–Chandler **D–H···A angle at the hydrogen** (Luzar & Chandler,
/// *Nature* **1996**, 379, 55), with the common defaults r(D···A) ≤ 3.5 Å and
/// ∠(D–H···A) ≥ 150° (i.e. ≤ 30° from the linear 180°). Both conventions encode
/// the same near-linear-bond intent; molrs uses the angle-at-H form because that
/// is the quantity the spec and the downstream lifetime TCFs reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HBondCriterion {
    /// Distance cutoff in ångström (applies to `dist_kind`).
    pub dist_cutoff: F,
    /// Which distance the cutoff gates.
    pub dist_kind: HBondDistanceKind,
    /// Minimum D–H···A angle in **degrees** (measured at the hydrogen).
    pub angle_cutoff: F,
    /// With [`HBondDistanceKind::HydrogenAcceptor`], the donor–acceptor
    /// distance must also stay within `dist_cutoff + donor_acceptor_margin`
    /// (Å) — the margin is about one D–H bond length. Unused with
    /// [`HBondDistanceKind::DonorAcceptor`].
    pub donor_acceptor_margin: F,
}

impl Default for HBondCriterion {
    fn default() -> Self {
        Self {
            dist_cutoff: 3.5,
            dist_kind: HBondDistanceKind::DonorAcceptor,
            angle_cutoff: 150.0,
            donor_acceptor_margin: 1.05,
        }
    }
}

impl HBondCriterion {
    /// Construct an explicit criterion; the donor–acceptor margin keeps its
    /// default (1.05 Å).
    pub fn new(dist_cutoff: F, dist_kind: HBondDistanceKind, angle_cutoff: F) -> Self {
        Self {
            dist_cutoff,
            dist_kind,
            angle_cutoff,
            ..Self::default()
        }
    }

    /// Set the donor–acceptor margin of the hydrogen–acceptor gate (Å).
    pub fn with_donor_acceptor_margin(mut self, margin: F) -> Self {
        self.donor_acceptor_margin = margin;
        self
    }

    /// Every field in its physical range.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite
    /// `dist_cutoff`, an `angle_cutoff` outside `[0, 180]` degrees, or a
    /// negative or non-finite `donor_acceptor_margin`.
    pub(crate) fn validate(&self) -> Result<(), ComputeError> {
        let reject = |field, value: F| {
            Err(ComputeError::OutOfRange {
                field,
                value: value.to_string(),
            })
        };
        if !(self.dist_cutoff.is_finite() && self.dist_cutoff > 0.0) {
            return reject("HBondCriterion::dist_cutoff", self.dist_cutoff);
        }
        if !(0.0..=180.0).contains(&self.angle_cutoff) {
            return reject("HBondCriterion::angle_cutoff (degrees)", self.angle_cutoff);
        }
        if !(self.donor_acceptor_margin.is_finite() && self.donor_acceptor_margin >= 0.0) {
            return reject(
                "HBondCriterion::donor_acceptor_margin",
                self.donor_acceptor_margin,
            );
        }
        Ok(())
    }
}

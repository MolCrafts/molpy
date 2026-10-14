//! Bond-orientation 2-D histogram on the unit sphere.

use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array2;
use std::f64::consts::PI;

use crate::compute::order::SphericalHarmonic;
use crate::compute::{Axis, Compute, ComputeError, is_half_shell, require_disp};

/// Bond-order diagram calculator.
///
/// Mirrors `freud.environment.BondOrientationalOrder`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/BondOrientationalOrder.cc)).
///
/// Configuration: the `(θ, φ)` bin counts. Data: one snapshot's neighbor
/// table with the `disp` column. For every neighbor pair `(i, j)` the bond
/// vector `r̂ = (r_j − r_i) / |…|` is converted to spherical angles `(θ, φ)`
/// (the same conversion the spherical harmonics of Steinhardt use) and
/// accumulated into a `(n_θ × n_φ)` histogram on the unit sphere. A half-shell
/// table also counts each reversed bond. We normalise by the solid angle of
/// each pixel (`sin(θ_c) · dθ · dφ`) so that, in the random-bond limit, the
/// output approaches a constant.
///
/// freud supports four normalisation modes (BOD, LBOD, OBCD, ABCD); only
/// the **Bond Order Diagram (BOD)** is implemented here, which is the
/// default mode. The remaining three flavours require either per-particle
/// orientations or per-pair query orientations and can be added in
/// follow-up phases.
#[derive(Debug, Clone, Copy)]
pub struct BondOrientationalOrder {
    theta: Axis,
    phi: Axis,
}

/// The data a [`BondOrientationalOrder`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct BondOrientationalOrderInput<'a> {
    /// One snapshot's neighbor table with the `disp` column.
    pub neighbors: &'a Neighbors,
}

impl BondOrientationalOrder {
    /// `n_theta × n_phi` angular bins over the unit sphere: `θ ∈ [0, π]`,
    /// `φ ∈ [−π, π]`, both attained at their ends, so both axes bin with the
    /// upper edge closed.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a zero bin count.
    pub fn new(n_theta: usize, n_phi: usize) -> Result<Self, ComputeError> {
        if n_theta == 0 || n_phi == 0 {
            return Err(ComputeError::OutOfRange {
                field: "BondOrientationalOrder bin counts",
                value: format!("({n_theta}, {n_phi})"),
            });
        }
        Ok(Self {
            theta: Axis::new(0.0, PI, n_theta)?,
            phi: Axis::new(-PI, PI, n_phi)?,
        })
    }

    /// Number of polar bins.
    pub fn n_theta(&self) -> usize {
        self.theta.n_bins()
    }

    /// Number of azimuthal bins.
    pub fn n_phi(&self) -> usize {
        self.phi.n_bins()
    }

    /// Count one bond direction in its (θ, φ) bin.
    fn push(&self, counts: &mut Array2<u64>, disp: [F; 3]) -> Result<(), ComputeError> {
        let Some((theta, phi)) = SphericalHarmonic::angles(disp) else {
            return Ok(());
        };
        // Both angles lie in their closed ranges by construction.
        if let (Some(it), Some(ip)) = (self.theta.index_closed(theta)?, self.phi.index_closed(phi)?)
        {
            counts[[it, ip]] += 1;
        }
        Ok(())
    }
}

impl Compute for BondOrientationalOrder {
    type Input<'a> = BondOrientationalOrderInput<'a>;
    type Output = BondOrientationalOrderResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp`;
    /// [`ComputeError::NonFinite`] for a non-finite bond.
    fn compute(
        &self,
        input: BondOrientationalOrderInput<'_>,
    ) -> Result<BondOrientationalOrderResult, ComputeError> {
        let nlist = input.neighbors;
        // Every bond's direction is binned; an indices-only table has none.
        let disp = require_disp(nlist)?;
        let symmetric = is_half_shell(nlist.mode());
        let (n_theta, n_phi) = (self.n_theta(), self.n_phi());

        let mut counts = Array2::<u64>::zeros((n_theta, n_phi));
        for row in disp.outer_iter() {
            let d = [row[0], row[1], row[2]];
            self.push(&mut counts, d)?;
            if symmetric {
                // j-side bond is reversed: (-dx, -dy, -dz).
                self.push(&mut counts, [-d[0], -d[1], -d[2]])?;
            }
        }

        let d_theta = self.theta.width();
        let d_phi = self.phi.width();
        let mut bond_order = Array2::<F>::zeros((n_theta, n_phi));
        for i in 0..n_theta {
            let theta_c = (i as F + 0.5) * d_theta;
            let solid = theta_c.sin() * d_theta * d_phi;
            for j in 0..n_phi {
                bond_order[[i, j]] = counts[[i, j]] as F / solid;
            }
        }

        Ok(BondOrientationalOrderResult {
            bond_order,
            raw_counts: counts,
            theta_edges: self.theta.edges().to_vec(),
            phi_edges: self.phi.edges().to_vec(),
        })
    }
}

/// One snapshot's bond-order histogram.
#[derive(Debug, Clone, Default)]
pub struct BondOrientationalOrderResult {
    /// `(n_θ, n_φ)` bond-count histogram normalised by solid angle.
    pub bond_order: Array2<F>,
    /// Raw bond counts per bin (before solid-angle normalisation).
    pub raw_counts: Array2<u64>,
    /// θ bin edges (length `n_θ + 1`), in radians.
    pub theta_edges: Vec<F>,
    /// φ bin edges (length `n_φ + 1`), in radians.
    pub phi_edges: Vec<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::SimBox;
    use ndarray::array;

    /// A self-query table over `positions` in an open cube of side `box_len`.
    fn table(positions: &[[F; 3]], box_len: F, cutoff: F) -> Neighbors {
        let simbox =
            SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap();
        neighbors_of(positions, &simbox, cutoff)
    }

    fn bod(n_theta: usize, n_phi: usize, nl: &Neighbors) -> BondOrientationalOrderResult {
        BondOrientationalOrder::new(n_theta, n_phi)
            .unwrap()
            .compute(BondOrientationalOrderInput { neighbors: nl })
            .unwrap()
    }

    #[test]
    fn single_bond_lands_in_correct_bin() {
        // Centre particle + neighbor at +z: θ = 0, φ undefined (arbitrary
        // φ bin). The bond and its symmetric reverse (centre as j) gives
        // bonds at θ = 0 and θ = π → bins 0 and n_θ-1.
        let nl = table(&[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]], 20.0, 1.5);
        let r = &bod(10, 10, &nl);
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
        // Top bin (θ ≈ 0) and bottom bin (θ ≈ π) each got one bond.
        let top: u64 = (0..10).map(|j| r.raw_counts[[0, j]]).sum();
        let bot: u64 = (0..10).map(|j| r.raw_counts[[9, j]]).sum();
        assert_eq!(top, 1);
        assert_eq!(bot, 1);
    }

    #[test]
    fn octahedral_bonds_distribute_across_axes() {
        // Centre + ±x, ±y, ±z neighbors. Bonds + reverses → 12 entries.
        let nl = table(
            &[
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [4.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
                [5.0, 4.0, 5.0],
                [5.0, 5.0, 6.0],
                [5.0, 5.0, 4.0],
            ],
            20.0,
            1.2,
        );
        let r = &bod(8, 8, &nl);
        let total: u64 = r.raw_counts.iter().copied().sum();
        // 6 unique bonds × 2 (self-query symmetric counterparts) = 12.
        assert_eq!(total, 12);
    }

    /// Bonds are directions: a table that kept only distances is refused.
    #[test]
    fn table_without_disp_is_bad_shape() {
        use molrs::core::NeighborColumns;
        let nl =
            table(&[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]], 20.0, 1.5).repack(NeighborColumns::DIST_SQ);
        let err = BondOrientationalOrder::new(10, 10)
            .unwrap()
            .compute(BondOrientationalOrderInput { neighbors: &nl })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }

    #[test]
    fn invalid_bins_error() {
        assert!(BondOrientationalOrder::new(0, 10).is_err());
        assert!(BondOrientationalOrder::new(10, 0).is_err());
    }
}

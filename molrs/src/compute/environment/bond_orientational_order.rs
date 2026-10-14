//! Bond-orientation 2-D histogram on the unit sphere.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array2;
use std::f64::consts::{PI, TAU};

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::compute::require_disp;

/// Bond-order diagram calculator.
///
/// Mirrors `freud.environment.BondOrientationalOrder`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/BondOrientationalOrder.cc)).
///
/// For every neighbor pair `(i, j)` the bond vector `r̂ = (r_j − r_i) / |…|`
/// is converted to spherical angles `(θ, φ)` and accumulated into a
/// `(n_θ × n_φ)` histogram on the unit sphere. We normalise by the solid
/// angle of each pixel (`sin(θ_c) · dθ · dφ`) so that, in the random-bond
/// limit, the output approaches a constant.
///
/// freud supports four normalisation modes (BOD, LBOD, OBCD, ABCD); only
/// the **Bond Order Diagram (BOD)** is implemented here, which is the
/// default mode. The remaining three flavours require either per-particle
/// orientations or per-pair query orientations and can be added in
/// follow-up phases.
#[derive(Debug, Clone, Copy)]
pub struct BondOrientationalOrder {
    n_theta: usize,
    n_phi: usize,
}

impl BondOrientationalOrder {
    /// `n_theta × n_phi` angular bins over the unit sphere.
    pub fn new(n_theta: usize, n_phi: usize) -> Result<Self, ComputeError> {
        if n_theta == 0 || n_phi == 0 {
            return Err(ComputeError::OutOfRange {
                field: "BondOrientationalOrder bin counts",
                value: format!("({n_theta}, {n_phi})"),
            });
        }
        Ok(Self { n_theta, n_phi })
    }

    pub fn n_theta(&self) -> usize {
        self.n_theta
    }
    pub fn n_phi(&self) -> usize {
        self.n_phi
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<BondOrientationalOrderResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let _ = xs_p; // suppress unused — we only need the frame to be
        // FrameAccess-valid; nlist already carries disp.

        let d_theta = PI / self.n_theta as F;
        let d_phi = TAU / self.n_phi as F;

        let theta_edges: Vec<F> = (0..=self.n_theta).map(|i| i as F * d_theta).collect();
        let phi_edges: Vec<F> = (0..=self.n_phi).map(|i| -PI + i as F * d_phi).collect();

        let mut counts = Array2::<u64>::zeros((self.n_theta, self.n_phi));
        let n_pairs = nlist.n_pairs();
        // Every bond's direction is binned; an indices-only table has none.
        let disp = require_disp(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        for k in 0..n_pairs {
            let dx = disp[[k, 0]];
            let dy = disp[[k, 1]];
            let dz = disp[[k, 2]];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r == 0.0 {
                continue;
            }
            push_angle(&mut counts, dx, dy, dz, r, self.n_theta, self.n_phi);
            if symmetric {
                // j-side bond is reversed: (-dx, -dy, -dz).
                push_angle(&mut counts, -dx, -dy, -dz, r, self.n_theta, self.n_phi);
            }
        }

        let mut bond_order = Array2::<F>::zeros((self.n_theta, self.n_phi));
        for i in 0..self.n_theta {
            let theta_c = (i as F + 0.5) * d_theta;
            let solid = theta_c.sin() * d_theta * d_phi;
            if solid <= 0.0 {
                continue;
            }
            for j in 0..self.n_phi {
                bond_order[[i, j]] = counts[[i, j]] as F / solid;
            }
        }

        Ok(BondOrientationalOrderResult {
            bond_order,
            raw_counts: counts,
            theta_edges,
            phi_edges,
        })
    }
}

#[inline]
fn push_angle(counts: &mut Array2<u64>, dx: F, dy: F, dz: F, r: F, n_theta: usize, n_phi: usize) {
    let theta = (dz / r).clamp(-1.0, 1.0).acos(); // [0, π]
    let phi = dy.atan2(dx); // [-π, π]
    let it = ((theta / PI) * n_theta as F) as usize;
    let it = it.min(n_theta - 1);
    let ip = (((phi + PI) / TAU) * n_phi as F) as usize;
    let ip = ip.min(n_phi - 1);
    counts[[it, ip]] += 1;
}

impl LegacyCompute for BondOrientationalOrder {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<BondOrientationalOrderResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<BondOrientationalOrderResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(nlists.par_iter())
                .map(|(f, nl)| self.one_frame(*f, nl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame bond-order histogram result.
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

impl ComputeResult for BondOrientationalOrderResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    #[test]
    fn single_bond_lands_in_correct_bin() {
        // Centre particle + neighbor at +z: θ = 0, φ undefined (arbitrary
        // φ bin). The bond and its symmetric reverse (centre as j) gives
        // bonds at θ = 0 and θ = π → bins 0 and n_θ-1.
        let frame = frame_with(&[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]], 20.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &BondOrientationalOrder::new(10, 10)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
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
        let frame = frame_with(
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
        );
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &BondOrientationalOrder::new(8, 8)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        // 6 unique bonds × 2 (self-query symmetric counterparts) = 12.
        assert_eq!(total, 12);
    }

    #[test]
    fn empty_input_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = BondOrientationalOrder::new(10, 10)
            .unwrap()
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn invalid_bins_error() {
        assert!(BondOrientationalOrder::new(0, 10).is_err());
        assert!(BondOrientationalOrder::new(10, 0).is_err());
    }
}

//! Per-frame geometric hydrogen-bond detection.

use molrs::core::FrameAccess;
use molrs::core::NeighborQuery;
use molrs::op::F;

use super::criterion::{HBondCriterion, HBondDistanceKind};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::{dot, norm, sub};
use molrs::core::{Mic, SimBox};

/// A single detected D–H···A hydrogen bond (atom indices into the frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HBond {
    /// Donor heavy-atom index.
    pub donor: u32,
    /// Bridging hydrogen index.
    pub hydrogen: u32,
    /// Acceptor heavy-atom index.
    pub acceptor: u32,
    /// Donor–acceptor distance r(D···A), ångström (minimum image).
    pub distance: F,
    /// D–H···A angle at the hydrogen, degrees.
    pub angle: F,
}

/// Hydrogen-bond detector over donor `(D, H)` pairs and acceptor atoms.
///
/// Stateless parameter bag: the donor/acceptor selections and the geometric
/// [`HBondCriterion`]. `compute` returns one bond list per frame.
///
/// Ported from the reference implementation `CHBond::AnalyzeStep` / the bond test in `src/hbond.cpp`
/// (lines ~900–965): candidate donor/acceptor pairs are gathered by a cutoff
/// neighbour search, then gated by the distance and angle criterion (see
/// [`HBondCriterion`]). molrs gathers candidates with the existing
/// [`NeighborQuery`] cross-query and evaluates the geometry under the minimum
/// image via [`Mic`](molrs::core::Mic) — the same MIC the rest of
/// `compute` uses.
#[derive(Debug, Clone)]
pub struct HBonds {
    /// Donor `(heavy, hydrogen)` atom-index pairs.
    pub donors: Vec<(u32, u32)>,
    /// Acceptor heavy-atom indices.
    pub acceptors: Vec<u32>,
    /// Geometric criterion.
    pub criterion: HBondCriterion,
}

/// Result of [`HBonds`]: the satisfied bonds per frame plus per-frame counts.
#[derive(Debug, Clone)]
pub struct HBondsResult {
    /// `per_frame[t]` is the list of bonds detected in frame `t`.
    pub per_frame: Vec<Vec<HBond>>,
    /// `counts[t] == per_frame[t].len()`.
    pub counts: Vec<usize>,
}

impl ComputeResult for HBondsResult {}

impl HBonds {
    /// Construct a detector. Convenience over struct-literal syntax.
    pub fn new(donors: Vec<(u32, u32)>, acceptors: Vec<u32>, criterion: HBondCriterion) -> Self {
        Self {
            donors,
            acceptors,
            criterion,
        }
    }

    fn detect_frame<FA: FrameAccess>(&self, frame: &FA) -> Result<Vec<HBond>, ComputeError> {
        if self.donors.is_empty() || self.acceptors.is_empty() {
            return Ok(Vec::new());
        }
        let (xp, yp, zp) = get_positions_ref(frame)?;
        let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
        let n = xs.len();
        let pos = |i: u32| -> [F; 3] {
            let i = i as usize;
            [xs[i], ys[i], zs[i]]
        };
        // Bounds-check selections against the frame.
        for &(d, h) in &self.donors {
            if d as usize >= n || h as usize >= n {
                return Err(ComputeError::OutOfRange {
                    field: "HBonds::donors atom index",
                    value: format!("{d}/{h} >= {n}"),
                });
            }
        }
        for &a in &self.acceptors {
            if a as usize >= n {
                return Err(ComputeError::OutOfRange {
                    field: "HBonds::acceptors atom index",
                    value: format!("{a} >= {n}"),
                });
            }
        }

        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);

        // Candidate search: query points are the donor heavy atom (DonorAcceptor)
        // or the bridging hydrogen (HydrogenAcceptor); reference points are the
        // acceptors. Gather each selection into contiguous SoA x/y/z columns
        // (molrs-native layout) and hand them to `NeighborQuery`'s SoA entry —
        // no interleaved `Array2`. It returns cross pairs within `dist_cutoff`.
        let na = self.acceptors.len();
        let (mut acc_x, mut acc_y, mut acc_z) = (
            Vec::with_capacity(na),
            Vec::with_capacity(na),
            Vec::with_capacity(na),
        );
        for &a in &self.acceptors {
            let p = pos(a);
            acc_x.push(p[0]);
            acc_y.push(p[1]);
            acc_z.push(p[2]);
        }
        let nd = self.donors.len();
        let (mut q_x, mut q_y, mut q_z) = (
            Vec::with_capacity(nd),
            Vec::with_capacity(nd),
            Vec::with_capacity(nd),
        );
        for &(don, hyd) in &self.donors {
            let src = match self.criterion.dist_kind {
                HBondDistanceKind::DonorAcceptor => don,
                HBondDistanceKind::HydrogenAcceptor => hyd,
            };
            let p = pos(src);
            q_x.push(p[0]);
            q_y.push(p[1]);
            q_z.push(p[2]);
        }

        let cutoff = self.criterion.dist_cutoff;
        let nlist = match frame.simbox_ref() {
            Some(sb) => NeighborQuery::from_columns(sb, &acc_x, &acc_y, &acc_z, cutoff)
                .query_columns(&q_x, &q_y, &q_z),
            None => NeighborQuery::unbounded_columns(&acc_x, &acc_y, &acc_z, cutoff)
                .query_columns(&q_x, &q_y, &q_z),
        };

        let qi = nlist.query_point_indices();
        let pj = nlist.point_indices();
        let mut bonds = Vec::new();
        for k in 0..nlist.n_pairs() {
            let (donor, hydrogen) = self.donors[qi[k] as usize];
            let acceptor = self.acceptors[pj[k] as usize];
            // Exclude self-bonds (acceptor coincides with this donor's atoms).
            if acceptor == donor || acceptor == hydrogen {
                continue;
            }
            let dpos = pos(donor);
            let hpos = pos(hydrogen);
            let apos = pos(acceptor);

            // r(D···A) under MIC.
            let v_da = mic.apply(sub(apos, dpos));
            let r_da = norm(v_da);
            let dist_ok = match self.criterion.dist_kind {
                // NeighborQuery already enforced r(D···A) ≤ cutoff.
                HBondDistanceKind::DonorAcceptor => true,
                // NeighborQuery enforced r(H···A); still require r(D···A) finite.
                HBondDistanceKind::HydrogenAcceptor => r_da <= self.criterion.dist_cutoff + 1.05,
            };

            // D–H···A angle at the hydrogen: angle between H→D and H→A.
            let v_hd = mic.apply(sub(dpos, hpos));
            let v_ha = mic.apply(sub(apos, hpos));
            let n_hd = norm(v_hd);
            let n_ha = norm(v_ha);
            if n_hd == 0.0 || n_ha == 0.0 {
                // Degenerate geometry (coincident atoms) — not a bond, no NaN.
                continue;
            }
            let cos = (dot(v_hd, v_ha) / (n_hd * n_ha)).clamp(-1.0, 1.0);
            let angle = cos.acos().to_degrees();

            if dist_ok && angle >= self.criterion.angle_cutoff {
                bonds.push(HBond {
                    donor,
                    hydrogen,
                    acceptor,
                    distance: r_da,
                    angle,
                });
            }
        }
        Ok(bonds)
    }
}

impl LegacyCompute for HBonds {
    type Args<'a> = ();
    type Output = HBondsResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<HBondsResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }

        // Each frame's detection is independent (own NeighborQuery + geometry),
        // so fan out over frames. `collect::<Result<Vec<_>>>()` on `par_iter`
        // keeps the per-frame order and short-circuits on the first error, so
        // the bond lists are identical to the serial pass.
        let serial = || {
            frames
                .iter()
                .map(|frame| self.detect_frame(*frame))
                .collect::<Result<Vec<_>, _>>()
        };
        #[cfg(feature = "rayon")]
        let per_frame: Vec<Vec<HBond>> = if frames.len() >= 4 {
            use rayon::prelude::*;
            frames
                .par_iter()
                .map(|frame| self.detect_frame(*frame))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            serial()?
        };
        #[cfg(not(feature = "rayon"))]
        let per_frame: Vec<Vec<HBond>> = serial()?;

        let counts = per_frame.iter().map(Vec::len).collect();
        Ok(HBondsResult { per_frame, counts })
    }
}

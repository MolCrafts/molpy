//! Ring piercing: bonds that pass through a ring, under periodic boundaries.
//!
//! A bond threaded through a ring is a topological defect of a built
//! structure. Bonds do not break during a minimization or a classical
//! dynamics run, so the bond can never be pulled back out: whatever placed it
//! there has to place it again. A bond *pierces* a ring when its segment
//! crosses the ring's mean plane at a point closer to the ring's centroid than
//! a fraction of the ring's mean radius, and it shares no atom with the ring.
//! Every vector is taken by minimum image, so a ring or a bond cut by a face
//! of the cell is whole.

use molrs::core::{NeighborQuery, SimBox};
use molrs::op::vec3::{dot, norm, sub};
use molrs::op::{F, Vec3, eigh_sym_3x3};
use ndarray::{Array1, Array2, ArrayView2};

use crate::compute::{Compute, ComputeError};

/// Which bonds pierce which rings, in one periodic snapshot.
///
/// Configuration: `radius_fraction`, the share of a ring's mean radius (the
/// mean distance of its atoms from its centroid) inside which a crossing of
/// its mean plane is a piercing; `1.0` takes the ring's own mean radius.
/// Data: positions, the cell, the bonds and the rings (each an atom index
/// list, three atoms or more — the small rings of a ring perception).
///
/// A ring's centroid is the mean of its atoms unwrapped about its first atom;
/// its plane is the least-squares plane through them (normal: the eigenvector
/// of the smallest eigenvalue of their scatter matrix). Candidate pairs come
/// from a neighbor search of the bond midpoints around the ring centroids,
/// whose radius is derived from the data: no bond whose midpoint lies further
/// than `radius_fraction · r_max + ℓ_max / 2` from a centroid can reach the
/// disc (`r_max` the largest ring radius, `ℓ_max` the longest bond).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingPiercing {
    radius_fraction: F,
}

impl RingPiercing {
    /// A crossing closer than `radius_fraction` of the ring's mean radius to
    /// its centroid pierces it.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `radius_fraction` is positive and
    /// finite.
    pub fn new(radius_fraction: F) -> Result<Self, ComputeError> {
        if !(radius_fraction.is_finite() && radius_fraction > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "RingPiercing::radius_fraction (positive, finite)",
                value: radius_fraction.to_string(),
            });
        }
        Ok(Self { radius_fraction })
    }

    /// The share of a ring's mean radius a crossing must fall inside.
    pub fn radius_fraction(&self) -> F {
        self.radius_fraction
    }
}

/// The data a [`RingPiercing`] analyses: one snapshot.
#[derive(Clone, Copy)]
pub struct RingPiercingInput<'a> {
    /// Atom positions, `N × 3` (Å), wrapped or not.
    pub positions: ArrayView2<'a, F>,
    /// The periodic cell.
    pub simbox: &'a SimBox,
    /// Bonds as `(i, j)` atom index pairs.
    pub bonds: &'a [[usize; 2]],
    /// Rings as atom index lists.
    pub rings: &'a [Vec<usize>],
}

/// Every piercing, as `(ring, bond)` rows sorted by ring then bond.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RingPiercingResult {
    /// The pierced ring of each piercing, an index into the input rings.
    pub ring_idx: Array1<usize>,
    /// The piercing bond of each piercing, an index into the input bonds.
    pub bond_idx: Array1<usize>,
}

impl RingPiercingResult {
    /// Number of piercings.
    pub fn n_piercings(&self) -> usize {
        self.ring_idx.len()
    }
}

/// A ring's centroid, unit plane normal and mean radius.
struct Disc {
    centre: Vec3,
    normal: Vec3,
    radius: F,
}

fn point(positions: ArrayView2<'_, F>, i: usize) -> Vec3 {
    [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]]
}

fn disc(positions: ArrayView2<'_, F>, simbox: &SimBox, ring: &[usize]) -> Disc {
    let mic = simbox.mic();
    let first = point(positions, ring[0]);
    let offsets: Vec<Vec3> = ring
        .iter()
        .map(|&i| mic.apply(sub(point(positions, i), first)))
        .collect();
    let n = offsets.len() as F;
    let mean: Vec3 = std::array::from_fn(|a| offsets.iter().map(|o| o[a]).sum::<F>() / n);
    let local: Vec<Vec3> = offsets.iter().map(|o| sub(*o, mean)).collect();
    let scatter = std::array::from_fn(|a| {
        std::array::from_fn(|b| local.iter().map(|l| l[a] * l[b]).sum::<F>())
    });
    // Eigenvalues descending, eigenvectors as columns: the last is the normal.
    let (_, vectors) = eigh_sym_3x3(&scatter);
    Disc {
        centre: std::array::from_fn(|a| first[a] + mean[a]),
        normal: [vectors[0][2], vectors[1][2], vectors[2][2]],
        radius: local.iter().map(|l| norm(*l)).sum::<F>() / n,
    }
}

impl Compute for RingPiercing {
    type Input<'a> = RingPiercingInput<'a>;
    type Output = RingPiercingResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for positions that are not `N × 3`;
    /// [`ComputeError::NonFinite`] for a non-finite position;
    /// [`ComputeError::OutOfRange`] for an atom index past `N` or a ring of
    /// fewer than three atoms.
    fn compute(&self, input: RingPiercingInput<'_>) -> Result<RingPiercingResult, ComputeError> {
        let positions = input.positions;
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("{:?}", positions.shape()),
            });
        }
        if let Some(i) = positions
            .rows()
            .into_iter()
            .position(|row| row.iter().any(|x| !x.is_finite()))
        {
            return Err(ComputeError::NonFinite {
                where_: "RingPiercing positions",
                index: i,
            });
        }
        for &[i, j] in input.bonds {
            if i >= n || j >= n {
                return Err(ComputeError::OutOfRange {
                    field: "RingPiercing bond index",
                    value: format!("({i}, {j}) with {n} atoms"),
                });
            }
        }
        for (r, ring) in input.rings.iter().enumerate() {
            if ring.len() < 3 || ring.iter().any(|&i| i >= n) {
                return Err(ComputeError::OutOfRange {
                    field: "RingPiercing ring (three atoms or more, each below N)",
                    value: format!("ring {r} = {ring:?} with {n} atoms"),
                });
            }
        }
        let empty = RingPiercingResult {
            ring_idx: Array1::zeros(0),
            bond_idx: Array1::zeros(0),
        };
        if input.rings.is_empty() || input.bonds.is_empty() {
            return Ok(empty);
        }

        let simbox = input.simbox;
        let mic = simbox.mic();
        let discs: Vec<Disc> = input
            .rings
            .iter()
            .map(|ring| disc(positions, simbox, ring))
            .collect();
        let vectors: Vec<Vec3> = input
            .bonds
            .iter()
            .map(|&[i, j]| mic.apply(sub(point(positions, j), point(positions, i))))
            .collect();
        let middles = Array2::from_shape_fn((input.bonds.len(), 3), |(b, a)| {
            positions[[input.bonds[b][0], a]] + 0.5 * vectors[b][a]
        });
        let centres = Array2::from_shape_fn((discs.len(), 3), |(r, a)| discs[r].centre[a]);
        let reach = self.radius_fraction * discs.iter().map(|d| d.radius).fold(0.0, F::max)
            + 0.5 * vectors.iter().map(|v| norm(*v)).fold(0.0, F::max);
        if reach <= 0.0 {
            return Ok(empty);
        }
        let near = NeighborQuery::new(simbox, middles.view(), reach)
            .map_err(|e| ComputeError::OutOfRange {
                field: "RingPiercing search radius",
                value: e.to_string(),
            })?
            .query(centres.view());

        let mut found: Vec<(usize, usize)> = Vec::new();
        for (&r, &b) in near.query_point_indices().iter().zip(near.point_indices()) {
            let (r, b) = (r as usize, b as usize);
            let [i, j] = input.bonds[b];
            if input.rings[r].contains(&i) || input.rings[r].contains(&j) {
                continue;
            }
            let disc = &discs[r];
            // The bond's start, in the image nearest the centroid.
            let start = mic.apply(sub(point(positions, i), disc.centre));
            let before = dot(start, disc.normal);
            let after = before + dot(vectors[b], disc.normal);
            if before * after >= 0.0 {
                continue;
            }
            let t = before / (before - after);
            let cross: Vec3 = std::array::from_fn(|a| start[a] + t * vectors[b][a]);
            let height = dot(cross, disc.normal);
            let radial: Vec3 = std::array::from_fn(|a| cross[a] - height * disc.normal[a]);
            if norm(radial) < self.radius_fraction * disc.radius {
                found.push((r, b));
            }
        }
        found.sort_unstable();
        Ok(RingPiercingResult {
            ring_idx: found.iter().map(|&(r, _)| r).collect(),
            bond_idx: found.iter().map(|&(_, b)| b).collect(),
        })
    }
}

crate::compute::spec::compute_spec! {
    RingPiercing as "RingPiercing";
    params |_s| {
        radius_fraction: Float;
    }
    inputs {
        positions: Positions;
        simbox: SimBox;
        bonds: IndexPairs;
        rings: IndexLists;
    }
}

#[cfg(test)]
mod tests;

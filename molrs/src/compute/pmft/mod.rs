//! Potentials of mean force and torque (PMFT): one engine, four coordinate
//! systems.
//!
//! A PMFT bins every neighbor pair's bond in a reference frame attached to
//! one of the two particles and reports the free-energy surface
//! `−ln(ρ / ρ_ref)` of the binned pair density. Ported from `freud.pmft`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/)).
//! What changes between the four is only the coordinate system — what a bond
//! *is* in that frame — so the analysis is one generic type, [`Pmft<C>`], over
//! a [`PairCoordinate`]:
//!
//! | Coordinate | Bins a bond as | Orientations |
//! |------------|----------------|--------------|
//! | [`R12`] | `(r, θ₁, θ₂)` — distance + the bond angle in each particle's frame (2-D) | planar angles, required |
//! | [`Xy`] | `(x, y)` — the bond in the query particle's frame (2-D) | planar angles, or the lab frame |
//! | [`Xyt`] | `(x, y, θ)` — the bond in the query frame + the relative orientation (2-D) | planar angles, required |
//! | [`Xyz`] | `(x, y, z)` — the bond in the query particle's frame (3-D) | quaternions, or the lab frame |
//!
//! `Pmft<C>` owns everything the four share, once: the box (the neighbor
//! table's own, [`Neighbors::simbox`]), the half-shell test, binning through
//! [`Axis`] (half-open bins), the reference density `ρ_ref`, the `−ln`
//! transform and the edges. Its `compute` takes one snapshot's neighbor table
//! and orientation array; many snapshots go through
//! [`Batch`](crate::compute::Batch). Orientations are what the caller reads
//! from its frame — planar angles (radians) or `(w, x, y, z)` quaternions.

mod r12;
mod xy;
mod xyt;
mod xyz;

pub use r12::R12;
pub use xy::Xy;
pub use xyt::Xyt;
pub use xyz::Xyz;

use std::f64::consts::TAU;

use molrs::core::{BoxKind, Neighbors, SimBox};
use molrs::op::{F, Vec3};
use ndarray::{Array, Array1, ArrayD, Dimension, IxDyn};

use crate::compute::{Axis, Compute, ComputeError, is_half_shell, require_disp};

/// A coordinate system a [`Pmft`] bins pair bonds in.
///
/// It names how one particle's orientation is written, how many coordinates a
/// bond has, the bond's coordinates given the two particles' orientations, and
/// the measure of the whole coordinate space the reference density spreads
/// over.
pub trait PairCoordinate: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// One particle's orientation: a planar angle (`F`, radians) or a unit
    /// quaternion (`Quat`).
    type Orientation: Copy + Send + Sync;

    /// One snapshot's orientations as the caller passes them: `ArrayView1`
    /// of `N` angles, or `ArrayView2` of `N × 4` quaternions.
    type Orientations<'a>: Copy + Send + Sync;

    /// The dimension of the binned grid (`Ix2`, `Ix3`).
    type Dim: Dimension;

    /// Number of coordinates per bond.
    const DIM: usize;

    /// A particle's orientation when none are given — the lab frame — or
    /// `None` when the coordinates are undefined without orientations.
    const LAB_FRAME: Option<Self::Orientation>;

    /// Validate one snapshot's orientations into one value per particle.
    ///
    /// # Errors
    /// Whatever the representation's guard raises (a non-finite angle, a
    /// zero quaternion, a wrong shape).
    fn orientations(
        orientations: Self::Orientations<'_>,
    ) -> Result<Vec<Self::Orientation>, ComputeError>;

    /// The `DIM` binned axes.
    fn axes(&self) -> &[Axis];

    /// The coordinates of the bond `disp = r_j − r_i` (Å) seen from `i`,
    /// whose orientation is `oi`, towards `j`, whose orientation is `oj`,
    /// written into `out` (`DIM` long). `false` when the bond has no
    /// coordinates (no direction where one is needed).
    fn coords(
        &self,
        disp: Vec3,
        oi: Self::Orientation,
        oj: Self::Orientation,
        out: &mut [F],
    ) -> bool;

    /// The measure of the coordinate space a uniform pair density spreads
    /// over: the box area (2-D) or volume (3-D), times the range of every
    /// angle coordinate.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a box the coordinates do not support.
    fn ref_measure(&self, simbox: &SimBox) -> Result<F, ComputeError>;

    /// The measure of the bin centred at `center`: the product of the axis
    /// widths, times the Jacobian of curvilinear coordinates.
    fn bin_measure(&self, center: &[F]) -> F {
        let _ = center;
        self.axes().iter().map(Axis::width).product()
    }
}

/// Potential of mean force and torque in the coordinate system `C`.
///
/// Configuration: the coordinate system and its axes (built by the
/// constructor of each instantiation, e.g. `Pmft::<Xy>::new(x_max, y_max,
/// n_x, n_y)`). Data: one snapshot's neighbor table — with the `disp` column
/// and the box it was searched in — and the particles' orientations.
///
/// Every pair of the table is binned at its coordinates seen from `i`; a
/// half-shell table also bins each pair seen from `j` (the reversed bond, the
/// two orientations swapped). The density of a bin is its count over its
/// measure; the reference density is the number of ordered pairs,
/// `N(N − 1)` for a self-query and `N_q · N_p` for a cross-query, over the
/// coordinate space's measure; the PMF is `−ln(ρ / ρ_ref)`, `+∞` for an
/// empty bin.
#[derive(Debug, Clone)]
pub struct Pmft<C> {
    coordinate: C,
}

impl<C: PairCoordinate> Pmft<C> {
    /// The coordinate system and its axes.
    pub fn coordinate(&self) -> &C {
        &self.coordinate
    }
}

/// The data a [`Pmft`] analyses.
pub struct PmftInput<'a, C: PairCoordinate> {
    /// One snapshot's neighbor table, with the `disp` column and its box.
    pub pairs: &'a Neighbors,
    /// The particles' orientations, one per particle — the same array
    /// indexes `i` and `j`, so a cross-query table needs equal point sets.
    /// `None` is the lab frame, for the coordinates that have one.
    pub orientations: Option<C::Orientations<'a>>,
}

impl<C: PairCoordinate> Clone for PmftInput<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: PairCoordinate> Copy for PmftInput<'_, C> {}

/// One snapshot's PMFT in the coordinate system `C`; every array is indexed
/// by bin, in the order of the coordinate's axes.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize),
    serde(bound(serialize = "C::Dim: serde::Serialize"))
)]
pub struct PmftResult<C: PairCoordinate> {
    /// Pair density per bin: count over bin measure.
    pub density: Array<F, C::Dim>,
    /// Pair count per bin.
    pub raw_counts: Array<u64, C::Dim>,
    /// `−ln(ρ / ρ_ref)` per bin; `+∞` where nothing was binned.
    pub pmf: Array<F, C::Dim>,
    /// Bin edges per axis (each `n_bins + 1` long).
    pub edges: Vec<Array1<F>>,
}

impl<C: PairCoordinate> Pmft<C> {
    /// Bin one bond seen from a particle oriented `oi` towards one oriented
    /// `oj`.
    fn push(
        &self,
        counts: &mut ArrayD<u64>,
        disp: Vec3,
        oi: C::Orientation,
        oj: C::Orientation,
        point: &mut [F],
        bin: &mut [usize],
    ) -> Result<(), ComputeError> {
        if !self.coordinate.coords(disp, oi, oj, point) {
            return Ok(());
        }
        for ((slot, axis), &x) in bin.iter_mut().zip(self.coordinate.axes()).zip(point.iter()) {
            match axis.index(x)? {
                Some(b) => *slot = b,
                None => return Ok(()),
            }
        }
        counts[IxDyn(bin)] += 1;
        Ok(())
    }

    /// The orientation of every particle, validated, or the lab frame's.
    fn orientations(&self, input: &PmftInput<'_, C>) -> Result<Vec<C::Orientation>, ComputeError> {
        let (n_q, n_p) = (input.pairs.n_query_points(), input.pairs.n_points());
        let Some(given) = input.orientations else {
            let lab = C::LAB_FRAME.ok_or_else(|| ComputeError::BadShape {
                expected: format!(
                    "orientations for {:?}: its coordinates need them",
                    self.coordinate
                ),
                got: "None".into(),
            })?;
            return Ok(vec![lab; n_q.max(n_p)]);
        };
        let orientations = C::orientations(given)?;
        for (n, what) in [
            (n_q, "orientations vs query points"),
            (n_p, "orientations vs points"),
        ] {
            if orientations.len() != n {
                return Err(ComputeError::DimensionMismatch {
                    expected: n,
                    got: orientations.len(),
                    what,
                });
            }
        }
        Ok(orientations)
    }
}

impl<C: PairCoordinate> Compute for Pmft<C> {
    type Input<'a> = PmftInput<'a, C>;
    type Output = PmftResult<C>;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp` or without a
    /// box, or missing orientations the coordinates need;
    /// [`ComputeError::DimensionMismatch`] for orientations not one per
    /// particle; [`ComputeError::OutOfRange`] for an unsupported box or a pair
    /// index past the point counts; [`ComputeError::NonFinite`] for a
    /// non-finite coordinate; and the orientation guards' errors.
    fn compute(&self, input: PmftInput<'_, C>) -> Result<PmftResult<C>, ComputeError> {
        let pairs = input.pairs;
        let simbox = pairs.simbox().ok_or_else(|| ComputeError::BadShape {
            expected: "a neighbor table that records the box it was searched in".into(),
            got: "a table without a box (from_pairs without with_simbox)".into(),
        })?;
        let disp = require_disp(pairs)?;
        let symmetric = is_half_shell(pairs.mode());
        let orientations = self.orientations(&input)?;

        let axes = self.coordinate.axes();
        let shape: Vec<usize> = axes.iter().map(Axis::n_bins).collect();
        let mut counts = ArrayD::<u64>::zeros(IxDyn(&shape));
        let mut point = vec![0.0; C::DIM];
        let mut bin = vec![0_usize; C::DIM];
        for (k, ((&i, &j), row)) in pairs
            .query_point_indices()
            .iter()
            .zip(pairs.point_indices())
            .zip(disp.outer_iter())
            .enumerate()
        {
            let (i, j) = (i as usize, j as usize);
            if i >= orientations.len() || j >= orientations.len() {
                return Err(ComputeError::OutOfRange {
                    field: "pair index",
                    value: format!("({i}, {j}) of pair {k} past the table's point counts"),
                });
            }
            let d = [row[0], row[1], row[2]];
            let (oi, oj) = (orientations[i], orientations[j]);
            self.push(&mut counts, d, oi, oj, &mut point, &mut bin)?;
            if symmetric {
                self.push(
                    &mut counts,
                    [-d[0], -d[1], -d[2]],
                    oj,
                    oi,
                    &mut point,
                    &mut bin,
                )?;
            }
        }

        let (n_q, n_p) = (pairs.n_query_points() as F, pairs.n_points() as F);
        let n_ordered_pairs = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let rho_ref = n_ordered_pairs / self.coordinate.ref_measure(simbox)?;

        let centers: Vec<Array1<F>> = axes.iter().map(Axis::centers).collect();
        let mut density = ArrayD::<F>::zeros(IxDyn(&shape));
        let mut pmf = ArrayD::<F>::from_elem(IxDyn(&shape), F::INFINITY);
        let mut center = vec![0.0; C::DIM];
        for (index, &count) in counts.indexed_iter() {
            for (a, c) in center.iter_mut().enumerate() {
                *c = centers[a][index[a]];
            }
            let rho = count as F / self.coordinate.bin_measure(&center);
            density[&index] = rho;
            if rho > 0.0 && rho_ref > 0.0 {
                pmf[&index] = -(rho / rho_ref).ln();
            }
        }

        Ok(PmftResult {
            density: fixed_dim(density)?,
            raw_counts: fixed_dim(counts)?,
            pmf: fixed_dim(pmf)?,
            edges: axes.iter().map(Axis::edges).collect(),
        })
    }
}

/// A dynamic-dimension grid as the coordinate's fixed dimension.
fn fixed_dim<T, D: Dimension>(grid: ArrayD<T>) -> Result<Array<T, D>, ComputeError> {
    let ndim = grid.ndim();
    grid.into_dimensionality::<D>()
        .map_err(|e| ComputeError::BadShape {
            expected: format!("a {}-D grid", D::NDIM.unwrap_or(ndim)),
            got: e.to_string(),
        })
}

/// The edge lengths of an orthorhombic box — the only shape the PMFT
/// reference measures are written for.
fn ortho_lengths(simbox: &SimBox) -> Result<[F; 3], ComputeError> {
    match simbox.kind() {
        BoxKind::Ortho { len, .. } => Ok([len[0], len[1], len[2]]),
        BoxKind::Triclinic => Err(ComputeError::OutOfRange {
            field: "Pmft simbox",
            value: "triclinic boxes are not supported".into(),
        }),
    }
}

/// An angle wrapped into `[0, 2π)` (radians), the range the angle axes span;
/// `2π` itself (a rounding of a tiny negative angle) is the angle 0.
fn wrap_angle(a: F) -> F {
    let w = a.rem_euclid(TAU);
    if w < TAU { w } else { 0.0 }
}

/// The `±max` axis of a Cartesian body-frame coordinate.
fn symmetric_axis(field: &'static str, max: F, n: usize) -> Result<Axis, ComputeError> {
    if !(max.is_finite() && max > 0.0) {
        return Err(ComputeError::OutOfRange {
            field,
            value: max.to_string(),
        });
    }
    Axis::new(-max, max, n)
}

/// A `[0, 2π)` angle axis.
fn angle_axis(n: usize) -> Result<Axis, ComputeError> {
    Axis::new(0.0, TAU, n)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use ndarray::array;

    /// A self-query table over `positions` in an open box of side `box_len`
    /// (`box_len × box_len × lz`).
    pub(crate) fn table(positions: &[[F; 3]], box_len: F, lz: F, cutoff: F) -> Neighbors {
        let simbox = SimBox::ortho(
            array![box_len, box_len, lz],
            array![0.0, 0.0, 0.0],
            [false; 3],
        )
        .unwrap();
        neighbors_of(positions, &simbox, cutoff)
    }

    #[test]
    fn wrap_angle_stays_below_two_pi() {
        assert_eq!(wrap_angle(-1e-300), 0.0);
        assert!(
            (wrap_angle(-std::f64::consts::FRAC_PI_2) - 1.5 * std::f64::consts::PI).abs() < 1e-15
        );
        assert_eq!(wrap_angle(TAU), 0.0);
    }

    /// A table nothing searched in a box carries none; the reference density
    /// needs one.
    #[test]
    fn a_table_without_a_box_is_bad_shape() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};
        let nl = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let err = Pmft::<Xy>::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }

    #[test]
    fn a_triclinic_box_is_out_of_range() {
        let simbox = SimBox::new(
            array![[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            array![0.0, 0.0, 0.0],
            [false; 3],
        )
        .unwrap();
        let nl = neighbors_of(&[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], &simbox, 1.5);
        let err = Pmft::<Xy>::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }
}

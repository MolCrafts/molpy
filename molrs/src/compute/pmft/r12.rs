//! 2-D `(r, θ₁, θ₂)` Pair Mode Fourier Transform.

// Parallel iteration over (frames, nlists, orientations) by index reads
// more clearly than nested zips.
#![allow(clippy::needless_range_loop)]

use crate::compute::ComputeResult;
use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array3;
use std::f64::consts::TAU;

use super::orientation::wrap_2pi;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_disp;

/// `PmftR12` analyzer.
///
/// Mirrors `freud.pmft.PmftR12`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftR12.cc)).
///
/// 2-D pair distribution function on polar bond coordinates plus the
/// relative angle each particle makes with the bond. For a neighbor pair
/// `(i, j)` with lab-frame bond vector `r_ij = r_j − r_i`:
///
/// - `r       = |r_ij|`                              (radial distance)
/// - `θ_lab   = atan2(r_ij.y, r_ij.x)`               (lab-frame bond angle)
/// - `t1      = wrap(θ_lab − orient_i,  [0, 2π))`    (bond in i's frame)
/// - `t2      = wrap(θ_lab + π − orient_j, [0, 2π))` (reverse bond in j's frame)
///
/// The triplet `(r, t1, t2)` is binned into a 3-D histogram on
/// `[0, r_max] × [0, 2π) × [0, 2π)`. PMF is `−ln(ρ / ρ_ref)` per cell.
///
/// The system is 2-D in the sense that `r_ij.z` is ignored and the
/// orientations are scalar angles (radians). Caller is responsible for
/// the planar configuration.
#[derive(Debug, Clone, Copy)]
pub struct PmftR12 {
    r_max: F,
    n_r: usize,
    n_t1: usize,
    n_t2: usize,
}

impl PmftR12 {
    /// Radial range `r_max` (Å); `n_r × n_t1 × n_t2` (r, θ₁, θ₂) bins.
    pub fn new(r_max: F, n_r: usize, n_t1: usize, n_t2: usize) -> Result<Self, ComputeError> {
        if r_max.is_nan() || r_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftR12::r_max",
                value: r_max.to_string(),
            });
        }
        if n_r == 0 || n_t1 == 0 || n_t2 == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftR12 bin counts",
                value: format!("n_r={n_r}, n_t1={n_t1}, n_t2={n_t2}"),
            });
        }
        Ok(Self {
            r_max,
            n_r,
            n_t1,
            n_t2,
        })
    }
}

/// Per-frame args for PmftR12: parallel `&[Neighbors]` and
/// per-particle 2-D orientations (radians).
pub struct PmftR12Args<'a> {
    pub nlists: &'a [Neighbors],
    pub orientations: &'a [Vec<F>],
}

impl PmftR12 {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
        orientations: &[F],
    ) -> Result<PmftR12Result, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PmftR12::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dr = self.r_max / self.n_r as F;
        let dt1 = TAU / self.n_t1 as F;
        let dt2 = TAU / self.n_t2 as F;
        let bin_vol = dr * dt1 * dt2;

        let mut counts = Array3::<u64>::zeros((self.n_r, self.n_t1, self.n_t2));
        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // The pair coordinate is built from the bond vector itself.
        let disp = require_disp(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        for k in 0..n_pairs {
            let vx = disp[[k, 0]];
            let vy = disp[[k, 1]];
            let r = (vx * vx + vy * vy).sqrt();
            if r >= self.r_max || r == 0.0 {
                continue;
            }
            let i = i_idx[k] as usize;
            let j = j_idx[k] as usize;
            if i >= orientations.len() || j >= orientations.len() {
                return Err(ComputeError::DimensionMismatch {
                    expected: i.max(j) + 1,
                    got: orientations.len(),
                    what: "PmftR12 orientations length",
                });
            }
            let theta_lab = vy.atan2(vx);
            let t1 = wrap_2pi(theta_lab - orientations[i]);
            let t2 = wrap_2pi(theta_lab + std::f64::consts::PI - orientations[j]);
            let br = ((r / dr) as usize).min(self.n_r - 1);
            let b1 = ((t1 / dt1) as usize).min(self.n_t1 - 1);
            let b2 = ((t2 / dt2) as usize).min(self.n_t2 - 1);
            counts[[br, b1, b2]] += 1;
            if symmetric {
                // Reverse bond on the j-side: (−vx, −vy) → same r,
                // angles swap (t1' = wrap(theta_lab+π − orient_j),
                // t2' = wrap(theta_lab − orient_i)).
                let t1p = wrap_2pi(theta_lab + std::f64::consts::PI - orientations[j]);
                let t2p = wrap_2pi(theta_lab - orientations[i]);
                let b1p = ((t1p / dt1) as usize).min(self.n_t1 - 1);
                let b2p = ((t2p / dt2) as usize).min(self.n_t2 - 1);
                counts[[br, b1p, b2p]] += 1;
            }
        }

        let n_q = nlist.n_query_points() as F;
        let n_p = nlist.n_points() as F;
        let n_pairs_total = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let area_box = lx * ly;
        let rho_ref = if area_box > 0.0 {
            // Per-pair density in 2-D, marginalised uniformly over the two
            // orientation axes.
            n_pairs_total / area_box / (TAU * TAU)
        } else {
            0.0
        };

        let mut density = Array3::<F>::zeros((self.n_r, self.n_t1, self.n_t2));
        let mut pmf = Array3::<F>::from_elem((self.n_r, self.n_t1, self.n_t2), F::INFINITY);
        for ir in 0..self.n_r {
            // 2-D radial shell: divide by 2π · r_c · dr  (cylindrical for 2D).
            let r_c = (ir as F + 0.5) * dr;
            let shell = 2.0 * std::f64::consts::PI * r_c * bin_vol / dr; // shell · dt1 · dt2
            for i1 in 0..self.n_t1 {
                for i2 in 0..self.n_t2 {
                    let rho = counts[[ir, i1, i2]] as F / shell.max(1e-12);
                    density[[ir, i1, i2]] = rho;
                    if rho > 0.0 && rho_ref > 0.0 {
                        pmf[[ir, i1, i2]] = -(rho / rho_ref).ln();
                    }
                }
            }
        }

        let r_edges: Vec<F> = (0..=self.n_r).map(|i| i as F * dr).collect();
        let t1_edges: Vec<F> = (0..=self.n_t1).map(|i| i as F * dt1).collect();
        let t2_edges: Vec<F> = (0..=self.n_t2).map(|i| i as F * dt2).collect();

        Ok(PmftR12Result {
            density,
            raw_counts: counts,
            pmf,
            r_edges,
            t1_edges,
            t2_edges,
        })
    }
}

impl LegacyCompute for PmftR12 {
    type Args<'a> = PmftR12Args<'a>;
    type Output = Vec<PmftR12Result>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: PmftR12Args<'a>,
    ) -> Result<Vec<PmftR12Result>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let nf = frames.len();
        if args.nlists.len() != nf || args.orientations.len() != nf {
            return Err(ComputeError::DimensionMismatch {
                expected: nf,
                got: args.nlists.len().min(args.orientations.len()),
                what: "PmftR12 frame-aligned inputs",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if nf >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return (0..nf)
                .into_par_iter()
                .map(|k| self.one_frame(frames[k], &args.nlists[k], &args.orientations[k]))
                .collect();
        }

        let mut out = Vec::with_capacity(nf);
        for k in 0..nf {
            out.push(self.one_frame(frames[k], &args.nlists[k], &args.orientations[k])?);
        }
        Ok(out)
    }
}

/// Per-frame PmftR12 result.
#[derive(Debug, Clone, Default)]
pub struct PmftR12Result {
    pub density: Array3<F>,
    pub raw_counts: Array3<u64>,
    pub pmf: Array3<F>,
    pub r_edges: Vec<F>,
    pub t1_edges: Vec<F>,
    pub t2_edges: Vec<F>,
}

impl ComputeResult for PmftR12Result {}

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
        frame.simbox = Some(
            SimBox::ortho(
                array![box_len, box_len, 1.0_f64],
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> Neighbors {
        nlist_from_frame(frame, cutoff)
    }

    #[test]
    fn two_particles_land_in_one_bin() {
        // Two particles along +x; both orientations = 0. Bond from i to j
        // has θ_lab = 0 → t1 = 0, t2 = π. The symmetric reverse contributes
        // a second count at t1 = π, t2 = 0.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = build_nlist(&frame, 1.5);
        let r = &PmftR12::new(2.0, 4, 4, 4)
            .unwrap()
            .compute(
                &[&frame],
                PmftR12Args {
                    nlists: std::slice::from_ref(&nl),
                    orientations: std::slice::from_ref(&vec![0.0_f64, 0.0]),
                },
            )
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn out_of_range_dropped() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 10.0);
        let nl = build_nlist(&frame, 5.0);
        let r = &PmftR12::new(1.0, 4, 4, 4)
            .unwrap()
            .compute(
                &[&frame],
                PmftR12Args {
                    nlists: std::slice::from_ref(&nl),
                    orientations: std::slice::from_ref(&vec![0.0_f64, 0.0]),
                },
            )
            .unwrap()[0];
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(PmftR12::new(0.0, 4, 4, 4).is_err());
        assert!(PmftR12::new(1.0, 0, 4, 4).is_err());
    }
}

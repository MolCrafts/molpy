//! 2-D `(x, y, θ)` Pair Mode Fourier Transform.

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

/// `PmftXyt` analyzer.
///
/// Mirrors `freud.pmft.PmftXyt`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXyt.cc)).
///
/// For a 2-D system of oriented particles: bin each neighbor pair `(i, j)`
/// by the bond vector **rotated into the query particle's local frame**
/// `(x, y)` and the **relative orientation angle** `θ = orient_j − orient_i`.
///
/// Algorithm:
/// 1. `r_local = R(−orient_i) · r_ij`           (rotate bond into i's frame)
/// 2. `x = r_local.x`, `y = r_local.y`
/// 3. `θ = wrap(orient_j − orient_i, [0, 2π))`
/// 4. Bin into `[−x_max, x_max] × [−y_max, y_max] × [0, 2π)`
///
/// PMF is `−ln(ρ / ρ_ref)`. The lab z-axis is ignored; orientations are
/// scalar 2-D angles in radians.
#[derive(Debug, Clone, Copy)]
pub struct PmftXyt {
    x_max: F,
    y_max: F,
    n_x: usize,
    n_y: usize,
    n_t: usize,
}

impl PmftXyt {
    /// Body-frame window `±x_max × ±y_max` (Å); `n_x × n_y × n_t` (x, y, θ) bins.
    pub fn new(
        x_max: F,
        y_max: F,
        n_x: usize,
        n_y: usize,
        n_t: usize,
    ) -> Result<Self, ComputeError> {
        if x_max.is_nan() || y_max.is_nan() || x_max <= 0.0 || y_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftXyt ranges",
                value: format!("x_max={x_max}, y_max={y_max}"),
            });
        }
        if n_x == 0 || n_y == 0 || n_t == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftXyt bin counts",
                value: format!("n_x={n_x}, n_y={n_y}, n_t={n_t}"),
            });
        }
        Ok(Self {
            x_max,
            y_max,
            n_x,
            n_y,
            n_t,
        })
    }
}

pub struct PmftXytArgs<'a> {
    pub nlists: &'a [Neighbors],
    pub orientations: &'a [Vec<F>],
}

impl PmftXyt {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
        orientations: &[F],
    ) -> Result<PmftXytResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PmftXyt::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dx = 2.0 * self.x_max / self.n_x as F;
        let dy = 2.0 * self.y_max / self.n_y as F;
        let dt = TAU / self.n_t as F;
        let bin_vol = dx * dy * dt;

        let mut counts = Array3::<u64>::zeros((self.n_x, self.n_y, self.n_t));
        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // The pair coordinate is built from the bond vector itself.
        let disp = require_disp(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        let push = |xl: F, yl: F, t: F, counts: &mut Array3<u64>| {
            if xl.abs() >= self.x_max || yl.abs() >= self.y_max {
                return;
            }
            let bx = (((xl + self.x_max) / dx) as usize).min(self.n_x - 1);
            let by = (((yl + self.y_max) / dy) as usize).min(self.n_y - 1);
            let bt = ((t / dt) as usize).min(self.n_t - 1);
            counts[[bx, by, bt]] += 1;
        };

        for k in 0..n_pairs {
            let vx = disp[[k, 0]];
            let vy = disp[[k, 1]];
            let i = i_idx[k] as usize;
            let j = j_idx[k] as usize;
            if i >= orientations.len() || j >= orientations.len() {
                return Err(ComputeError::DimensionMismatch {
                    expected: i.max(j) + 1,
                    got: orientations.len(),
                    what: "PmftXyt orientations length",
                });
            }
            // i-side: rotate bond into i's frame, t = orient_j − orient_i.
            let ci = orientations[i].cos();
            let si = orientations[i].sin();
            let xl_i = ci * vx + si * vy;
            let yl_i = -si * vx + ci * vy;
            let t_i = wrap_2pi(orientations[j] - orientations[i]);
            push(xl_i, yl_i, t_i, &mut counts);

            if symmetric {
                // j-side: reverse bond (−vx, −vy), rotated into j's frame.
                let cj = orientations[j].cos();
                let sj = orientations[j].sin();
                let xl_j = cj * (-vx) + sj * (-vy);
                let yl_j = -sj * (-vx) + cj * (-vy);
                let t_j = wrap_2pi(orientations[i] - orientations[j]);
                push(xl_j, yl_j, t_j, &mut counts);
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
            n_pairs_total / area_box / TAU
        } else {
            0.0
        };

        let mut density = Array3::<F>::zeros((self.n_x, self.n_y, self.n_t));
        let mut pmf = Array3::<F>::from_elem((self.n_x, self.n_y, self.n_t), F::INFINITY);
        for ix in 0..self.n_x {
            for iy in 0..self.n_y {
                for it in 0..self.n_t {
                    let rho = counts[[ix, iy, it]] as F / bin_vol;
                    density[[ix, iy, it]] = rho;
                    if rho > 0.0 && rho_ref > 0.0 {
                        pmf[[ix, iy, it]] = -(rho / rho_ref).ln();
                    }
                }
            }
        }

        let x_edges: Vec<F> = (0..=self.n_x).map(|i| -self.x_max + i as F * dx).collect();
        let y_edges: Vec<F> = (0..=self.n_y).map(|i| -self.y_max + i as F * dy).collect();
        let t_edges: Vec<F> = (0..=self.n_t).map(|i| i as F * dt).collect();

        Ok(PmftXytResult {
            density,
            raw_counts: counts,
            pmf,
            x_edges,
            y_edges,
            t_edges,
        })
    }
}

impl LegacyCompute for PmftXyt {
    type Args<'a> = PmftXytArgs<'a>;
    type Output = Vec<PmftXytResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: PmftXytArgs<'a>,
    ) -> Result<Vec<PmftXytResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let nf = frames.len();
        if args.nlists.len() != nf || args.orientations.len() != nf {
            return Err(ComputeError::DimensionMismatch {
                expected: nf,
                got: args.nlists.len().min(args.orientations.len()),
                what: "PmftXyt frame-aligned inputs",
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

/// Per-frame PmftXyt result.
#[derive(Debug, Clone, Default)]
pub struct PmftXytResult {
    pub density: Array3<F>,
    pub raw_counts: Array3<u64>,
    pub pmf: Array3<F>,
    pub x_edges: Vec<F>,
    pub y_edges: Vec<F>,
    pub t_edges: Vec<F>,
}

impl ComputeResult for PmftXytResult {}

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

    #[test]
    fn zero_orientations_match_lab_frame() {
        // Two particles along +x, both orientations = 0 → local-frame bond
        // is (+1, 0); t = 0. The reverse contributes (−1, 0, 0).
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &PmftXyt::new(2.0, 2.0, 8, 8, 8)
            .unwrap()
            .compute(
                &[&frame],
                PmftXytArgs {
                    nlists: std::slice::from_ref(&nl),
                    orientations: std::slice::from_ref(&vec![0.0_f64, 0.0]),
                },
            )
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn rotated_query_rotates_local_bond() {
        // Particle 0 oriented at +π/2 (looking in +y direction), particle 1
        // at lab (1, 0, 0). In particle 0's frame, the bond runs along its
        // -y axis (i.e. local (x, y) = (0, -1)).
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &PmftXyt::new(2.0, 2.0, 8, 8, 4)
            .unwrap()
            .compute(
                &[&frame],
                PmftXytArgs {
                    nlists: std::slice::from_ref(&nl),
                    orientations: std::slice::from_ref(&vec![std::f64::consts::FRAC_PI_2, 0.0]),
                },
            )
            .unwrap()[0];
        // The bond from particle 0 lands in a bin with y < 0 (negative side).
        // Find the bin holding particle-0's contribution.
        let mut found = false;
        for ix in 0..8 {
            for iy in 0..4 {
                // y < 0 → iy < 4
                for it in 0..4 {
                    if r.raw_counts[[ix, iy, it]] > 0 {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "rotated-frame bond should land in y < 0 bins");
    }

    #[test]
    fn invalid_args_error() {
        assert!(PmftXyt::new(0.0, 1.0, 4, 4, 4).is_err());
        assert!(PmftXyt::new(1.0, 1.0, 0, 4, 4).is_err());
        assert!(PmftXyt::new(1.0, 1.0, 4, 4, 0).is_err());
    }
}

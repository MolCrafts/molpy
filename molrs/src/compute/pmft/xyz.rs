//! 3-D `(x, y, z)` Pair Mode Fourier Transform.

use crate::compute::ComputeResult;
use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array3;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_disp;
use crate::op::{quat_conj, rotate_by_quat};

/// `PmftXyz` analyzer.
///
/// Mirrors `freud.pmft.PmftXyz`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXyz.cc)).
///
/// For every neighbor pair the bond vector `(dx, dy, dz)` is binned into
/// a 3-D histogram on
/// `[−x_max, x_max] × [−y_max, y_max] × [−z_max, z_max]`. PMF is
/// `−ln(ρ(x, y, z) / ρ_ref)` with `ρ_ref = N² / V_box`. Empty bins → `+∞`.
///
/// When per-particle orientations are supplied as quaternions via
/// [`PmftXyzArgs::query_orientations`], every bond is rotated into the
/// query particle's local frame before binning (matches freud's
/// `query_orientations` argument). Without orientations the analyzer
/// works in the lab frame.
#[derive(Debug, Clone, Copy)]
pub struct PmftXyz {
    x_max: F,
    y_max: F,
    z_max: F,
    n_x: usize,
    n_y: usize,
    n_z: usize,
}

impl PmftXyz {
    /// Body-frame window `±x_max × ±y_max × ±z_max` (Å); `n_x × n_y × n_z` bins.
    pub fn new(
        x_max: F,
        y_max: F,
        z_max: F,
        n_x: usize,
        n_y: usize,
        n_z: usize,
    ) -> Result<Self, ComputeError> {
        if x_max.is_nan()
            || y_max.is_nan()
            || z_max.is_nan()
            || x_max <= 0.0
            || y_max <= 0.0
            || z_max <= 0.0
        {
            return Err(ComputeError::OutOfRange {
                field: "PmftXyz ranges",
                value: format!("x_max={x_max}, y_max={y_max}, z_max={z_max}"),
            });
        }
        if n_x == 0 || n_y == 0 || n_z == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftXyz bin counts",
                value: format!("n_x={n_x}, n_y={n_y}, n_z={n_z}"),
            });
        }
        Ok(Self {
            x_max,
            y_max,
            z_max,
            n_x,
            n_y,
            n_z,
        })
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
        orientations: Option<&[[F; 4]]>,
    ) -> Result<PmftXyzResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly, lz) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PmftXyz::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dx = 2.0 * self.x_max / self.n_x as F;
        let dy = 2.0 * self.y_max / self.n_y as F;
        let dz = 2.0 * self.z_max / self.n_z as F;
        let bin_vol = dx * dy * dz;

        let mut counts = Array3::<u64>::zeros((self.n_x, self.n_y, self.n_z));
        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // The pair coordinate is built from the bond vector itself.
        let disp = require_disp(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        let push = |vx: F, vy: F, vz: F, counts: &mut Array3<u64>| {
            if vx.abs() >= self.x_max || vy.abs() >= self.y_max || vz.abs() >= self.z_max {
                return;
            }
            let bx = (((vx + self.x_max) / dx) as usize).min(self.n_x - 1);
            let by = (((vy + self.y_max) / dy) as usize).min(self.n_y - 1);
            let bz = (((vz + self.z_max) / dz) as usize).min(self.n_z - 1);
            counts[[bx, by, bz]] += 1;
        };

        for k in 0..n_pairs {
            let vx = disp[[k, 0]];
            let vy = disp[[k, 1]];
            let vz = disp[[k, 2]];
            let (xl_i, yl_i, zl_i) = match orientations {
                None => (vx, vy, vz),
                Some(o) => {
                    let i = i_idx[k] as usize;
                    if i >= o.len() {
                        return Err(ComputeError::DimensionMismatch {
                            expected: i + 1,
                            got: o.len(),
                            what: "PmftXyz orientations length",
                        });
                    }
                    // Rotate the lab-frame bond into i's local frame: use
                    // q⁻¹ · v · q (with q⁻¹ = q_conj for unit quaternions).
                    let r = rotate_by_quat(quat_conj(o[i]), [vx, vy, vz]);
                    (r[0], r[1], r[2])
                }
            };
            push(xl_i, yl_i, zl_i, &mut counts);
            if symmetric {
                let (xl_j, yl_j, zl_j) = match orientations {
                    None => (-vx, -vy, -vz),
                    Some(o) => {
                        let j = j_idx[k] as usize;
                        if j >= o.len() {
                            return Err(ComputeError::DimensionMismatch {
                                expected: j + 1,
                                got: o.len(),
                                what: "PmftXyz orientations length",
                            });
                        }
                        let r = rotate_by_quat(quat_conj(o[j]), [-vx, -vy, -vz]);
                        (r[0], r[1], r[2])
                    }
                };
                push(xl_j, yl_j, zl_j, &mut counts);
            }
        }

        let n_q = nlist.n_query_points() as F;
        let n_p = nlist.n_points() as F;
        let n_pairs_total = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let v_box = lx * ly * lz;
        let rho_ref = if v_box > 0.0 {
            n_pairs_total / v_box
        } else {
            0.0
        };

        let shape = (self.n_x, self.n_y, self.n_z);
        let mut density = Array3::<F>::zeros(shape);
        let mut pmf = Array3::<F>::from_elem(shape, F::INFINITY);
        for ix in 0..self.n_x {
            for iy in 0..self.n_y {
                for iz in 0..self.n_z {
                    let rho = counts[[ix, iy, iz]] as F / bin_vol;
                    density[[ix, iy, iz]] = rho;
                    if rho > 0.0 && rho_ref > 0.0 {
                        pmf[[ix, iy, iz]] = -(rho / rho_ref).ln();
                    }
                }
            }
        }

        let x_edges: Vec<F> = (0..=self.n_x).map(|i| -self.x_max + i as F * dx).collect();
        let y_edges: Vec<F> = (0..=self.n_y).map(|i| -self.y_max + i as F * dy).collect();
        let z_edges: Vec<F> = (0..=self.n_z).map(|i| -self.z_max + i as F * dz).collect();

        Ok(PmftXyzResult {
            density,
            raw_counts: counts,
            pmf,
            x_edges,
            y_edges,
            z_edges,
        })
    }
}

/// `Args` for [`PmftXyz`]. When `query_orientations` is `Some`, each entry
/// is a per-frame `Vec<[F;4]>` of unit quaternions `(w, x, y, z)` used to
/// rotate every bond into the query particle's local frame before binning.
pub struct PmftXyzArgs<'a> {
    pub nlists: &'a [Neighbors],
    pub query_orientations: Option<&'a [Vec<[F; 4]>]>,
}

impl LegacyCompute for PmftXyz {
    type Args<'a> = PmftXyzArgs<'a>;
    type Output = Vec<PmftXyzResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: PmftXyzArgs<'a>,
    ) -> Result<Vec<PmftXyzResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != args.nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: args.nlists.len(),
                what: "neighbor-list count",
            });
        }
        if let Some(o) = args.query_orientations
            && o.len() != frames.len()
        {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: o.len(),
                what: "PmftXyz orientations frame count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .enumerate()
                .map(|(k, f)| {
                    let nl = &args.nlists[k];
                    let o = args.query_orientations.map(|o| o[k].as_slice());
                    self.one_frame(*f, nl, o)
                })
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (k, f) in frames.iter().enumerate() {
            let nl = &args.nlists[k];
            let o = args.query_orientations.map(|o| o[k].as_slice());
            out.push(self.one_frame(*f, nl, o)?);
        }
        Ok(out)
    }
}

/// Per-frame PmftXyz result.
#[derive(Debug, Clone, Default)]
pub struct PmftXyzResult {
    pub density: Array3<F>,
    pub raw_counts: Array3<u64>,
    pub pmf: Array3<F>,
    pub x_edges: Vec<F>,
    pub y_edges: Vec<F>,
    pub z_edges: Vec<F>,
}

impl ComputeResult for PmftXyzResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
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
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    fn build_nlist(frame: &Frame, cutoff: F) -> Neighbors {
        nlist_from_frame(frame, cutoff)
    }

    #[test]
    fn antiparallel_bonds_populate_symmetric_bins() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [0.5, 0.4, 0.3]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PmftXyz::new(1.0, 1.0, 1.0, 8, 8, 8)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyzArgs {
                    nlists: &[nl],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn out_of_range_dropped() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 5.0);
        let r = &PmftXyz::new(1.0, 1.0, 1.0, 4, 4, 4)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyzArgs {
                    nlists: &[nl],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(PmftXyz::new(0.0, 1.0, 1.0, 4, 4, 4).is_err());
        assert!(PmftXyz::new(1.0, 1.0, 1.0, 0, 4, 4).is_err());
    }

    #[test]
    fn quaternion_orientations_rotate_bond_into_local_frame() {
        // Particle 0 at origin oriented at 90° about +z
        // (quaternion (cos 45°, 0, 0, sin 45°)). Particle 1 at lab
        // (1, 0, 0). In particle 0's local frame the bond runs along
        // its −y axis: rotating the +x lab vector by q⁻¹ gives (0, -1, 0).
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let q0 = [
            std::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_4.sin(),
        ];
        let identity = [1.0_f64, 0.0, 0.0, 0.0];
        let orient = vec![q0, identity];

        let r = &PmftXyz::new(1.5, 1.5, 1.5, 6, 6, 6)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyzArgs {
                    nlists: std::slice::from_ref(&nl),
                    query_orientations: Some(std::slice::from_ref(&orient)),
                },
            )
            .unwrap()[0];

        // Particle 0's contribution rotates into y < 0. Confirm the
        // overall y-binning shifted into the negative half.
        let mut found = false;
        for ix in 0..6 {
            for iy in 0..3 {
                for iz in 0..6 {
                    if r.raw_counts[[ix, iy, iz]] > 0 {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "quaternion-rotated bond should land in y < 0 bins");
    }
}

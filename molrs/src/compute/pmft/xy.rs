//! 2-D `(x, y)` Pair Mode Fourier Transform.

use crate::compute::ComputeResult;
use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array2;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_disp;

/// `PmftXy` analyzer.
///
/// Mirrors `freud.pmft.PmftXy`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXy.cc)).
///
/// For every neighbor pair the bond vector `(dx, dy)` (ignoring `dz`) is
/// binned into a 2-D histogram on `[−x_max, x_max] × [−y_max, y_max]`.
/// The result is the **PMF** estimate
///
/// ```text
///   PMF(x, y) = −ln( ρ(x, y) / ρ_ref )
/// ```
///
/// where `ρ_ref` is the bulk number density (`N / (Lx · Ly)` for an
/// orthorhombic 2-D box). Empty bins return `+∞` for the PMF; the raw
/// density and per-bin counts are also exposed.
///
/// When per-particle 2-D orientations are supplied via
/// [`PmftXyArgs::query_orientations`], every bond is rotated into the
/// query particle's local frame before binning (matches freud's
/// `query_orientations` argument). Without orientations the analyzer
/// works in the lab frame.
#[derive(Debug, Clone, Copy)]
pub struct PmftXy {
    x_max: F,
    y_max: F,
    n_x: usize,
    n_y: usize,
}

impl PmftXy {
    /// Body-frame window `±x_max × ±y_max` (Å); `n_x × n_y` bins.
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize) -> Result<Self, ComputeError> {
        if x_max.is_nan() || x_max <= 0.0 || y_max.is_nan() || y_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftXy ranges",
                value: format!("x_max={x_max}, y_max={y_max}"),
            });
        }
        if n_x == 0 || n_y == 0 {
            return Err(ComputeError::OutOfRange {
                field: "PmftXy bin counts",
                value: format!("n_x={n_x}, n_y={n_y}"),
            });
        }
        Ok(Self {
            x_max,
            y_max,
            n_x,
            n_y,
        })
    }

    pub fn x_max(&self) -> F {
        self.x_max
    }
    pub fn y_max(&self) -> F {
        self.y_max
    }
    pub fn n_x(&self) -> usize {
        self.n_x
    }
    pub fn n_y(&self) -> usize {
        self.n_y
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
        orientations: Option<&[F]>,
    ) -> Result<PmftXyResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "PmftXy::simbox",
                    value: "triclinic boxes not supported".into(),
                });
            }
        };

        let dx = 2.0 * self.x_max / self.n_x as F;
        let dy = 2.0 * self.y_max / self.n_y as F;
        let bin_area = dx * dy;

        let mut counts = Array2::<u64>::zeros((self.n_x, self.n_y));
        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // The pair coordinate is built from the bond vector itself.
        let disp = require_disp(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        let push = |dxp: F, dyp: F, counts: &mut Array2<u64>| {
            if dxp.abs() >= self.x_max || dyp.abs() >= self.y_max {
                return;
            }
            let bx = (((dxp + self.x_max) / dx) as usize).min(self.n_x - 1);
            let by = (((dyp + self.y_max) / dy) as usize).min(self.n_y - 1);
            counts[[bx, by]] += 1;
        };

        for k in 0..n_pairs {
            let vx = disp[[k, 0]];
            let vy = disp[[k, 1]];
            // i-side accumulation: rotate bond into i's local frame if
            // orientations are supplied.
            let (xl_i, yl_i) = match orientations {
                None => (vx, vy),
                Some(o) => {
                    let i = i_idx[k] as usize;
                    if i >= o.len() {
                        return Err(ComputeError::DimensionMismatch {
                            expected: i + 1,
                            got: o.len(),
                            what: "PmftXy orientations length",
                        });
                    }
                    let c = o[i].cos();
                    let s = o[i].sin();
                    (c * vx + s * vy, -s * vx + c * vy)
                }
            };
            push(xl_i, yl_i, &mut counts);
            if symmetric {
                let (xl_j, yl_j) = match orientations {
                    None => (-vx, -vy),
                    Some(o) => {
                        let j = j_idx[k] as usize;
                        if j >= o.len() {
                            return Err(ComputeError::DimensionMismatch {
                                expected: j + 1,
                                got: o.len(),
                                what: "PmftXy orientations length",
                            });
                        }
                        let c = o[j].cos();
                        let s = o[j].sin();
                        (c * -vx + s * -vy, -s * -vx + c * -vy)
                    }
                };
                push(xl_j, yl_j, &mut counts);
            }
        }

        // ρ_ref = N_query · (N_points − 1) / area  (per-pair density in the box).
        let n_q = nlist.n_query_points() as F;
        let n_p = nlist.n_points() as F;
        let n_pairs_total = if symmetric {
            n_p * (n_p - 1.0)
        } else {
            n_q * n_p
        };
        let area_box = lx * ly;
        let rho_ref = if area_box > 0.0 {
            n_pairs_total / area_box
        } else {
            0.0
        };

        let mut density = Array2::<F>::zeros((self.n_x, self.n_y));
        let mut pmf = Array2::<F>::from_elem((self.n_x, self.n_y), F::INFINITY);
        for ix in 0..self.n_x {
            for iy in 0..self.n_y {
                let rho = counts[[ix, iy]] as F / bin_area;
                density[[ix, iy]] = rho;
                if rho > 0.0 && rho_ref > 0.0 {
                    pmf[[ix, iy]] = -(rho / rho_ref).ln();
                }
            }
        }

        let x_edges: Vec<F> = (0..=self.n_x).map(|i| -self.x_max + i as F * dx).collect();
        let y_edges: Vec<F> = (0..=self.n_y).map(|i| -self.y_max + i as F * dy).collect();

        Ok(PmftXyResult {
            density,
            raw_counts: counts,
            pmf,
            x_edges,
            y_edges,
        })
    }
}

/// `Args` for [`PmftXy`]. When `query_orientations` is `Some`, each entry
/// is a per-frame `Vec<F>` of 2-D angles (radians) used to rotate every
/// bond into the query particle's local frame before binning.
pub struct PmftXyArgs<'a> {
    pub nlists: &'a [Neighbors],
    pub query_orientations: Option<&'a [Vec<F>]>,
}

impl<'a> From<&'a [Neighbors]> for PmftXyArgs<'a> {
    fn from(v: &'a [Neighbors]) -> Self {
        Self {
            nlists: v,
            query_orientations: None,
        }
    }
}

impl LegacyCompute for PmftXy {
    type Args<'a> = PmftXyArgs<'a>;
    type Output = Vec<PmftXyResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: PmftXyArgs<'a>,
    ) -> Result<Vec<PmftXyResult>, ComputeError> {
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
                what: "PmftXy orientations frame count",
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

/// Per-frame PmftXy result.
#[derive(Debug, Clone, Default)]
pub struct PmftXyResult {
    /// Number-density histogram, `(n_x, n_y)`, normalised to the bin area
    /// and total expected pair count.
    pub density: Array2<F>,
    /// Raw pair counts per bin.
    pub raw_counts: Array2<u64>,
    /// Potential of mean force, `−ln(density / ρ_ref)`. Empty bins → `+∞`.
    pub pmf: Array2<F>,
    /// `x` bin edges (length `n_x + 1`).
    pub x_edges: Vec<F>,
    /// `y` bin edges (length `n_y + 1`).
    pub y_edges: Vec<F>,
}

impl ComputeResult for PmftXyResult {}

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
    fn two_particles_land_in_symmetric_bins() {
        // Two particles separated by (1, 0). In a self-query NL the pair
        // is visited once but accumulated symmetrically, so we expect
        // count = 1 in (+1, 0) bin and count = 1 in (−1, 0) bin.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PmftXy::new(2.0, 2.0, 8, 8)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyArgs {
                    nlists: &[nl],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
        // Bin for (+1, 0): dx = 0.5 → bx = ((1 + 2)/0.5) = 6.
        let bx_pos = ((1.0_f64 + 2.0) / 0.5) as usize;
        let by_zero = ((0.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_pos, by_zero]], 1);
        let bx_neg = ((-1.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_neg, by_zero]], 1);
    }

    #[test]
    fn pmf_is_finite_only_in_occupied_bins() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let r = &PmftXy::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyArgs {
                    nlists: &[nl],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        let mut finite_count = 0;
        for v in r.pmf.iter() {
            if v.is_finite() {
                finite_count += 1;
            }
        }
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn out_of_range_pairs_dropped() {
        // Bond length 5 → outside the (x_max=2, y_max=2) box → no count.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 6.0);
        let r = &PmftXy::new(2.0, 2.0, 8, 8)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyArgs {
                    nlists: &[nl],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(PmftXy::new(0.0, 2.0, 4, 4).is_err());
        assert!(PmftXy::new(2.0, 2.0, 0, 4).is_err());
    }

    #[test]
    fn empty_input_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = PmftXy::new(2.0, 2.0, 4, 4)
            .unwrap()
            .compute(
                &frames,
                PmftXyArgs {
                    nlists: &[],
                    query_orientations: None,
                },
            )
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn orientations_rotate_bond_into_local_frame() {
        // Particle 0 at origin, oriented at +π/2 (looking in +y). Particle 1
        // at (1, 0, 0). In particle 0's local frame the bond is (0, -1, 0),
        // so it lands in a `y < 0` bin. Without orientations the same bond
        // would land in `x > 0`.
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let orient = vec![std::f64::consts::FRAC_PI_2, 0.0];

        let with_orient = &PmftXy::new(2.0, 2.0, 8, 8)
            .unwrap()
            .compute(
                &[&frame],
                PmftXyArgs {
                    nlists: std::slice::from_ref(&nl),
                    query_orientations: Some(std::slice::from_ref(&orient)),
                },
            )
            .unwrap()[0];

        // The i-side contribution from particle 0 must be in a y < 0 bin
        // when orientations are applied.
        let mut found_y_negative = false;
        for ix in 0..8 {
            for iy in 0..4 {
                if with_orient.raw_counts[[ix, iy]] > 0 {
                    found_y_negative = true;
                }
            }
        }
        assert!(
            found_y_negative,
            "rotated-frame bond should land in y < 0 bins"
        );
    }

    /// The `frames.len() >= PAR_THRESHOLD` rayon branch must match the
    /// serial path exactly, in frame order (enumerate + per-frame nlist).
    #[test]
    fn parallel_matches_serial() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, [false; 3]);
        let nl = build_nlist(&frame, 1.5);
        let p = PmftXy::new(2.0, 2.0, 8, 8).unwrap();
        let solo = p
            .compute(
                &[&frame],
                PmftXyArgs {
                    nlists: std::slice::from_ref(&nl),
                    query_orientations: None,
                },
            )
            .unwrap();
        let nls = vec![nl.clone(), nl];
        let par = p
            .compute(
                &[&frame, &frame],
                PmftXyArgs {
                    nlists: &nls,
                    query_orientations: None,
                },
            )
            .unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].raw_counts, solo[0].raw_counts);
        assert_eq!(par[1].raw_counts, solo[0].raw_counts);
    }
}

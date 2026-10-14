//! Per-particle local number density in a sphere of radius `r_max`.

use crate::compute::ComputeResult;
use molrs::core::FOUR_THIRDS_PI;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::compute::require_dist_sq;

/// Local-density calculator.
///
/// Mirrors `freud.density.LocalDensity`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/LocalDensity.cc)).
///
/// For each query point `i` the analyzer counts neighbors within `r_max`
/// and reports `density_i = count_i / (4/3 π r_max³)` (number per unit
/// volume).
///
/// freud also supports a `diameter` correction that subtracts the "hard-
/// sphere fraction" near the edge of the cutoff sphere — for two unit
/// spheres at distance `r` the overlap on the boundary linearly interpolates
/// a partial count between 0 and 1. We replicate that smoothing: each
/// neighbor `j` contributes
///
/// ```text
///   weight_j = clamp((r_max + diameter/2 − r_ij) / diameter, 0, 1)
/// ```
///
/// which collapses to the standard `1.0` count when `diameter = 0`. The
/// identical formula appears in `LocalDensity::compute` in freud.
#[derive(Debug, Clone, Copy)]
pub struct LocalDensity {
    r_max: F,
    diameter: F,
}

impl LocalDensity {
    /// New analyzer with a cutoff `r_max` and no smoothing (`diameter = 0`).
    pub fn new(r_max: F) -> Result<Self, ComputeError> {
        if r_max.is_nan() || r_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "LocalDensity::r_max",
                value: r_max.to_string(),
            });
        }
        Ok(Self {
            r_max,
            diameter: 0.0,
        })
    }

    /// Apply a "hard-sphere diameter" smoothing at the cutoff edge.
    pub fn with_diameter(mut self, diameter: F) -> Self {
        debug_assert!(diameter >= 0.0, "diameter must be ≥ 0");
        self.diameter = diameter;
        self
    }

    pub fn r_max(&self) -> F {
        self.r_max
    }
    pub fn diameter(&self) -> F {
        self.diameter
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<LocalDensityResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n_query = xs_p.slice().len();

        let i_idx = nlist.query_point_indices();
        let n_pairs = nlist.n_pairs();
        // Each neighbor's contribution is weighted by its distance.
        let dist_sq = require_dist_sq(nlist)?;

        let mut num = vec![0.0_f64; n_query];

        // freud convention: each neighbor pair is visited once from the query
        // side. For a self-query Neighbors we get i<j pairs only, so we
        // must add the symmetric contribution. For a cross-query nlist we
        // take i as the query point and j as the reference.
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        let half_diam = self.diameter * 0.5;
        let inv_diam = if self.diameter > 0.0 {
            1.0 / self.diameter
        } else {
            0.0
        };

        for k in 0..n_pairs {
            let r = dist_sq[k].sqrt();
            if r > self.r_max + half_diam {
                continue;
            }
            let weight = if self.diameter == 0.0 {
                if r <= self.r_max { 1.0 } else { 0.0 }
            } else {
                ((self.r_max + half_diam - r) * inv_diam).clamp(0.0, 1.0)
            };
            let i = i_idx[k] as usize;
            num[i] += weight;
            if symmetric {
                let j = nlist.point_indices()[k] as usize;
                num[j] += weight;
            }
        }

        let vol = FOUR_THIRDS_PI * self.r_max.powi(3);
        let inv_vol = if vol > 0.0 { 1.0 / vol } else { 0.0 };
        let density: Vec<F> = num.iter().map(|&c| c * inv_vol).collect();

        Ok(LocalDensityResult {
            n_neighbors: num,
            density,
        })
    }
}

impl LegacyCompute for LocalDensity {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<LocalDensityResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<LocalDensityResult>, ComputeError> {
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
        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame local-density result.
#[derive(Debug, Clone, Default)]
pub struct LocalDensityResult {
    /// Fractional (or integer when `diameter = 0`) neighbor count per particle.
    pub n_neighbors: Vec<F>,
    /// Number density per particle: `n_neighbors / (4/3 π r_max³)`.
    pub density: Vec<F>,
}

impl ComputeResult for LocalDensityResult {}

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

    #[test]
    fn isolated_particle_has_zero_density() {
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 20.0, [false; 3]);
        let nl = nlist_from_frame(&frame, 2.0);
        let r = &LocalDensity::new(2.0)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        assert_eq!(r.n_neighbors[0], 0.0);
        assert_eq!(r.density[0], 0.0);
    }

    #[test]
    fn pair_counts_symmetric_in_self_query() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0, [false; 3]);
        let nl = nlist_from_frame(&frame, 2.0);
        let r = &LocalDensity::new(2.0)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        // Each of the two particles has exactly 1 neighbor within 2.0.
        assert!((r.n_neighbors[0] - 1.0).abs() < 1e-12);
        assert!((r.n_neighbors[1] - 1.0).abs() < 1e-12);
        let v = FOUR_THIRDS_PI * 2.0_f64.powi(3);
        assert!((r.density[0] - 1.0 / v).abs() < 1e-12);
    }

    #[test]
    fn diameter_smoothing_clamps_at_edge() {
        // Two particles at exactly r = r_max + diameter/2 → weight ≈ 0.
        let r_max = 3.0;
        let diameter = 1.0;
        let r = r_max + diameter / 2.0 - 1e-9;
        let frame = frame_with(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]], 20.0, [false; 3]);
        let nl = nlist_from_frame(&frame, r + 1.0);
        let res = &LocalDensity::new(r_max)
            .unwrap()
            .with_diameter(diameter)
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        assert!(
            res.n_neighbors[0] < 1e-8,
            "weight at edge should be ≈ 0, got {}",
            res.n_neighbors[0]
        );

        // At r = r_max − diameter/2 → weight = 1 (fully inside).
        let r_in = r_max - diameter / 2.0;
        let frame2 = frame_with(&[[0.0, 0.0, 0.0], [r_in, 0.0, 0.0]], 20.0, [false; 3]);
        let nl2 = nlist_from_frame(&frame2, r + 1.0);
        let res2 = &LocalDensity::new(r_max)
            .unwrap()
            .with_diameter(diameter)
            .compute(&[&frame2], &[nl2])
            .unwrap()[0];
        assert!((res2.n_neighbors[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_r_max_errors() {
        assert!(LocalDensity::new(0.0).is_err());
        assert!(LocalDensity::new(-1.0).is_err());
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = LocalDensity::new(2.0)
            .unwrap()
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}

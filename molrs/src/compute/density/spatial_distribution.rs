//! Spatial Distribution Function (SDF): 3-D number density of a target species
//! in the body-fixed frame of a reference molecule, accumulated over a
//! trajectory.

use molrs::core::{Mic, SimBox};
use molrs::op::F;
use ndarray::{Array3, Array4, ArrayView2};

use crate::compute::{Accumulate, Compute, ComputeError, RegularGrid, Stream};
use crate::op::vec3::{normalize, sub};
use crate::op::{DEFAULT_GAP_TOL, Freedom, centroid, superpose};
use crate::op::{Rigid, transform_point};
use crate::op::{Vec3, to_vec3};

/// Spatial Distribution Function calculator.
///
/// Configuration: the body-frame [`RegularGrid`] the targets are binned on
/// (typically [`RegularGrid::centered`] about the reference centre of mass)
/// and an optional bulk density. Data, per frame
/// ([`SpatialDistributionFrame`]): the positions, an optional box, the rigid
/// reference selection with its canonical `template`, the target selection,
/// and optional orientation pairs.
///
/// Unlike [`GaussianDensity`](super::GaussianDensity), which is lab-frame,
/// the SDF first superimposes each frame's reference atoms onto the template
/// (Horn superposition, [`crate::op::superpose`]), applies the same rigid
/// rotation to the surrounding target atoms — after minimum-image unwrapping
/// relative to the reference centre of mass — and only then bins them into
/// the grid. The result is the familiar "density cloud" of *where, relative
/// to this molecule*, a second species sits.
///
/// # reference implementation provenance
///
/// - The reference-frame fix on a 3-atom reference set mirrors the reference
///   implementation's global `g_iFixMol` / `g_iFixAtom[0..2]` alignment (the
///   SDF "fix" in `engine.cpp`), here realized as a least-squares quaternion
///   superposition so the whole reference set — not just three atoms — is used.
/// - 3-D voxel accumulation follows the reference implementation's
///   `C3DF::AddToBin` (`src/3df.cpp`): nearest-voxel deposition, out-of-grid
///   samples skipped.
/// - The per-voxel mean-orientation field is `Σ value / count` exactly as
///   `CSDFMap::Finish` averages its value bins (`src/sdfmap.cpp:418-424`).
///
/// Density normalization: `ρ(voxel) = counts / (n_frames · ΔV)` in Å⁻³; the
/// optional bulk-normalized `g_SDF = ρ / ρ_bulk` is the SDF analogue of RDF's
/// `g(r)` and tends to 1 far from the reference for an unstructured target.
///
/// A [`Stream`]: `compute` is its accumulator folded over the frames.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialDistribution {
    grid: RegularGrid<3>,
    /// Optional bulk number density (Å⁻³) for the `g_SDF` normalization.
    bulk_density: Option<F>,
}

/// One frame of a [`SpatialDistribution`].
#[derive(Debug, Clone, Copy)]
pub struct SpatialDistributionFrame<'a> {
    /// Positions, `n_atoms × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box for minimum-image displacements; `None` for open boundaries.
    pub simbox: Option<&'a SimBox>,
    /// Indices of the rigid reference atoms (at least 3).
    pub reference: &'a [usize],
    /// Canonical body-frame geometry of the reference, one row per reference
    /// atom (`reference.len() × 3`).
    pub template: ArrayView2<'a, F>,
    /// Indices of the target atoms binned on the grid.
    pub target: &'a [usize],
    /// Optional `(head, tail)` atom-index pairs, one per target in the same
    /// order: the per-voxel mean body-frame direction of the unit vector
    /// `head − tail` (minimum image) is reported.
    pub orientation: Option<&'a [(usize, usize)]>,
}

impl SpatialDistribution {
    /// An SDF binned on the body-frame `grid`.
    pub fn new(grid: RegularGrid<3>) -> Self {
        Self {
            grid,
            bulk_density: None,
        }
    }

    /// Set the bulk number density (Å⁻³) used to form `g_SDF = ρ / ρ_bulk`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite density
    /// (it used to be ignored silently).
    pub fn with_bulk_density(mut self, rho: F) -> Result<Self, ComputeError> {
        if !(rho.is_finite() && rho > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::bulk_density",
                value: rho.to_string(),
            });
        }
        self.bulk_density = Some(rho);
        Ok(self)
    }

    /// The body-frame grid.
    pub fn grid(&self) -> &RegularGrid<3> {
        &self.grid
    }
}

impl Compute for SpatialDistribution {
    type Input<'a> = &'a [SpatialDistributionFrame<'a>];
    type Output = SpatialDistributionResult;

    fn compute(&self, frames: Self::Input<'_>) -> Result<SpatialDistributionResult, ComputeError> {
        self.accumulator().fold(frames.iter().copied())
    }
}

impl Stream for SpatialDistribution {
    type Accumulator = SpatialDistributionAccumulator;

    fn accumulator(&self) -> SpatialDistributionAccumulator {
        SpatialDistributionAccumulator {
            counts: Array3::zeros(self.grid.dims()),
            orientation: None,
            config: self.clone(),
            n_frames: 0,
        }
    }
}

/// The frame-by-frame form of a [`SpatialDistribution`].
#[derive(Debug, Clone)]
pub struct SpatialDistributionAccumulator {
    config: SpatialDistribution,
    counts: Array3<F>,
    /// Per-voxel orientation sums and counts, present once the first frame
    /// brought orientation pairs.
    orientation: Option<(Array4<F>, Array3<F>)>,
    n_frames: usize,
}

impl SpatialDistributionAccumulator {
    /// Validate one frame's selections against its positions.
    fn check(&self, frame: &SpatialDistributionFrame<'_>) -> Result<(), ComputeError> {
        let n_atoms = frame.positions.nrows();
        if frame.positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("N × {}", frame.positions.ncols()),
            });
        }
        if frame.reference.len() < 3 {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::reference",
                value: frame.reference.len().to_string(),
            });
        }
        if frame.template.nrows() != frame.reference.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frame.reference.len(),
                got: frame.template.nrows(),
                what: "SDF template rows",
            });
        }
        if frame.template.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: frame.template.ncols(),
                what: "SDF template columns",
            });
        }
        if let Some(pairs) = frame.orientation
            && pairs.len() != frame.target.len()
        {
            return Err(ComputeError::DimensionMismatch {
                expected: frame.target.len(),
                got: pairs.len(),
                what: "SDF orientation pairs",
            });
        }
        let latched = self.n_frames > 0;
        if latched && self.orientation.is_some() != frame.orientation.is_some() {
            return Err(ComputeError::BadShape {
                expected: format!(
                    "orientation pairs {} (as frame 0)",
                    if self.orientation.is_some() {
                        "on every frame"
                    } else {
                        "on no frame"
                    }
                ),
                got: format!("frame {} differs", self.n_frames),
            });
        }
        let indices = frame.reference.iter().chain(frame.target).chain(
            frame
                .orientation
                .into_iter()
                .flatten()
                .flat_map(|(h, t)| [h, t]),
        );
        for &i in indices {
            if i >= n_atoms {
                return Err(ComputeError::OutOfRange {
                    field: "SpatialDistribution atom index",
                    value: format!("{i} (positions hold {n_atoms} atoms)"),
                });
            }
        }
        Ok(())
    }
}

impl Accumulate for SpatialDistributionAccumulator {
    type Item<'a> = SpatialDistributionFrame<'a>;
    type Output = SpatialDistributionResult;

    /// Bin one frame's targets in the reference's body frame.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for fewer than 3 reference atoms, an atom
    /// index outside the positions (it used to panic), or a superposition
    /// that leaves the orientation undetermined (e.g. a collinear template);
    /// [`ComputeError::DimensionMismatch`] for a template or orientation
    /// array of the wrong size; [`ComputeError::BadShape`] when orientation
    /// pairs come and go between frames; [`ComputeError::NonFinite`] for a
    /// non-finite body-frame coordinate. A rejected frame bins nothing.
    fn push(&mut self, frame: SpatialDistributionFrame<'_>) -> Result<(), ComputeError> {
        self.check(&frame)?;
        let mic = frame.simbox.map_or(Mic::Free, SimBox::mic);
        let row = |i: usize| -> Vec3 {
            [
                frame.positions[[i, 0]],
                frame.positions[[i, 1]],
                frame.positions[[i, 2]],
            ]
        };

        // Align this frame's reference set onto the template.
        // `superpose(frame, template)` gives R with R·(frameᵢ − c_f) ≈
        // (templateᵢ − c_t): the lab → body-frame rotation.
        let template: Vec<Vec3> = frame.template.rows().into_iter().map(to_vec3).collect();
        let ref_coords: Vec<Vec3> = frame.reference.iter().map(|&i| row(i)).collect();
        let weights = vec![1.0; ref_coords.len()];
        let fit = superpose(&ref_coords, &template, &weights, DEFAULT_GAP_TOL).map_err(|e| {
            ComputeError::OutOfRange {
                field: "SpatialDistribution::reference",
                value: e.to_string(),
            }
        })?;
        if fit.freedom != Freedom::Unique {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::reference",
                value: format!(
                    "superposition leaves the orientation undetermined ({:?}); \
                     the template must span a plane",
                    fit.freedom
                ),
            });
        }
        let body_rotation = Rigid {
            rotation: fit.rigid.rotation,
            translation: [0.0; 3],
        };
        // Reference COM (lab frame) = centroid of the reference atoms.
        let com = centroid(&ref_coords, &weights).ok_or(ComputeError::OutOfRange {
            field: "SpatialDistribution::reference",
            value: "empty reference selection".into(),
        })?;

        // Locate every target before touching the grids, so a rejected
        // frame leaves them alone.
        let mut hits: Vec<([usize; 3], Option<Vec3>)> = Vec::with_capacity(frame.target.len());
        for (t, &ai) in frame.target.iter().enumerate() {
            // Minimum-image vector COM → target, then rotate into body frame.
            let disp = mic.apply(sub(row(ai), com));
            let body = transform_point(&body_rotation, disp);
            let Some(voxel) = self.config.grid.index_of(body)? else {
                continue;
            };
            let direction = frame.orientation.and_then(|pairs| {
                let (head, tail) = pairs[t];
                normalize(mic.apply(sub(row(head), row(tail))))
                    .map(|u| transform_point(&body_rotation, u))
            });
            hits.push((voxel, direction));
        }

        if self.n_frames == 0 && frame.orientation.is_some() {
            let [nx, ny, nz] = self.config.grid.dims();
            self.orientation = Some((Array4::zeros((nx, ny, nz, 3)), Array3::zeros((nx, ny, nz))));
        }
        for ([ix, iy, iz], direction) in hits {
            self.counts[[ix, iy, iz]] += 1.0;
            if let (Some((sum, count)), Some(bu)) = (self.orientation.as_mut(), direction) {
                for d in 0..3 {
                    sum[[ix, iy, iz, d]] += bu[d];
                }
                count[[ix, iy, iz]] += 1.0;
            }
        }
        self.n_frames += 1;
        Ok(())
    }

    /// # Errors
    /// [`ComputeError::EmptyInput`] before any frame.
    fn finish(&self) -> Result<SpatialDistributionResult, ComputeError> {
        if self.n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let voxel_volume = self.config.grid.voxel_volume();
        let denom = self.n_frames as F * voxel_volume;
        let density = self.counts.mapv(|c| c / denom);
        let g_sdf = self
            .config
            .bulk_density
            .map(|rho| density.mapv(|d| d / rho));
        let orientation = self.orientation.as_ref().map(|(sum, count)| {
            let mut mean = Array4::<F>::zeros(sum.raw_dim());
            for ((ix, iy, iz), &c) in count.indexed_iter() {
                if c > 0.0 {
                    for d in 0..3 {
                        mean[[ix, iy, iz, d]] = sum[[ix, iy, iz, d]] / c;
                    }
                }
            }
            mean
        });
        Ok(SpatialDistributionResult {
            counts: self.counts.clone(),
            density,
            g_sdf,
            orientation,
            grid: self.config.grid.clone(),
            voxel_volume,
            n_frames: self.n_frames,
        })
    }
}

/// Accumulated SDF grid plus optional bulk-normalized density and orientation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialDistributionResult {
    /// Raw voxel counts, summed across frames.
    pub counts: Array3<F>,
    /// Number density per voxel (Å⁻³): `counts / (n_frames · ΔV)`.
    pub density: Array3<F>,
    /// Bulk-normalized `g_SDF = density / ρ_bulk`, if a bulk density was set.
    pub g_sdf: Option<Array3<F>>,
    /// Per-voxel mean body-frame orientation vector `(nx, ny, nz, 3)`, if
    /// orientation pairs were supplied. Zero on empty voxels.
    pub orientation: Option<Array4<F>>,
    /// The body-frame grid.
    pub grid: RegularGrid<3>,
    /// Single-voxel volume, Å³.
    pub voxel_volume: F,
    /// Number of frames accumulated.
    pub n_frames: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// 4 × 4 × 4 grid of unit voxels spanning `[−2, 2]³` about the COM.
    fn unit_grid() -> RegularGrid<3> {
        RegularGrid::centered([4.0, 4.0, 4.0], [4, 4, 4]).unwrap()
    }

    fn frame<'a>(
        positions: &'a Array2<F>,
        reference: &'a [usize],
        template: &'a Array2<F>,
        target: &'a [usize],
    ) -> SpatialDistributionFrame<'a> {
        SpatialDistributionFrame {
            positions: positions.view(),
            simbox: None,
            reference,
            template: template.view(),
            target,
            orientation: None,
        }
    }

    /// The rotated-molecule fixture: template in the xy-plane, lab copy
    /// rotated by +90° about z and shifted to (5, 5, 5), one target.
    fn rotated() -> (Array2<F>, Array2<F>) {
        let template = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, -2.0, 0.0]];
        let positions = array![
            [5.0, 6.0, 5.0],
            [3.0, 5.0, 5.0],
            [7.0, 4.0, 5.0],
            [4.5, 6.5, 5.5]
        ];
        (positions, template)
    }

    #[test]
    fn two_reference_atoms_are_refused() {
        // Two points fix only a line: the spin about it is undetermined.
        let template = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let positions = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let err = SpatialDistribution::new(unit_grid())
            .compute(&[frame(&positions, &[0, 1], &template, &[])])
            .unwrap_err();
        assert!(
            matches!(
                err,
                ComputeError::OutOfRange {
                    field: "SpatialDistribution::reference",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn collinear_reference_is_refused_per_frame() {
        // Three collinear points: superposition leaves a free spin about the
        // line (Freedom::Spin), so the body frame is undetermined.
        let template = array![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let positions = array![
            [4.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 5.5, 5.0]
        ];
        let err = SpatialDistribution::new(unit_grid())
            .compute(&[frame(&positions, &[0, 1, 2], &template, &[3])])
            .unwrap_err();
        assert!(
            matches!(
                err,
                ComputeError::OutOfRange {
                    field: "SpatialDistribution::reference",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn template_without_three_columns_is_dimension_mismatch() {
        let template = array![[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]];
        let positions = Array2::<F>::zeros((3, 3));
        let err = SpatialDistribution::new(unit_grid())
            .compute(&[frame(&positions, &[0, 1, 2], &template, &[])])
            .unwrap_err();
        assert!(
            matches!(
                err,
                ComputeError::DimensionMismatch {
                    expected: 3,
                    got: 2,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn target_lands_in_body_frame_voxel_after_z_rotation() {
        // Template (body frame), centroid at the origin, spans the xy-plane.
        // Lab copy = Rz(+90°)·template + (5, 5, 5); Rz(+90°): (x, y) → (−y, x).
        //   (1, 0, 0) → (5, 6, 5); (0, 2, 0) → (3, 5, 5); (−1, −2, 0) → (7, 4, 5).
        // COM = (5, 5, 5). Lab → body rotation is Rz(−90°): (x, y) → (y, −x).
        // Target displacement (−0.5, 1.5, 0.5) → body (1.5, 0.5, 0.5).
        // Shifted by +2 → (3.5, 2.5, 2.5) → voxel (3, 2, 2).
        // (Identity would give voxel (1, 3, 2); the wrong sense Rz(+90°), (0, 1, 2).)
        let (positions, template) = rotated();
        let res = SpatialDistribution::new(unit_grid())
            .compute(&[frame(&positions, &[0, 1, 2], &template, &[3])])
            .unwrap();
        assert_eq!(res.counts[[3, 2, 2]], 1.0);
        assert_eq!(res.counts.sum(), 1.0);
    }

    /// An index past the positions used to panic; it is an error.
    #[test]
    fn unchecked_index_is_out_of_range() {
        let (positions, template) = rotated();
        let sdf = SpatialDistribution::new(unit_grid());
        for (reference, target) in [([0usize, 1, 9], [3usize]), ([0, 1, 2], [4])] {
            assert!(matches!(
                sdf.compute(&[frame(&positions, &reference, &template, &target)]),
                Err(ComputeError::OutOfRange {
                    field: "SpatialDistribution atom index",
                    ..
                })
            ));
        }
        let pairs = [(3usize, 7usize)];
        let mut with_pairs = frame(&positions, &[0, 1, 2], &template, &[3]);
        with_pairs.orientation = Some(&pairs);
        assert!(matches!(
            sdf.compute(&[with_pairs]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// Orientation pairs are `(head, tail)`, axis = head − tail; the mean
    /// body-frame direction lands in the target's voxel.
    #[test]
    fn orientation_is_head_minus_tail_in_the_body_frame() {
        let (positions, template) = rotated();
        // Head = target (3), tail = reference atom 1 at (3, 5, 5):
        // head − tail = (1.5, 1.5, 0.5); body (Rz(−90°)) = (1.5, −1.5, 0.5).
        let pairs = [(3usize, 1usize)];
        let mut f = frame(&positions, &[0, 1, 2], &template, &[3]);
        f.orientation = Some(&pairs);
        let res = SpatialDistribution::new(unit_grid()).compute(&[f]).unwrap();
        let o = res.orientation.unwrap();
        let n = (1.5_f64 * 1.5 * 2.0 + 0.25).sqrt();
        let want = [1.5 / n, -1.5 / n, 0.5 / n];
        for d in 0..3 {
            assert!((o[[3, 2, 2, d]] - want[d]).abs() < 1e-12, "component {d}");
        }
    }

    #[test]
    fn density_and_bulk_normalization() {
        let (positions, template) = rotated();
        let sdf = SpatialDistribution::new(unit_grid())
            .with_bulk_density(0.5)
            .unwrap();
        let f = frame(&positions, &[0, 1, 2], &template, &[3]);
        let res = sdf.compute(&[f, f]).unwrap();
        assert_eq!(res.n_frames, 2);
        assert_eq!(res.counts[[3, 2, 2]], 2.0);
        assert_eq!(res.density[[3, 2, 2]], 1.0);
        assert_eq!(res.g_sdf.unwrap()[[3, 2, 2]], 2.0);
        assert!(matches!(
            SpatialDistribution::new(unit_grid()).with_bulk_density(0.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}

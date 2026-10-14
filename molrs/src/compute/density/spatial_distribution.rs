//! Spatial Distribution Function (SDF): 3-D number density of a target species
//! in the body-fixed frame of a reference molecule, accumulated over a
//! trajectory.

use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::{Array2, Array3, Array4};

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::{normalize, sub};
use crate::op::{DEFAULT_GAP_TOL, Freedom, centroid, superpose};
use crate::op::{Rigid, transform_point};
use crate::op::{Vec3, to_vec3};
use molrs::core::{Mic, SimBox};

/// A regular axis-aligned voxel grid centred on the reference COM.
///
/// `extent[d]` is the **full** side length (Å) along axis `d`; the grid spans
/// `[−extent/2, +extent/2]` about the COM with `n[d]` voxels per axis.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// Voxels per axis.
    pub n: [usize; 3],
    /// Full side length per axis, Å.
    pub extent: [F; 3],
}

impl GridSpec {
    /// Voxel side lengths (Å) per axis.
    fn voxel_size(&self) -> [F; 3] {
        [
            self.extent[0] / self.n[0] as F,
            self.extent[1] / self.n[1] as F,
            self.extent[2] / self.n[2] as F,
        ]
    }

    /// Single voxel volume, Å³.
    fn voxel_volume(&self) -> F {
        let v = self.voxel_size();
        v[0] * v[1] * v[2]
    }

    /// Map a body-frame coordinate (relative to the COM) to a voxel index,
    /// or `None` if it falls outside the grid.
    fn index(&self, r: [F; 3]) -> Option<[usize; 3]> {
        let vs = self.voxel_size();
        let mut idx = [0usize; 3];
        for d in 0..3 {
            let shifted = r[d] + 0.5 * self.extent[d];
            if shifted < 0.0 {
                return None;
            }
            let i = (shifted / vs[d]).floor() as isize;
            if i < 0 || i >= self.n[d] as isize {
                return None;
            }
            idx[d] = i as usize;
        }
        Some(idx)
    }
}

/// Spatial Distribution Function calculator.
///
/// All inputs live on `&self` (the [`LegacyCompute::Args`] are `()`): a rigid
/// reference atom selection and its canonical `template`, a target selection,
/// the grid, and optional bulk density / orientation vectors.
///
/// Unlike [`GaussianDensity`](super::gaussian_density::GaussianDensity), which
/// is lab-frame, the SDF first superimposes each frame's reference atoms onto a
/// canonical template (Horn superposition, [`crate::op::superpose`]), applies the same
/// rigid rotation to the surrounding target atoms — after minimum-image
/// unwrapping relative to the reference centre of mass — and only then bins
/// them into a grid centred on that COM. The result is the familiar "density
/// cloud" of *where, relative to this molecule*, a second species sits.
///
/// # reference implementation provenance
///
/// - The reference-frame fix on a 3-atom reference set mirrors the reference implementation's global
///   `g_iFixMol` / `g_iFixAtom[0..2]` alignment (the SDF "fix" in `engine.cpp`),
///   here realized as a least-squares quaternion superposition so the whole
///   reference set — not just three atoms — is used and no BLAS is needed.
/// - 3-D voxel accumulation follows the reference implementation's `C3DF::AddToBin` (`src/3df.cpp`):
///   nearest-voxel deposition, out-of-grid samples skipped.
/// - The per-voxel mean-orientation field is `Σ value / count` exactly as
///   `CSDFMap::Finish` averages its value bins (`src/sdfmap.cpp:418-424`).
///
/// Density normalization: `ρ(voxel) = counts / (n_frames · ΔV)` in Å⁻³; the
/// optional bulk-normalized `g_SDF = ρ / ρ_bulk` is the SDF analogue of RDF's
/// `g(r)` and tends to 1 far from the reference for an unstructured target.
#[derive(Debug, Clone)]
pub struct SpatialDistribution {
    reference: Vec<usize>,
    /// Canonical reference geometry, one row per reference atom.
    template: Vec<Vec3>,
    target: Vec<usize>,
    grid: GridSpec,
    /// Optional `(tail, head)` atom-index pairs (parallel to `target`) defining
    /// a per-target vector whose body-frame mean is mapped per voxel.
    orientation: Option<Vec<(usize, usize)>>,
    /// Optional bulk number density (Å⁻³) for the `g_SDF` normalization.
    bulk_density: Option<F>,
}

impl SpatialDistribution {
    /// New SDF over a `reference` selection (with canonical `template`,
    /// `reference.len() × 3`) and a `target` selection, binned into `grid`.
    ///
    /// # Errors
    ///
    /// [`ComputeError::DimensionMismatch`] if `template` rows ≠ `reference`
    /// length or `template` does not have 3 columns;
    /// [`ComputeError::OutOfRange`] for a degenerate grid or fewer than 3
    /// reference atoms. (A frame whose superposition leaves the orientation
    /// undetermined, e.g. a collinear template, is refused per-frame with
    /// [`ComputeError::OutOfRange`].)
    pub fn new(
        reference: Vec<usize>,
        template: Array2<F>,
        target: Vec<usize>,
        grid: GridSpec,
    ) -> Result<Self, ComputeError> {
        if template.nrows() != reference.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: reference.len(),
                got: template.nrows(),
                what: "SDF template rows",
            });
        }
        if template.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: template.ncols(),
                what: "SDF template columns",
            });
        }
        if reference.len() < 3 {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::reference",
                value: reference.len().to_string(),
            });
        }
        if grid.n.contains(&0) || grid.extent.iter().any(|&e| e <= 0.0 || e.is_nan()) {
            return Err(ComputeError::OutOfRange {
                field: "SpatialDistribution::grid",
                value: format!("n={:?}, extent={:?}", grid.n, grid.extent),
            });
        }
        let template = template.rows().into_iter().map(to_vec3).collect();
        Ok(Self {
            reference,
            template,
            target,
            grid,
            orientation: None,
            bulk_density: None,
        })
    }

    /// Attach `(tail, head)` index pairs (one per target atom, same order) so
    /// the result carries a per-voxel mean body-frame orientation of the unit
    /// `head − tail` vector.
    pub fn with_orientation(mut self, pairs: Vec<(usize, usize)>) -> Self {
        self.orientation = Some(pairs);
        self
    }

    /// Set the bulk number density (Å⁻³) used to form `g_SDF = ρ / ρ_bulk`.
    pub fn with_bulk_density(mut self, rho: F) -> Self {
        self.bulk_density = Some(rho);
        self
    }

    fn reference_coords(&self, xs: &[F], ys: &[F], zs: &[F]) -> Vec<Vec3> {
        self.reference
            .iter()
            .map(|&i| [xs[i], ys[i], zs[i]])
            .collect()
    }

    /// Bin one frame's targets into `counts` (and, when supplied, the orientation
    /// accumulators). Factored out of [`LegacyCompute::compute`] so the same body serves
    /// both the serial loop and the frame-parallel fast path.
    ///
    /// The minimum-image convention is resolved **once per frame** here (box kind
    /// and PBC mask), so the per-target displacement is pure scalar arithmetic.
    fn accumulate_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        counts: &mut Array3<F>,
        mut orient: Option<(&mut Array4<F>, &mut Array3<F>)>,
    ) -> Result<(), ComputeError> {
        let simbox = frame.simbox_ref();
        let mic = simbox.map_or(Mic::Free, SimBox::mic);
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();

        // Align this frame's reference set onto the template.
        // `superpose(frame, template)` gives R with R·(frameᵢ − c_f) ≈
        // (templateᵢ − c_t): the lab → body-frame rotation.
        let ref_coords = self.reference_coords(xs, ys, zs);
        let weights = vec![1.0; ref_coords.len()];
        let fit =
            superpose(&ref_coords, &self.template, &weights, DEFAULT_GAP_TOL).map_err(|e| {
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

        // Reference COM (lab frame) = centroid of the reference atoms. The
        // weights are all 1 and `superpose` accepted the set, so it is non-empty.
        let com = centroid(&ref_coords, &weights).ok_or(ComputeError::OutOfRange {
            field: "SpatialDistribution::reference",
            value: "empty reference selection".into(),
        })?;

        for (t, &ai) in self.target.iter().enumerate() {
            // Minimum-image vector COM → target, then rotate into body frame.
            let disp = mic.apply(sub([xs[ai], ys[ai], zs[ai]], com));
            let body = transform_point(&body_rotation, disp);
            let Some([ix, iy, iz]) = self.grid.index(body) else {
                continue;
            };
            counts[[ix, iy, iz]] += 1.0;

            if let Some((osum, ocount)) = orient.as_mut()
                && let Some(pairs) = &self.orientation
            {
                let (tail, head) = pairs[t];
                let v = mic.apply(sub(
                    [xs[head], ys[head], zs[head]],
                    [xs[tail], ys[tail], zs[tail]],
                ));
                if let Some(u) = normalize(v) {
                    let bu = transform_point(&body_rotation, u);
                    for d in 0..3 {
                        osum[[ix, iy, iz, d]] += bu[d];
                    }
                    ocount[[ix, iy, iz]] += 1.0;
                }
            }
        }
        Ok(())
    }
}

impl LegacyCompute for SpatialDistribution {
    type Args<'a> = ();
    type Output = SpatialDistributionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(pairs) = &self.orientation
            && pairs.len() != self.target.len()
        {
            return Err(ComputeError::DimensionMismatch {
                expected: self.target.len(),
                got: pairs.len(),
                what: "SDF orientation pairs",
            });
        }

        let [nx, ny, nz] = self.grid.n;
        let mut counts = Array3::<F>::zeros((nx, ny, nz));
        let mut orient_sum: Option<Array4<F>> = self
            .orientation
            .as_ref()
            .map(|_| Array4::zeros((nx, ny, nz, 3)));
        let mut orient_count: Option<Array3<F>> = self
            .orientation
            .as_ref()
            .map(|_| Array3::zeros((nx, ny, nz)));

        // Frame-parallel fast path when no orientation field is requested: per-voxel
        // counts are integer (+1.0) increments, so summing per-frame partial grids
        // in frame order is bit-identical to the serial accumulation. With an
        // orientation field the per-voxel vector sum is floating and order-sensitive,
        // so that case stays serial to keep results exactly reproducible.
        #[cfg(feature = "rayon")]
        let parallel = self.orientation.is_none() && frames.len() >= 4;
        #[cfg(not(feature = "rayon"))]
        let parallel = false;
        if parallel {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                let partials: Result<Vec<Array3<F>>, ComputeError> = frames
                    .par_iter()
                    .map(|frame| {
                        let mut c = Array3::<F>::zeros((nx, ny, nz));
                        self.accumulate_frame(*frame, &mut c, None)?;
                        Ok(c)
                    })
                    .collect();
                for partial in partials? {
                    counts += &partial;
                }
            }
        } else {
            for frame in frames {
                let orient = orient_sum.as_mut().zip(orient_count.as_mut());
                self.accumulate_frame(*frame, &mut counts, orient)?;
            }
        }

        let mut result = SpatialDistributionResult {
            counts,
            density: Array3::zeros((nx, ny, nz)),
            g_sdf: None,
            orientation: None,
            orient_sum,
            orient_count,
            n: self.grid.n,
            extent: self.grid.extent,
            voxel_volume: self.grid.voxel_volume(),
            n_frames: frames.len(),
            bulk_density: self.bulk_density,
            finalized: false,
        };
        result.finalize();
        Ok(result)
    }
}

/// Accumulated SDF grid plus optional bulk-normalized density and orientation.
#[derive(Debug, Clone)]
pub struct SpatialDistributionResult {
    /// Raw voxel counts, summed across frames.
    pub counts: Array3<F>,
    /// Number density per voxel (Å⁻³): `counts / (n_frames · ΔV)`.
    pub density: Array3<F>,
    /// Bulk-normalized `g_SDF = density / ρ_bulk`, if a bulk density was set.
    pub g_sdf: Option<Array3<F>>,
    /// Per-voxel mean body-frame orientation vector `(nx, ny, nz, 3)`, if an
    /// orientation selection was supplied. Zero on empty voxels.
    pub orientation: Option<Array4<F>>,
    orient_sum: Option<Array4<F>>,
    orient_count: Option<Array3<F>>,
    /// Voxels per axis.
    pub n: [usize; 3],
    /// Full grid side length per axis, Å.
    pub extent: [F; 3],
    /// Single-voxel volume, Å³.
    pub voxel_volume: F,
    /// Number of frames accumulated.
    pub n_frames: usize,
    bulk_density: Option<F>,
    finalized: bool,
}

impl ComputeResult for SpatialDistributionResult {
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        let denom = self.n_frames.max(1) as F * self.voxel_volume;
        self.density = self.counts.mapv(|c| c / denom);
        if let Some(rho) = self.bulk_density
            && rho > 0.0
        {
            self.g_sdf = Some(self.density.mapv(|d| d / rho));
        }
        if let (Some(sum), Some(count)) = (&self.orient_sum, &self.orient_count) {
            let [nx, ny, nz] = self.n;
            let mut mean = Array4::<F>::zeros((nx, ny, nz, 3));
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let c = count[[ix, iy, iz]];
                        if c > 0.0 {
                            for d in 0..3 {
                                mean[[ix, iy, iz, d]] = sum[[ix, iy, iz, d]] / c;
                            }
                        }
                    }
                }
            }
            self.orientation = Some(mean);
        }
        self.finalized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    /// 4 × 4 × 4 grid of unit voxels spanning `[−2, 2]³` about the COM.
    fn unit_grid() -> GridSpec {
        GridSpec {
            n: [4, 4, 4],
            extent: [4.0, 4.0, 4.0],
        }
    }

    #[test]
    fn two_reference_atoms_are_refused() {
        // Two points fix only a line: the spin about it is undetermined.
        let template = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let err = SpatialDistribution::new(vec![0, 1], template, vec![], unit_grid()).unwrap_err();
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
        let sdf = SpatialDistribution::new(vec![0, 1, 2], template, vec![3], unit_grid()).unwrap();
        let frame = frame_with(&[
            [4.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 5.5, 5.0],
        ]);
        let err = sdf.compute(&[&frame], ()).unwrap_err();
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
        let err =
            SpatialDistribution::new(vec![0, 1, 2], template, vec![], unit_grid()).unwrap_err();
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
        let template = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, -2.0, 0.0]];
        let sdf = SpatialDistribution::new(vec![0, 1, 2], template, vec![3], unit_grid()).unwrap();
        // Lab copy = Rz(+90°)·template + (5, 5, 5); Rz(+90°): (x, y) → (−y, x).
        //   (1, 0, 0) → (5, 6, 5); (0, 2, 0) → (3, 5, 5); (−1, −2, 0) → (7, 4, 5).
        // COM = (5, 5, 5). Lab → body rotation is Rz(−90°): (x, y) → (y, −x).
        // Target displacement (−0.5, 1.5, 0.5) → body (1.5, 0.5, 0.5).
        // Shifted by +2 → (3.5, 2.5, 2.5) → voxel (3, 2, 2).
        // (Identity would give voxel (1, 3, 2); the wrong sense Rz(+90°), (0, 1, 2).)
        let frame = frame_with(&[
            [5.0, 6.0, 5.0],
            [3.0, 5.0, 5.0],
            [7.0, 4.0, 5.0],
            [4.5, 6.5, 5.5],
        ]);
        let res = sdf.compute(&[&frame], ()).unwrap();
        assert_eq!(res.counts[[3, 2, 2]], 1.0);
        assert_eq!(res.counts.sum(), 1.0);
    }
}

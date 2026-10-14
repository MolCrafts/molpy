//! Pair structure, densities on grids and diffraction — the WASM face of
//! `molrs::compute::{rdf, density, diffraction}`.
//!
//! `Rdf` is a frame-form compute: `compute` takes an array of per-frame
//! `{ pairs }` objects and folds them one at a time, so every frame is
//! normalised by its own box (the one its `Neighbors` table carries).
//!
//! ```js
//! const nl = new NeighborList(6.0);
//! const frames = trajectory.map((frame) => { nl.build(frame); return { pairs: nl.neighbors() }; });
//! const gr = new Rdf(120, 6.0, 0.0).compute(frames).toJSON();
//! ```

use molrs::compute::{self as mc, Accumulate, Compute, Stream};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::{Given, fixed};
use super::handle::{described, output};
use super::wire::{Fields, records};

output! {
    /// g(r): `bin_edges`, `bin_centers`, `rdf`, the raw counts `n_r`, and the
    /// normalisation (`n_points`, `volume`, `n_frames`, …).
    RdfResult => mc::RdfResult
}

output! {
    /// The pair correlation of two per-item values, binned by distance.
    CorrelationFunctionResult => mc::CorrelationFunctionResult
}

output! {
    /// Per-item neighbour count and local density.
    LocalDensityResult => mc::LocalDensityResult
}

output! {
    /// A Gaussian-smeared density on a regular grid.
    GaussianDensityResult => mc::GaussianDensityResult
}

output! {
    /// Occupied voxels of a sphere voxelization, with raw counts.
    SphereVoxelizationResult => mc::SphereVoxelizationResult
}

output! {
    /// S(k) by the Debye formula: `k_values`, `sk`, `n_particles`.
    StaticStructureFactorDebyeResult => mc::StaticStructureFactorDebyeResult
}

output! {
    /// S(k) by direct summation: `k_magnitudes`, `sk`, `n_particles`.
    StaticStructureFactorDirectResult => mc::StaticStructureFactorDirectResult
}

output! {
    /// A 2-D diffraction pattern and the projected image it came from.
    DiffractionPatternResult => mc::DiffractionPatternResult
}

/// Radial distribution function, folded over frames.
#[wasm_bindgen]
pub struct Rdf {
    inner: mc::Rdf,
}

#[wasm_bindgen]
impl Rdf {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_bins: usize,
        r_max: F,
        r_min: F,
        dimensionality: Option<u8>,
    ) -> Result<Rdf, JsError> {
        let inner = mc::Rdf::new(n_bins, r_max, r_min)?
            .given(dimensionality, |c, v| c.with_dimensionality(v))?;
        Ok(Self { inner })
    }

    /// `[{ pairs }, …]`: one `Neighbors` table per frame.
    pub fn compute(&self, frames: &JsValue) -> Result<RdfResult, JsError> {
        let mut accumulator = self.inner.accumulator();
        for frame in records(frames, "Rdf")? {
            let pairs = Fields::new(&frame, "Rdf")?.neighbors("pairs")?;
            accumulator.push(&*pairs)?;
        }
        Ok(accumulator.finish()?.into())
    }
}

/// Distance-binned correlation of two per-item values.
#[wasm_bindgen]
pub struct CorrelationFunction {
    inner: mc::CorrelationFunction,
}

#[wasm_bindgen]
impl CorrelationFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<CorrelationFunction, JsError> {
        Ok(Self {
            inner: mc::CorrelationFunction::new(n_bins, r_max, r_min)?,
        })
    }

    /// `{ pairs, values_a, values_b }`.
    pub fn compute(&self, data: &JsValue) -> Result<CorrelationFunctionResult, JsError> {
        let data = Fields::new(data, "CorrelationFunction")?;
        let pairs = data.neighbors("pairs")?;
        let values_a = data.reals("values_a")?;
        let values_b = data.reals("values_b")?;
        let input = mc::CorrelationFunctionInput {
            pairs: &*pairs,
            values_a: values_a.view(),
            values_b: values_b.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Smoothed local number density around each item.
#[wasm_bindgen]
pub struct LocalDensity {
    inner: mc::LocalDensity,
}

#[wasm_bindgen]
impl LocalDensity {
    #[wasm_bindgen(constructor)]
    pub fn new(r_max: F, diameter: Option<F>) -> Result<LocalDensity, JsError> {
        let inner = mc::LocalDensity::new(r_max)?.given(diameter, |c, v| c.with_diameter(v))?;
        Ok(Self { inner })
    }

    /// `{ pairs }`.
    pub fn compute(&self, data: &JsValue) -> Result<LocalDensityResult, JsError> {
        let data = Fields::new(data, "LocalDensity")?;
        let pairs = data.neighbors("pairs")?;
        let input = mc::LocalDensityInput { pairs: &*pairs };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Gaussian-smeared particle density on a grid spanning the box.
#[wasm_bindgen]
pub struct GaussianDensity {
    inner: mc::GaussianDensity,
}

#[wasm_bindgen]
impl GaussianDensity {
    /// `dims` is the voxel count per axis (3 values).
    #[wasm_bindgen(constructor)]
    pub fn new(dims: Vec<usize>, sigma: F, r_max: Option<F>) -> Result<GaussianDensity, JsError> {
        let dims = fixed::<usize, 3>(&dims, "GaussianDensity dims (3 values)")?;
        let inner = mc::GaussianDensity::new(dims, sigma)?.given(r_max, |c, v| c.with_r_max(v))?;
        Ok(Self { inner })
    }

    /// `{ positions, simbox }`.
    pub fn compute(&self, data: &JsValue) -> Result<GaussianDensityResult, JsError> {
        let data = Fields::new(data, "GaussianDensity")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.simbox("simbox")?;
        let input = mc::GaussianDensityInput {
            positions: positions.view(),
            simbox: &simbox,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Voxels within `rMax` of any particle.
#[wasm_bindgen]
pub struct SphereVoxelization {
    inner: mc::SphereVoxelization,
}

#[wasm_bindgen]
impl SphereVoxelization {
    /// `dims` is the voxel count per axis (3 values).
    #[wasm_bindgen(constructor)]
    pub fn new(dims: Vec<usize>, r_max: F) -> Result<SphereVoxelization, JsError> {
        let dims = fixed::<usize, 3>(&dims, "SphereVoxelization dims (3 values)")?;
        Ok(Self {
            inner: mc::SphereVoxelization::new(dims, r_max)?,
        })
    }

    /// `{ positions, simbox }`.
    pub fn compute(&self, data: &JsValue) -> Result<SphereVoxelizationResult, JsError> {
        let data = Fields::new(data, "SphereVoxelization")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.simbox("simbox")?;
        let input = mc::SphereVoxelizationInput {
            positions: positions.view(),
            simbox: &simbox,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// S(k) by the Debye formula at the given `kValues`.
#[wasm_bindgen]
pub struct StaticStructureFactorDebye {
    inner: mc::StaticStructureFactorDebye,
}

#[wasm_bindgen]
impl StaticStructureFactorDebye {
    #[wasm_bindgen(constructor)]
    pub fn new(
        k_values: Vec<F>,
        sinc_tolerance: Option<F>,
    ) -> Result<StaticStructureFactorDebye, JsError> {
        let inner = mc::StaticStructureFactorDebye::new(&k_values)?
            .given(sinc_tolerance, |c, v| c.with_sinc_tolerance(v))?;
        Ok(Self { inner })
    }

    /// `{ positions }`.
    pub fn compute(&self, data: &JsValue) -> Result<StaticStructureFactorDebyeResult, JsError> {
        let data = Fields::new(data, "StaticStructureFactorDebye")?;
        let positions = data.matrix("positions", Some(3))?;
        let input = mc::StaticStructureFactorDebyeInput {
            positions: positions.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// S(k) by direct summation over k-vectors.
#[wasm_bindgen]
pub struct StaticStructureFactorDirect {
    inner: mc::StaticStructureFactorDirect,
}

#[wasm_bindgen]
impl StaticStructureFactorDirect {
    /// Explicit k-vectors, flat `[kx, ky, kz, …]`.
    #[wasm_bindgen(constructor)]
    pub fn new(k_vecs: Vec<F>) -> Result<StaticStructureFactorDirect, JsError> {
        let (vectors, rest) = k_vecs.as_chunks::<3>();
        if !rest.is_empty() {
            return Err(mc::ComputeError::DimensionMismatch {
                expected: k_vecs.len() - rest.len(),
                got: k_vecs.len(),
                what: "StaticStructureFactorDirect k_vecs (3 values per vector)",
            }
            .into());
        }
        Ok(Self {
            inner: mc::StaticStructureFactorDirect::new(vectors)?,
        })
    }

    /// The isotropic form: the box's reciprocal lattice up to `kMax`, binned
    /// into `nBins` shells.
    pub fn isotropic(k_max: F, n_bins: usize) -> Result<StaticStructureFactorDirect, JsError> {
        Ok(Self {
            inner: mc::StaticStructureFactorDirect::isotropic(k_max, n_bins)?,
        })
    }

    /// `{ positions, simbox? }`.
    pub fn compute(&self, data: &JsValue) -> Result<StaticStructureFactorDirectResult, JsError> {
        let data = Fields::new(data, "StaticStructureFactorDirect")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.opt_simbox("simbox")?;
        let input = mc::StaticStructureFactorDirectInput {
            positions: positions.view(),
            simbox: simbox.as_ref(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// A 2-D diffraction pattern of a projected Gaussian density.
#[wasm_bindgen]
pub struct DiffractionPattern {
    inner: mc::DiffractionPattern,
}

#[wasm_bindgen]
impl DiffractionPattern {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_grid: usize,
        sigma: F,
        axis: Option<usize>,
        cutoff: Option<F>,
    ) -> Result<DiffractionPattern, JsError> {
        let inner = mc::DiffractionPattern::new(n_grid, sigma)?
            .given(axis, |c, v| c.with_axis(v))?
            .given(cutoff, |c, v| c.with_cutoff(v))?;
        Ok(Self { inner })
    }

    /// `{ positions, simbox }`.
    pub fn compute(&self, data: &JsValue) -> Result<DiffractionPatternResult, JsError> {
        let data = Fields::new(data, "DiffractionPattern")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.simbox("simbox")?;
        let input = mc::DiffractionPatternInput {
            positions: positions.view(),
            simbox: &simbox,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    Rdf => mc::Rdf,
    CorrelationFunction => mc::CorrelationFunction,
    LocalDensity => mc::LocalDensity,
    GaussianDensity => mc::GaussianDensity,
    SphereVoxelization => mc::SphereVoxelization,
    StaticStructureFactorDebye => mc::StaticStructureFactorDebye,
    StaticStructureFactorDirect => mc::StaticStructureFactorDirect,
    DiffractionPattern => mc::DiffractionPattern,
);

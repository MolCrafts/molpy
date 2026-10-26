//! Radical Voronoi tessellation and its consumers — the WASM face of
//! `molrs::compute::voronoi`.
//!
//! ```js
//! const cells = new RadicalVoronoi().compute({ positions, simbox, radii });
//! const voids = new VoronoiVoidAnalysis().compute({
//!     cells, is_void: mask, box_volume: simbox.volume(),
//! });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::RegularGrid;
use super::handle::{described, output};
use super::wire::Fields;
use crate::core::NDArray;

output! {
    /// One radical Voronoi cell per generator: `volumes`, `faces`. The input
    /// of `VoronoiDomainAnalysis` and `VoronoiVoidAnalysis`.
    VoronoiCells => mc::VoronoiCells
}

output! {
    /// Voronoi-integrated charges, dipoles and references per molecule. The
    /// input of `FiniteFieldPolarizability`.
    MolecularMoments => mc::MolecularMoments
}

output! {
    /// Face-connected domains of each group: `sizes`, `count`,
    /// `largest_fraction`, `domain_of`. A `Partition`.
    VoronoiDomainResult => mc::VoronoiDomainResult
}

output! {
    /// Cavities of the void cells: `cavity_volumes`, `total_void_volume`,
    /// `void_fraction`.
    VoronoiVoidResult => mc::VoronoiVoidResult
}

/// Radical (Laguerre) Voronoi tessellation of weighted points.
#[wasm_bindgen]
pub struct RadicalVoronoi {
    inner: mc::RadicalVoronoi,
}

#[wasm_bindgen]
impl RadicalVoronoi {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> RadicalVoronoi {
        Self {
            inner: mc::RadicalVoronoi::new(),
        }
    }

    /// `{ positions, simbox, radii }`.
    pub fn compute(&self, data: &JsValue) -> Result<VoronoiCells, JsError> {
        let data = Fields::new(data, "RadicalVoronoi")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.simbox("simbox")?;
        let radii = data.reals("radii")?;
        let input = mc::RadicalVoronoiInput {
            positions: positions.view(),
            simbox: &simbox,
            radii: radii.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Molecular charges and dipoles integrated over Voronoi cells of a density.
#[wasm_bindgen]
pub struct VoronoiIntegration {
    inner: mc::VoronoiIntegration,
}

#[wasm_bindgen]
impl VoronoiIntegration {
    /// `unit` is a [`LengthUnit`](mc::LengthUnit) name: the unit of the
    /// positions, box and grid.
    #[wasm_bindgen(constructor)]
    pub fn new(unit: &str) -> Result<VoronoiIntegration, JsError> {
        Ok(Self {
            inner: mc::VoronoiIntegration::new(unit.parse()?),
        })
    }

    /// `{ positions, radii, nuclear_charges, partition, simbox, grid, density }`:
    /// `grid` a `RegularGrid`, `density` its `nx × ny × nz` values.
    pub fn compute(&self, data: &JsValue) -> Result<MolecularMoments, JsError> {
        let data = Fields::new(data, "VoronoiIntegration")?;
        let positions = data.matrix("positions", Some(3))?;
        let radii = data.reals("radii")?;
        let nuclear_charges = data.reals("nuclear_charges")?;
        let partition = data.partition("partition")?;
        let simbox = data.simbox("simbox")?;
        let grid = data.handle::<RegularGrid>("grid", "RegularGrid")?;
        let density = data.cube("density")?;
        let input = mc::VoronoiIntegrationInput {
            positions: positions.view(),
            radii: radii.view(),
            nuclear_charges: nuclear_charges.view(),
            partition: partition.as_dyn(),
            simbox: &simbox,
            grid: &grid.inner,
            density: density.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Face-connected domains of each group over Voronoi cells.
#[wasm_bindgen]
pub struct VoronoiDomainAnalysis {
    inner: mc::VoronoiDomainAnalysis,
}

#[wasm_bindgen]
impl VoronoiDomainAnalysis {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> VoronoiDomainAnalysis {
        Self {
            inner: mc::VoronoiDomainAnalysis::new(),
        }
    }

    /// `{ cells, partition }`: `cells` a `VoronoiCells`.
    pub fn compute(&self, data: &JsValue) -> Result<VoronoiDomainResult, JsError> {
        let data = Fields::new(data, "VoronoiDomainAnalysis")?;
        let cells = data.handle::<VoronoiCells>("cells", "VoronoiCells")?;
        let partition = data.partition("partition")?;
        let input = mc::VoronoiDomainInput {
            cells: &cells.inner,
            partition: partition.as_dyn(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Connected cavities of the cells marked void.
#[wasm_bindgen]
pub struct VoronoiVoidAnalysis {
    inner: mc::VoronoiVoidAnalysis,
}

#[wasm_bindgen]
impl VoronoiVoidAnalysis {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> VoronoiVoidAnalysis {
        Self {
            inner: mc::VoronoiVoidAnalysis::new(),
        }
    }

    /// `{ cells, is_void, box_volume }`.
    pub fn compute(&self, data: &JsValue) -> Result<VoronoiVoidResult, JsError> {
        let data = Fields::new(data, "VoronoiVoidAnalysis")?;
        let cells = data.handle::<VoronoiCells>("cells", "VoronoiCells")?;
        let is_void = data.mask("is_void")?;
        let input = mc::VoronoiVoidInput {
            cells: &cells.inner,
            is_void: &is_void,
            box_volume: data.scalar("box_volume")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Molecular polarizabilities from moments under `±field`.
#[wasm_bindgen]
pub struct FiniteFieldPolarizability {
    inner: mc::FiniteFieldPolarizability,
}

#[wasm_bindgen]
impl FiniteFieldPolarizability {
    #[wasm_bindgen(constructor)]
    pub fn new(field: F) -> Result<FiniteFieldPolarizability, JsError> {
        Ok(Self {
            inner: mc::FiniteFieldPolarizability::new(field)?,
        })
    }

    /// `{ plus, minus }`: the `MolecularMoments` under `+field` and `−field`;
    /// returns one polarizability row per molecule.
    pub fn compute(&self, data: &JsValue) -> Result<NDArray, JsError> {
        let data = Fields::new(data, "FiniteFieldPolarizability")?;
        let plus = data.handle::<MolecularMoments>("plus", "MolecularMoments")?;
        let minus = data.handle::<MolecularMoments>("minus", "MolecularMoments")?;
        let input = mc::FiniteFieldPolarizabilityInput {
            plus: &plus.inner,
            minus: &minus.inner,
        };
        Ok(NDArray::from_ndarray(&self.inner.compute(input)?))
    }
}

described!(
    RadicalVoronoi => mc::RadicalVoronoi,
    VoronoiIntegration => mc::VoronoiIntegration,
    VoronoiDomainAnalysis => mc::VoronoiDomainAnalysis,
    VoronoiVoidAnalysis => mc::VoronoiVoidAnalysis,
    FiniteFieldPolarizability => mc::FiniteFieldPolarizability,
);

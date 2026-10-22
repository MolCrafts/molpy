//! Local environments — the WASM face of `molrs::compute::environment`.
//!
//! ```js
//! const env = new EnvironmentMatch(0.3).compute({ neighbors });
//! const com = new CenterOfMass().compute({ positions, partition: env, masses });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// The bond-orientational order diagram on a (θ, φ) grid.
    BondOrientationalOrderResult => mc::BondOrientationalOrderResult
}

output! {
    /// Spherical-harmonic descriptors per particle (`l_max`, `n_sphs`,
    /// `features`). A `FeatureMatrix` for `Pca` / `Kmeans`.
    LocalDescriptorsResult => mc::LocalDescriptorsResult
}

output! {
    /// Projections of each bond onto the reference vectors.
    LocalBondProjectionResult => mc::LocalBondProjectionResult
}

output! {
    /// Environment clusters: `cluster_idx`, `n_clusters`, `fingerprints`. A
    /// `Partition` for the group computes.
    EnvironmentMatchResult => mc::EnvironmentMatchResult
}

output! {
    /// Angles between every query and every reference orientation.
    AngularSeparationGlobalResult => mc::AngularSeparationGlobalResult
}

output! {
    /// Angle between the orientations of each neighbour pair.
    AngularSeparationNeighborResult => mc::AngularSeparationNeighborResult
}

/// Bond-orientational order diagram.
#[wasm_bindgen]
pub struct BondOrientationalOrder {
    inner: mc::BondOrientationalOrder,
}

#[wasm_bindgen]
impl BondOrientationalOrder {
    #[wasm_bindgen(constructor)]
    pub fn new(n_theta: usize, n_phi: usize) -> Result<BondOrientationalOrder, JsError> {
        Ok(Self {
            inner: mc::BondOrientationalOrder::new(n_theta, n_phi)?,
        })
    }

    /// `{ neighbors }`.
    pub fn compute(&self, data: &JsValue) -> Result<BondOrientationalOrderResult, JsError> {
        let data = Fields::new(data, "BondOrientationalOrder")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::BondOrientationalOrderInput {
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Spherical-harmonic descriptors up to `lMax`.
#[wasm_bindgen]
pub struct LocalDescriptors {
    inner: mc::LocalDescriptors,
}

#[wasm_bindgen]
impl LocalDescriptors {
    #[wasm_bindgen(constructor)]
    pub fn new(l_max: u32) -> LocalDescriptors {
        Self {
            inner: mc::LocalDescriptors::new(l_max),
        }
    }

    /// `{ neighbors }`.
    pub fn compute(&self, data: &JsValue) -> Result<LocalDescriptorsResult, JsError> {
        let data = Fields::new(data, "LocalDescriptors")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::LocalDescriptorsInput {
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Bond projections onto reference vectors, in each particle's frame.
#[wasm_bindgen]
pub struct LocalBondProjection {
    inner: mc::LocalBondProjection,
}

#[wasm_bindgen]
impl LocalBondProjection {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> LocalBondProjection {
        Self {
            inner: mc::LocalBondProjection::new(),
        }
    }

    /// `{ neighbors, proj_vectors, orientations? }`: `proj_vectors` `M × 3`,
    /// `orientations` `N × 4`.
    pub fn compute(&self, data: &JsValue) -> Result<LocalBondProjectionResult, JsError> {
        let data = Fields::new(data, "LocalBondProjection")?;
        let neighbors = data.neighbors("neighbors")?;
        let proj_vectors = data.matrix("proj_vectors", Some(3))?;
        let orientations = data.opt_matrix("orientations", Some(4))?;
        let input = mc::LocalBondProjectionInput {
            neighbors: &neighbors,
            proj_vectors: proj_vectors.view(),
            orientations: orientations.as_ref().map(|o| o.view()),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Cluster particles whose neighbour environments match within an RMSD.
#[wasm_bindgen]
pub struct EnvironmentMatch {
    inner: mc::EnvironmentMatch,
}

#[wasm_bindgen]
impl EnvironmentMatch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rmsd_threshold: F,
        registration: Option<bool>,
        max_neighbors_for_registration: Option<usize>,
    ) -> Result<EnvironmentMatch, JsError> {
        let inner = mc::EnvironmentMatch::new(rmsd_threshold)?
            .given(registration, |c, v| Ok(c.with_registration(v)))?
            .given(max_neighbors_for_registration, |c, v| {
                c.with_max_neighbors_for_registration(v)
            })?;
        Ok(Self { inner })
    }

    /// `{ neighbors }`.
    pub fn compute(&self, data: &JsValue) -> Result<EnvironmentMatchResult, JsError> {
        let data = Fields::new(data, "EnvironmentMatch")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::EnvironmentMatchInput {
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Angles between query orientations and a reference set.
#[wasm_bindgen]
pub struct AngularSeparationGlobal {
    inner: mc::AngularSeparationGlobal,
}

#[wasm_bindgen]
impl AngularSeparationGlobal {
    #[wasm_bindgen(constructor)]
    pub fn new(equivalent_orientations: Option<bool>) -> Result<AngularSeparationGlobal, JsError> {
        let inner = mc::AngularSeparationGlobal::new().given(equivalent_orientations, |c, v| {
            Ok(c.with_equivalent_orientations(v))
        })?;
        Ok(Self { inner })
    }

    /// `{ query, reference }`: quaternions, `N × 4` and `M × 4`.
    pub fn compute(&self, data: &JsValue) -> Result<AngularSeparationGlobalResult, JsError> {
        let data = Fields::new(data, "AngularSeparationGlobal")?;
        let query = data.matrix("query", Some(4))?;
        let reference = data.matrix("reference", Some(4))?;
        let input = mc::AngularSeparationGlobalInput {
            query: query.view(),
            reference: reference.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Angles between the orientations of neighbouring particles.
#[wasm_bindgen]
pub struct AngularSeparationNeighbor {
    inner: mc::AngularSeparationNeighbor,
}

#[wasm_bindgen]
impl AngularSeparationNeighbor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        equivalent_orientations: Option<bool>,
    ) -> Result<AngularSeparationNeighbor, JsError> {
        let inner = mc::AngularSeparationNeighbor::new()
            .given(equivalent_orientations, |c, v| {
                Ok(c.with_equivalent_orientations(v))
            })?;
        Ok(Self { inner })
    }

    /// `{ neighbors, query_orientations, point_orientations }`: quaternions,
    /// `N × 4`.
    pub fn compute(&self, data: &JsValue) -> Result<AngularSeparationNeighborResult, JsError> {
        let data = Fields::new(data, "AngularSeparationNeighbor")?;
        let neighbors = data.neighbors("neighbors")?;
        let query_orientations = data.matrix("query_orientations", Some(4))?;
        let point_orientations = data.matrix("point_orientations", Some(4))?;
        let input = mc::AngularSeparationNeighborInput {
            neighbors: &neighbors,
            query_orientations: query_orientations.view(),
            point_orientations: point_orientations.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    BondOrientationalOrder => mc::BondOrientationalOrder,
    LocalDescriptors => mc::LocalDescriptors,
    LocalBondProjection => mc::LocalBondProjection,
    EnvironmentMatch => mc::EnvironmentMatch,
    AngularSeparationGlobal => mc::AngularSeparationGlobal,
    AngularSeparationNeighbor => mc::AngularSeparationNeighbor,
);

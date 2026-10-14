//! Cluster shape: centers, center of mass, gyration / inertia tensors and
//! radius of gyration — WASM face of the `molrs::compute` shape family.

use super::ClusterResult;
use crate::core::frame::Frame;
use crate::core::nd_array::JsFloatArray;
use molrs::compute::LegacyCompute;
use molrs::compute::{
    CenterOfMass as RsCenterOfMass, CenterOfMassResult as RsCenterOfMassResult,
    ClusterCenters as RsClusterCenters, GyrationTensor as RsGyrationTensor,
    InertiaTensor as RsInertiaTensor, RadiusOfGyration as RsRadiusOfGyration,
};
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Geometric cluster centers with minimum image convention.
///
/// # Example (JavaScript)
///
/// ```js
/// const centers = new ClusterCenters().compute(frame, clusterResult);
/// // Float64Array [x0,y0,z0, x1,y1,z1, ...]
/// ```
#[wasm_bindgen(js_name = ClusterCenters)]
pub struct ClusterCenters {
    inner: RsClusterCenters,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = ClusterCenters)]
impl ClusterCenters {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsClusterCenters::new(),
        }
    }

    /// Compute geometric centers. Returns a flat float typed array `[x0,y0,z0, ...]`.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let clusters_vec = vec![cluster_result.inner.clone()];
            let mut results = self
                .inner
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("ClusterCenters: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("ClusterCenters: empty result"))?;
            Ok(first
                .centers
                .iter()
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect())
        })
    }
}

// ===========================================================================
// CenterOfMass — Mass-weighted cluster centers
// ===========================================================================

/// Result of center-of-mass computation.
///
/// # Example (JavaScript)
///
/// ```js
/// const com = new CenterOfMass().compute(frame, clusterResult);
/// com.centersOfMass();   // Float64Array [x0,y0,z0, ...]
/// com.clusterMasses();   // Float64Array
/// ```
#[wasm_bindgen(js_name = CenterOfMassResult)]
pub struct CenterOfMassResult {
    inner: RsCenterOfMassResult,
}

#[wasm_bindgen(js_class = CenterOfMassResult)]
impl CenterOfMassResult {
    /// Zero-copy `Float64Array` view of mass-weighted centers, flat
    /// `[x0,y0,z0, x1,y1,z1, ...]`. **Invalidated** on WASM memory growth.
    #[wasm_bindgen(js_name = centersOfMass)]
    pub fn centers_of_mass(&self) -> JsFloatArray {
        // SAFETY: Vec<[F; 3]> is contiguous; `as_flattened` is safe.
        unsafe { JsFloatArray::view(self.inner.centers_of_mass.as_flattened()) }
    }

    /// Zero-copy `Float64Array` view of total mass per cluster.
    /// **Invalidated** on WASM memory growth.
    #[wasm_bindgen(js_name = clusterMasses)]
    pub fn cluster_masses(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(&self.inner.cluster_masses) }
    }

    /// Number of clusters.
    #[wasm_bindgen(getter, js_name = nClusters)]
    pub fn n_clusters(&self) -> usize {
        self.inner.centers_of_mass.len()
    }
}

/// Mass-weighted cluster center calculator.
#[wasm_bindgen(js_name = CenterOfMass)]
pub struct CenterOfMass {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = CenterOfMass)]
impl CenterOfMass {
    /// Create a center-of-mass calculator.
    ///
    /// Pass `null` for uniform masses, or a float typed array of per-particle masses.
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute centers of mass.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<CenterOfMassResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let mut results = calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("CenterOfMass: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("CenterOfMass: empty result"))?;
            Ok(CenterOfMassResult { inner: first })
        })
    }
}

// ===========================================================================
// GyrationTensor
// ===========================================================================

/// Gyration tensor per cluster.
///
/// Returns flat array: `[g00,g01,g02, g10,g11,g12, g20,g21,g22, ...]` per cluster.
#[wasm_bindgen(js_name = GyrationTensor)]
pub struct GyrationTensor {
    inner: RsGyrationTensor,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = GyrationTensor)]
impl GyrationTensor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsGyrationTensor::new(),
        }
    }

    /// Compute gyration tensors. Returns a flat float typed array (9 values per cluster).
    ///
    /// Internally computes the cluster geometric centers (via
    /// [`RsClusterCenters`]): the compute trait takes them as a required
    /// upstream, and this single-frame wasm API hides that detail.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let clusters_vec = vec![cluster_result.inner.clone()];
            let centers = RsClusterCenters::new()
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("GyrationTensor centers: {e}")))?;
            let mut tensors = self
                .inner
                .compute(&[rs_frame], (&clusters_vec, &centers))
                .map_err(|e| JsValue::from_str(&format!("GyrationTensor: {e}")))?;
            let first = tensors
                .pop()
                .ok_or_else(|| JsValue::from_str("GyrationTensor: empty result"))?;
            Ok(first
                .0
                .iter()
                .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
                .collect())
        })
    }
}

// ===========================================================================
// InertiaTensor
// ===========================================================================

/// Moment of inertia tensor per cluster.
#[wasm_bindgen(js_name = InertiaTensor)]
pub struct InertiaTensor {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = InertiaTensor)]
impl InertiaTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute inertia tensors. Returns a flat float typed array (9 values per cluster).
    ///
    /// Internally computes the cluster centers of mass (via
    /// [`RsCenterOfMass`]): the compute trait takes them as a required
    /// upstream, and this single-frame wasm API hides that detail.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let com_calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let coms = com_calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("InertiaTensor COM: {e}")))?;
            let calc = if let Some(ref ms) = self.masses {
                RsInertiaTensor::new().with_masses(ms)
            } else {
                RsInertiaTensor::new()
            };
            let mut tensors = calc
                .compute(&[rs_frame], (&clusters_vec, &coms))
                .map_err(|e| JsValue::from_str(&format!("InertiaTensor: {e}")))?;
            let first = tensors
                .pop()
                .ok_or_else(|| JsValue::from_str("InertiaTensor: empty result"))?;
            Ok(first
                .0
                .iter()
                .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
                .collect())
        })
    }
}

// ===========================================================================
// RadiusOfGyration
// ===========================================================================

/// Radius of gyration per cluster.
#[wasm_bindgen(js_name = RadiusOfGyration)]
pub struct RadiusOfGyration {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = RadiusOfGyration)]
impl RadiusOfGyration {
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute radii of gyration. Returns a float typed array of length `nClusters`.
    ///
    /// Internally computes the cluster centers of mass so the single-frame
    /// wasm signature `(frame, cluster)` stays stable despite the new
    /// compute trait needing explicit COM upstream.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let com_calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let coms = com_calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("RadiusOfGyration COM: {e}")))?;
            let calc = if let Some(ref ms) = self.masses {
                RsRadiusOfGyration::new().with_masses(ms)
            } else {
                RsRadiusOfGyration::new()
            };
            let mut radii = calc
                .compute(&[rs_frame], (&clusters_vec, &coms))
                .map_err(|e| JsValue::from_str(&format!("RadiusOfGyration: {e}")))?;
            let first = radii
                .pop()
                .ok_or_else(|| JsValue::from_str("RadiusOfGyration: empty result"))?;
            Ok(first.0)
        })
    }
}

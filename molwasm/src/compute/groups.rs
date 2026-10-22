//! Groups and features — the WASM face of `molrs::compute::{cluster, shape,
//! decomposition, kmeans}`.
//!
//! A `Partition` field takes a label array or any result that groups items
//! (`ClusterResult`, `KmeansResult`, …); a `FeatureMatrix` field takes a
//! shaped array or any result that is one feature row per item.
//!
//! ```js
//! const clusters = new Cluster(5).compute({ pairs });
//! const shape = new GyrationTensor().compute({ positions, simbox, partition: clusters, weights });
//! console.log(shape.radiusOfGyration());
//!
//! const pca = new Pca(2).compute({ features: steinhardtResult });
//! const kmeans = new Kmeans(3).compute({ features: pca });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// Connected components: `cluster_idx` (−1 for an item in no cluster),
    /// `n_clusters`, `cluster_sizes`, `cluster_keys`. A `Partition`.
    ClusterResult => mc::ClusterResult
}

output! {
    /// Per-group centre of mass: `groups`, `centers`, `total_masses`.
    CenterOfMassResult => mc::CenterOfMassResult
}

output! {
    /// Per-group gyration tensor: `groups`, `tensors` (`G × 3 × 3`).
    GyrationTensorResult => mc::GyrationTensorResult
}

#[wasm_bindgen]
impl GyrationTensorResult {
    /// Radius of gyration of every group.
    #[wasm_bindgen(js_name = radiusOfGyration)]
    pub fn radius_of_gyration(&self) -> Vec<F> {
        self.inner.radius_of_gyration().to_vec()
    }
}

output! {
    /// Per-group inertia tensor: `groups`, `tensors` (`G × 3 × 3`).
    InertiaTensorResult => mc::InertiaTensorResult
}

output! {
    /// Principal components: `coords` (items × components) and `variance`.
    /// A `FeatureMatrix`.
    PcaResult => mc::PcaResult
}

output! {
    /// k-means `labels` and `centroids`. A `Partition`.
    KmeansResult => mc::KmeansResult
}

/// Connected components of a neighbour graph.
#[wasm_bindgen]
pub struct Cluster {
    inner: mc::Cluster,
}

#[wasm_bindgen]
impl Cluster {
    #[wasm_bindgen(constructor)]
    pub fn new(min_cluster_size: Option<usize>) -> Result<Cluster, JsError> {
        let inner =
            mc::Cluster::new().given(min_cluster_size, |c, v| c.with_min_cluster_size(v))?;
        Ok(Self { inner })
    }

    /// `{ pairs, merge? }`: `merge` a partition whose groups are joined whole.
    pub fn compute(&self, data: &JsValue) -> Result<ClusterResult, JsError> {
        let data = Fields::new(data, "Cluster")?;
        let pairs = data.neighbors("pairs")?;
        let merge = data.opt_partition("merge")?;
        let input = mc::ClusterInput {
            pairs: &pairs,
            merge: merge.as_ref().map(|m| m.as_dyn()),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Mass-weighted centre of every group, unwrapped across the box.
#[wasm_bindgen]
pub struct CenterOfMass {
    inner: mc::CenterOfMass,
}

#[wasm_bindgen]
impl CenterOfMass {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> CenterOfMass {
        Self {
            inner: mc::CenterOfMass::new(),
        }
    }

    /// `{ positions, simbox?, partition, masses }`.
    pub fn compute(&self, data: &JsValue) -> Result<CenterOfMassResult, JsError> {
        let data = Fields::new(data, "CenterOfMass")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.opt_simbox("simbox")?;
        let partition = data.partition("partition")?;
        let masses = data.reals("masses")?;
        let input = mc::CenterOfMassInput {
            positions: positions.view(),
            simbox: simbox.as_ref(),
            partition: partition.as_dyn(),
            masses: masses.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Weighted gyration tensor of every group.
#[wasm_bindgen]
pub struct GyrationTensor {
    inner: mc::GyrationTensor,
}

#[wasm_bindgen]
impl GyrationTensor {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> GyrationTensor {
        Self {
            inner: mc::GyrationTensor::new(),
        }
    }

    /// `{ positions, simbox?, partition, weights }`.
    pub fn compute(&self, data: &JsValue) -> Result<GyrationTensorResult, JsError> {
        let data = Fields::new(data, "GyrationTensor")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.opt_simbox("simbox")?;
        let partition = data.partition("partition")?;
        let weights = data.reals("weights")?;
        let input = mc::GyrationTensorInput {
            positions: positions.view(),
            simbox: simbox.as_ref(),
            partition: partition.as_dyn(),
            weights: weights.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Inertia tensor of every group.
#[wasm_bindgen]
pub struct InertiaTensor {
    inner: mc::InertiaTensor,
}

#[wasm_bindgen]
impl InertiaTensor {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> InertiaTensor {
        Self {
            inner: mc::InertiaTensor::new(),
        }
    }

    /// `{ positions, simbox?, partition, masses }`.
    pub fn compute(&self, data: &JsValue) -> Result<InertiaTensorResult, JsError> {
        let data = Fields::new(data, "InertiaTensor")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.opt_simbox("simbox")?;
        let partition = data.partition("partition")?;
        let masses = data.reals("masses")?;
        let input = mc::InertiaTensorInput {
            positions: positions.view(),
            simbox: simbox.as_ref(),
            partition: partition.as_dyn(),
            masses: masses.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Principal component analysis of a feature matrix.
#[wasm_bindgen]
pub struct Pca {
    inner: mc::Pca,
}

#[wasm_bindgen]
impl Pca {
    #[wasm_bindgen(constructor)]
    pub fn new(n_components: usize) -> Result<Pca, JsError> {
        Ok(Self {
            inner: mc::Pca::new(n_components)?,
        })
    }

    /// `{ features }`.
    pub fn compute(&self, data: &JsValue) -> Result<PcaResult, JsError> {
        let data = Fields::new(data, "Pca")?;
        let features = data.features("features")?;
        let input = mc::PcaInput {
            features: features.as_dyn(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// k-means clustering of a feature matrix.
#[wasm_bindgen]
pub struct Kmeans {
    inner: mc::Kmeans,
}

#[wasm_bindgen]
impl Kmeans {
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, max_iter: Option<usize>, seed: Option<u64>) -> Result<Kmeans, JsError> {
        let inner = mc::Kmeans::new(k)?
            .given(max_iter, |c, v| c.with_max_iter(v))?
            .given(seed, |c, v| Ok(c.with_seed(v)))?;
        Ok(Self { inner })
    }

    /// `{ features }`.
    pub fn compute(&self, data: &JsValue) -> Result<KmeansResult, JsError> {
        let data = Fields::new(data, "Kmeans")?;
        let features = data.features("features")?;
        let input = mc::KmeansInput {
            features: features.as_dyn(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    Cluster => mc::Cluster,
    CenterOfMass => mc::CenterOfMass,
    GyrationTensor => mc::GyrationTensor,
    InertiaTensor => mc::InertiaTensor,
    Pca => mc::Pca,
    Kmeans => mc::Kmeans,
);

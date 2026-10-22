//! The two data kinds several upstream types can fill: a `Partition` (a label
//! array or any result that groups items) and a `FeatureMatrix` (a shaped
//! array or any result that is one row of features per item).

use std::rc::Rc;

use js_sys::Array;
use molrs::compute::{self as mc, ComputeError, FeatureMatrix, Partition};
use molrs::op::F;
use ndarray::Array2;
use wasm_bindgen::JsValue;
use wasm_bindgen::convert::TryFromJsValue;

use super::environment::{EnvironmentMatchResult, LocalDescriptorsResult};
use super::groups::{ClusterResult, KmeansResult, PcaResult};
use super::order::SteinhardtResult;
use super::wire::Fields;

/// A partition of the items into groups, as given to `compute`.
pub(crate) enum PartitionArg {
    Labels(Vec<u32>),
    Cluster(Rc<mc::ClusterResult>),
    Kmeans(Rc<mc::KmeansResult>),
    Environment(Rc<mc::EnvironmentMatchResult>),
    #[cfg(feature = "voronoi")]
    Domain(Rc<mc::VoronoiDomainResult>),
}

impl PartitionArg {
    /// The partition the core `Input` borrows.
    pub(crate) fn as_dyn(&self) -> &(dyn Partition + Sync) {
        match self {
            Self::Labels(labels) => labels,
            Self::Cluster(result) => &**result,
            Self::Kmeans(result) => &**result,
            Self::Environment(result) => &**result,
            #[cfg(feature = "voronoi")]
            Self::Domain(result) => &**result,
        }
    }
}

/// A feature matrix, one row per item, as given to `compute`.
pub(crate) enum FeaturesArg {
    Matrix(Array2<F>),
    Pca(Rc<mc::PcaResult>),
    Steinhardt(Rc<mc::SteinhardtResult>),
    Descriptors(Rc<mc::LocalDescriptorsResult>),
}

impl FeaturesArg {
    /// The matrix the core `Input` borrows.
    pub(crate) fn as_dyn(&self) -> &(dyn FeatureMatrix + Sync) {
        match self {
            Self::Matrix(matrix) => matrix,
            Self::Pca(result) => &**result,
            Self::Steinhardt(result) => &**result,
            Self::Descriptors(result) => &**result,
        }
    }
}

/// `value` as the first of a chain of handle classes it is an instance of.
/// Each attempt hands the value back when it is not that class.
macro_rules! first_handle {
    ($value:expr, $($class:ty => $variant:expr),+ $(,)?) => {{
        let value: JsValue = $value;
        $(
            let value = match <$class as TryFromJsValue>::try_from_js_value(value) {
                Ok(handle) => return Ok($variant(handle.inner)),
                Err(value) => value,
            };
        )+
        drop(value);
    }};
}

const PARTITION: &str =
    "a label array, ClusterResult, KmeansResult, EnvironmentMatchResult or VoronoiDomainResult";

const FEATURES: &str =
    "a shaped array (rows × features), PcaResult, SteinhardtResult or LocalDescriptorsResult";

impl Fields<'_> {
    /// A partition (`Partition`).
    pub(crate) fn partition(&self, key: &str) -> Result<PartitionArg, ComputeError> {
        let value = self.value(key, PARTITION)?;
        self.partition_of(key, &value)
    }

    /// An optional partition.
    pub(crate) fn opt_partition(&self, key: &str) -> Result<Option<PartitionArg>, ComputeError> {
        self.opt_value(key)?
            .map(|value| self.partition_of(key, &value))
            .transpose()
    }

    fn partition_of(&self, key: &str, value: &JsValue) -> Result<PartitionArg, ComputeError> {
        if Array::is_array(value) || js_sys::ArrayBuffer::is_view(value) {
            return Ok(PartitionArg::Labels(self.naturals(key)?));
        }
        if let Some(dup) = super::wire::duplicate(value) {
            first_handle!(
                dup,
                ClusterResult => PartitionArg::Cluster,
                KmeansResult => PartitionArg::Kmeans,
                EnvironmentMatchResult => PartitionArg::Environment,
            );
            #[cfg(feature = "voronoi")]
            if let Some(dup) = super::wire::duplicate(value) {
                first_handle!(dup, super::voronoi::VoronoiDomainResult => PartitionArg::Domain);
            }
        }
        Err(super::wire::mismatch(&self.name(key), PARTITION, value))
    }

    /// A feature matrix (`FeatureMatrix`).
    pub(crate) fn features(&self, key: &str) -> Result<FeaturesArg, ComputeError> {
        let value = self.value(key, FEATURES)?;
        if let Some(dup) = super::wire::duplicate(&value) {
            first_handle!(
                dup,
                PcaResult => FeaturesArg::Pca,
                SteinhardtResult => FeaturesArg::Steinhardt,
                LocalDescriptorsResult => FeaturesArg::Descriptors,
            );
        }
        self.matrix(key, None).map(FeaturesArg::Matrix)
    }
}

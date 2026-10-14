//! Python arguments → core `compute` inputs.
//!
//! The data roles a core `Input` declares that are not a plain array or
//! number each have one adapter here:
//!
//! - a **pair source** ([`PairSourceArg`]): a `Neighbors` table or a built
//!   `NeighborList` engine, both core `PairSource`s;
//! - a **partition** ([`PartitionArg`]): an unsigned-int label array or any
//!   result that is a core `Partition`;
//! - a **feature matrix** ([`FeatureArg`]): a 2-D float array or any result
//!   that is a core `FeatureMatrix`;
//! - **boxes**, **unit presets** and **index tables**.
//!
//! A choice by name is parsed with the core `FromStr` (`"all".parse::<Origins>()`)
//! at the call site; nothing here retypes a choice's names.

use molrs::compute::{ComputeError, FeatureMatrix, Partition};
use molrs::core::{PairSource, SimBox, UnitPreset, lookup_unit_preset};
use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::cluster::PyClusterResult;
use super::decomposition::PyPcaResult;
use super::environment::{PyEnvironmentMatchResult, PyLocalDescriptorsResult};
use super::error::Result;
use super::kmeans::PyKmeansResult;
use super::order::PySteinhardtResult;
use super::voronoi::PyVoronoiDomainResult;
use crate::core::neighborlist::{PyNeighborList, PyNeighbors};
use crate::core::simbox::PyBox;
use crate::core::units::PyUnitPreset;

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// A core `PairSource`: a materialized `Neighbors` table or a `NeighborList`
/// engine streaming its pairs.
pub(crate) enum PairSourceArg<'py> {
    Table(PyRef<'py, PyNeighbors>),
    Engine(PyRef<'py, PyNeighborList>),
}

impl<'py> PairSourceArg<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(table) = obj.extract::<PyRef<'py, PyNeighbors>>() {
            return Ok(Self::Table(table));
        }
        if let Ok(engine) = obj.extract::<PyRef<'py, PyNeighborList>>() {
            return Ok(Self::Engine(engine));
        }
        Err(PyTypeError::new_err(format!(
            "a pair source is a Neighbors or a NeighborList, got {}",
            type_name(obj)
        )))
    }

    /// Every element of `objs` as a pair source.
    pub(crate) fn extract_all(objs: &[Bound<'py, PyAny>]) -> PyResult<Vec<Self>> {
        objs.iter().map(Self::extract).collect()
    }

    pub(crate) fn source(&self) -> PyResult<&(dyn PairSource + Sync)> {
        match self {
            Self::Table(table) => Ok(&table.inner),
            Self::Engine(engine) => match engine.inner.as_ref() {
                Some(engine) => Ok(engine),
                None => Err(PyValueError::new_err(
                    "NeighborList has been moved into a VerletSkin; build a new NeighborList",
                )),
            },
        }
    }
}

/// Borrow every pair source of `args`.
pub(crate) fn sources<'a>(
    args: &'a [PairSourceArg<'_>],
) -> PyResult<Vec<&'a (dyn PairSource + Sync)>> {
    args.iter().map(PairSourceArg::source).collect()
}

/// A core `Partition`: per-item group labels, or a result that partitions.
pub(crate) enum PartitionArg<'py> {
    Labels(Vec<u64>),
    Cluster(PyRef<'py, PyClusterResult>),
    Kmeans(PyRef<'py, PyKmeansResult>),
    EnvironmentMatch(PyRef<'py, PyEnvironmentMatchResult>),
    VoronoiDomain(PyRef<'py, PyVoronoiDomainResult>),
}

impl<'py> PartitionArg<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(result) = obj.extract::<PyRef<'py, PyClusterResult>>() {
            return Ok(Self::Cluster(result));
        }
        if let Ok(result) = obj.extract::<PyRef<'py, PyKmeansResult>>() {
            return Ok(Self::Kmeans(result));
        }
        if let Ok(result) = obj.extract::<PyRef<'py, PyEnvironmentMatchResult>>() {
            return Ok(Self::EnvironmentMatch(result));
        }
        if let Ok(result) = obj.extract::<PyRef<'py, PyVoronoiDomainResult>>() {
            return Ok(Self::VoronoiDomain(result));
        }
        if let Some(labels) = unsigned_labels(obj) {
            return Ok(Self::Labels(labels));
        }
        Err(PyTypeError::new_err(format!(
            "a partition is a 1-D unsigned-int label array or a partition result \
             (ClusterResult, KmeansResult, EnvironmentMatchResult, VoronoiDomainResult), \
             got {}",
            type_name(obj)
        )))
    }

    pub(crate) fn partition(&self) -> &(dyn Partition + Sync) {
        match self {
            Self::Labels(labels) => labels,
            Self::Cluster(result) => &result.inner,
            Self::Kmeans(result) => &result.inner,
            Self::EnvironmentMatch(result) => &result.inner,
            Self::VoronoiDomain(result) => &result.inner,
        }
    }
}

/// The labels of a 1-D numpy array of an unsigned integer dtype.
fn unsigned_labels(obj: &Bound<'_, PyAny>) -> Option<Vec<u64>> {
    fn widen<T: numpy::Element + Copy + Into<u64>>(obj: &Bound<'_, PyAny>) -> Option<Vec<u64>> {
        let array = obj.extract::<PyReadonlyArray1<'_, T>>().ok()?;
        Some(array.as_array().iter().map(|&label| label.into()).collect())
    }
    widen::<u64>(obj)
        .or_else(|| widen::<u32>(obj))
        .or_else(|| widen::<u16>(obj))
        .or_else(|| widen::<u8>(obj))
}

/// A core `FeatureMatrix`: a 2-D float array, or a result that is one.
pub(crate) enum FeatureArg<'py> {
    Matrix(Array2<f64>),
    Steinhardt(PyRef<'py, PySteinhardtResult>),
    LocalDescriptors(PyRef<'py, PyLocalDescriptorsResult>),
    Pca(PyRef<'py, PyPcaResult>),
}

impl<'py> FeatureArg<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(result) = obj.extract::<PyRef<'py, PySteinhardtResult>>() {
            return Ok(Self::Steinhardt(result));
        }
        if let Ok(result) = obj.extract::<PyRef<'py, PyLocalDescriptorsResult>>() {
            return Ok(Self::LocalDescriptors(result));
        }
        if let Ok(result) = obj.extract::<PyRef<'py, PyPcaResult>>() {
            return Ok(Self::Pca(result));
        }
        if let Ok(matrix) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
            return Ok(Self::Matrix(matrix.as_array().to_owned()));
        }
        Err(PyTypeError::new_err(format!(
            "a feature matrix is a 2-D float64 array or a descriptor result \
             (SteinhardtResult, LocalDescriptorsResult, PcaResult), got {}",
            type_name(obj)
        )))
    }

    pub(crate) fn features(&self) -> &(dyn FeatureMatrix + Sync) {
        match self {
            Self::Matrix(matrix) => matrix,
            Self::Steinhardt(result) => &result.inner,
            Self::LocalDescriptors(result) => &result.inner,
            Self::Pca(result) => &result.inner,
        }
    }
}

/// The core boxes of a list of `Box`es.
pub(crate) fn boxes(boxes: &[PyRef<'_, PyBox>]) -> Vec<SimBox> {
    boxes.iter().map(|simbox| simbox.inner.clone()).collect()
}

/// A unit preset given by name (``"real"``) or as a `molrs.core.UnitPreset`.
pub(crate) fn units(obj: &Bound<'_, PyAny>) -> PyResult<UnitPreset> {
    if let Ok(preset) = obj.extract::<PyRef<'_, PyUnitPreset>>() {
        return Ok(preset.inner.clone());
    }
    let name: &str = obj.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "units is a preset name or a UnitPreset, got {}",
            type_name(obj)
        ))
    })?;
    lookup_unit_preset(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown unit preset {name:?}")))
}

/// An `M × k` table of item tuples from a sequence of equal-length rows.
pub(crate) fn tuples(rows: Vec<Vec<usize>>, what: &'static str) -> Result<Array2<usize>> {
    let width = rows.first().map_or(0, Vec::len);
    if let Some(row) = rows.iter().find(|row| row.len() != width) {
        return Err(ComputeError::DimensionMismatch {
            expected: width,
            got: row.len(),
            what,
        }
        .into());
    }
    let n_rows = rows.len();
    let flat: Vec<usize> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, width), flat).map_err(|error| {
        ComputeError::BadShape {
            expected: format!("{what}: a rectangular table"),
            got: error.to_string(),
        }
        .into()
    })
}

use crate::io::mrec::zarr_storage::frame_io::join_path;
use molrs::core::DType;
use molrs::core::MolRsError;
use std::sync::Arc;
use zarrs::array::{Array, CodecOptions};
use zarrs::group::Group;
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};

use super::block_arrays::{DENSE_PIPELINE, Knobs};
use super::growth::{GrowthArray, land_values};
use super::layout::{
    NSTEP_ATTRIBUTE, STEP_ARRAY, STEP_PROGRESSION_ATTRIBUTE, TIME_PROGRESSION_ATTRIBUTE,
    TRAJECTORY_GROUP,
};
use super::store_access::{array_exists, read_whole};

// ---------------------------------------------------------------------------
// Dense per-frame series: progression or array
// ---------------------------------------------------------------------------

/// A value type a dense per-frame series (`step`, `time`) holds.
pub(super) trait Series:
    Copy + PartialEq + zarrs::array::Element + zarrs::array::ElementOwned
{
    const DTYPE: DType;
    /// `start + i * stride`, or `None` when it does not fit the type.
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self>;
    /// The stride from `a` to `b`, or `None` when it does not fit the type.
    fn stride_between(a: Self, b: Self) -> Option<Self>;
    fn to_json(self) -> Option<serde_json::Value>;
    fn from_json(value: &serde_json::Value) -> Option<Self>;
}

impl Series for i64 {
    const DTYPE: DType = DType::I64;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        let i = i64::try_from(i).ok()?;
        start.checked_add(stride.checked_mul(i)?)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        b.checked_sub(a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        Some(self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_i64()
    }
}

impl Series for f64 {
    const DTYPE: DType = DType::Float;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        Some(start + (i as f64) * stride)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        Some(b - a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        // JSON has no NaN / infinity; such a series goes to an array.
        self.is_finite().then(|| self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_f64()
    }
}

/// An arithmetic progression `start + i * stride` recorded as a trajectory
/// group attribute — how `step` and `time` are kept while they are regular.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Progression<T> {
    start: T,
    /// Unknown until the second value.
    stride: Option<T>,
}

impl<T: Series> Progression<T> {
    /// Parse the attribute form `{"start": …, "stride": …}`.
    pub(super) fn from_attribute(value: &serde_json::Value) -> Option<Self> {
        let start = T::from_json(value.get("start")?)?;
        let stride = value.get("stride").and_then(T::from_json);
        Some(Self { start, stride })
    }

    fn to_attribute(self) -> Option<serde_json::Value> {
        let mut object = serde_json::Map::new();
        object.insert("start".to_string(), self.start.to_json()?);
        if let Some(stride) = self.stride {
            object.insert("stride".to_string(), stride.to_json()?);
        }
        Some(serde_json::Value::Object(object))
    }

    /// The `i`-th value, or `None` past what the progression can say.
    fn nth(&self, i: u64) -> Option<T> {
        if i == 0 {
            return Some(self.start);
        }
        T::nth(self.start, self.stride?, i)
    }

    /// The first `count` values.
    fn values(&self, count: u64) -> Result<Vec<T>, MolRsError> {
        (0..count)
            .map(|i| {
                self.nth(i).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "progression attribute cannot produce value {i}: stride unknown or overflow"
                    ))
                })
            })
            .collect()
    }
}

/// Where a dense per-frame series lives: in the trajectory group's attribute
/// while it is regular, in its own array once it is not.
pub(super) enum Track<T: Series> {
    Regular {
        progression: Option<Progression<T>>,
        /// Values landed so far.
        count: u64,
    },
    Array(Box<GrowthArray>),
}

impl<T: Series> Track<T> {
    pub(super) fn fresh() -> Self {
        Self::Regular {
            progression: None,
            count: 0,
        }
    }

    /// Whether appending `values` keeps the series regular, and the
    /// progression it becomes.
    fn extended(&self, values: &[T]) -> Option<Progression<T>> {
        let Self::Regular { progression, count } = self else {
            return None;
        };
        let mut progression = *progression;
        for (index, &value) in (*count..).zip(values.iter()) {
            progression = match progression {
                None => Some(Progression {
                    start: value,
                    stride: None,
                }),
                Some(Progression {
                    start,
                    stride: None,
                }) => Some(Progression {
                    start,
                    stride: Some(T::stride_between(start, value)?),
                }),
                Some(p) => {
                    if p.nth(index)? != value {
                        return None;
                    }
                    Some(p)
                }
            };
            // The attribute must be able to spell it.
            progression?.to_attribute()?;
        }
        progression
    }

    /// The last value landed.
    pub(super) fn last(
        &self,
        store: &ReadableWritableListableStorage,
        path: &str,
    ) -> Result<Option<T>, MolRsError> {
        match self {
            Self::Regular { progression, count } => Ok(match (progression, *count) {
                (Some(p), n) if n > 0 => p.nth(n - 1),
                _ => None,
            }),
            Self::Array(array) => {
                if array.rows == 0 {
                    return Ok(None);
                }
                let values: Vec<T> = read_whole(store, path)?;
                Ok(values.get(array.rows as usize - 1).copied())
            }
        }
    }

    /// Land `values`: extend the progression, or materialize the array (backfilling the regular history) and append.
    pub(super) fn land(
        &mut self,
        store: &ReadableWritableListableStorage,
        path: &str,
        knobs: Knobs,
        values: Vec<T>,
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        if let Some(progression) = self.extended(&values) {
            let Self::Regular { count, .. } = self else {
                unreachable!("extended() is Some only for a regular track")
            };
            let count = *count + values.len() as u64;
            *self = Self::Regular {
                progression: Some(progression),
                count,
            };
            return Ok(());
        }
        if let Self::Regular { progression, count } = self {
            let mut array = GrowthArray::create(
                store,
                path,
                T::DTYPE,
                &[],
                knobs.dense(),
                serde_json::Map::new(),
                DENSE_PIPELINE,
            )?;
            if *count > 0 {
                let history = progression
                    .as_ref()
                    .ok_or_else(|| MolRsError::zarr("regular track without a progression"))?
                    .values(*count)?;
                let n = history.len() as u64;
                land_values!(array, n, history, options);
            }
            *self = Self::Array(Box::new(array));
        }
        let Self::Array(array) = self else {
            unreachable!("materialized above")
        };
        let n = values.len() as u64;
        land_values!(array, n, values, options);
        Ok(())
    }

    /// The attribute this track wants on the trajectory group, if any.
    pub(super) fn attribute(&self) -> Option<serde_json::Value> {
        match self {
            Self::Regular {
                progression: Some(p),
                count,
            } if *count > 0 => p.to_attribute(),
            _ => None,
        }
    }
}

/// Read a dense series back: the progression attribute when present, else
/// the array cut to `nstep`, else `None`.
pub(super) fn read_track<T: Series, S>(
    store: &Arc<S>,
    attributes: &serde_json::Map<String, serde_json::Value>,
    attribute: &str,
    path: &str,
    nstep: u64,
) -> Result<Option<Vec<T>>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(value) = attributes.get(attribute) {
        let progression = Progression::<T>::from_attribute(value).ok_or_else(|| {
            MolRsError::zarr(format!(
                "trajectory attribute {attribute:?} is not a progression"
            ))
        })?;
        return Ok(Some(progression.values(nstep)?));
    }
    if array_exists(store, path)? {
        let mut values: Vec<T> = read_whole(store, path)?;
        values.truncate(nstep as usize);
        return Ok(Some(values));
    }
    Ok(None)
}

/// The commit marker and the regular series, read off the trajectory group.
pub(super) struct TrajectoryAttributes {
    pub(super) nstep: Option<u64>,
    pub(super) all: serde_json::Map<String, serde_json::Value>,
}

pub(super) fn trajectory_attributes<S>(store: &Arc<S>) -> Result<TrajectoryAttributes, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
    let all = group.attributes().clone();
    let nstep = match all.get(NSTEP_ATTRIBUTE) {
        None => None,
        Some(value) => Some(value.as_u64().ok_or_else(|| {
            MolRsError::zarr(format!(
                "{TRAJECTORY_GROUP}: {NSTEP_ATTRIBUTE} must be a non-negative integer, found \
                 {value}"
            ))
        })?),
    };
    // A progression is only ever written in the same metadata update as the
    // commit marker, so one without it is a malformed store.
    if nstep.is_none() {
        for progression in [STEP_PROGRESSION_ATTRIBUTE, TIME_PROGRESSION_ATTRIBUTE] {
            if all.contains_key(progression) {
                return Err(MolRsError::zarr(format!(
                    "{TRAJECTORY_GROUP} carries {progression} without {NSTEP_ATTRIBUTE}: a \
                     progression is written with the commit marker, never alone"
                )));
            }
        }
    }
    Ok(TrajectoryAttributes { nstep, all })
}

/// The committed frame count: the `nstep` attribute, or — for a store from a
/// writer that kept no marker attribute — the length of the `step` array.
pub(super) fn committed_frames<S>(
    store: &Arc<S>,
    attributes: &TrajectoryAttributes,
) -> Result<u64, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(nstep) = attributes.nstep {
        return Ok(nstep);
    }
    let path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
    if array_exists(store, &path)? {
        return Ok(Array::open(store.clone(), &path)?
            .shape()
            .first()
            .copied()
            .unwrap_or(0));
    }
    Ok(0)
}

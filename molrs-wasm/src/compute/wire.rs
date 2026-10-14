//! The two directions of the compute seam: reading a `compute` call's data
//! object into the core `Input`, and writing a core value back to JavaScript.
//!
//! A `compute` call takes **one object** whose properties are the fields of
//! the core `Input` struct, under the same snake_case names
//! [`ComputeSpec`](molrs::compute::ComputeSpec) lists as `inputs` — the
//! keyword-argument form of the core struct literal. [`Fields`] reads it; how
//! each field is spelled in JavaScript follows its `InputKind`:
//!
//! | `InputKind` | JavaScript value |
//! |---|---|
//! | `Scalar`, `Index` | `number` |
//! | `Scalars`, `Angles` | `Float64Array` or `number[]` |
//! | `Indices`, `IndexPairs`, `Tuples` | `Uint32Array` or `number[]`, flat |
//! | `Mask` | `Uint8Array` or `boolean[]` |
//! | `Positions`, `Vectors`, `Quaternions`, `Series` | a shaped array (below) |
//! | `Trajectory`, `Field` | a shaped array of rank 3 |
//! | `Images` | `Int32Array` or `number[]`, flat, the shape of the positions |
//! | `SimBox` / `Boxes` | a `Box` / an array of `Box` |
//! | `Neighbors`, `PairSource` / `NeighborFrames`, `PairSourceFrames` | a `Neighbors` / an array of them |
//! | `Partition` | a label array (`Uint32Array`), or a `ClusterResult`, `KmeansResult`, `EnvironmentMatchResult`, `VoronoiDomainResult` |
//! | `FeatureMatrix` | a shaped array, or a `PcaResult`, `SteinhardtResult`, `LocalDescriptorsResult` |
//! | `Upstream(T)` / `UpstreamList(T)` | the handle `T` a compute returned / an array of them |
//! | `Choice` | one of the canonical names |
//!
//! A **shaped array** is an [`NDArray`](crate::core::NDArray), or an object
//! `{ dim, data }` — the form every array field of a result takes in its
//! `toJSON()`, so a result's array feeds the next compute unchanged — or,
//! where the trailing dimension is fixed (positions and vectors are `N × 3`,
//! quaternions `N × 4`), a flat array.
//!
//! Handles (`Box`, `Neighbors`, `NDArray`, result handles) are borrowed through
//! their `clone()`: the call works on a second handle and the caller's handle
//! stays usable.
//!
//! Every failure here is a [`ComputeError`], the same type the core returns,
//! and reaches JavaScript through `wasm_bindgen::JsError` as a thrown `Error`.

use std::rc::Rc;
use std::str::FromStr;

use js_sys::{Array, Float64Array, Function, Reflect, Uint32Array};
use molrs::compute::ComputeError;
use molrs::core::{Neighbors as RsNeighbors, SimBox};
use molrs::op::F;
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use serde::Serialize;
use wasm_bindgen::JsCast;
use wasm_bindgen::convert::TryFromJsValue;
use wasm_bindgen::prelude::*;

use crate::core::{Box as JsBox, NDArray, Neighbors};

/// A core value as a plain JavaScript object: structs become objects with the
/// Rust field names, `ndarray` arrays `{ v, dim, data }`, enums their serde
/// form.
pub(crate) fn to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsError> {
    Ok(value.serialize(&serde_wasm_bindgen::Serializer::json_compatible())?)
}

/// What a JavaScript value is, for an error message.
fn describe(value: &JsValue) -> String {
    if value.is_undefined() {
        "undefined".into()
    } else if value.is_null() {
        "null".into()
    } else if Array::is_array(value) {
        "an array".into()
    } else {
        match value.js_typeof().as_string() {
            Some(kind) => kind,
            None => "a value".into(),
        }
    }
}

/// The error for a value that is not what the field takes.
pub(crate) fn mismatch(what: &str, expected: &str, got: &JsValue) -> ComputeError {
    ComputeError::BadShape {
        expected: format!("{what} as {expected}"),
        got: describe(got),
    }
}

/// A second handle to `value`, from its `clone()`, if it has one.
pub(crate) fn duplicate(value: &JsValue) -> Option<JsValue> {
    if !value.is_object() {
        return None;
    }
    let method = Reflect::get(value, &JsValue::from_str("clone"))
        .ok()?
        .dyn_into::<Function>()
        .ok()?;
    method.call0(value).ok()
}

/// The bound class `H` behind `value`, through a second handle.
pub(crate) fn handle<H: TryFromJsValue>(
    value: &JsValue,
    what: &str,
    class: &str,
) -> Result<H, ComputeError> {
    duplicate(value)
        .and_then(|dup| H::try_from_js_value(dup).ok())
        .ok_or_else(|| mismatch(what, class, value))
}

/// One bound class `H` per element of the array `value`.
pub(crate) fn handles<H: TryFromJsValue>(
    value: &JsValue,
    what: &str,
    class: &str,
) -> Result<Vec<H>, ComputeError> {
    if !Array::is_array(value) {
        return Err(mismatch(what, &format!("an array of {class}"), value));
    }
    Array::from(value)
        .iter()
        .map(|item| handle(&item, what, class))
        .collect()
}

/// `value` read as reals: a typed array or an array of numbers.
fn reals(value: &JsValue) -> Option<Vec<F>> {
    if let Some(typed) = value.dyn_ref::<Float64Array>() {
        return Some(typed.to_vec());
    }
    (Array::is_array(value) || js_sys::ArrayBuffer::is_view(value))
        .then(|| Float64Array::new(value).to_vec())
}

/// `value` read as non-negative integers that fit `u32`.
fn naturals(value: &JsValue) -> Option<Vec<u32>> {
    if let Some(typed) = value.dyn_ref::<Uint32Array>() {
        return Some(typed.to_vec());
    }
    if !(Array::is_array(value) || js_sys::ArrayBuffer::is_view(value)) {
        return None;
    }
    Array::from(value)
        .iter()
        .map(|item| {
            let x = item.as_f64()?;
            (x.fract() == 0.0 && (0.0..=F::from(u32::MAX)).contains(&x)).then_some(x as u32)
        })
        .collect()
}

/// `value` read as integers that fit `i32`.
fn integers(value: &JsValue) -> Option<Vec<i32>> {
    if let Some(typed) = value.dyn_ref::<js_sys::Int32Array>() {
        return Some(typed.to_vec());
    }
    if !(Array::is_array(value) || js_sys::ArrayBuffer::is_view(value)) {
        return None;
    }
    Array::from(value)
        .iter()
        .map(|item| {
            let x = item.as_f64()?;
            (x.fract() == 0.0 && (F::from(i32::MIN)..=F::from(i32::MAX)).contains(&x))
                .then_some(x as i32)
        })
        .collect()
}

/// `value` read as a shaped real array: an `NDArray`, a `{ dim, data }`
/// object, or a flat array (rank 1).
fn shaped(value: &JsValue) -> Option<ArrayD<F>> {
    let (data, dim): (Vec<F>, Vec<usize>) = if let Some(flat) = reals(value) {
        let len = flat.len();
        (flat, vec![len])
    } else if let Some(Ok(array)) = duplicate(value).map(NDArray::try_from_js_value) {
        let (data, dim) = array.into_parts();
        (data, dim.into_vec())
    } else if value.is_object() {
        let dim = Reflect::get(value, &JsValue::from_str("dim")).ok()?;
        let data = Reflect::get(value, &JsValue::from_str("data")).ok()?;
        let dim = naturals(&dim)?.into_iter().map(|d| d as usize).collect();
        (reals(&data)?, dim)
    } else {
        return None;
    };
    ArrayD::from_shape_vec(IxDyn(&dim), data).ok()
}

/// The `compute` data object of one compute type.
pub(crate) struct Fields<'a> {
    object: &'a JsValue,
    owner: &'static str,
}

impl<'a> Fields<'a> {
    /// Read `object` as the data of `owner`'s `compute`.
    pub(crate) fn new(object: &'a JsValue, owner: &'static str) -> Result<Self, ComputeError> {
        if object.is_object() && !Array::is_array(object) {
            Ok(Self { object, owner })
        } else {
            Err(mismatch(
                &format!("{owner}.compute data"),
                "an object of named fields",
                object,
            ))
        }
    }

    fn what(&self, key: &str) -> String {
        format!("{} input `{key}`", self.owner)
    }

    fn fail(&self, key: &str, expected: &str, got: &JsValue) -> ComputeError {
        mismatch(&self.what(key), expected, got)
    }

    /// The field `key`; `None` when it is absent, `undefined` or `null`.
    fn get(&self, key: &str) -> Result<Option<JsValue>, ComputeError> {
        let value = Reflect::get(self.object, &JsValue::from_str(key))
            .map_err(|err| self.fail(key, "a readable property", &err))?;
        Ok((!value.is_undefined() && !value.is_null()).then_some(value))
    }

    /// The field `key`, which must be given.
    fn require(&self, key: &str, expected: &str) -> Result<JsValue, ComputeError> {
        self.get(key)?
            .ok_or_else(|| self.fail(key, expected, &JsValue::UNDEFINED))
    }

    /// Read the optional field `key` with `read`.
    fn optional<T>(
        &self,
        key: &str,
        read: impl FnOnce(&JsValue) -> Result<T, ComputeError>,
    ) -> Result<Option<T>, ComputeError> {
        self.get(key)?.as_ref().map(read).transpose()
    }

    // ------------------------------------------------------------- scalars

    /// A real number (`Scalar`).
    pub(crate) fn scalar(&self, key: &str) -> Result<F, ComputeError> {
        let value = self.require(key, "a number")?;
        value
            .as_f64()
            .ok_or_else(|| self.fail(key, "a number", &value))
    }

    /// An optional real number.
    pub(crate) fn opt_scalar(&self, key: &str) -> Result<Option<F>, ComputeError> {
        self.optional(key, |value| {
            value
                .as_f64()
                .ok_or_else(|| self.fail(key, "a number", value))
        })
    }

    /// A non-negative integer (`Index`).
    pub(crate) fn index(&self, key: &str) -> Result<usize, ComputeError> {
        let expected = "a non-negative integer";
        let value = self.require(key, expected)?;
        value
            .as_f64()
            .filter(|x| x.fract() == 0.0 && *x >= 0.0 && *x <= F::from(u32::MAX))
            .map(|x| x as usize)
            .ok_or_else(|| self.fail(key, expected, &value))
    }

    /// One of a choice's canonical names, parsed by the core `FromStr`.
    pub(crate) fn choice<E: FromStr<Err = ComputeError>>(
        &self,
        key: &str,
    ) -> Result<E, ComputeError> {
        let value = self.require(key, "a name")?;
        value
            .as_string()
            .ok_or_else(|| self.fail(key, "a name", &value))?
            .parse()
    }

    // -------------------------------------------------------------- arrays

    /// A 1-D array of reals (`Scalars`, `Angles`).
    pub(crate) fn reals(&self, key: &str) -> Result<Array1<F>, ComputeError> {
        let value = self.require(key, "a Float64Array")?;
        self.reals_of(key, &value)
    }

    /// An optional 1-D array of reals.
    pub(crate) fn opt_reals(&self, key: &str) -> Result<Option<Array1<F>>, ComputeError> {
        self.optional(key, |value| self.reals_of(key, value))
    }

    fn reals_of(&self, key: &str, value: &JsValue) -> Result<Array1<F>, ComputeError> {
        let array = shaped(value)
            .ok_or_else(|| self.fail(key, "a Float64Array", value))?
            .into_dimensionality()
            .map_err(|_| self.fail(key, "a rank-1 array", value))?;
        Ok(array)
    }

    /// A flat list of non-negative integers (`Indices`).
    pub(crate) fn naturals(&self, key: &str) -> Result<Vec<u32>, ComputeError> {
        let expected = "a Uint32Array";
        let value = self.require(key, expected)?;
        naturals(&value).ok_or_else(|| self.fail(key, expected, &value))
    }

    /// A flat list of indices (`Indices`), as `usize`.
    pub(crate) fn indices(&self, key: &str) -> Result<Vec<usize>, ComputeError> {
        Ok(self
            .naturals(key)?
            .into_iter()
            .map(|i| i as usize)
            .collect())
    }

    /// A flat list of `(i, j)` pairs (`IndexPairs`).
    pub(crate) fn pairs(&self, key: &str) -> Result<Vec<(u32, u32)>, ComputeError> {
        let flat = self.naturals(key)?;
        let (pairs, rest) = flat.as_chunks::<2>();
        if !rest.is_empty() {
            return Err(ComputeError::DimensionMismatch {
                expected: flat.len() - 1,
                got: flat.len(),
                what: "index pairs (an even count)",
            });
        }
        Ok(pairs.iter().map(|p| (p[0], p[1])).collect())
    }

    /// An optional flat list of `(i, j)` pairs, as `usize`.
    pub(crate) fn opt_index_pairs(
        &self,
        key: &str,
    ) -> Result<Option<Vec<(usize, usize)>>, ComputeError> {
        if self.get(key)?.is_none() {
            return Ok(None);
        }
        let pairs = self.pairs(key)?;
        Ok(Some(
            pairs
                .into_iter()
                .map(|(i, j)| (i as usize, j as usize))
                .collect(),
        ))
    }

    /// A flat `M × arity` table of item tuples (`Tuples`).
    pub(crate) fn tuples(&self, key: &str, arity: usize) -> Result<Array2<usize>, ComputeError> {
        let value = self.require(key, "a Uint32Array")?;
        self.tuples_of(key, &value, arity)
    }

    fn tuples_of(
        &self,
        key: &str,
        value: &JsValue,
        arity: usize,
    ) -> Result<Array2<usize>, ComputeError> {
        let flat: Vec<usize> = naturals(value)
            .ok_or_else(|| self.fail(key, "a Uint32Array", value))?
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let rows = flat.len() / arity;
        Array2::from_shape_vec((rows, arity), flat).map_err(|_| ComputeError::BadShape {
            expected: format!("{}: a multiple of {arity} indices", self.what(key)),
            got: describe(value),
        })
    }

    /// One flat tuple table per axis (`TuplesList`), each of its arity.
    pub(crate) fn tuples_list(
        &self,
        key: &str,
        arities: &[usize],
    ) -> Result<Vec<Array2<usize>>, ComputeError> {
        let expected = "an array of Uint32Array, one per axis";
        let value = self.require(key, expected)?;
        if !Array::is_array(&value) || Array::from(&value).length() as usize != arities.len() {
            return Err(self.fail(key, expected, &value));
        }
        Array::from(&value)
            .iter()
            .zip(arities)
            .map(|(table, &arity)| self.tuples_of(key, &table, arity))
            .collect()
    }

    /// A per-item flag mask (`Mask`).
    #[cfg(feature = "voronoi")]
    pub(crate) fn mask(&self, key: &str) -> Result<Vec<bool>, ComputeError> {
        let expected = "a Uint8Array or boolean[]";
        let value = self.require(key, expected)?;
        if let Some(bytes) = value.dyn_ref::<js_sys::Uint8Array>() {
            return Ok(bytes.to_vec().into_iter().map(|b| b != 0).collect());
        }
        if !Array::is_array(&value) {
            return Err(self.fail(key, expected, &value));
        }
        Array::from(&value)
            .iter()
            .map(|flag| flag.as_bool())
            .collect::<Option<_>>()
            .ok_or_else(|| self.fail(key, expected, &value))
    }

    /// A rank-2 real array; with `cols`, a flat array is read as rows of
    /// `cols` (`Positions`, `Vectors`, `Quaternions`, `Series`).
    pub(crate) fn matrix(&self, key: &str, cols: Option<usize>) -> Result<Array2<F>, ComputeError> {
        let value = self.require(key, "a shaped array")?;
        self.matrix_of(key, &value, cols)
    }

    /// An optional rank-2 real array.
    pub(crate) fn opt_matrix(
        &self,
        key: &str,
        cols: Option<usize>,
    ) -> Result<Option<Array2<F>>, ComputeError> {
        self.optional(key, |value| self.matrix_of(key, value, cols))
    }

    fn matrix_of(
        &self,
        key: &str,
        value: &JsValue,
        cols: Option<usize>,
    ) -> Result<Array2<F>, ComputeError> {
        let array = shaped(value).ok_or_else(|| self.fail(key, "a shaped array", value))?;
        let array = match (array.ndim(), cols) {
            (1, Some(cols)) => {
                let len = array.len();
                array
                    .into_shape_with_order(IxDyn(&[len / cols.max(1), cols]))
                    .map_err(|_| ComputeError::BadShape {
                        expected: format!("{}: a multiple of {cols} values", self.what(key)),
                        got: format!("{len} values"),
                    })?
            }
            _ => array,
        };
        array
            .into_dimensionality()
            .map_err(|_| self.fail(key, "a rank-2 array", value))
    }

    /// A rank-3 real array (`Trajectory`, `Field`).
    pub(crate) fn cube(&self, key: &str) -> Result<Array3<F>, ComputeError> {
        let value = self.require(key, "a rank-3 shaped array")?;
        self.cube_of(key, &value)
    }

    /// An optional rank-3 real array.
    pub(crate) fn opt_cube(&self, key: &str) -> Result<Option<Array3<F>>, ComputeError> {
        self.optional(key, |value| self.cube_of(key, value))
    }

    fn cube_of(&self, key: &str, value: &JsValue) -> Result<Array3<F>, ComputeError> {
        shaped(value)
            .and_then(|array| array.into_dimensionality().ok())
            .ok_or_else(|| self.fail(key, "a rank-3 shaped array", value))
    }

    /// Optional integer image flags of the given shape (`Images`).
    pub(crate) fn opt_images(
        &self,
        key: &str,
        shape: (usize, usize, usize),
    ) -> Result<Option<Array3<i32>>, ComputeError> {
        self.optional(key, |value| {
            let flat = integers(value).ok_or_else(|| self.fail(key, "an Int32Array", value))?;
            let got = flat.len();
            Array3::from_shape_vec(shape, flat).map_err(|_| ComputeError::DimensionMismatch {
                expected: shape.0 * shape.1 * shape.2,
                got,
                what: "image flags (the positions' T × N × 3)",
            })
        })
    }

    // ------------------------------------------------------------- handles

    /// A bound handle of class `H`.
    pub(crate) fn handle<H: TryFromJsValue>(
        &self,
        key: &str,
        class: &str,
    ) -> Result<H, ComputeError> {
        let value = self.require(key, class)?;
        handle(&value, &self.what(key), class)
    }

    /// An optional bound handle of class `H`.
    pub(crate) fn opt_handle<H: TryFromJsValue>(
        &self,
        key: &str,
        class: &str,
    ) -> Result<Option<H>, ComputeError> {
        self.optional(key, |value| handle(value, &self.what(key), class))
    }

    /// An array of bound handles of class `H`.
    pub(crate) fn handles<H: TryFromJsValue>(
        &self,
        key: &str,
        class: &str,
    ) -> Result<Vec<H>, ComputeError> {
        let value = self.require(key, &format!("an array of {class}"))?;
        handles(&value, &self.what(key), class)
    }

    /// The field `key` itself, for a reader that accepts several forms.
    pub(crate) fn value(&self, key: &str, expected: &str) -> Result<JsValue, ComputeError> {
        self.require(key, expected)
    }

    /// The optional field `key` itself.
    pub(crate) fn opt_value(&self, key: &str) -> Result<Option<JsValue>, ComputeError> {
        self.get(key)
    }

    /// What a field is called in an error message.
    pub(crate) fn name(&self, key: &str) -> String {
        self.what(key)
    }

    /// A box (`SimBox`).
    pub(crate) fn simbox(&self, key: &str) -> Result<SimBox, ComputeError> {
        Ok(self.handle::<JsBox>(key, "Box")?.inner)
    }

    /// An optional box.
    pub(crate) fn opt_simbox(&self, key: &str) -> Result<Option<SimBox>, ComputeError> {
        Ok(self.opt_handle::<JsBox>(key, "Box")?.map(|b| b.inner))
    }

    /// One box per frame (`Boxes`).
    pub(crate) fn boxes(&self, key: &str) -> Result<Vec<SimBox>, ComputeError> {
        Ok(self
            .handles::<JsBox>(key, "Box")?
            .into_iter()
            .map(|b| b.inner)
            .collect())
    }

    /// Optional boxes, one per frame.
    pub(crate) fn opt_boxes(&self, key: &str) -> Result<Option<Vec<SimBox>>, ComputeError> {
        if self.get(key)?.is_none() {
            return Ok(None);
        }
        self.boxes(key).map(Some)
    }

    /// A neighbour table (`Neighbors`, `PairSource`).
    pub(crate) fn neighbors(&self, key: &str) -> Result<Rc<RsNeighbors>, ComputeError> {
        Ok(self.handle::<Neighbors>(key, "Neighbors")?.inner)
    }

    /// One neighbour table per frame (`NeighborFrames`, `PairSourceFrames`).
    pub(crate) fn neighbor_frames(&self, key: &str) -> Result<Vec<Rc<RsNeighbors>>, ComputeError> {
        Ok(self
            .handles::<Neighbors>(key, "Neighbors")?
            .into_iter()
            .map(|n| n.inner)
            .collect())
    }
}

/// The records of a frame-form compute (`InputForm::Frames`): an array of
/// objects, each with the fields its spec lists.
pub(crate) fn records(frames: &JsValue, owner: &'static str) -> Result<Vec<JsValue>, ComputeError> {
    if Array::is_array(frames) {
        Ok(Array::from(frames).iter().collect())
    } else {
        Err(mismatch(
            &format!("{owner}.compute frames"),
            "an array of per-frame objects",
            frames,
        ))
    }
}

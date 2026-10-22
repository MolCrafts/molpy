//! Owned float multi-dimensional array for WASM-JS interop.
//!
//! [`NDArray`] bridges the gap between Rust's ndarray types and
//! JavaScript typed arrays by storing both the flat data and shape
//! metadata. It is used for bulk coordinate data (e.g., Nx3 positions)
//! that is too large or structured for individual typed-array columns.

use molrs::op::F;
use ndarray::{Array2, ArrayRef, ArrayView2, Dimension};
use wasm_bindgen::prelude::*;

pub(crate) type JsFloatArray = js_sys::Float64Array;

/// Owned float array with ndarray-compatible shape metadata.
///
/// Stores a flat `Vec<F>` together with a shape descriptor (e.g.,
/// `[N, 3]` for an Nx3 coordinate matrix). Used for passing
/// multi-dimensional numeric data across the WASM boundary.
///
/// # Memory layout
///
/// Data is stored in row-major (C) order, matching ndarray's default
/// and JavaScript's float typed-array convention.
///
/// # Example (JavaScript)
///
/// ```js
/// // Create a 2x3 zero array
/// const arr = new NDArray([2, 3]);
/// arr.copyFrom(floatArray);
///
/// // Or from existing data
/// const arr2 = NDArray.from(floatArray, [1, 3]);
///
/// // Get data back
/// const copy = arr.toCopy();       // safe owned copy
/// const view = arr.toTypedArray(); // zero-copy (invalidated on alloc)
/// ```
#[wasm_bindgen]
pub struct NDArray {
    data: Vec<F>,
    shape: Box<[usize]>,
}

#[wasm_bindgen]
impl NDArray {
    /// Create a zero-initialized array with the given shape.
    ///
    /// The total number of elements is the product of all dimensions.
    ///
    /// # Arguments
    ///
    /// * `shape` - Array of dimension sizes (e.g., `[10, 3]` for 10 rows x 3 columns)
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const coords = new NDArray([100, 3]); // 100 atoms, 3D
    /// console.log(coords.len()); // 300
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(shape: Box<[usize]>) -> Self {
        let len: usize = shape.iter().product();
        let data = vec![0.0; len];
        Self { data, shape }
    }

    /// Create a `NDArray` from an existing JS float typed array.
    ///
    /// # Arguments
    ///
    /// * `data` - Source float typed array (`Float64Array`)
    /// * `shape` - Optional shape. If omitted, defaults to `[data.length]` (1D).
    ///
    /// # Returns
    ///
    /// A new `NDArray` owning a copy of the data.
    ///
    /// # Errors
    ///
    /// Throws if `shape` product does not equal `data.length`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const arr = NDArray.from(floatArray, [2, 3]);
    /// console.log(arr.shape()); // [2, 3]
    /// ```
    #[wasm_bindgen(js_name = from)]
    pub fn from_js(data: &JsFloatArray, shape: Option<Box<[usize]>>) -> Result<NDArray, JsValue> {
        let shape = shape.unwrap_or_else(|| Box::new([data.length() as usize]));
        let expected: usize = shape.iter().product();
        if expected != data.length() as usize {
            return Err(JsValue::from_str(&format!(
                "Shape mismatch: shape product {} but data length {}",
                expected,
                data.length()
            )));
        }
        Ok(NDArray {
            data: data.to_vec(),
            shape,
        })
    }

    /// Return the total number of elements (product of all shape dimensions).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const arr = new NDArray([10, 3]);
    /// console.log(arr.len()); // 30
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check whether the array contains no elements.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a copy of the shape metadata as a JS array.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const s = arr.shape(); // e.g., [10, 3]
    /// ```
    pub fn shape(&self) -> Box<[usize]> {
        self.shape.clone()
    }

    /// An independent copy of this array.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> NDArray {
        NDArray {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }

    /// Return a raw pointer to the underlying data buffer.
    ///
    /// This is intended for advanced interop with other WASM modules
    /// that need direct memory access. The pointer is only valid as
    /// long as this `NDArray` is alive and no WASM memory growth
    /// has occurred.
    pub fn ptr(&self) -> *const F {
        self.data.as_ptr()
    }

    /// The element dtype, `"float"` (core `DType::Float.name()`; stored as
    /// `f64`), as `Block.dtype` names a float column.
    pub fn dtype(&self) -> String {
        molrs::core::DType::Float.name().to_string()
    }

    /// Overwrite the array contents from a JS float typed array.
    ///
    /// The source array must have exactly the same number of elements
    /// as this `NDArray` (i.e., the shape is preserved).
    ///
    /// # Arguments
    ///
    /// * `arr` - Source float typed array with matching length
    ///
    /// # Errors
    ///
    /// Throws if `arr.length` does not match this array's element count.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const wa = new NDArray([3]);
    /// wa.copyFrom(floatArray);
    /// ```
    #[wasm_bindgen(js_name = copyFrom)]
    pub fn copy_from(&mut self, arr: &JsFloatArray) -> Result<(), JsValue> {
        if arr.length() as usize != self.data.len() {
            return Err(JsValue::from_str(&format!(
                "Array length mismatch: expected {}, got {}",
                self.data.len(),
                arr.length()
            )));
        }
        arr.copy_to(&mut self.data);
        Ok(())
    }

    /// Zero-copy float typed-array view over this array's backing storage.
    ///
    /// **Warning**: The returned view becomes **invalid** if WASM linear
    /// memory grows (due to any allocation). Use [`toCopy`](NDArray::to_copy)
    /// if you need to keep the data.
    ///
    /// # Safety (internal)
    ///
    /// Uses the corresponding JS float typed-array `view` constructor,
    /// which creates an unowned view into
    /// WASM memory. The view must not outlive the `NDArray` and must
    /// not be used after any allocation that could trigger memory growth.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const view = arr.toTypedArray(); // use immediately
    /// // Do NOT allocate between view creation and use
    /// ```
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> JsFloatArray {
        // SAFETY:
        // - `self.data` is contiguous and lives in WASM linear memory.
        // - JS callers must treat this as a short-lived view.
        unsafe { JsFloatArray::view(self.data.as_slice()) }
    }

    /// Create an owned JS float typed-array copy of the data.
    ///
    /// The returned array is an independent copy that is safe to store
    /// and use regardless of subsequent WASM memory operations.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const copy = arr.toCopy(); // safe to keep indefinitely
    /// ```
    #[wasm_bindgen(js_name = toCopy)]
    pub fn to_copy(&self) -> JsFloatArray {
        JsFloatArray::from(self.data.as_slice())
    }

    /// Compute the sum of all elements.
    ///
    /// Primarily useful for quick sanity checks and testing.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const arr = NDArray.from(floatArray);
    /// console.log(arr.sum()); // 6.0
    /// ```
    pub fn sum(&self) -> F {
        self.data.iter().sum()
    }
}

// Internal methods not exposed to JavaScript
impl NDArray {
    pub(crate) fn from_vec(data: Vec<F>, shape: Box<[usize]>) -> Self {
        Self { data, shape }
    }

    /// An array of any rank, in row-major order.
    pub(crate) fn from_ndarray<D: Dimension>(array: &ArrayRef<F, D>) -> Self {
        Self {
            data: array.iter().copied().collect(),
            shape: array.shape().into(),
        }
    }

    /// The flat row-major data and the shape.
    pub(crate) fn into_parts(self) -> (Vec<F>, Box<[usize]>) {
        (self.data, self.shape)
    }

    pub(crate) fn as_array2(&self, rows: usize, cols: usize) -> Result<ArrayView2<'_, F>, String> {
        if rows * cols != self.data.len() {
            return Err(format!(
                "Shape mismatch: {}x{} = {} but data has {} elements",
                rows,
                cols,
                rows * cols,
                self.data.len()
            ));
        }
        ArrayView2::from_shape((rows, cols), &self.data)
            .map_err(|e| format!("Failed to create array view: {}", e))
    }

    pub(crate) fn from_array2(arr: Array2<F>) -> Self {
        let shape = Box::new([arr.nrows(), arr.ncols()]);
        let (data, _offset) = arr.into_raw_vec_and_offset();
        Self { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::{JsFloatArray, NDArray};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_array_basic_ops() {
        let mut view = NDArray::new(Box::new([2_usize, 3_usize]));
        assert_eq!(view.len(), 6);
        assert_eq!(&*view.shape(), &[2, 3]);

        let data = JsFloatArray::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]);
        view.copy_from(&data).expect("copy_from failed");
        assert!((view.sum() - 21.0).abs() < 1.0e-5);

        let js_array = view.to_copy();
        assert_eq!(js_array.length(), 6);
        assert!((js_array.get_index(0) - 1.0).abs() < 1.0e-5);
    }
}

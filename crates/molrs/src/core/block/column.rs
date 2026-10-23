//! Column representation for heterogeneous data: [`Column`] over [`ColumnArray`].

use std::any::Any;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;

use ndarray::ArrayD;

use num_complex::Complex;

use super::dtype::{BlockDtype, DType};
use crate::op::{F, I, Idx};

/// Walk every column variant, binding the inner column array.
macro_rules! map_column {
    ($col:expr, $holder:ident => $body:expr) => {
        match $col {
            Column::Float($holder) => $body,
            Column::I8($holder) => $body,
            Column::I16($holder) => $body,
            Column::Int($holder) => $body,
            Column::I64($holder) => $body,
            Column::U8($holder) => $body,
            Column::U16($holder) => $body,
            Column::U32($holder) => $body,
            Column::Uint($holder) => $body,
            Column::Bool($holder) => $body,
            Column::String($holder) => $body,
            Column::C64($holder) => $body,
            Column::C128($holder) => $body,
        }
    };
}

/// Wrapper around `ArrayD<T>` that optionally defers buffer ownership to a
/// foreign allocator.
///
/// # Storage model
///
/// [`Column`] variants wrap `Arc<ColumnArray<T>>` so cloning a Column is an
/// O(1) refcount bump rather than a deep copy. The `ColumnArray` is a thin
/// wrapper that makes the underlying `ArrayD<T>` either:
///
/// * **Rust-owned** — the normal path; `ArrayD<T>`'s backing `Vec<T>` was
///   allocated by Rust's allocator and is dropped normally when the column array
///   drops.
/// * **Foreign-borrowed** — the buffer was allocated by *some other* allocator
///   (e.g., numpy's). The column array fakes an `ArrayD<T>` pointing at that memory
///   and skips the `Vec::drop` on column array drop. Instead, it holds an opaque
///   "keep-alive" object (e.g., a `Py<PyArrayDyn<T>>`) whose own `Drop` is
///   responsible for releasing the memory via the foreign allocator.
///
/// Readers don't need to know which storage is active: they access the inner
/// array via `Deref<Target = ArrayD<T>>`, and `get::<T>()`, `shape()`, `view()`
/// etc work identically. Every mutable getter first detaches from foreign
/// storage (copy-on-write) before returning a mutable reference, so mutation
/// never reaches into foreign memory.
pub struct ColumnArray<T> {
    array: ManuallyDrop<ArrayD<T>>,
    /// Optional keep-alive for a foreign-allocated buffer.
    ///
    /// When `Some`, the `array` field's `Vec<T>` points at memory managed by
    /// some other allocator (e.g., numpy). On drop we must NOT run
    /// `Vec::drop` on that buffer; we drop the keeper instead, and trust that
    /// the keeper's own `Drop` releases the buffer through its native API.
    ///
    /// When `None`, the `array` is Rust-owned and drops normally.
    foreign_keeper: Option<Box<dyn Any + Send + Sync>>,
}

impl<T> ColumnArray<T> {
    /// Create a column array owning a Rust-allocated `ArrayD<T>`. This is the normal
    /// path: the column array drops the inner `ArrayD` normally.
    pub fn from_owned(arr: ArrayD<T>) -> Self {
        Self {
            array: ManuallyDrop::new(arr),
            foreign_keeper: None,
        }
    }

    /// Create a column array borrowing a foreign-allocated buffer.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    ///
    /// * The `arr`'s backing memory was allocated by the same allocator that
    ///   `keeper`'s `Drop` impl will call into. Typically the caller built
    ///   `arr` via `Vec::from_raw_parts(ptr, len, len)` where `ptr` was
    ///   obtained from `keeper` (e.g., numpy array data pointer).
    /// * The `keeper` keeps the foreign memory alive for at least as long as
    ///   this column array exists.
    /// * The foreign buffer will not be mutated or reallocated while any
    ///   reader holds a reference to it through this column array.
    ///
    /// When the column array drops, the inner `ArrayD`'s `Vec` is *not* dropped
    /// (which would invoke Rust's allocator on foreign memory — UB). Instead,
    /// `keeper` is dropped, and its own `Drop` impl releases the memory.
    pub unsafe fn from_foreign<K: Any + Send + Sync>(arr: ArrayD<T>, keeper: K) -> Self {
        Self {
            array: ManuallyDrop::new(arr),
            foreign_keeper: Some(Box::new(keeper)),
        }
    }

    /// Is this column array backed by foreign memory?
    pub fn is_foreign(&self) -> bool {
        self.foreign_keeper.is_some()
    }

    /// Direct reference to the inner `ArrayD<T>`.
    #[inline]
    pub fn array(&self) -> &ArrayD<T> {
        &self.array
    }
}

impl<T> Deref for ColumnArray<T> {
    type Target = ArrayD<T>;
    #[inline]
    fn deref(&self) -> &ArrayD<T> {
        &self.array
    }
}

impl<T: Clone> Clone for ColumnArray<T> {
    /// Cloning always produces a **Rust-owned** column array (deep-copies the inner
    /// array out of any foreign buffer). This is the fundamental guarantee
    /// that lets `Arc::make_mut`-style APIs return a `&mut ArrayD<T>` without
    /// risking mutation of foreign memory.
    fn clone(&self) -> Self {
        Self::from_owned(ArrayD::clone(&self.array))
    }
}

impl<T> Drop for ColumnArray<T> {
    fn drop(&mut self) {
        if let Some(keeper) = self.foreign_keeper.take() {
            // Foreign-backed: release the keep-alive, let its Drop free the
            // underlying buffer via the foreign allocator.
            // The inner `ArrayD`'s `Vec` is NOT dropped — that would call
            // Rust's allocator on foreign memory.
            drop(keeper);
        } else {
            // Rust-owned: drop the inner ArrayD normally.
            // SAFETY: `foreign_keeper` is None, so `array` is a real owned
            // ArrayD allocated by Rust's global allocator, and this is the
            // only place `array` is dropped.
            unsafe { ManuallyDrop::drop(&mut self.array) }
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for ColumnArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnArray")
            .field("shape", &self.array.shape())
            .field("foreign", &self.is_foreign())
            .finish()
    }
}

/// Internal enum representing a column of data in a Block.
///
/// Inner values are `Arc<ColumnArray<T>>` so `Column::clone()` and
/// `Block::clone()` are cheap: they bump refcounts instead of copying
/// scalar data. [`Column::from`] an owned `ArrayD<T>` wraps it in a
/// Rust-owned column array.
///
/// The typed getter [`get::<T>`](Self::get) returns a plain `&ArrayD<T>`
/// reference via deref; the Arc+column array layering is transparent to callers.
/// The mutable getter [`get_mut::<T>`](Self::get_mut) goes through
/// [`ColumnArray::clone`] copy-on-write so the
/// returned `&mut ArrayD<T>` always refers to Rust-owned memory.
#[derive(Clone)]
pub enum Column {
    /// Floating point column using the compute scalar [`F`] (`f64`).
    Float(Arc<ColumnArray<F>>),
    /// Signed 8-bit integer column.
    I8(Arc<ColumnArray<i8>>),
    /// Signed 16-bit integer column.
    I16(Arc<ColumnArray<i16>>),
    /// Signed integer column using the domain scalar [`I`] (`i32`).
    Int(Arc<ColumnArray<I>>),
    /// Signed 64-bit integer column.
    I64(Arc<ColumnArray<i64>>),
    /// Boolean column
    Bool(Arc<ColumnArray<bool>>),
    /// Unsigned integer column using the identifier scalar [`Idx`] (`u64`).
    Uint(Arc<ColumnArray<Idx>>),
    /// 8-bit unsigned integer column
    U8(Arc<ColumnArray<u8>>),
    /// 16-bit unsigned integer column.
    U16(Arc<ColumnArray<u16>>),
    /// 32-bit unsigned integer column.
    U32(Arc<ColumnArray<u32>>),
    /// String column
    String(Arc<ColumnArray<String>>),
    /// Complex pair of `f32` (numpy `complex64`).
    C64(Arc<ColumnArray<Complex<f32>>>),
    /// Complex pair of `f64` (numpy `complex128`).
    C128(Arc<ColumnArray<Complex<f64>>>),
}

/// Force an `Arc<ColumnArray<T>>` to be (1) Rust-owned and (2) uniquely
/// referenced, so callers can mutate the inner `ArrayD<T>` safely. Clones if
/// necessary. Returns `&mut ArrayD<T>`.
fn realize_owned_mut<T: Clone>(arc: &mut Arc<ColumnArray<T>>) -> &mut ArrayD<T> {
    // Step 1: if column array is foreign-backed, clone to detach from foreign memory.
    // Cloning always produces a Rust-owned column array (see ColumnArray::clone).
    if arc.is_foreign() {
        *arc = Arc::new((**arc).clone());
    }
    // Step 2: make Arc unique via Arc::make_mut (clones column array if shared, which
    // in turn deep-clones the ArrayD — preserving current CoW semantics).
    let holder = Arc::make_mut(arc);
    // Step 3: column array is now guaranteed Rust-owned AND uniquely referenced.
    // ManuallyDrop's DerefMut gives auto-deref to &mut ArrayD<T>.
    &mut holder.array
}

impl Column {
    /// Returns the number of rows (axis-0 length) of this column.
    ///
    /// Returns `None` if the array has rank 0 (which should never happen
    /// in a valid Block, as rank-0 arrays are rejected during insertion).
    pub fn n_rows(&self) -> Option<usize> {
        map_column!(self, a => a.shape().first().copied())
    }

    /// Returns the data type of this column.
    pub fn dtype(&self) -> DType {
        match self {
            Column::Float(_) => DType::Float,
            Column::I8(_) => DType::I8,
            Column::I16(_) => DType::I16,
            Column::Int(_) => DType::Int,
            Column::I64(_) => DType::I64,
            Column::Bool(_) => DType::Bool,
            Column::Uint(_) => DType::Uint,
            Column::U8(_) => DType::U8,
            Column::U16(_) => DType::U16,
            Column::U32(_) => DType::U32,
            Column::String(_) => DType::String,
            Column::C64(_) => DType::C64,
            Column::C128(_) => DType::C128,
        }
    }

    /// Returns the shape of the underlying array.
    pub fn shape(&self) -> &[usize] {
        map_column!(self, a => a.shape())
    }

    /// Owned little-endian byte buffer of this column's numeric backing store,
    /// in row-major (standard) layout.
    ///
    /// Returns `None` only for the [`Column::String`] variant, whose
    /// variable-length elements have no fixed byte representation. A strided or
    /// sliced numeric column is materialized into standard layout first, so any
    /// valid numeric column yields `Some` with length
    /// `product(shape) * size_of::<element>()`. `Bool` is emitted as one byte
    /// per element (`0`/`1`).
    pub fn raw_bytes(&self) -> Option<Vec<u8>> {
        fn le_numeric<T: Clone, const N: usize>(
            h: &ColumnArray<T>,
            to_bytes: impl Fn(&T) -> [u8; N],
        ) -> Vec<u8> {
            let a = h.array().as_standard_layout();
            let mut out = Vec::with_capacity(a.len() * N);
            for v in a.iter() {
                out.extend_from_slice(&to_bytes(v));
            }
            out
        }
        match self {
            Column::Float(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::I8(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::I16(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Int(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::I64(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Uint(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::U8(h) => Some(h.array().as_standard_layout().iter().copied().collect()),
            Column::U16(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::U32(h) => Some(le_numeric(h, |v| v.to_le_bytes())),
            Column::Bool(h) => Some(
                h.array()
                    .as_standard_layout()
                    .iter()
                    .map(|&b| b as u8)
                    .collect(),
            ),
            Column::String(_) => None,
            Column::C64(h) => Some(le_numeric(h, |v| {
                let mut bytes = [0u8; 8];
                bytes[..4].copy_from_slice(&v.re.to_le_bytes());
                bytes[4..].copy_from_slice(&v.im.to_le_bytes());
                bytes
            })),
            Column::C128(h) => Some(le_numeric(h, |v| {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&v.re.to_le_bytes());
                bytes[8..].copy_from_slice(&v.im.to_le_bytes());
                bytes
            })),
        }
    }

    /// Gather rows at `indices` (along axis 0) into a new owned Column of the
    /// same dtype. Backs [`Block::select_rows`](crate::core::Block::select_rows)
    /// and the sort path. String rows are cloned.
    pub fn select_rows(&self, indices: &[usize]) -> Column {
        use ndarray::Axis;
        map_column!(self, h => Column::from(h.array().select(Axis(0), indices)))
    }

    /// Whether rows `a` and `b` (along axis 0) hold equal values, element by
    /// element under the dtype's `==` (so a float `NaN` equals nothing).
    /// Validity is the caller's: a null row's stored value is compared as is.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is out of range, or the column is 0-dimensional.
    pub fn rows_equal(&self, a: usize, b: usize) -> bool {
        use ndarray::Axis;
        map_column!(self, h => h.array().index_axis(Axis(0), a) == h.array().index_axis(Axis(0), b))
    }

    /// Is this column backed by a foreign (non-Rust) buffer?
    pub fn is_foreign(&self) -> bool {
        map_column!(self, a => a.is_foreign())
    }

    /// A copy that shares no buffer with `self` or with a foreign owner.
    ///
    /// [`Clone`] is an `Arc` bump, which is a copy for every Rust writer
    /// (writes go through copy-on-write) but not for a writer that holds the
    /// buffer itself — a numpy view handed out by a binding, or the numpy array
    /// a foreign-backed column was forged from. This copies the elements into a
    /// new Rust-owned buffer.
    pub fn deep_copy(&self) -> Column {
        fn owned<T: Clone>(holder: &Arc<ColumnArray<T>>) -> Arc<ColumnArray<T>> {
            Arc::new(ColumnArray::clone(holder))
        }
        match self {
            Column::Float(h) => Column::Float(owned(h)),
            Column::I8(h) => Column::I8(owned(h)),
            Column::I16(h) => Column::I16(owned(h)),
            Column::Int(h) => Column::Int(owned(h)),
            Column::I64(h) => Column::I64(owned(h)),
            Column::Bool(h) => Column::Bool(owned(h)),
            Column::Uint(h) => Column::Uint(owned(h)),
            Column::U8(h) => Column::U8(owned(h)),
            Column::U16(h) => Column::U16(owned(h)),
            Column::U32(h) => Column::U32(owned(h)),
            Column::String(h) => Column::String(owned(h)),
            Column::C64(h) => Column::C64(owned(h)),
            Column::C128(h) => Column::C128(owned(h)),
        }
    }

    /// The array, when this column holds `T` elements; `None` for any other
    /// dtype.
    ///
    /// ```
    /// use molrs::core::Column;
    /// use ndarray::Array1;
    ///
    /// let col = Column::from(Array1::from_vec(vec![1.0_f64, 2.0]).into_dyn());
    /// assert_eq!(col.get::<f64>().unwrap()[[1]], 2.0);
    /// assert!(col.get::<i32>().is_none());
    /// ```
    pub fn get<T: BlockDtype>(&self) -> Option<&ArrayD<T>> {
        T::holder(self).map(|h| h.array())
    }

    /// The array, mutably, when this column holds `T` elements; `None` for any
    /// other dtype.
    ///
    /// Copy-on-write: a shared or foreign-backed buffer is cloned first, so the
    /// returned reference always refers to uniquely held Rust-owned memory.
    pub fn get_mut<T: BlockDtype>(&mut self) -> Option<&mut ArrayD<T>> {
        T::holder_mut(self).map(realize_owned_mut)
    }

    /// Resize this column along axis 0 to `new_nrows`.
    ///
    /// See the main doc on `Block::resize`. If the underlying column array is
    /// shared or foreign-backed, this replaces the column array with a fresh
    /// Rust-owned copy.
    pub fn resize(&mut self, new_nrows: usize) {
        let current = self.shape()[0];
        if new_nrows == current {
            return;
        }
        match self {
            Column::Float(a) => resize_array(a, current, new_nrows),
            Column::I8(a) => resize_array(a, current, new_nrows),
            Column::I16(a) => resize_array(a, current, new_nrows),
            Column::Int(a) => resize_array(a, current, new_nrows),
            Column::I64(a) => resize_array(a, current, new_nrows),
            Column::Uint(a) => resize_array(a, current, new_nrows),
            Column::U8(a) => resize_array(a, current, new_nrows),
            Column::U16(a) => resize_array(a, current, new_nrows),
            Column::U32(a) => resize_array(a, current, new_nrows),
            Column::Bool(a) => resize_array(a, current, new_nrows),
            Column::String(a) => resize_array(a, current, new_nrows),
            Column::C64(a) => resize_array(a, current, new_nrows),
            Column::C128(a) => resize_array(a, current, new_nrows),
        }
    }
}

/// Wrap an owned array in a Rust-owned column of `T`'s dtype.
impl<T: BlockDtype> From<ArrayD<T>> for Column {
    fn from(arr: ArrayD<T>) -> Self {
        T::wrap(Arc::new(ColumnArray::from_owned(arr)))
    }
}

/// Wrap a pre-built column array — the zero-copy path for a binding that
/// forged a foreign-backed one (see [`ColumnArray::from_foreign`]).
impl<T: BlockDtype> From<ColumnArray<T>> for Column {
    fn from(holder: ColumnArray<T>) -> Self {
        T::wrap(Arc::new(holder))
    }
}

fn resize_array<T: Clone + Default>(a: &mut Arc<ColumnArray<T>>, current: usize, new_nrows: usize) {
    use ndarray::{Axis, IxDyn, concatenate};
    let view = a.view();
    let new_arr = if new_nrows < current {
        view.slice_axis(Axis(0), (..new_nrows).into()).to_owned()
    } else {
        let mut pad_shape = a.shape().to_vec();
        pad_shape[0] = new_nrows - current;
        let pad = ArrayD::<T>::default(IxDyn(&pad_shape));
        concatenate(Axis(0), &[view, pad.view()]).unwrap()
    };
    *a = Arc::new(ColumnArray::from_owned(new_arr));
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column::{:?}(shape={:?})", self.dtype(), self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{F, I, Idx};
    use ndarray::{Array1, ArrayD};

    // ---- helpers ----

    fn float_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![0.0 as F; n]).into_dyn())
    }

    fn int_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![0 as I; n]).into_dyn())
    }

    fn bool_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![false; n]).into_dyn())
    }

    fn uint_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![0 as Idx; n]).into_dyn())
    }

    fn u8_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![0u8; n]).into_dyn())
    }

    fn string_col(n: usize) -> Column {
        Column::from(Array1::from_vec(vec![String::new(); n]).into_dyn())
    }

    // ---- deep_copy ----

    #[test]
    fn deep_copy_shares_no_buffer_and_keeps_the_values() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn());
        let shallow = col.clone();
        let deep = col.deep_copy();

        let (Column::Float(a), Column::Float(s), Column::Float(d)) = (&col, &shallow, &deep) else {
            panic!("float columns stay float");
        };
        assert!(Arc::ptr_eq(a, s), "clone is the Arc bump");
        assert!(!Arc::ptr_eq(a, d), "deep_copy owns a new holder");
        assert_ne!(a.array().as_ptr(), d.array().as_ptr());
        assert_eq!(
            d.array().as_slice_memory_order(),
            Some(&[1.0 as F, 2.0][..])
        );
        assert_eq!(deep.dtype(), DType::Float);
    }

    // ---- nrows / dtype / shape ----

    #[test]
    fn test_nrows() {
        assert_eq!(float_col(5).n_rows(), Some(5));
        assert_eq!(int_col(3).n_rows(), Some(3));
        assert_eq!(bool_col(7).n_rows(), Some(7));
        assert_eq!(uint_col(2).n_rows(), Some(2));
        assert_eq!(u8_col(4).n_rows(), Some(4));
        assert_eq!(string_col(1).n_rows(), Some(1));

        let rank0 = Column::from(ArrayD::<F>::from_elem(vec![], 1.0));
        assert_eq!(rank0.n_rows(), None);
    }

    #[test]
    fn test_dtype() {
        assert_eq!(float_col(1).dtype(), DType::Float);
        assert_eq!(int_col(1).dtype(), DType::Int);
        assert_eq!(bool_col(1).dtype(), DType::Bool);
        assert_eq!(uint_col(1).dtype(), DType::Uint);
        assert_eq!(u8_col(1).dtype(), DType::U8);
        assert_eq!(string_col(1).dtype(), DType::String);
    }

    #[test]
    fn test_shape() {
        assert_eq!(float_col(4).shape(), &[4]);
        let col2d = Column::from(ArrayD::<I>::from_elem(vec![3, 2], 0));
        assert_eq!(col2d.shape(), &[3, 2]);
    }

    // ---- typed accessors ----

    #[test]
    fn test_get_float_on_float() {
        let col = float_col(3);
        assert!(col.get::<f64>().is_some());
        assert_eq!(col.get::<f64>().unwrap().len(), 3);
    }

    #[test]
    fn test_get_float_on_wrong_type() {
        assert!(int_col(2).get::<f64>().is_none());
        assert!(bool_col(2).get::<f64>().is_none());
        assert!(uint_col(2).get::<f64>().is_none());
        assert!(u8_col(2).get::<f64>().is_none());
        assert!(string_col(2).get::<f64>().is_none());
    }

    #[test]
    fn test_get_int() {
        let col = int_col(4);
        assert!(col.get::<i32>().is_some());
        assert_eq!(col.get::<i32>().unwrap().len(), 4);
        assert!(float_col(1).get::<i32>().is_none());
        assert!(bool_col(1).get::<i32>().is_none());
    }

    #[test]
    fn test_get_bool() {
        let col = bool_col(2);
        assert!(col.get::<bool>().is_some());
        assert_eq!(col.get::<bool>().unwrap().len(), 2);
        assert!(float_col(1).get::<bool>().is_none());
        assert!(int_col(1).get::<bool>().is_none());
    }

    #[test]
    fn test_get_uint() {
        let col = uint_col(6);
        assert!(col.get::<u64>().is_some());
        assert_eq!(col.get::<u64>().unwrap().len(), 6);
        assert!(float_col(1).get::<u64>().is_none());
        assert!(int_col(1).get::<u64>().is_none());
    }

    #[test]
    fn test_get_u8() {
        let col = u8_col(3);
        assert!(col.get::<u8>().is_some());
        assert_eq!(col.get::<u8>().unwrap().len(), 3);
        assert!(float_col(1).get::<u8>().is_none());
        assert!(uint_col(1).get::<u8>().is_none());
    }

    #[test]
    fn test_get_string() {
        let col = string_col(2);
        assert!(col.get::<String>().is_some());
        assert_eq!(col.get::<String>().unwrap().len(), 2);
        assert!(float_col(1).get::<String>().is_none());
        assert!(int_col(1).get::<String>().is_none());
    }

    #[test]
    fn test_get_float_mut() {
        let mut col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn());
        {
            let arr = col.get_mut::<f64>().unwrap();
            arr[0] = 99.0;
        }
        let arr = col.get::<f64>().unwrap();
        assert!((arr[0] - 99.0).abs() < F::EPSILON);
        let mut int = int_col(1);
        assert!(int.get_mut::<f64>().is_none());
    }

    #[test]
    fn test_debug_format() {
        let dbg = format!("{:?}", float_col(3));
        assert!(dbg.contains("Column::Float"));
        assert!(dbg.contains("shape="));
    }

    // ---- Arc semantics ----

    #[test]
    fn test_clone_shares_buffer() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let cloned = col.clone();
        let p1 = col.get::<f64>().unwrap().as_ptr();
        let p2 = cloned.get::<f64>().unwrap().as_ptr();
        assert_eq!(p1, p2, "clone must share the buffer");
    }

    #[test]
    fn test_get_float_mut_cow_when_shared() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let mut cloned = col.clone();
        cloned.get_mut::<f64>().unwrap()[0] = 99.0;
        assert_eq!(col.get::<f64>().unwrap()[0], 1.0);
        assert_eq!(cloned.get::<f64>().unwrap()[0], 99.0);
    }

    // ---- Foreign column array correctness ----

    /// Test that a foreign-backed column array reads the right data without
    /// dropping the foreign memory (we use `Vec<f64>` itself as the foreign
    /// "allocator" — its Drop is what will free memory when the keeper drops).
    #[test]
    fn test_foreign_holder_readable() {
        // "Foreign" keeper = a Rust Vec, but we exercise the ManuallyDrop +
        // Drop path (keeper drops at end, Vec::drop runs via the keeper's own
        // Drop, not via ManuallyDrop::drop).
        let source: Vec<F> = vec![1.0, 2.0, 3.0, 4.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        // Forge ArrayD pointing at `source`'s memory. SAFETY: `source` outlives
        // the column array (kept alive as keeper). We construct cap=len so ndarray
        // won't try to grow/realloc.
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        let holder = unsafe { ColumnArray::from_foreign(forged, source) };
        let col = Column::from(holder);
        assert!(col.is_foreign());
        let arr = col.get::<f64>().unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_foreign_holder_cow_on_mut() {
        // Set up a foreign column array, then mutate. The mutation should detach
        // (CoW) and the source Vec should remain intact.
        let source: Vec<F> = vec![10.0, 20.0, 30.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        // Keep the source alive for the assertion below: store a clone in the
        // keeper, so the original Vec isn't consumed when we check it.
        let source_clone = source.clone();
        let holder = unsafe { ColumnArray::from_foreign(forged, source_clone) };
        let mut col = Column::from(holder);
        assert!(col.is_foreign());

        // Mutate the column. This should CoW into a Rust-owned column array.
        col.get_mut::<f64>().unwrap()[0] = 999.0;
        assert!(!col.is_foreign(), "after mut, holder must be Rust-owned");
        assert_eq!(col.get::<f64>().unwrap()[0], 999.0);

        // Source Vec is untouched.
        assert_eq!(source[0], 10.0);
    }

    #[test]
    fn test_foreign_holder_clone_detaches() {
        let source: Vec<F> = vec![7.0, 8.0];
        let ptr = source.as_ptr() as *mut F;
        let len = source.len();
        let forged = unsafe {
            let vec = Vec::from_raw_parts(ptr, len, len);
            ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
        };
        // The keepalive is the buffer the forged view points into.
        let holder = unsafe { ColumnArray::from_foreign(forged, source) };
        // Clone produces a Rust-owned column array.
        let holder_clone = holder.clone();
        assert!(holder.is_foreign());
        assert!(!holder_clone.is_foreign());
        assert_eq!(holder_clone.array().as_slice().unwrap(), &[7.0, 8.0]);
    }

    // ---- raw_bytes ----

    #[test]
    fn test_raw_bytes_lengths() {
        assert_eq!(float_col(3).raw_bytes().map(|b| b.len()), Some(24));
        assert_eq!(int_col(4).raw_bytes().map(|b| b.len()), Some(16));
        // Identity columns are 8 bytes wide: an identifier that wraps is not
        // an identifier, so `Idx` is 64-bit.
        assert_eq!(uint_col(2).raw_bytes().map(|b| b.len()), Some(16));
        assert_eq!(u8_col(5).raw_bytes().map(|b| b.len()), Some(5));
        assert_eq!(bool_col(7).raw_bytes().map(|b| b.len()), Some(7));
        assert!(string_col(3).raw_bytes().is_none());
    }

    #[test]
    fn test_raw_bytes_little_endian_values() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn());
        let b = col.raw_bytes().unwrap();
        assert_eq!(&b[0..8], &1.0f64.to_le_bytes());
        assert_eq!(&b[8..16], &2.0f64.to_le_bytes());
    }

    #[test]
    fn test_raw_bytes_multidim_length() {
        // A [2,3] int column flattens to 6 elements * 4 bytes.
        let col = Column::from(
            ArrayD::<I>::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![0 as I; 6]).unwrap(),
        );
        assert_eq!(col.raw_bytes().map(|b| b.len()), Some(24));
    }

    #[test]
    fn test_raw_bytes_strided_materializes() {
        // A strided (sliced) numeric column still yields Some, in logical order.
        let base = Array1::from_vec(vec![1.0 as F, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let strided = base.slice_move(ndarray::s![..;2]).into_dyn(); // [1, 3, 5]
        let col = Column::from(strided);
        let b = col.raw_bytes().unwrap();
        assert_eq!(b.len(), 24);
        assert_eq!(&b[0..8], &1.0f64.to_le_bytes());
        assert_eq!(&b[8..16], &3.0f64.to_le_bytes());
        assert_eq!(&b[16..24], &5.0f64.to_le_bytes());
    }
}

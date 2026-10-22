//! Zero-copy borrowed view of a [`Column`].
//!
//! `ColumnView<'a>` borrows the underlying ndarray data from a [`Column`] without
//! copying, providing read-only access with the same API surface as `Column`.

use ndarray::ArrayViewD;
use num_complex::Complex;

use super::column::Column;
use super::dtype::{BlockDtype, DType};
use crate::op::{F, I, Idx};

macro_rules! map_view {
    ($view:expr, $arr:ident => $body:expr) => {
        match $view {
            ColumnView::Float($arr) => $body,
            ColumnView::I8($arr) => $body,
            ColumnView::I16($arr) => $body,
            ColumnView::Int($arr) => $body,
            ColumnView::I64($arr) => $body,
            ColumnView::U8($arr) => $body,
            ColumnView::U16($arr) => $body,
            ColumnView::U32($arr) => $body,
            ColumnView::Uint($arr) => $body,
            ColumnView::Bool($arr) => $body,
            ColumnView::String($arr) => $body,
            ColumnView::C64($arr) => $body,
            ColumnView::C128($arr) => $body,
        }
    };
}

/// A borrowed, read-only view of a [`Column`].
///
/// Each variant holds an `ArrayViewD` that borrows from the corresponding
/// `ArrayD` inside an owned `Column`. No data is copied.
#[derive(Clone)]
pub enum ColumnView<'a> {
    /// Borrowed float column.
    Float(ArrayViewD<'a, F>),
    /// Borrowed `i8` column.
    I8(ArrayViewD<'a, i8>),
    /// Borrowed `i16` column.
    I16(ArrayViewD<'a, i16>),
    /// Borrowed signed integer column.
    Int(ArrayViewD<'a, I>),
    /// Borrowed `i64` column.
    I64(ArrayViewD<'a, i64>),
    /// Borrowed boolean column.
    Bool(ArrayViewD<'a, bool>),
    /// Borrowed unsigned integer column.
    Uint(ArrayViewD<'a, Idx>),
    /// Borrowed u8 column.
    U8(ArrayViewD<'a, u8>),
    /// Borrowed `u16` column.
    U16(ArrayViewD<'a, u16>),
    /// Borrowed `u32` column.
    U32(ArrayViewD<'a, u32>),
    /// Borrowed string column.
    String(ArrayViewD<'a, String>),
    /// Borrowed `complex64` column.
    C64(ArrayViewD<'a, Complex<f32>>),
    /// Borrowed `complex128` column.
    C128(ArrayViewD<'a, Complex<f64>>),
}

impl<'a> ColumnView<'a> {
    /// Returns the number of rows (axis-0 length) of this column view.
    ///
    /// Returns `None` if the array has rank 0.
    pub fn n_rows(&self) -> Option<usize> {
        map_view!(self, a => a.shape().first().copied())
    }

    /// Returns the data type of this column view.
    pub fn dtype(&self) -> DType {
        match self {
            ColumnView::Float(_) => DType::Float,
            ColumnView::I8(_) => DType::I8,
            ColumnView::I16(_) => DType::I16,
            ColumnView::Int(_) => DType::Int,
            ColumnView::I64(_) => DType::I64,
            ColumnView::Bool(_) => DType::Bool,
            ColumnView::Uint(_) => DType::Uint,
            ColumnView::U8(_) => DType::U8,
            ColumnView::U16(_) => DType::U16,
            ColumnView::U32(_) => DType::U32,
            ColumnView::String(_) => DType::String,
            ColumnView::C64(_) => DType::C64,
            ColumnView::C128(_) => DType::C128,
        }
    }

    /// Returns the shape of the underlying array view.
    pub fn shape(&self) -> &[usize] {
        map_view!(self, a => a.shape())
    }

    /// Reborrow this view for the lifetime of `self`.
    ///
    /// A `ColumnView` stored inside a [`BlockView`](super::BlockView) keeps the
    /// original column's lifetime. Callers that only hold `&self` need a view
    /// that ends with that borrow.
    pub fn reborrow(&self) -> ColumnView<'_> {
        match self {
            ColumnView::Float(a) => ColumnView::Float(a.view()),
            ColumnView::I8(a) => ColumnView::I8(a.view()),
            ColumnView::I16(a) => ColumnView::I16(a.view()),
            ColumnView::Int(a) => ColumnView::Int(a.view()),
            ColumnView::I64(a) => ColumnView::I64(a.view()),
            ColumnView::Bool(a) => ColumnView::Bool(a.view()),
            ColumnView::Uint(a) => ColumnView::Uint(a.view()),
            ColumnView::U8(a) => ColumnView::U8(a.view()),
            ColumnView::U16(a) => ColumnView::U16(a.view()),
            ColumnView::U32(a) => ColumnView::U32(a.view()),
            ColumnView::String(a) => ColumnView::String(a.view()),
            ColumnView::C64(a) => ColumnView::C64(a.view()),
            ColumnView::C128(a) => ColumnView::C128(a.view()),
        }
    }

    /// The borrowed array, when this view holds `T` elements; `None` for any
    /// other dtype. The read-only twin of [`Column::get`].
    pub fn get<T: BlockDtype>(&self) -> Option<ArrayViewD<'a, T>> {
        T::view_of(self)
    }

    /// Format one row as EXTXYZ property tokens.
    pub fn xyz_tokens(&self, row: usize) -> Vec<String> {
        use ndarray::Axis;
        match self {
            ColumnView::Bool(a) => a
                .index_axis(Axis(0), row)
                .iter()
                .map(|v| if *v { "T" } else { "F" }.to_string())
                .collect(),
            ColumnView::String(a) => a.index_axis(Axis(0), row).iter().cloned().collect(),
            _ => map_view!(self, a => {
                a.index_axis(Axis(0), row)
                    .iter()
                    .map(|v| v.to_string())
                    .collect()
            }),
        }
    }

    /// Creates an owned [`Column`] by cloning the viewed data.
    pub fn to_owned(&self) -> Column {
        map_view!(self, a => Column::from(a.to_owned()))
    }
}

impl<'a> From<&'a Column> for ColumnView<'a> {
    fn from(col: &'a Column) -> Self {
        match col {
            Column::Float(a) => ColumnView::Float(a.view()),
            Column::I8(a) => ColumnView::I8(a.view()),
            Column::I16(a) => ColumnView::I16(a.view()),
            Column::Int(a) => ColumnView::Int(a.view()),
            Column::I64(a) => ColumnView::I64(a.view()),
            Column::Bool(a) => ColumnView::Bool(a.view()),
            Column::Uint(a) => ColumnView::Uint(a.view()),
            Column::U8(a) => ColumnView::U8(a.view()),
            Column::U16(a) => ColumnView::U16(a.view()),
            Column::U32(a) => ColumnView::U32(a.view()),
            Column::String(a) => ColumnView::String(a.view()),
            Column::C64(a) => ColumnView::C64(a.view()),
            Column::C128(a) => ColumnView::C128(a.view()),
        }
    }
}

impl std::fmt::Debug for ColumnView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ColumnView::{:?}(shape={:?})",
            self.dtype(),
            self.shape()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_from_column_float() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Float);
        assert_eq!(view.n_rows(), Some(3));
        assert_eq!(view.shape(), &[3]);
        assert!(view.get::<f64>().is_some());
        assert!(view.get::<i32>().is_none());
    }

    #[test]
    fn test_from_column_int() {
        let col = Column::from(Array1::from_vec(vec![1 as I, 2, 3]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Int);
        assert!(view.get::<i32>().is_some());
        assert!(view.get::<f64>().is_none());
    }

    #[test]
    fn test_from_column_bool() {
        let col = Column::from(Array1::from_vec(vec![true, false]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Bool);
        assert!(view.get::<bool>().is_some());
    }

    #[test]
    fn test_from_column_uint() {
        let col = Column::from(Array1::from_vec(vec![1 as Idx, 2]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::Uint);
        assert!(view.get::<u64>().is_some());
    }

    #[test]
    fn test_from_column_u8() {
        let col = Column::from(Array1::from_vec(vec![1u8, 2]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::U8);
        assert!(view.get::<u8>().is_some());
    }

    #[test]
    fn test_from_column_string() {
        let col = Column::from(Array1::from_vec(vec!["a".to_string(), "b".to_string()]).into_dyn());
        let view = ColumnView::from(&col);
        assert_eq!(view.dtype(), DType::String);
        assert!(view.get::<String>().is_some());
    }

    #[test]
    fn test_to_owned_roundtrip() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        let owned = view.to_owned();
        assert_eq!(owned.dtype(), DType::Float);
        assert_eq!(owned.n_rows(), Some(3));
        assert_eq!(
            owned.get::<f64>().unwrap().as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_zero_copy() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn());
        let view = ColumnView::from(&col);
        // The view's data pointer should match the original column's data pointer
        let orig_ptr = col.get::<f64>().unwrap().as_ptr();
        let view_ptr = view.get::<f64>().unwrap().as_ptr();
        assert_eq!(orig_ptr, view_ptr);
    }

    #[test]
    fn projections_cover_the_narrow_and_complex_variants() {
        let i64_col = Column::from(Array1::from_vec(vec![1_i64, 2]).into_dyn());
        let i64_view = ColumnView::from(&i64_col);
        assert_eq!(i64_view.get::<i64>().unwrap().as_slice().unwrap(), &[1, 2]);
        assert!(i64_view.get::<i32>().is_none());

        let c = Column::from(Array1::from_vec(vec![Complex::new(1.0, 2.0)]).into_dyn());
        let view = ColumnView::from(&c);
        let arr = view.get::<num_complex::Complex<f64>>().unwrap();
        assert_eq!(arr[[0]].re, 1.0);
        assert!(view.get::<num_complex::Complex<f32>>().is_none());
    }
}

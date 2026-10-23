//! Data type enumeration and trait for Block columns.

use std::sync::Arc;

use ndarray::ArrayViewD;
use num_complex::Complex;

use super::column::{Column, ColumnArray};
use super::column_view::ColumnView;
use crate::op::{F, I, Idx};

/// Supported data types for Block columns.
///
/// [`name`](Self::name) is the one spelling of a dtype everywhere it is
/// written down — the serde wire, a `.mrec` schema, every binding (C
/// `MOLC_D_TYPE_<NAME>`, JS / Python the name strings): molrec's
/// concrete-width names `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
/// `u32`, `u64`, `bool`, `string`, `c64`, `c128`.
///
/// The variants for the domain aliases [`F`] / [`I`] / [`Idx`] are named
/// [`DType::Float`] / [`DType::Int`] / [`DType::Uint`]. Floats are always the compute scalar [`F`]
/// (`f64`): there is no `f16`/`f32` variant, and a narrow float on disk is
/// **refused**, not promoted. Every other variant is a storage width: a column
/// that arrived as `i64` has to leave as that width, not as the compute scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// Floating point using the compute scalar [`F`] (`f64`).
    Float,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed integer using the domain scalar [`I`] (`i32`).
    Int,
    /// Signed 64-bit integer.
    I64,
    /// Boolean
    Bool,
    /// Unsigned integer using the identifier scalar [`Idx`] (`u64`).
    Uint,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// String
    String,
    /// Complex pair of `f32` (numpy `complex64`).
    C64,
    /// Complex pair of `f64` (numpy `complex128`).
    C128,
}

impl DType {
    /// Every variant, in declaration order.
    pub const ALL: [DType; 13] = [
        DType::Float,
        DType::I8,
        DType::I16,
        DType::Int,
        DType::I64,
        DType::Bool,
        DType::Uint,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::String,
        DType::C64,
        DType::C128,
    ];

    /// The variant whose [`name`](Self::name) is `name`, or `None`.
    pub fn from_name(name: &str) -> Option<DType> {
        Self::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// The dtype's one spelling: molrec's concrete-width name (`f64`,
    /// `i32`, `u64`, `string`, …).
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::Int => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
            DType::Uint => "u64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::String => "string",
            DType::C64 => "c64",
            DType::C128 => "c128",
        }
    }

    /// Bytes one element of this dtype occupies in storage, or `None` for
    /// [`DType::String`].
    ///
    /// The widths are the ones [`Column::raw_bytes`] already emits — `Bool` is
    /// one byte per element, `Complex64` a pair of `f32`, `Complex128` a pair
    /// of `f64` — and the `None` mirrors the `None` that method returns for a
    /// string column: variable-length elements have no fixed byte width, so
    /// anything sizing storage by bytes has to branch on their absence.
    ///
    /// [`Column::raw_bytes`]: super::column::Column::raw_bytes
    pub fn itemsize(&self) -> Option<usize> {
        match self {
            DType::Float => Some(8),
            DType::I8 => Some(1),
            DType::I16 => Some(2),
            DType::Int => Some(4),
            DType::I64 => Some(8),
            DType::Bool => Some(1),
            DType::Uint => Some(8),
            DType::U8 => Some(1),
            DType::U16 => Some(2),
            DType::U32 => Some(4),
            DType::String => None,
            DType::C64 => Some(8),
            DType::C128 => Some(16),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Element types a Block column can hold.
///
/// One implementation per [`DType`] variant: [`F`], `i8`, `i16`, [`I`],
/// `i64`, `bool`, [`Idx`], `u8`, `u16`, `u32`, `String`, `Complex<f32>`,
/// `Complex<f64>`. It is what makes the typed column surface generic —
/// [`Column::from`] an `ArrayD<T>`, [`Column::get::<T>`](Column::get),
/// [`Column::get_mut::<T>`](Column::get_mut),
/// [`ColumnView::get::<T>`](super::ColumnView::get). The trait is sealed: the
/// set of dtypes is closed.
pub trait BlockDtype: Clone + Send + Sync + Sized + 'static + sealed::Storage {
    /// The [`DType`] of a column of this element type.
    fn dtype() -> DType;
}

pub(super) mod sealed {
    use std::sync::Arc;

    use ndarray::ArrayViewD;

    use super::super::column::{Column, ColumnArray};
    use super::super::column_view::ColumnView;

    /// The variant plumbing behind [`BlockDtype`](super::BlockDtype): which
    /// `Column` / `ColumnView` variant holds this element type.
    pub trait Storage: Sized {
        /// Wrap a column array in this type's `Column` variant.
        fn wrap(holder: Arc<ColumnArray<Self>>) -> Column;
        /// The column array, when `col` is this type's variant.
        fn holder(col: &Column) -> Option<&Arc<ColumnArray<Self>>>;
        /// The column array, mutably, when `col` is this type's variant.
        fn holder_mut(col: &mut Column) -> Option<&mut Arc<ColumnArray<Self>>>;
        /// The borrowed array, when `view` is this type's variant.
        fn view_of<'a>(view: &ColumnView<'a>) -> Option<ArrayViewD<'a, Self>>;
    }
}

macro_rules! impl_block_dtype {
    ($ty:ty, $variant:ident) => {
        impl BlockDtype for $ty {
            fn dtype() -> DType {
                DType::$variant
            }
        }
        impl sealed::Storage for $ty {
            fn wrap(holder: Arc<ColumnArray<Self>>) -> Column {
                Column::$variant(holder)
            }
            fn holder(col: &Column) -> Option<&Arc<ColumnArray<Self>>> {
                match col {
                    Column::$variant(h) => Some(h),
                    _ => None,
                }
            }
            fn holder_mut(col: &mut Column) -> Option<&mut Arc<ColumnArray<Self>>> {
                match col {
                    Column::$variant(h) => Some(h),
                    _ => None,
                }
            }
            fn view_of<'a>(view: &ColumnView<'a>) -> Option<ArrayViewD<'a, Self>> {
                match view {
                    ColumnView::$variant(a) => Some(a.clone()),
                    _ => None,
                }
            }
        }
    };
}

impl_block_dtype!(F, Float);
impl_block_dtype!(i8, I8);
impl_block_dtype!(i16, I16);
impl_block_dtype!(I, Int);
impl_block_dtype!(i64, I64);
impl_block_dtype!(bool, Bool);
impl_block_dtype!(Idx, Uint);
impl_block_dtype!(u8, U8);
impl_block_dtype!(u16, U16);
impl_block_dtype!(u32, U32);
impl_block_dtype!(String, String);
impl_block_dtype!(Complex<f32>, C64);
impl_block_dtype!(Complex<f64>, C128);

#[cfg(test)]
mod tests {
    use super::DType;

    /// Every fixed-width variant paired with the bytes one element occupies.
    ///
    /// These are the storage widths [`Column::raw_bytes`] already emits
    /// (`block/column.rs:428`): `Bool` is one byte per element, `Complex64` is
    /// a pair of `f32` and `Complex128` a pair of `f64`. The three domain
    /// aliases carry their concrete scalars from `core/types.rs` — `F = f64`
    /// (8), `I = i32` (4), `Idx = u64` (8) — not the width their name suggests.
    ///
    /// [`Column::raw_bytes`]: super::column::Column::raw_bytes
    const FIXED_WIDTH: [(DType, usize); 12] = [
        (DType::Float, 8),
        (DType::I8, 1),
        (DType::I16, 2),
        (DType::Int, 4),
        (DType::I64, 8),
        (DType::Bool, 1),
        (DType::Uint, 8),
        (DType::U8, 1),
        (DType::U16, 2),
        (DType::U32, 4),
        (DType::C64, 8),
        (DType::C128, 16),
    ];

    #[test]
    fn fixed_width_variants_report_their_byte_width() {
        for (dtype, width) in FIXED_WIDTH {
            assert_eq!(dtype.itemsize(), Some(width), "itemsize of {dtype}");
        }
    }

    #[test]
    fn all_lists_every_variant_once_in_order() {
        for (i, dtype) in DType::ALL.into_iter().enumerate() {
            assert_eq!(dtype as usize, i, "DType::ALL[{i}] is {dtype}");
        }
        // The last declared variant closes the list, so a variant appended
        // to the enum without joining `ALL` fails here.
        assert_eq!(DType::C128 as usize + 1, DType::ALL.len());
    }

    #[test]
    fn from_name_inverts_name() {
        for dtype in DType::ALL {
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
        }
        assert_eq!(DType::from_name("f32"), None);
    }

    #[test]
    fn string_has_no_itemsize() {
        // Variable-length elements have no fixed byte representation. This is
        // the same absence `Column::raw_bytes` reports for a string column, and
        // it is what a byte-target chunk planner has to branch on.
        assert_eq!(DType::String.itemsize(), None);
    }

    #[test]
    fn the_width_table_covers_every_variant() {
        // `DType` is `#[non_exhaustive]`, but inside the defining crate this
        // match is exhaustive: a new variant stops this module compiling until
        // someone decides whether it has a fixed width.
        fn is_fixed_width(dtype: DType) -> bool {
            match dtype {
                DType::String => false,
                DType::Float
                | DType::I8
                | DType::I16
                | DType::Int
                | DType::I64
                | DType::Bool
                | DType::Uint
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::C64
                | DType::C128 => true,
            }
        }

        let mut named: Vec<&str> = FIXED_WIDTH.iter().map(|(d, _)| d.name()).collect();
        named.sort_unstable();
        named.dedup();
        assert_eq!(
            named.len(),
            FIXED_WIDTH.len(),
            "FIXED_WIDTH lists a variant twice"
        );
        for (dtype, _) in FIXED_WIDTH {
            assert!(is_fixed_width(dtype), "{dtype} is not a fixed-width dtype");
        }
        // 12 fixed-width rows plus `String` is the whole enum, so a variant
        // missing from the table cannot hide behind the loop above.
        assert!(!is_fixed_width(DType::String));
    }
}

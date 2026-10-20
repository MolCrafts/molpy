//! Block: dict-like keyed arrays with consistent axis-0 length and heterogeneous types.

mod column;
mod dtype;
mod error;

mod access;
mod block_view;
mod column_view;

pub use access::BlockAccess;
pub use block_view::BlockView;
pub use column::{Column, ColumnArray};
pub use column_view::ColumnView;
pub use dtype::{BlockDtype, DType};
pub use error::BlockError;

use indexmap::IndexMap;
use std::ops::{Index, IndexMut};

/// A dictionary from string keys to ndarray arrays with a consistent axis-0 length.
///
/// This Block supports heterogeneous column types (float, int, bool).
///
/// `shape` is optional structural metadata that lets a Block declare itself
/// as N-dimensional (e.g. a 3D volumetric grid). Columns themselves are
/// stored row-major, so a grid block with `shape = [Nx, Ny, Nz]` carries
/// columns of axis-0 length `Nx * Ny * Nz` — `shape` only tells consumers
/// how to unflatten that index. When `shape` is `None`, the block is a
/// plain row table and `block.shape()` reports `vec![n_rows]`.
///
/// # Nullable columns
///
/// A [`Column`] is dense: every row carries a value of the column's type.
/// A column may keep, *beside* the values, a per-row validity mask saying
/// which of those values mean anything — the mask lives in a side map on the
/// block, not in the column, so a consumer that knows nothing about
/// nullability reads the filled values exactly as it did before. The mask is
/// attached to a column of any dtype when it is [inserted](Self::insert), or
/// afterwards with [`set_validity`](Self::set_validity). Ask [`validity`](Self::validity) for the mask; it is
/// `Some` iff at least one row of that column is null.
///
/// # Column order
///
/// Columns iterate in insertion order — the order a file or a caller wrote
/// them. Re-inserting an existing key replaces its column in place,
/// [`remove`](Self::remove) keeps the remaining columns in their relative
/// order, and [`rename_column`](Self::rename_column) keeps the renamed column
/// where it was.
///
/// # Examples
///
/// ```
/// use molrs::core::Block;
/// use molrs::op::{F, Idx};
/// use ndarray::Array1;
///
/// let mut block = Block::new();
///
/// // Insert different types - generic dispatch handles the conversion
/// let pos = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
/// let ids = Array1::from_vec(vec![10 as Idx, 20 as Idx, 30 as Idx]).into_dyn();
///
/// block.insert("pos", pos, None).unwrap();
/// block.insert("id", ids, None).unwrap();
///
/// // The column comes back whole. dtype is a property of that column.
/// let pos_ref = block.get("pos").and_then(|c| c.get::<f64>()).unwrap();
/// let ids_ref = block.get("id").and_then(|c| c.get::<u64>()).unwrap();
///
/// assert_eq!(block.n_rows(), Some(3));
/// assert_eq!(block.len(), 2);
/// ```
#[derive(Default, Clone)]
pub struct Block {
    map: IndexMap<String, Column>,
    /// Per-column validity masks, each of length `n_rows`. A column absent from
    /// this map is fully valid; see [`Block::insert`].
    validity: IndexMap<String, Vec<bool>>,
    /// Declared [precision](crate::core::check_precision) per `f64` column. A
    /// column absent from this map is stored as given.
    precision: IndexMap<String, f64>,
    /// Declared row-reference target per `u64` column (`targets`): the block
    /// its values index, `<block>` or `/<section>/<block>`.
    targets: IndexMap<String, String>,
    n_rows: Option<usize>,
    shape: Option<Vec<usize>>,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in &self.map {
            let mut dtype_shape = format!("{}(shape={:?})", v.dtype(), v.shape());
            if let Some(p) = self.precision.get(k) {
                dtype_shape.push_str(&format!(" precision={p}"));
            }
            match self.validity.get(k) {
                Some(mask) => {
                    let nulls = mask.iter().filter(|&&valid| !valid).count();
                    map.entry(k, &format!("{dtype_shape} nulls={nulls}"))
                }
                None => map.entry(k, &dtype_shape),
            };
        }
        map.finish()
    }
}

impl Block {
    /// Creates an empty Block.
    pub fn new() -> Self {
        Self {
            map: IndexMap::new(),
            validity: IndexMap::new(),
            precision: IndexMap::new(),
            targets: IndexMap::new(),
            n_rows: None,
            shape: None,
        }
    }

    /// Number of keys (columns).
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if there are no arrays in the block.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the common axis-0 length of all arrays, or `None` if empty.
    #[inline]
    pub fn n_rows(&self) -> Option<usize> {
        self.n_rows
    }

    /// Explicit N-D structural shape, if one was declared.
    ///
    /// `None` for a plain row table. Distinct from [`shape`](Self::shape),
    /// which reports `vec![n_rows]` when this is unset.
    #[inline]
    pub fn structural_shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }

    /// Returns the structural shape of the block.
    ///
    /// - For plain row tables (atoms, bonds): `vec![n_rows]` — a single
    ///   axis whose length is the row count.
    /// - For N-D blocks (volumetric grids): the explicitly-set shape,
    ///   e.g. `vec![Nx, Ny, Nz]`.
    /// - For empty blocks: `vec![]`.
    ///
    /// The product of the returned shape always equals `n_rows.unwrap_or(0)`.
    /// This API is uniform across atoms / bonds / grid blocks — the
    /// difference is the rank of the returned vector, not whether the
    /// accessor exists.
    pub fn shape(&self) -> Vec<usize> {
        match (&self.shape, self.n_rows) {
            (Some(s), _) => s.clone(),
            (None, Some(n)) => vec![n],
            (None, None) => Vec::new(),
        }
    }

    /// Declare this block as N-dimensional with the given `shape`.
    ///
    /// `shape` must have at least one axis and `shape.iter().product()`
    /// must equal the block's current `n_rows` (when the block has columns).
    /// This does **not** change column storage — columns remain row-major
    /// 1D buffers of length `product(shape)`. `shape` is structural
    /// metadata used by consumers (e.g. the volumetric renderer) to
    /// unflatten the row index back into N-D coordinates.
    ///
    /// Passing an empty slice clears the shape, reverting the block to
    /// plain-row-table semantics.
    ///
    /// Returns an error without changing the block if the product overflows
    /// `usize` or disagrees with the row count.
    pub fn set_shape(&mut self, shape: &[usize]) -> Result<(), BlockError> {
        if shape.is_empty() {
            self.shape = None;
            return Ok(());
        }
        let prod = shape
            .iter()
            .try_fold(1_usize, |n, &axis| n.checked_mul(axis))
            .ok_or_else(|| BlockError::validation("shape product overflows the row count"))?;
        if let Some(n_rows) = self.n_rows {
            if prod != n_rows {
                return Err(BlockError::validation(format!(
                    "shape product {} does not match block n_rows {}",
                    prod, n_rows
                )));
            }
        } else {
            // Block is empty — adopt n_rows = product(shape) so subsequent
            // inserts validate against the flattened length.
            self.n_rows = Some(prod);
        }
        self.shape = Some(shape.to_vec());
        Ok(())
    }

    /// Returns true if the Block contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Inserts a column under `key`, enforcing consistent axis-0 length.
    ///
    /// `col` is anything that converts into a [`Column`]: an owned
    /// `ArrayD<T>` of any [`BlockDtype`], a pre-built [`Column`] (the zero-copy
    /// path for moving a column between blocks), or a [`ColumnArray`].
    ///
    /// `validity` is the column's per-row validity mask, or `None` for a
    /// column whose every row holds a value. `mask[i] == false` marks row `i`
    /// as holding *no* value: the array still carries something at that row —
    /// whatever the caller put there, typically the type's default — and every
    /// reader that does not ask for the mask sees that filled value. An
    /// all-`true` mask states nothing `None` does not, so it is dropped rather
    /// than stored: [`validity`](Self::validity) returns `Some` **iff** at
    /// least one row of `key` is null.
    ///
    /// Re-inserting an existing key replaces its column outright, mask
    /// included.
    ///
    /// # Errors
    ///
    /// - [`BlockError::RankZero`] if the column has rank 0;
    /// - [`BlockError::ValidityLength`] if `validity` does not have exactly one
    ///   entry per row of the column;
    /// - [`BlockError::SchemaDtype`] / [`BlockError::SchemaShape`] if `key` is
    ///   a canonical column the value does not fit;
    /// - [`BlockError::RaggedAxis0`] if the column's axis-0 length doesn't
    ///   match the Block's existing `n_rows`.
    ///
    /// The block is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::{F, I, Idx};
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    ///
    /// // Insert float array
    /// let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
    /// block.insert("x", arr_float, None).unwrap();
    ///
    /// // Insert int array with same n_rows
    /// let arr_int = Array1::from_vec(vec![10 as Idx, 20 as Idx]).into_dyn();
    /// block.insert("id", arr_int, None).unwrap();
    ///
    /// // `id` is UInt in the Frame schema, so a signed column is refused —
    /// // the vocabulary binds the key wherever it appears, without the block
    /// // knowing whether it is `atoms` or `bonds`.
    /// let signed = Array1::from_vec(vec![10 as I, 20 as I]).into_dyn();
    /// assert!(block.insert("id", signed, None).is_err());
    ///
    /// // This would error - different n_rows
    /// let arr_bad = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
    /// assert!(block.insert("bad", arr_bad, None).is_err());
    ///
    /// // A nullable column: rows 1 and 2 hold no value.
    /// let frag = Array1::from_vec(vec![7 as I, 0]).into_dyn();
    /// block.insert("frag_id", frag, Some(vec![true, false])).unwrap();
    /// assert_eq!(block.validity("frag_id"), Some(&[true, false][..]));
    /// // The values stay readable: row 1 reads as the filled 0.
    /// assert_eq!(block.get("frag_id").and_then(|c| c.get::<I>()).unwrap()[[1]], 0);
    /// ```
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        col: impl Into<Column>,
        validity: Option<Vec<bool>>,
    ) -> Result<(), BlockError> {
        let key = key.into();
        let col = col.into();
        let Some(&len0) = col.shape().first() else {
            return Err(BlockError::RankZero { key });
        };
        if let Some(mask) = &validity
            && mask.len() != len0
        {
            return Err(BlockError::ValidityLength {
                key,
                expected: len0,
                got: mask.len(),
            });
        }
        check_schema(&key, col.dtype(), col.shape())?;

        match self.n_rows {
            // The first insertion defines n_rows.
            None => self.n_rows = Some(len0),
            Some(expected) if len0 != expected => {
                return Err(BlockError::RaggedAxis0 {
                    key,
                    expected,
                    got: len0,
                });
            }
            Some(_) => {}
        }

        let col = promote_canonical_uint(&key, col);
        // An insert replaces the column outright, mask included: the rows it
        // describes are gone.
        self.validity.shift_remove(&key);
        self.keep_precision_if_float(&key, col.dtype());
        self.keep_target_if_uint(&key, col.dtype());
        self.map.insert(key.clone(), col);
        if let Some(mask) = validity {
            self.put_validity(key, mask);
        }
        Ok(())
    }

    /// The validity mask of column `key`, or `None` when the column is absent
    /// or every one of its rows holds a value.
    ///
    /// `mask[i] == false` means row `i` holds no value; see
    /// [`insert`](Self::insert).
    #[inline]
    pub fn validity(&self, key: &str) -> Option<&[bool]> {
        self.validity.get(key).map(Vec::as_slice)
    }

    /// Attach `mask` as the validity mask of the already-inserted column
    /// `key`, whatever its dtype.
    ///
    /// This makes an already-inserted column nullable (or changes its mask):
    /// [`insert`](Self::insert) the values, then mark the rows that hold no
    /// value. `mask[i] == false`
    /// marks row `i` null. The mask replaces any mask the column had, and the
    /// normalisation [`validity`](Self::validity) documents applies: an
    /// all-`true` mask clears it.
    ///
    /// # Errors
    ///
    /// [`BlockError::MissingColumn`] if `key` names no column, and
    /// [`BlockError::ValidityLength`] if `mask` does not have exactly one
    /// entry per row. The block is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::{Block, Column};
    /// use ndarray::ArrayD;
    ///
    /// let mut block = Block::new();
    /// let labels = ArrayD::from_shape_vec(vec![2], vec!["a".to_string(), String::new()]).unwrap();
    /// block.insert("label", Column::from(labels), None).unwrap();
    /// block.set_validity("label", vec![true, false]).unwrap();
    /// assert_eq!(block.validity("label"), Some(&[true, false][..]));
    ///
    /// block.set_validity("label", vec![true, true]).unwrap(); // clears it
    /// assert_eq!(block.validity("label"), None);
    /// ```
    pub fn set_validity(&mut self, key: &str, mask: Vec<bool>) -> Result<(), BlockError> {
        if !self.map.contains_key(key) {
            return Err(BlockError::MissingColumn {
                key: key.to_owned(),
            });
        }
        let rows = self.n_rows.unwrap_or(0);
        if mask.len() != rows {
            return Err(BlockError::ValidityLength {
                key: key.to_owned(),
                expected: rows,
                got: mask.len(),
            });
        }
        self.put_validity(key.to_owned(), mask);
        Ok(())
    }

    /// Record `mask` for `key`, dropping it when it marks no row null — the
    /// normalisation [`validity`](Self::validity) documents.
    fn put_validity(&mut self, key: String, mask: Vec<bool>) {
        if mask.iter().all(|&valid| valid) {
            self.validity.shift_remove(&key);
        } else {
            self.validity.insert(key, mask);
        }
    }

    /// A declared precision survives a column being replaced by another `f64`
    /// column (the declaration is about the key's values, and a precision
    /// column rewritten in place is still one); any other dtype drops it.
    fn keep_precision_if_float(&mut self, key: &str, dtype: DType) {
        if dtype != DType::Float {
            self.precision.shift_remove(key);
        }
    }

    /// A declared target survives a column being replaced by another `u64`
    /// column; any other dtype drops it.
    fn keep_target_if_uint(&mut self, key: &str, dtype: DType) {
        if dtype != DType::Uint {
            self.targets.shift_remove(key);
        }
    }

    /// Declare that the `u64` column `key` holds 0-based row indices into
    /// `target`: `<block>` of the same frame, or `/<section>/<block>` of a
    /// frame-shaped section of the same record (molrec "row references").
    ///
    /// The relation endpoints `atomi` … `atoml` reference `atoms` without a
    /// declaration; a declaration overrides that default, and names the
    /// target of any other referencing column (`members.atom`). The
    /// declaration follows the column like a [precision](Self::set_precision)
    /// does, and is persisted as the block group's `targets` attribute.
    ///
    /// # Errors
    ///
    /// - [`BlockError::MissingColumn`] when the block has no column `key`;
    /// - [`BlockError::Validation`] when the column is not `u64`, or `target`
    ///   is neither `<block>` nor `/<section>/<block>`, or names a trajectory
    ///   block (`/trajectory/…`), whose row count is not fixed.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use ndarray::Array1;
    ///
    /// let mut members = Block::new();
    /// members.insert("ibead", Array1::from_vec(vec![0_u64, 0]).into_dyn(), None).unwrap();
    /// members.insert("atom", Array1::from_vec(vec![3_u64, 4]).into_dyn(), None).unwrap();
    /// members.set_target("atom", "/frame/atoms").unwrap();
    /// assert_eq!(members.target("atom"), Some("/frame/atoms"));
    /// assert!(members.set_target("atom", "/trajectory/atoms").is_err());
    /// ```
    pub fn set_target(&mut self, key: &str, target: &str) -> Result<(), BlockError> {
        let col = self.map.get(key).ok_or_else(|| BlockError::MissingColumn {
            key: key.to_owned(),
        })?;
        if col.dtype() != DType::Uint {
            return Err(BlockError::validation(format!(
                "column '{key}' is {}; a row reference is a u64 column",
                col.dtype()
            )));
        }
        crate::core::schema::check_target(target)
            .map_err(|e| BlockError::validation(format!("column '{key}': {e}")))?;
        self.targets.insert(key.to_owned(), target.to_owned());
        Ok(())
    }

    /// The declared target of column `key`, or `None` when it declares none.
    pub fn target(&self, key: &str) -> Option<&str> {
        self.targets.get(key).map(String::as_str)
    }

    /// Withdraw the declared target of column `key`, returning it.
    pub fn clear_target(&mut self, key: &str) -> Option<String> {
        self.targets.shift_remove(key)
    }

    /// Every declared target, as `(column, target)` in declaration order.
    pub fn targets(&self) -> impl Iterator<Item = (&str, &str)> {
        self.targets.iter().map(|(k, t)| (k.as_str(), t.as_str()))
    }

    /// Declare the [precision](crate::core::check_precision) of the `f64` column
    /// `key`: an absolute tolerance in the column's own units. A writer rounds
    /// the column's values to the binary grid it implies before storing them;
    /// the in-memory values are not touched.
    ///
    /// The declaration follows the column: a rename carries it, removing the
    /// column drops it, and replacing the column with one of another dtype
    /// drops it.
    ///
    /// # Errors
    ///
    /// - [`BlockError::MissingColumn`] when the block has no column `key`;
    /// - [`BlockError::Validation`] when the column is not `f64`, or when
    ///   `precision` is not finite and within `[2^-1000, 2^1000]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![0.1234_f64, 1.5]).into_dyn(), None).unwrap();
    /// block.set_precision("x", 1e-3).unwrap();
    /// assert_eq!(block.precision("x"), Some(1e-3));
    /// assert!(block.set_precision("x", 0.0).is_err());
    /// ```
    pub fn set_precision(&mut self, key: &str, precision: f64) -> Result<(), BlockError> {
        let col = self.map.get(key).ok_or_else(|| BlockError::MissingColumn {
            key: key.to_owned(),
        })?;
        if col.dtype() != DType::Float {
            return Err(BlockError::validation(format!(
                "column '{key}' is {}; only an f64 column declares a precision",
                col.dtype()
            )));
        }
        if !crate::core::precision::is_admissible(precision) {
            return Err(BlockError::validation(format!(
                "column '{key}': {}",
                crate::core::precision::inadmissible(precision)
            )));
        }
        self.precision.insert(key.to_owned(), precision);
        Ok(())
    }

    /// The declared precision of column `key`, or `None` when it declares
    /// none (or is absent).
    pub fn precision(&self, key: &str) -> Option<f64> {
        self.precision.get(key).copied()
    }

    /// Withdraw the declared precision of column `key`, returning it.
    pub fn clear_precision(&mut self, key: &str) -> Option<f64> {
        self.precision.shift_remove(key)
    }

    /// Every declared precision, as `(column, precision)` in declaration
    /// order.
    pub fn precisions(&self) -> impl Iterator<Item = (&str, f64)> {
        self.precision.iter().map(|(k, &p)| (k.as_str(), p))
    }

    /// New Block with rows gathered at `indices` (along axis 0), preserving the
    /// column set and dtypes. Errors if any index is out of range. This is the
    /// Rust-native row select/gather backing the Python `Block[rows]` path.
    ///
    /// Validity masks are gathered with their columns, so a null cell stays
    /// null wherever the gather moved it.
    pub fn select_rows(&self, indices: &[usize]) -> Result<Block, BlockError> {
        let n_rows = self.n_rows.unwrap_or(0);
        if let Some(&bad) = indices.iter().find(|&&i| i >= n_rows) {
            return Err(BlockError::validation(format!(
                "row index {bad} out of range (n_rows={n_rows})"
            )));
        }
        let mut out = Block::new();
        for (k, col) in &self.map {
            out.insert(k.clone(), col.select_rows(indices), None)?;
        }
        for (k, mask) in &self.validity {
            out.put_validity(k.clone(), indices.iter().map(|&i| mask[i]).collect());
        }
        out.precision = self.precision.clone();
        out.targets = self.targets.clone();
        out.n_rows = Some(indices.len());
        Ok(out)
    }

    /// New Block holding only the `keys` columns, with their validity masks. Errors naming the first key the block lacks — a
    /// selection never silently skips a column.
    pub fn select_columns(&self, keys: &[&str]) -> Result<Block, BlockError> {
        let mut out = Block::new();
        for &key in keys {
            let col = self
                .map
                .get(key)
                .ok_or_else(|| BlockError::validation(format!("column '{key}' not found")))?;
            out.insert(key.to_owned(), col.clone(), None)?;
            if let Some(mask) = self.validity.get(key) {
                out.put_validity(key.to_owned(), mask.clone());
            }
            if let Some(&p) = self.precision.get(key) {
                out.precision.insert(key.to_owned(), p);
            }
            if let Some(target) = self.targets.get(key) {
                out.targets.insert(key.to_owned(), target.clone());
            }
        }
        out.n_rows = self.n_rows;
        out.shape = self.shape.clone();
        Ok(out)
    }

    /// Row order that sorts the block by the `key` column (ascending, or the
    /// ascending order reversed when `reverse` — matching numpy's
    /// `argsort()[::-1]`). Floats use a total order (NaN at an extreme).
    pub fn sort_indices(&self, key: &str, reverse: bool) -> Result<Vec<usize>, BlockError> {
        let n_rows = self.n_rows.unwrap_or(0);
        let col = self
            .map
            .get(key)
            .ok_or_else(|| BlockError::validation(format!("sort key '{key}' not found")))?;
        let mut order = sort_order(col, n_rows);
        if reverse {
            order.reverse();
        }
        Ok(order)
    }

    /// New Block sorted by the `key` column (original unchanged).
    pub fn sort_by(&self, key: &str, reverse: bool) -> Result<Block, BlockError> {
        let order = self.sort_indices(key, reverse)?;
        self.select_rows(&order)
    }

    /// The column for `key`, or `None` when the key is absent.
    ///
    /// Project a dtype with [`Column::get::<T>`](Column::get). `None` from a
    /// projection means the column has a different dtype, which is
    /// not the same as a missing key.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Column> {
        self.map.get(key)
    }

    /// A mutable column for `key`, or `None` when the key is absent.
    ///
    /// Project a dtype with [`Column::get_mut::<T>`](Column::get_mut). `None`
    /// from a projection means the column has a different dtype.
    ///
    /// # Warning
    ///
    /// Mutating the column's shape through this reference is allowed but NOT
    /// revalidated. It's the caller's responsibility to maintain axis-0 consistency.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Column> {
        self.map.get_mut(key)
    }

    /// Removes and returns the column for `key`, if present.
    ///
    /// The column's validity mask goes with it.
    ///
    /// If the Block becomes empty after removal, resets `n_rows` and
    /// `shape` to `None`.
    pub fn remove(&mut self, key: &str) -> Option<Column> {
        self.validity.shift_remove(key);
        self.precision.shift_remove(key);
        self.targets.shift_remove(key);
        let out = self.map.shift_remove(key);
        if self.map.is_empty() {
            self.n_rows = None;
            self.shape = None;
        }
        out
    }

    /// Renames a column from `old_key` to `new_key`.
    ///
    /// Returns `true` if the column was successfully renamed, `false` if `old_key` doesn't exist
    /// or `new_key` already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None).unwrap();
    ///
    /// block.rename_column("x", "position_x").unwrap();
    /// assert!(!block.contains_key("x"));
    /// assert!(block.contains_key("position_x"));
    /// ```
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> Result<(), BlockError> {
        let Some(column) = self.map.get(old_key) else {
            return Err(BlockError::Validation {
                message: format!("cannot rename: no column '{old_key}'"),
            });
        };
        if self.map.contains_key(new_key) {
            return Err(BlockError::Validation {
                message: format!("cannot rename '{old_key}' to '{new_key}': target already exists"),
            });
        }
        // Renaming is a write into `new_key`, so the column being moved must
        // satisfy the target key's spec. Without this the whole vocabulary is
        // one rename away from being bypassed: write an int under an unspecified
        // key, then rename it onto `type`. The alias layer is built on this
        // method, which is exactly why it has to check.
        check_schema(new_key, column.dtype(), column.shape())?;

        let (index, _, column) = self.map.shift_remove_full(old_key).expect("checked above");
        self.map.shift_insert(index, new_key.to_string(), column);
        // The rows did not move, so neither did their nullability.
        if let Some((index, _, mask)) = self.validity.shift_remove_full(old_key) {
            self.validity.shift_insert(index, new_key.to_string(), mask);
        }
        if let Some((index, _, p)) = self.precision.shift_remove_full(old_key) {
            self.precision.shift_insert(index, new_key.to_string(), p);
        }
        if let Some((index, _, target)) = self.targets.shift_remove_full(old_key) {
            self.targets
                .shift_insert(index, new_key.to_string(), target);
        }
        Ok(())
    }

    /// Moves column `key` to position `index` among the columns; the columns
    /// between its old and new position shift by one, every other keeps its
    /// place.
    ///
    /// `Err(BlockError::Validation)` when there is no column `key` or `index`
    /// is not a column position.
    pub fn move_column(&mut self, key: &str, index: usize) -> Result<(), BlockError> {
        let Some(from) = self.map.get_index_of(key) else {
            return Err(BlockError::Validation {
                message: format!("cannot move: no column '{key}'"),
            });
        };
        if index >= self.map.len() {
            return Err(BlockError::Validation {
                message: format!(
                    "cannot move '{key}' to position {index}: the block has {} columns",
                    self.map.len()
                ),
            });
        }
        self.map.move_index(from, index);
        Ok(())
    }

    /// A copy whose columns share no buffer with this block.
    ///
    /// [`Clone`] shares every column (`Arc` bump); see [`Column::deep_copy`]
    /// for when that is not a copy. Validity masks, `n_rows` and the
    /// structural shape travel unchanged.
    pub fn deep_copy(&self) -> Block {
        Block {
            map: self
                .map
                .iter()
                .map(|(key, col)| (key.clone(), col.deep_copy()))
                .collect(),
            validity: self.validity.clone(),
            precision: self.precision.clone(),
            targets: self.targets.clone(),
            n_rows: self.n_rows,
            shape: self.shape.clone(),
        }
    }

    /// Clears the Block, removing all keys and resetting `n_rows` / `shape`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.validity.clear();
        self.precision.clear();
        self.targets.clear();
        self.n_rows = None;
        self.shape = None;
    }

    /// Returns an iterator over (&str, &Column).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|k| k.as_str())
    }

    /// Returns an iterator over column references.
    pub fn values(&self) -> impl Iterator<Item = &Column> {
        self.map.values()
    }

    /// Returns the data type of the column with the given key, if it exists.
    pub fn dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|c| c.dtype())
    }

    /// Resize all columns along axis 0 to `new_nrows`.
    ///
    /// - **Shrink** (`new_nrows` < current): slices each column to keep the first `new_nrows` rows.
    /// - **Grow** (`new_nrows` > current): extends each column with default values
    ///   (0.0 for Float, 0 for Int/UInt/U8, false for Bool, empty string for String).
    /// - **Same size**: no-op, returns `Ok(())`.
    /// - **Empty block** (no columns): sets `n_rows` without touching columns.
    ///
    /// Multi-dimensional columns (e.g. Nx3 positions) are resized only along
    /// axis 0; trailing dimensions are preserved.
    ///
    /// # Arguments
    /// * `new_nrows` - The desired number of rows after resize.
    ///
    /// # Returns
    /// * `Ok(())` on success.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut block = Block::new();
    /// block.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(), None).unwrap();
    ///
    /// block.resize(4).unwrap();
    /// assert_eq!(block.n_rows(), Some(4));
    /// let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
    /// assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
    /// ```
    pub fn resize(&mut self, new_nrows: usize) -> Result<(), crate::core::MolRsError> {
        if self.is_empty() {
            if self.n_rows != Some(new_nrows) {
                self.shape = None;
            }
            self.n_rows = Some(new_nrows);
            return Ok(());
        }

        let current = self.n_rows.unwrap_or(0);
        if new_nrows == current {
            return Ok(());
        }

        for col in self.map.values_mut() {
            col.resize(new_nrows);
        }
        // A grown row carries the type's default, which is the one thing a
        // mask exists to distinguish from a value, so it is grown as null;
        // a shrunk row's mask entry goes with the row.
        let masks: Vec<String> = self.validity.keys().cloned().collect();
        for key in masks {
            let mut mask = self.validity.shift_remove(&key).unwrap_or_default();
            mask.resize(new_nrows, false);
            self.put_validity(key, mask);
        }
        self.n_rows = Some(new_nrows);
        // N-D shape becomes meaningless once axis-0 row count is changed
        // by a 1D resize. Callers that want to preserve a grid shape must
        // re-declare it via `set_shape` after resizing.
        self.shape = None;
        Ok(())
    }

    /// Merge another block into this one by concatenating columns along axis-0.
    ///
    /// Both blocks must have the same set of column keys and matching dtypes.
    /// The resulting block will have n_rows = self.n_rows + other.n_rows.
    ///
    /// # Arguments
    /// * `other` - The block to merge into this one
    ///
    /// # Returns
    /// * `Ok(())` if merge succeeds
    /// * `Err(BlockError)` if blocks have incompatible columns
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut block1 = Block::new();
    /// block1.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(), None).unwrap();
    ///
    /// let mut block2 = Block::new();
    /// block2.insert("x", Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn(), None).unwrap();
    ///
    /// block1.merge(&block2).unwrap();
    /// assert_eq!(block1.n_rows(), Some(4));
    /// ```
    pub fn merge(&mut self, other: &Block) -> Result<(), BlockError> {
        if self.is_empty() && other.is_empty() {
            self.n_rows = self
                .n_rows
                .or(other.n_rows)
                .map(|_| self.n_rows.unwrap_or(0) + other.n_rows.unwrap_or(0));
            self.shape = None;
            return Ok(());
        }
        // An unsized empty block is an accumulator; a sized one still owns
        // rows and must obey the same-key contract rather than lose them.
        if other.is_empty() && other.n_rows.unwrap_or(0) == 0 {
            return Ok(());
        }
        if self.is_empty() && self.n_rows.unwrap_or(0) == 0 {
            self.adopt(other);
            return Ok(());
        }

        self.ensure_same_keys(other)?;
        for key in self.keys() {
            if self.target(key) != other.target(key) {
                return Err(BlockError::validation(format!(
                    "cannot merge column '{key}': conflicting row-reference targets"
                )));
            }
        }
        let new_map = self.concat_with(other)?;

        // Update n_rows. As with `resize`, an explicit N-D shape becomes
        // meaningless once axis-0 grows; the merged block falls back to a
        // plain row table unless the caller re-declares a shape.
        let self_rows = self.n_rows.unwrap();
        let other_rows = other.n_rows.unwrap();
        let new_nrows = self_rows + other_rows;
        self.map = new_map;
        self.n_rows = Some(new_nrows);
        self.shape = None;
        self.concat_validity(other, self_rows, other_rows);

        Ok(())
    }

    /// Row-wise union of `parts`: the rows of every part, in order, under the
    /// union of their columns.
    ///
    /// - **Columns** come out in first-seen order: every column of the first
    ///   part, then the columns only a later part introduces.
    /// - **A column a part lacks** is filled, for that part's rows, with the
    ///   dtype's default (`0`, `false`, `""`) *and marked null* in the
    ///   column's validity mask — the fill is a placeholder, not a value. A
    ///   part's own mask travels with its rows; a column that ends up with no
    ///   null row carries no mask.
    /// - **Row count** is the sum of the parts' [`n_rows`](Self::n_rows). A
    ///   part with no columns still contributes its declared rows (a block
    ///   [`resize`](Self::resize)d while empty), which are null in every
    ///   column.
    /// - The result is a plain row table: no part's structural shape
    ///   survives, as with [`merge`](Self::merge).
    ///
    /// Stacking zero parts gives an empty block. The parts are not modified;
    /// columns are copied into new buffers.
    ///
    /// # Errors
    ///
    /// - [`BlockError::StackDtype`] when a part carries a column under a
    ///   dtype other than the first carrier's. Widths are not unified: `i32`
    ///   beside `i64` is refused, since molrs never coerces a column.
    /// - [`BlockError::StackShape`] when a part's column has a different
    ///   per-row shape (the axes after axis 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut a = Block::new();
    /// a.insert("x", Array1::from_vec(vec![0.0 as F, 1.0]).into_dyn(), None).unwrap();
    /// a.insert("type", Array1::from_vec(vec!["A".to_string(), "B".into()]).into_dyn(), None).unwrap();
    /// let mut b = Block::new();
    /// b.insert("x", Array1::from_vec(vec![2.0 as F]).into_dyn(), None).unwrap();
    ///
    /// let s = Block::stack([&a, &b]).unwrap();
    /// assert_eq!(s.n_rows(), Some(3));
    /// assert_eq!(s.keys().collect::<Vec<_>>(), ["x", "type"]);
    /// // `b` had no `type`: its row is filled with "" and marked null.
    /// assert_eq!(s.get("type").and_then(|c| c.get::<String>()).unwrap()[[2]], "");
    /// assert_eq!(s.validity("type"), Some(&[true, true, false][..]));
    /// assert_eq!(s.validity("x"), None);
    /// ```
    pub fn stack<'a>(parts: impl IntoIterator<Item = &'a Block>) -> Result<Block, BlockError> {
        let parts: Vec<&Block> = parts.into_iter().collect();
        // Each key's first carrier, which fixes its dtype and per-row shape.
        let mut union: IndexMap<&str, &Column> = IndexMap::new();
        for (index, part) in parts.iter().enumerate() {
            for (key, col) in part.iter() {
                let Some(first) = union.get(key) else {
                    union.insert(key, col);
                    continue;
                };
                let first_part = parts.iter().find(|p| p.contains_key(key)).unwrap();
                if first_part.target(key) != part.target(key) {
                    return Err(BlockError::validation(format!(
                        "cannot stack column '{key}': part {index} has a conflicting row-reference target"
                    )));
                }
                if first.dtype() != col.dtype() {
                    return Err(BlockError::StackDtype {
                        key: key.to_owned(),
                        part: index,
                        expected: first.dtype(),
                        got: col.dtype(),
                    });
                }
                if first.shape()[1..] != col.shape()[1..] {
                    return Err(BlockError::StackShape {
                        key: key.to_owned(),
                        part: index,
                        expected: first.shape()[1..].to_vec(),
                        got: col.shape()[1..].to_vec(),
                    });
                }
            }
        }

        let rows: Vec<usize> = parts.iter().map(|b| b.n_rows.unwrap_or(0)).collect();
        let total: usize = rows.iter().sum();
        let mut out = Block::new();
        for (&key, &first) in &union {
            let mut pieces = Vec::with_capacity(parts.len());
            let mut mask: Option<Vec<bool>> = None;
            let mut offset = 0;
            for (part, &n) in parts.iter().zip(&rows) {
                let (piece, part_mask) = match part.get(key) {
                    Some(col) => (col.clone(), part.validity(key)),
                    None => {
                        let mut filler = first.select_rows(&[]);
                        filler.resize(n);
                        (filler, (n > 0).then_some(&[][..]))
                    }
                };
                pieces.push(piece);
                if let Some(part_mask) = part_mask {
                    let mask = mask.get_or_insert_with(|| vec![true; total]);
                    let span = &mut mask[offset..offset + n];
                    if part_mask.is_empty() {
                        span.fill(false);
                    } else {
                        span.copy_from_slice(part_mask);
                    }
                }
                offset += n;
            }
            out.insert(key, concat_columns(key, &pieces)?, None)?;
            if let Some(mask) = mask {
                out.put_validity(key.to_owned(), mask);
            }
            // The first part that declares a precision for the column speaks
            // for the stacked one.
            if let Some(p) = parts.iter().find_map(|part| part.precision(key)) {
                out.precision.insert(key.to_owned(), p);
            }
            if let Some(target) = parts.iter().find_map(|part| part.target(key)) {
                out.targets.insert(key.to_owned(), target.to_owned());
            }
        }
        if out.is_empty() && !parts.is_empty() {
            out.n_rows = Some(total);
        }
        Ok(out)
    }

    /// The `x` / `y` / `z` columns as one `N × 3` array, row `i` holding row
    /// `i`'s position.
    ///
    /// The array is a copy; write positions back with
    /// [`set_coords`](Self::set_coords).
    ///
    /// # Errors
    ///
    /// [`BlockError::MissingColumn`] naming the first of `x`, `y`, `z` the
    /// block lacks, and [`BlockError::SchemaDtype`] for one not stored as
    /// [`F`](crate::op::F).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::array;
    ///
    /// let mut block = Block::new();
    /// block.set_coords(array![[0.0 as F, 1.0, 2.0], [3.0, 4.0, 5.0]].view()).unwrap();
    /// assert_eq!(block.coords().unwrap(), array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    pub fn coords(&self) -> Result<crate::op::Fnx3, BlockError> {
        use crate::core::keys::COORDS;
        let n = self.n_rows.unwrap_or(0);
        let mut out = crate::op::Fnx3::zeros((n, 3));
        for (axis, key) in COORDS.into_iter().enumerate() {
            let col = self.get(key).ok_or_else(|| BlockError::MissingColumn {
                key: key.to_owned(),
            })?;
            let values = col.get::<f64>().ok_or_else(|| BlockError::SchemaDtype {
                key: key.to_owned(),
                expected: DType::Float,
                got: col.dtype(),
            })?;
            out.column_mut(axis)
                .iter_mut()
                .zip(values.iter())
                .for_each(|(d, &v)| *d = v);
        }
        Ok(out)
    }

    /// Write an `N × 3` array into the `x` / `y` / `z` columns.
    ///
    /// Each column is replaced by a new [`F`](crate::op::F) column (any
    /// validity mask it had goes with it) and keeps its position among the
    /// columns; a missing one is appended. The row count must match the
    /// block's, unless the block has no columns yet.
    ///
    /// # Errors
    ///
    /// - [`BlockError::Validation`] when `coords` does not have exactly three
    ///   columns.
    /// - [`BlockError::RaggedAxis0`] naming `x` when `coords` has a different
    ///   row count from the block's.
    ///
    /// The block is unchanged on error.
    pub fn set_coords(&mut self, coords: crate::op::Fnx3View<'_>) -> Result<(), BlockError> {
        use crate::core::keys::COORDS;
        if coords.ncols() != 3 {
            return Err(BlockError::validation(format!(
                "coordinates must be an N x 3 array, got shape {:?}",
                coords.shape()
            )));
        }
        // `x` goes first, so a row-count mismatch is refused before any
        // column is replaced.
        for (axis, key) in COORDS.into_iter().enumerate() {
            self.insert(key, coords.column(axis).to_owned().into_dyn(), None)?;
        }
        Ok(())
    }

    /// Become `other`: merging into an empty block adopts its columns, masks,
    /// row count and shape wholesale instead of concatenating anything.
    fn adopt(&mut self, other: &Block) {
        self.map = other.map.clone();
        self.validity = other.validity.clone();
        self.precision = other.precision.clone();
        self.targets = other.targets.clone();
        self.n_rows = other.n_rows;
        self.shape = other.shape.clone();
    }

    /// Reject operands that do not name the same columns, so that every merged
    /// column is built from two halves and none is silently dropped.
    fn ensure_same_keys(&self, other: &Block) -> Result<(), BlockError> {
        let self_keys: std::collections::HashSet<_> = self.keys().collect();
        let other_keys: std::collections::HashSet<_> = other.keys().collect();

        if self_keys != other_keys {
            return Err(BlockError::validation(format!(
                "Cannot merge blocks with different keys. Self has {:?}, other has {:?}",
                self_keys, other_keys
            )));
        }
        Ok(())
    }

    /// Concatenate every column with its namesake in `other`, keeping the
    /// all-or-nothing invariant: no map is returned unless every pair merged.
    fn concat_with(&self, other: &Block) -> Result<IndexMap<String, Column>, BlockError> {
        let mut new_map = IndexMap::with_capacity(self.map.len());
        for key in self.keys() {
            let pair = [self.map[key].clone(), other.map[key].clone()];
            new_map.insert(key.to_string(), concat_columns(key, &pair)?);
        }
        Ok(new_map)
    }

    /// Concatenate `other`'s validity masks onto `self`'s, mirroring the column
    /// concatenation [`merge`](Self::merge) just performed.
    ///
    /// A column masked on one side only is fully valid on the other, so that
    /// half of the joint mask is materialised as `true` rather than lost.
    fn concat_validity(&mut self, other: &Block, self_rows: usize, other_rows: usize) {
        let keys: Vec<String> = self.map.keys().cloned().collect();
        for key in keys {
            if !self.validity.contains_key(&key) && !other.validity.contains_key(&key) {
                continue;
            }
            let mut mask = match self.validity.get(&key) {
                Some(mask) => mask.clone(),
                None => vec![true; self_rows],
            };
            match other.validity.get(&key) {
                Some(tail) => mask.extend_from_slice(tail),
                None => mask.extend(std::iter::repeat_n(true, other_rows)),
            }
            self.put_validity(key, mask);
        }
    }
}

// Index trait for convenient access: block["key"]
impl Index<&str> for Block {
    type Output = Column;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

impl IndexMut<&str> for Block {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.get_mut(key)
            .unwrap_or_else(|| panic!("key '{}' not found in Block", key))
    }
}

/// Row order sorting one column ascending, keeping the invariant that every
/// dtype orders totally — floats by `total_cmp`, complex by (re, im).
fn sort_order(col: &Column, n_rows: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_rows).collect();
    match col {
        Column::Float(h) => {
            let v: Vec<crate::op::F> = h.array().iter().copied().collect();
            order.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        }
        Column::Int(h) => {
            let v: Vec<crate::op::I> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::I8(h) => {
            let v: Vec<i8> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::I16(h) => {
            let v: Vec<i16> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::I64(h) => {
            let v: Vec<i64> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::Uint(h) => {
            let v: Vec<crate::op::Idx> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::U8(h) => {
            let v: Vec<u8> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::U16(h) => {
            let v: Vec<u16> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::U32(h) => {
            let v: Vec<u32> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::Bool(h) => {
            let v: Vec<bool> = h.array().iter().copied().collect();
            order.sort_by_key(|&i| v[i]);
        }
        Column::String(h) => {
            let v: Vec<&String> = h.array().iter().collect();
            order.sort_by(|&i, &j| v[i].cmp(v[j]));
        }
        Column::C64(h) => {
            let v: Vec<_> = h.array().iter().copied().collect();
            order.sort_by(|&i, &j| {
                v[i].re
                    .total_cmp(&v[j].re)
                    .then_with(|| v[i].im.total_cmp(&v[j].im))
            });
        }
        Column::C128(h) => {
            let v: Vec<_> = h.array().iter().copied().collect();
            order.sort_by(|&i, &j| {
                v[i].re
                    .total_cmp(&v[j].re)
                    .then_with(|| v[i].im.total_cmp(&v[j].im))
            });
        }
    }
    order
}

/// Concatenate one key's pieces along axis-0, keeping the invariant that the
/// result carries the dtype every piece already shares.
///
/// The one concatenation both [`Block::merge`] (two pieces) and
/// [`Block::stack`] (one per part) run. `pieces` is non-empty.
fn concat_columns(key: &str, pieces: &[Column]) -> Result<Column, BlockError> {
    use ndarray::{Axis, concatenate};
    let dtype = pieces[0].dtype();
    if let Some(other) = pieces.iter().find(|c| c.dtype() != dtype) {
        return Err(BlockError::validation(format!(
            "Column '{key}' has incompatible dtypes: {dtype:?} vs {:?}",
            other.dtype()
        )));
    }
    macro_rules! cat {
        ($variant:ident, $into:path) => {{
            let views: Vec<_> = pieces
                .iter()
                .map(|c| match c {
                    Column::$variant(h) => h.view(),
                    _ => unreachable!("dtypes checked above"),
                })
                .collect();
            concatenate(Axis(0), &views).map($into)
        }};
    }
    let merged = match &pieces[0] {
        Column::Float(_) => cat!(Float, Column::from),
        Column::Int(_) => cat!(Int, Column::from),
        Column::I8(_) => cat!(I8, Column::from),
        Column::I16(_) => cat!(I16, Column::from),
        Column::I64(_) => cat!(I64, Column::from),
        Column::Uint(_) => cat!(Uint, Column::from),
        Column::U8(_) => cat!(U8, Column::from),
        Column::U16(_) => cat!(U16, Column::from),
        Column::U32(_) => cat!(U32, Column::from),
        Column::Bool(_) => cat!(Bool, Column::from),
        Column::String(_) => cat!(String, Column::from),
        Column::C64(_) => cat!(C64, Column::from),
        Column::C128(_) => cat!(C128, Column::from),
    };
    merged.map_err(|e| {
        BlockError::validation(format!("Failed to concatenate {dtype} column '{key}': {e}"))
    })
}

/// Reject a write that violates the canonical column vocabulary.
///
/// Keyed by column name alone, which is what lets this fire from `Block` —
/// a standalone block does not know whether it is about to become `atoms` or
/// `bonds`, but `atomi` is `UInt` in either. See
/// [`crate::core::schema`] for why a key that seems to need two dtypes is
/// two keys.
/// Identifiers (`id`, `atomi`, `type_id`, …) are [`Idx`]. A caller that
/// hands us a narrower unsigned array is naming the same quantity; store it
/// at identifier width so `Column::get::<u64>` and the writers that consume it agree.
///
/// This leniency is for the in-memory API only. The `*.mrec` store readers
/// refuse a canonical identifier stored at another width before it reaches
/// here, so a non-conforming store is reported rather than silently widened.
fn promote_canonical_uint(key: &str, col: Column) -> Column {
    use crate::op::Idx;
    let Some(spec) = crate::core::schema::column(key) else {
        return col;
    };
    if spec.dtype != DType::Uint {
        return col;
    }
    match col {
        Column::Uint(_) => col,
        Column::U8(h) => Column::from(h.array().mapv(Idx::from)),
        Column::U16(h) => Column::from(h.array().mapv(Idx::from)),
        Column::U32(h) => Column::from(h.array().mapv(Idx::from)),
        other => other,
    }
}

fn schema_dtype_admits(expected: DType, got: DType) -> bool {
    // The vocabulary names a quantity, not a storage width. `x` is a float
    // coordinate, stored as `F` — narrow floats are refused before they reach
    // a column, so a column is never f16/f32. Silent *family* changes (float
    // to int) stay illegal.
    match expected {
        DType::Float => got == DType::Float,
        DType::Int => matches!(got, DType::Int | DType::I8 | DType::I16 | DType::I64),
        DType::Uint => matches!(got, DType::Uint | DType::U8 | DType::U16 | DType::U32),
        DType::C64 | DType::C128 => {
            matches!(got, DType::C64 | DType::C128)
        }
        other => other == got,
    }
}

fn check_schema(key: &str, dtype: DType, shape: &[usize]) -> Result<(), BlockError> {
    let Some(spec) = crate::core::schema::column(key) else {
        return Ok(());
    };
    if !schema_dtype_admits(spec.dtype, dtype) {
        return Err(BlockError::SchemaDtype {
            key: key.to_string(),
            expected: spec.dtype,
            got: dtype,
        });
    }
    if !spec.shape.admits(shape) {
        return Err(BlockError::SchemaShape {
            key: key.to_string(),
            expected: spec.shape,
            got: shape.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_overflowing_structural_shape_is_rejected_without_changing_the_block() {
        let mut block = super::Block::new();
        block.set_shape(&[2, 3]).unwrap();
        assert!(block.set_shape(&[usize::MAX, 2]).is_err());
        assert_eq!(block.n_rows(), Some(6));
        assert_eq!(block.structural_shape(), Some(&[2, 3][..]));
    }
    use super::*;
    use crate::op::{F, I, Idx};
    use ndarray::{Array1, ArrayD};

    /// Block with float columns `c`, `a`, `b` inserted in that order, one row
    /// each, `a` masked null.
    fn cab() -> Block {
        let mut b = Block::new();
        b.insert("c", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        b.insert(
            "a",
            Array1::from_vec(vec![0.0 as F]).into_dyn(),
            Some(vec![false]),
        )
        .unwrap();
        b.insert("b", Array1::from_vec(vec![3.0 as F]).into_dyn(), None)
            .unwrap();
        b
    }

    #[test]
    fn keys_follow_insertion_order() {
        assert_eq!(cab().keys().collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    #[test]
    fn remove_keeps_the_remaining_keys_in_order() {
        let mut b = cab();
        b.insert("d", Array1::from_vec(vec![4.0 as F]).into_dyn(), None)
            .unwrap();
        b.remove("a");
        // shift_remove keeps [c, b, d]; swap_remove would give [c, d, b].
        assert_eq!(b.keys().collect::<Vec<_>>(), ["c", "b", "d"]);
    }

    #[test]
    fn reinserting_a_key_keeps_its_position() {
        let mut b = cab();
        b.insert("c", Array1::from_vec(vec![9.0 as F]).into_dyn(), None)
            .unwrap();
        assert_eq!(b.keys().collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    #[test]
    fn rename_column_keeps_the_column_in_place() {
        let mut b = cab();
        b.rename_column("a", "z").unwrap();
        assert_eq!(b.keys().collect::<Vec<_>>(), ["c", "z", "b"]);
    }

    #[test]
    fn move_column_shifts_only_the_columns_between() {
        let mut b = cab();
        b.move_column("b", 0).unwrap();
        assert_eq!(b.keys().collect::<Vec<_>>(), ["b", "c", "a"]);
        b.move_column("b", 2).unwrap();
        assert_eq!(b.keys().collect::<Vec<_>>(), ["c", "a", "b"]);
        assert!(b.move_column("zz", 0).is_err());
        assert!(b.move_column("b", 3).is_err());
    }

    #[test]
    fn row_select_copy_and_merge_keep_column_order() {
        let b = cab();
        let order = ["c", "a", "b"];
        assert_eq!(
            b.select_rows(&[0]).unwrap().keys().collect::<Vec<_>>(),
            order
        );
        assert_eq!(b.deep_copy().keys().collect::<Vec<_>>(), order);
        let mut merged = cab();
        merged.merge(&b).unwrap();
        assert_eq!(merged.keys().collect::<Vec<_>>(), order);
        assert_eq!(merged.validity("a"), Some(&[false, false][..]));
    }

    #[test]
    fn select_columns_follows_the_requested_order() {
        let picked = cab().select_columns(&["b", "c"]).unwrap();
        assert_eq!(picked.keys().collect::<Vec<_>>(), ["b", "c"]);
    }

    #[test]
    fn deep_copy_keeps_masks_rows_and_shape_on_new_buffers() {
        let mut b = Block::new();
        b.insert(
            "tag",
            Array1::from_vec(vec![1 as I, 0, 3, 0]).into_dyn(),
            Some(vec![true, false, true, false]),
        )
        .unwrap();
        b.set_shape(&[2, 2]).unwrap();

        let copy = b.deep_copy();

        assert_eq!(copy.validity("tag"), Some(&[true, false, true, false][..]));
        assert_eq!(copy.n_rows(), Some(4));
        assert_eq!(copy.structural_shape(), Some(&[2, 2][..]));
        assert_ne!(
            b.get("tag").and_then(|c| c.get::<i32>()).unwrap().as_ptr(),
            copy.get("tag")
                .and_then(|c| c.get::<i32>())
                .unwrap()
                .as_ptr()
        );
        assert_eq!(
            copy.get("tag")
                .and_then(|c| c.get::<i32>())
                .unwrap()
                .as_slice_memory_order(),
            Some(&[1 as I, 0, 3, 0][..])
        );
    }

    #[test]
    fn select_columns_keeps_exactly_the_named_columns() {
        let mut b = Block::new();
        b.insert("id", Array1::from_vec(vec![1 as Idx, 2]).into_dyn(), None)
            .unwrap();
        b.insert("x", Array1::from_vec(vec![1.5 as F, 2.5]).into_dyn(), None)
            .unwrap();
        b.insert(
            "charge",
            Array1::from_vec(vec![0.1 as F, -0.1]).into_dyn(),
            None,
        )
        .unwrap();

        let sel = b.select_columns(&["x", "id"]).unwrap();
        let mut keys: Vec<&str> = sel.keys().collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["id", "x"]);
        assert_eq!(sel.n_rows(), Some(2));

        let err = b.select_columns(&["x", "mass"]).unwrap_err();
        assert!(err.to_string().contains("'mass'"), "{err}");
    }

    #[test]
    fn test_select_rows_and_sort() {
        let mut b = Block::new();
        b.insert(
            "id",
            Array1::from_vec(vec![3 as Idx, 1, 2]).into_dyn(),
            None,
        )
        .unwrap();
        b.insert(
            "x",
            Array1::from_vec(vec![3.5 as F, 1.5, 2.5]).into_dyn(),
            None,
        )
        .unwrap();

        // select_rows gathers in order.
        let sel = b.select_rows(&[2, 0]).unwrap();
        assert_eq!(
            sel.get("id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![2, 3]
        );

        // out-of-range index errors.
        assert!(b.select_rows(&[5]).is_err());

        // sort by id ascending reorders all columns.
        let s = b.sort_by("id", false).unwrap();
        assert_eq!(
            s.get("id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            s.get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1.5, 2.5, 3.5]
        );

        // reverse = ascending reversed.
        let r = b.sort_by("id", true).unwrap();
        assert_eq!(
            r.get("id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );

        // unknown sort key errors.
        assert!(b.sort_by("nope", false).is_err());
    }

    #[test]
    fn test_insert_mixed_dtypes() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
        let arr_float_2 = Array1::from_vec(vec![4.0 as F, 5.0 as F, 6.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I, 20, 30]).into_dyn();
        let arr_bool = Array1::from_vec(vec![true, false, true]).into_dyn();

        assert!(block.insert("x", arr_float, None).is_ok());
        assert!(block.insert("y", arr_float_2, None).is_ok());
        assert!(block.insert("count", arr_i64, None).is_ok());
        assert!(block.insert("mask", arr_bool, None).is_ok());

        assert_eq!(block.len(), 4);
        assert_eq!(block.n_rows(), Some(3));
    }

    #[test]
    fn test_axis0_mismatch_error() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();

        block.insert("x", arr1, None).unwrap();
        let result = block.insert("y", arr2, None);

        assert!(result.is_err());
        match result {
            Err(BlockError::RaggedAxis0 { expected, got, .. }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected RaggedAxis0 error"),
        }
    }

    #[test]
    fn test_typed_getters() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I, 20]).into_dyn();

        block.insert("x", arr_float, None).unwrap();
        block.insert("count", arr_i64, None).unwrap();

        // Correct type access
        assert!(block.get("x").and_then(|c| c.get::<f64>()).is_some());
        assert!(block.get("count").and_then(|c| c.get::<i32>()).is_some());

        // Wrong type access returns None
        assert!(block.get("x").and_then(|c| c.get::<i32>()).is_none());
        assert!(block.get("count").and_then(|c| c.get::<f64>()).is_none());

        // Mutable access
        if let Some(x_mut) = block.get_mut("x").and_then(|c| c.get_mut::<f64>()) {
            x_mut[[0]] = 99.0;
        }
        assert_eq!(
            block.get("x").and_then(|c| c.get::<f64>()).unwrap()[[0]],
            99.0
        );
    }

    #[test]
    fn test_index_access() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block.insert("x", arr, None).unwrap();

        // Immutable index
        let col = &block["x"];
        assert_eq!(col.dtype(), DType::Float);

        // Mutable index
        let col_mut = &mut block["x"];
        if let Some(arr_mut) = col_mut.get_mut::<f64>() {
            arr_mut[[0]] = 42.0;
        }
        assert_eq!(
            block.get("x").and_then(|c| c.get::<f64>()).unwrap()[[0]],
            42.0
        );
    }

    #[test]
    #[should_panic(expected = "key 'missing' not found")]
    fn test_index_panic_on_missing_key() {
        let block = Block::new();
        let _ = &block["missing"];
    }

    #[test]
    fn test_remove_resets_nrows() {
        let mut block = Block::new();

        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block.insert("x", arr, None).unwrap();

        assert_eq!(block.n_rows(), Some(2));

        block.remove("x");
        assert_eq!(block.n_rows(), None);
        assert!(block.is_empty());
    }

    #[test]
    fn test_iter_keys_values() {
        let mut block = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![10 as I, 20]).into_dyn();

        block.insert("x", arr1, None).unwrap();
        block.insert("count", arr2, None).unwrap();

        let keys: Vec<&str> = block.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"x"));
        assert!(keys.contains(&"count"));

        let dtypes: Vec<DType> = block.values().map(|c| c.dtype()).collect();
        assert!(dtypes.contains(&DType::Float));
        assert!(dtypes.contains(&DType::Int));
    }

    #[test]
    fn test_rank_zero_error() {
        let mut block = Block::new();

        // Create a rank-0 array (scalar)
        let arr = ArrayD::<F>::zeros(vec![]);

        let result = block.insert("scalar", arr, None);
        assert!(result.is_err());
        match result {
            Err(BlockError::RankZero { key }) => {
                assert_eq!(key, "scalar");
            }
            _ => panic!("Expected RankZero error"),
        }
    }

    #[test]
    fn test_dtype_query() {
        let mut block = Block::new();

        let arr_float = Array1::from_vec(vec![1.0 as F]).into_dyn();
        let arr_i64 = Array1::from_vec(vec![10 as I]).into_dyn();

        block.insert("x", arr_float, None).unwrap();
        block.insert("count", arr_i64, None).unwrap();

        assert_eq!(block.dtype("x"), Some(DType::Float));
        assert_eq!(block.dtype("count"), Some(DType::Int));
        assert_eq!(block.dtype("missing"), None);
    }

    #[test]
    fn test_merge_basic() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn();

        block1.insert("x", arr1, None).unwrap();
        block2.insert("x", arr2, None).unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.n_rows(), Some(4));
        let x = block1.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge_empty_blocks() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        // Merge empty into empty
        block1.merge(&block2).unwrap();
        assert_eq!(block1.n_rows(), None);

        // Merge non-empty into empty
        let arr = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        block2.insert("x", arr, None).unwrap();
        block1.merge(&block2).unwrap();
        assert_eq!(block1.n_rows(), Some(2));

        // Merge empty into non-empty
        let block3 = Block::new();
        block1.merge(&block3).unwrap();
        assert_eq!(block1.n_rows(), Some(2));
    }

    #[test]
    fn test_merge_incompatible_keys() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn();

        block1.insert("x", arr1, None).unwrap();
        block2.insert("y", arr2, None).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_incompatible_dtypes() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        let arr1 = Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn();
        let arr2 = Array1::from_vec(vec![3 as I, 4]).into_dyn();

        block1.insert("value", arr1, None).unwrap();
        block2.insert("value", arr2, None).unwrap();

        let result = block1.merge(&block2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_multiple_columns() {
        let mut block1 = Block::new();
        let mut block2 = Block::new();

        block1
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        block1
            .insert("id", Array1::from_vec(vec![10 as Idx, 20]).into_dyn(), None)
            .unwrap();

        block2
            .insert(
                "x",
                Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        block2
            .insert("id", Array1::from_vec(vec![30 as Idx, 40]).into_dyn(), None)
            .unwrap();

        block1.merge(&block2).unwrap();

        assert_eq!(block1.n_rows(), Some(4));
        let x = block1.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        let id = block1.get("id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_rename_column() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "y",
                Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn(),
                None,
            )
            .unwrap();

        // Successful rename
        block.rename_column("x", "position_x").unwrap();
        assert!(!block.contains_key("x"));
        assert!(block.contains_key("position_x"));
        assert_eq!(
            block
                .get("position_x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename non-existent column
        assert!(block.rename_column("nonexistent", "new_name").is_err());

        // Try to rename to existing column name
        assert!(block.rename_column("position_x", "y").is_err());
    }

    #[test]
    fn test_has_typed_columns() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![1 as Idx, 2]).into_dyn(), None)
            .unwrap();
        block
            .insert(
                "res_seq",
                Array1::from_vec(vec![1 as I, 2]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "name",
                Array1::from_vec(vec!["CA".to_string(), "N".to_string()]).into_dyn(),
                None,
            )
            .unwrap();

        block
            .insert(
                "charge_i64",
                Array1::from_vec(vec![1_i64, 2]).into_dyn(),
                None,
            )
            .unwrap();

        assert_eq!(block.dtype("x"), Some(DType::Float));
        assert_eq!(block.dtype("id"), Some(DType::Uint));
        assert_eq!(block.dtype("res_seq"), Some(DType::Int));
        assert_eq!(block.dtype("charge_i64"), Some(DType::I64));
        assert_eq!(block.dtype("name"), Some(DType::String));
        assert_eq!(block.dtype("missing"), None);
    }

    #[test]
    fn test_resize_shrink() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0, 4.0]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "id",
                Array1::from_vec(vec![10 as Idx, 20, 30, 40]).into_dyn(),
                None,
            )
            .unwrap();

        block.resize(2).unwrap();

        assert_eq!(block.n_rows(), Some(2));
        let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0]);
        let id = block.get("id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20]);
    }

    #[test]
    fn test_resize_grow() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as Idx, 20]).into_dyn(), None)
            .unwrap();

        block.resize(4).unwrap();

        assert_eq!(block.n_rows(), Some(4));
        let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
        // Original data preserved, new rows are 0.0
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
        let id = block.get("id").and_then(|c| c.get::<u64>()).unwrap();
        // Original data preserved, new rows are 0
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 0, 0]);
    }

    #[test]
    fn test_resize_same() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();

        block.resize(3).unwrap();

        assert_eq!(block.n_rows(), Some(3));
        let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_empty() {
        let mut block = Block::new();

        block.resize(5).unwrap();
        assert_eq!(block.n_rows(), Some(5));
        assert!(block.is_empty());
    }

    #[test]
    fn test_resize_multidim() {
        use ndarray::Array2;

        let mut block = Block::new();
        // 4x3 position array
        let pos = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0 as F, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap()
        .into_dyn();
        block.insert("pos", pos, None).unwrap();

        // Shrink 4x3 -> 2x3
        block.resize(2).unwrap();
        assert_eq!(block.n_rows(), Some(2));
        let pos = block.get("pos").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(pos.shape(), &[2, 3]);
        assert_eq!(
            pos.as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Grow 2x3 -> 5x3
        block.resize(5).unwrap();
        assert_eq!(block.n_rows(), Some(5));
        let pos = block.get("pos").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(pos.shape(), &[5, 3]);
        // Original data followed by zeros
        assert_eq!(
            pos.as_slice_memory_order().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn test_resize_mixed_dtypes() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "id",
                Array1::from_vec(vec![10 as Idx, 20, 30]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "mask",
                Array1::from_vec(vec![true, false, true]).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "name",
                Array1::from_vec(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                    .into_dyn(),
                None,
            )
            .unwrap();

        // Grow from 3 to 5
        block.resize(5).unwrap();
        assert_eq!(block.n_rows(), Some(5));

        let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(
            x.as_slice_memory_order().unwrap(),
            &[1.0, 2.0, 3.0, 0.0, 0.0]
        );
        let id = block.get("id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20, 30, 0, 0]);
        let mask = block.get("mask").and_then(|c| c.get::<bool>()).unwrap();
        assert_eq!(
            mask.as_slice_memory_order().unwrap(),
            &[true, false, true, false, false]
        );
        let name = block.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(name[[0]], "a");
        assert_eq!(name[[1]], "b");
        assert_eq!(name[[2]], "c");
        assert_eq!(name[[3]], "");
        assert_eq!(name[[4]], "");

        // Shrink from 5 to 2
        block.resize(2).unwrap();
        assert_eq!(block.n_rows(), Some(2));

        let x = block.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(x.as_slice_memory_order().unwrap(), &[1.0, 2.0]);
        let id = block.get("id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(id.as_slice_memory_order().unwrap(), &[10, 20]);
        let mask = block.get("mask").and_then(|c| c.get::<bool>()).unwrap();
        assert_eq!(mask.as_slice_memory_order().unwrap(), &[true, false]);
        let name = block.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(name[[0]], "a");
        assert_eq!(name[[1]], "b");
    }

    // ---- nullable columns -------------------------------------------------

    #[test]
    fn insert_refuses_a_mask_that_does_not_cover_every_row() {
        let mut block = Block::new();
        let arr = Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn();
        assert!(block.insert("x", arr, Some(vec![true, false])).is_err());
    }

    #[test]
    fn insert_records_the_mask_it_was_given() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 0.0, 3.0]).into_dyn(),
                Some(vec![true, false, true]),
            )
            .unwrap();
        assert_eq!(block.validity("x"), Some(&[true, false, true][..]));
    }

    #[test]
    fn insert_leaves_the_column_without_a_mask() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        assert_eq!(block.validity("x"), None);
    }

    #[test]
    fn remove_drops_the_mask_with_the_column() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 0.0, 3.0]).into_dyn(),
                Some(vec![true, false, true]),
            )
            .unwrap();
        block.remove("x");
        block
            .insert(
                "x",
                Array1::from_vec(vec![4.0 as F, 5.0, 6.0]).into_dyn(),
                None,
            )
            .unwrap();
        assert_eq!(block.validity("x"), None);
    }

    #[test]
    fn clone_keeps_the_mask() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 0.0, 3.0]).into_dyn(),
                Some(vec![true, false, true]),
            )
            .unwrap();
        let copy = block.clone();
        assert_eq!(copy.validity("x"), Some(&[true, false, true][..]));
    }

    // ---- set_validity ----

    #[test]
    fn set_validity_masks_a_column_of_any_dtype() {
        let mut block = Block::new();
        let c = ArrayD::from_shape_vec(vec![2], vec![1i64, 0]).unwrap();
        block.insert("q", Column::from(c), None).unwrap();
        block.set_validity("q", vec![true, false]).unwrap();
        assert_eq!(block.validity("q"), Some(&[true, false][..]));
    }

    #[test]
    fn set_validity_refuses_a_missing_column_and_a_short_mask() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        assert!(matches!(
            block.set_validity("y", vec![true, false]),
            Err(BlockError::MissingColumn { .. })
        ));
        assert!(matches!(
            block.set_validity("x", vec![false]),
            Err(BlockError::ValidityLength { .. })
        ));
        assert_eq!(block.validity("x"), None);
    }

    // ---- stack ----

    fn floats(values: &[F]) -> ArrayD<F> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    #[test]
    fn stack_fills_a_missing_column_as_null() {
        let mut a = Block::new();
        a.insert("x", floats(&[0.0, 1.0]), None).unwrap();
        let mut b = Block::new();
        b.insert("x", floats(&[2.0]), None).unwrap();
        b.insert("q", floats(&[-1.0]), None).unwrap();

        let s = Block::stack([&a, &b]).unwrap();
        assert_eq!(s.keys().collect::<Vec<_>>(), ["x", "q"]);
        assert_eq!(
            s.get("q")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_slice_memory_order(),
            Some(&[0.0, 0.0, -1.0][..])
        );
        assert_eq!(s.validity("q"), Some(&[false, false, true][..]));
        assert_eq!(s.validity("x"), None);
    }

    #[test]
    fn stack_carries_each_parts_own_mask() {
        let mut a = Block::new();
        a.insert("x", floats(&[0.0, 1.0]), Some(vec![false, true]))
            .unwrap();
        let mut b = Block::new();
        b.insert("x", floats(&[2.0]), None).unwrap();
        let s = Block::stack([&b, &a]).unwrap();
        assert_eq!(s.validity("x"), Some(&[true, false, true][..]));
    }

    #[test]
    fn stack_counts_the_rows_of_a_columnless_part() {
        let mut empty = Block::new();
        empty.resize(2).unwrap();
        let mut b = Block::new();
        b.insert("x", floats(&[5.0]), None).unwrap();
        let s = Block::stack([&empty, &b]).unwrap();
        assert_eq!(s.n_rows(), Some(3));
        assert_eq!(s.validity("x"), Some(&[false, false, true][..]));

        let only_rows = Block::stack([&empty, &empty]).unwrap();
        assert_eq!((only_rows.len(), only_rows.n_rows()), (0, Some(4)));
        assert_eq!(Block::stack([]).unwrap().n_rows(), None);
    }

    #[test]
    fn stack_keeps_trailing_axes_of_a_filled_column() {
        let mut a = Block::new();
        a.insert("v", ArrayD::<F>::ones(vec![1, 3]), None).unwrap();
        let mut b = Block::new();
        b.insert("x", floats(&[1.0, 2.0]), None).unwrap();
        let s = Block::stack([&a, &b]).unwrap();
        assert_eq!(
            s.get("v").and_then(|c| c.get::<f64>()).unwrap().shape(),
            &[3, 3]
        );
    }

    #[test]
    fn stack_refuses_a_dtype_clash_naming_the_part() {
        let mut a = Block::new();
        a.insert("label", Array1::from_vec(vec![1 as I]).into_dyn(), None)
            .unwrap();
        let mut b = Block::new();
        b.insert("label", Array1::from_vec(vec![1i64]).into_dyn(), None)
            .unwrap();
        let err = Block::stack([&a, &a, &b]).unwrap_err();
        assert_eq!(
            err,
            BlockError::StackDtype {
                key: "label".into(),
                part: 2,
                expected: DType::Int,
                got: DType::I64,
            }
        );
    }

    #[test]
    fn stack_refuses_a_per_row_shape_clash() {
        let mut a = Block::new();
        a.insert("v", ArrayD::<F>::ones(vec![1, 3]), None).unwrap();
        let mut b = Block::new();
        b.insert("v", ArrayD::<F>::ones(vec![1, 2]), None).unwrap();
        assert!(matches!(
            Block::stack([&a, &b]),
            Err(BlockError::StackShape { part: 1, .. })
        ));
    }

    // ---- coords ----

    #[test]
    fn coords_gathers_x_y_z_row_by_row() {
        let mut block = Block::new();
        block.insert("x", floats(&[1.0, 2.0]), None).unwrap();
        block.insert("y", floats(&[0.5, 1.5]), None).unwrap();
        block.insert("z", floats(&[3.0, 4.0]), None).unwrap();
        assert_eq!(
            block.coords().unwrap(),
            ndarray::array![[1.0, 0.5, 3.0], [2.0, 1.5, 4.0]]
        );
    }

    #[test]
    fn coords_names_the_missing_axis() {
        let mut block = Block::new();
        block.insert("x", floats(&[1.0]), None).unwrap();
        assert_eq!(
            block.coords().unwrap_err(),
            BlockError::MissingColumn { key: "y".into() }
        );
    }

    #[test]
    fn set_coords_refuses_a_non_n_by_3_array() {
        let mut block = Block::new();
        let two = ndarray::Array2::<F>::zeros((2, 2));
        assert!(block.set_coords(two.view()).is_err());
        assert!(block.is_empty());
    }

    // ---- precision ----

    #[test]
    fn set_precision_declares_on_an_f64_column_only() {
        let mut block = Block::new();
        block.insert("x", floats(&[1.0, 2.0]), None).unwrap();
        block
            .insert("id", Array1::from_vec(vec![1 as Idx, 2]).into_dyn(), None)
            .unwrap();
        block.set_precision("x", 1e-3).unwrap();
        assert_eq!(block.precision("x"), Some(1e-3));
        assert!(block.set_precision("id", 1e-3).is_err());
        assert_eq!(
            block.set_precision("nope", 1e-3).unwrap_err(),
            BlockError::MissingColumn { key: "nope".into() }
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 2f64.powi(1001)] {
            assert!(block.set_precision("x", bad).is_err(), "accepted {bad}");
        }
        assert_eq!(block.precision("x"), Some(1e-3));
        assert_eq!(block.clear_precision("x"), Some(1e-3));
        assert_eq!(block.precision("x"), None);
    }

    #[test]
    fn precision_follows_its_column() {
        let mut block = Block::new();
        block.insert("x", floats(&[1.0, 2.0]), None).unwrap();
        block.insert("y", floats(&[3.0, 4.0]), None).unwrap();
        block.set_precision("x", 1e-3).unwrap();
        block.set_precision("y", 1e-2).unwrap();

        // A rename carries it.
        block.rename_column("x", "px").unwrap();
        assert_eq!(block.precision("px"), Some(1e-3));
        assert_eq!(block.precision("x"), None);

        // Replacing with another f64 column keeps it; another dtype drops it.
        block.insert("px", floats(&[5.0, 6.0]), None).unwrap();
        assert_eq!(block.precision("px"), Some(1e-3));
        block
            .insert("px", Array1::from_vec(vec![1 as I, 2]).into_dyn(), None)
            .unwrap();
        assert_eq!(block.precision("px"), None);

        // Row selection, column selection and copies keep it.
        assert_eq!(block.select_rows(&[1]).unwrap().precision("y"), Some(1e-2));
        assert_eq!(
            block.select_columns(&["y"]).unwrap().precision("y"),
            Some(1e-2)
        );
        assert_eq!(block.deep_copy().precision("y"), Some(1e-2));
        let stacked = Block::stack([&block, &block]).unwrap();
        assert_eq!(stacked.precision("y"), Some(1e-2));

        // Removing the column drops it.
        block.remove("y");
        assert_eq!(block.precision("y"), None);
    }

    // ---- targets ----

    #[test]
    fn set_target_declares_on_a_u64_column_with_a_well_formed_target() {
        let mut block = Block::new();
        block
            .insert("site", Array1::from_vec(vec![0 as Idx, 1]).into_dyn(), None)
            .unwrap();
        block.insert("x", floats(&[1.0, 2.0]), None).unwrap();
        block.set_target("site", "sites").unwrap();
        assert_eq!(block.target("site"), Some("sites"));
        assert!(block.set_target("x", "atoms").is_err());
        assert!(block.set_target("nope", "atoms").is_err());
        for bad in ["", "a/b", "/trajectory/atoms", "/frame"] {
            assert!(block.set_target("site", bad).is_err(), "{bad}");
        }
        block.rename_column("site", "ref").unwrap();
        assert_eq!(block.target("ref"), Some("sites"));
        assert_eq!(
            block.select_rows(&[1]).unwrap().target("ref"),
            Some("sites")
        );
        assert_eq!(block.deep_copy().target("ref"), Some("sites"));
        block.insert("ref", floats(&[0.0, 1.0]), None).unwrap();
        assert_eq!(block.target("ref"), None);
    }
}

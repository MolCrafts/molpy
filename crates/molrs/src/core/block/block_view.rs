//! Zero-copy borrowed view of a [`Block`]: [`BlockView`].

use indexmap::IndexMap;

use super::Block;
use super::column::Column;
use super::column_view::ColumnView;
use super::dtype::DType;

/// A borrowed, read-only view of a [`Block`], with the same API surface as
/// `Block` and without copying any array data.
///
/// Keys are `&str` references into the original `Block`'s key strings.
/// Values are [`ColumnView`]s that borrow the underlying array data.
pub struct BlockView<'a> {
    map: IndexMap<&'a str, ColumnView<'a>>,
    /// Borrowed validity masks of the viewed block's nullable columns.
    validity: IndexMap<&'a str, &'a [bool]>,
    /// Declared precisions of the viewed block's `f64` columns.
    precision: IndexMap<&'a str, f64>,
    /// Declared row-reference targets of the viewed block.
    targets: IndexMap<&'a str, &'a str>,
    /// Borrowed structural shape of the viewed block, if it declares one.
    shape: Option<&'a [usize]>,
    n_rows: Option<usize>,
}

impl<'a> BlockView<'a> {
    /// Creates an empty `BlockView`.
    pub fn new() -> Self {
        Self {
            map: IndexMap::new(),
            validity: IndexMap::new(),
            precision: IndexMap::new(),
            targets: IndexMap::new(),
            shape: None,
            n_rows: None,
        }
    }

    /// Inserts a column view under the given key.
    ///
    /// If the `BlockView` was empty, `n_rows` is set from the column's axis-0
    /// length. Subsequent insertions are not validated for consistency (caller
    /// is responsible).
    pub fn insert(&mut self, key: &'a str, col: ColumnView<'a>) {
        if self.n_rows.is_none() {
            self.n_rows = col.n_rows();
        }
        self.map.insert(key, col);
    }

    /// Number of columns in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the view contains no columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the common axis-0 length, or `None` if empty.
    #[inline]
    pub fn n_rows(&self) -> Option<usize> {
        self.n_rows
    }

    /// Returns `true` if the view contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// The column view for `key`, or `None` when the key is absent.
    ///
    /// Project a dtype with [`ColumnView::get::<T>`](ColumnView::get). `None`
    /// from a projection means the column has a different dtype.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&ColumnView<'a>> {
        self.map.get(key)
    }

    /// Returns an iterator over `(&str, &ColumnView)`.
    pub fn iter(&self) -> impl Iterator<Item = (&&'a str, &ColumnView<'a>)> {
        self.map.iter()
    }

    /// Returns an iterator over column keys.
    pub fn keys(&self) -> impl Iterator<Item = &&'a str> {
        self.map.keys()
    }

    /// Returns an iterator over column view references.
    pub fn values(&self) -> impl Iterator<Item = &ColumnView<'a>> {
        self.map.values()
    }

    /// The validity mask of column `key`, if it carries one.
    pub fn validity(&self, key: &str) -> Option<&'a [bool]> {
        self.validity.get(key).copied()
    }

    /// Every declared row-reference target, as `(column, target)`.
    pub fn targets(&self) -> Vec<(&'a str, &'a str)> {
        self.targets.iter().map(|(&k, &t)| (k, t)).collect()
    }

    /// Returns the data type of the column with the given key, if it exists.
    pub fn dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|c| c.dtype())
    }

    /// Creates an owned [`Block`] by cloning all viewed data, validity masks,
    /// declared precisions and targets, and structural shape included.
    ///
    /// # Errors
    /// Returns the column or metadata validation error if an independently
    /// assembled view cannot be represented by a valid Block. No columns are
    /// silently omitted.
    pub fn to_owned(&self) -> Result<Block, super::BlockError> {
        let mut block = Block::new();
        for (&key, col_view) in &self.map {
            let col: Column = col_view.to_owned();
            block.insert(key, col, None)?;
        }
        for (&key, &mask) in &self.validity {
            block.set_validity(key, mask.to_vec())?;
        }
        for (&key, &p) in &self.precision {
            block.set_precision(key, p)?;
        }
        for (&key, &target) in &self.targets {
            block.set_target(key, target)?;
        }
        if self.map.is_empty() {
            block.n_rows = self.n_rows;
        }
        if let Some(shape) = self.shape {
            block.set_shape(shape)?;
        }
        Ok(block)
    }
}

impl<'a> Default for BlockView<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> From<&'a Block> for BlockView<'a> {
    fn from(block: &'a Block) -> Self {
        let mut view = BlockView {
            map: IndexMap::with_capacity(block.len()),
            validity: IndexMap::new(),
            precision: block.precisions().collect(),
            targets: block.targets().collect(),
            shape: block.structural_shape(),
            n_rows: block.n_rows(),
        };
        for (key, col) in block.iter() {
            view.map.insert(key, ColumnView::from(col));
            if let Some(mask) = block.validity(key) {
                view.validity.insert(key, mask);
            }
        }
        view
    }
}

impl std::fmt::Debug for BlockView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in &self.map {
            map.entry(k, &format!("{}(shape={:?})", v.dtype(), v.shape()));
        }
        map.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{F, Idx};
    use ndarray::Array1;

    #[test]
    fn to_owned_keeps_column_order_masks_and_structural_shape() {
        let mut block = Block::new();
        block
            .insert("c", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        block
            .insert(
                "a",
                Array1::from_vec(vec![0.0 as F, 4.0]).into_dyn(),
                Some(vec![false, true]),
            )
            .unwrap();
        block.set_shape(&[1, 2]).unwrap();
        block.set_precision("c", 1e-3).unwrap();

        let owned = BlockView::from(&block).to_owned().unwrap();
        assert_eq!(owned.keys().collect::<Vec<_>>(), ["c", "a"]);
        assert_eq!(owned.validity("a"), Some(&[false, true][..]));
        assert_eq!(owned.precision("c"), Some(1e-3));
        assert_eq!(owned.structural_shape(), Some(&[1, 2][..]));
    }

    #[test]
    fn test_from_block() {
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

        let view = BlockView::from(&block);
        assert_eq!(view.len(), 2);
        assert_eq!(view.n_rows(), Some(3));
        assert!(view.contains_key("x"));
        assert!(view.contains_key("id"));
        assert!(!view.is_empty());
    }

    #[test]
    fn test_typed_getters() {
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

        let view = BlockView::from(&block);

        // Correct type
        assert!(view.get("x").and_then(|c| c.get::<f64>()).is_some());
        assert!(view.get("id").and_then(|c| c.get::<u64>()).is_some());

        // Wrong type
        assert!(view.get("x").and_then(|c| c.get::<i32>()).is_none());
        assert!(view.get("id").and_then(|c| c.get::<f64>()).is_none());

        // Missing key
        assert!(view.get("missing").and_then(|c| c.get::<f64>()).is_none());
    }

    #[test]
    fn test_to_owned_roundtrip() {
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

        let view = BlockView::from(&block);
        let owned = view.to_owned().unwrap();

        assert_eq!(owned.n_rows(), Some(3));
        assert_eq!(owned.len(), 2);
        assert_eq!(
            owned
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0, 3.0]
        );
        assert_eq!(
            owned
                .get("id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[10, 20, 30]
        );
    }

    #[test]
    fn test_zero_copy() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();

        let view = BlockView::from(&block);
        let orig_ptr = block
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap()
            .as_ptr();
        let view_ptr = view.get("x").and_then(|c| c.get::<f64>()).unwrap().as_ptr();
        assert_eq!(orig_ptr, view_ptr);
    }

    #[test]
    fn test_empty_view() {
        let view = BlockView::new();
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.n_rows(), None);
    }

    #[test]
    fn test_iter_keys() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        block
            .insert("id", Array1::from_vec(vec![10 as Idx, 20]).into_dyn(), None)
            .unwrap();

        let view = BlockView::from(&block);
        let keys: Vec<&&str> = view.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_dtype_query() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();

        let view = BlockView::from(&block);
        assert_eq!(view.dtype("x"), Some(DType::Float));
        assert_eq!(view.dtype("missing"), None);
    }
}

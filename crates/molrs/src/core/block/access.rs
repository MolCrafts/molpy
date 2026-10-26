//! Unified access trait for owned and borrowed block types: [`BlockAccess`].

use super::Block;
use super::block_view::BlockView;
use super::column_view::ColumnView;
use super::dtype::DType;

/// Unified read-only access for [`Block`] and [`BlockView`].
///
/// A column comes back whole; project a dtype with
/// [`ColumnView::get::<T>`](ColumnView::get).
pub trait BlockAccess {
    /// The column for `key`, or `None` when the key is absent.
    ///
    /// `None` from a later `get::<T>` projection means the column has a different
    /// dtype, which is not the same as a missing key.
    fn column<'a>(&'a self, key: &str) -> Option<ColumnView<'a>>;
    /// Returns the common axis-0 length, or `None` if empty.
    fn n_rows(&self) -> Option<usize>;
    /// Number of columns.
    fn len(&self) -> usize;
    /// Returns `true` if there are no columns.
    fn is_empty(&self) -> bool;
    /// Returns `true` if the block contains the specified key.
    fn contains_key(&self, key: &str) -> bool;
    /// Returns column keys as a `Vec`.
    fn column_keys(&self) -> Vec<&str>;
    /// Returns the data type of the column with the given key, if it exists.
    fn column_dtype(&self, key: &str) -> Option<DType>;
    /// Returns the shape of the column with the given key, if it exists.
    fn column_shape(&self, key: &str) -> Option<Vec<usize>>;
    /// The validity mask of column `key`, or `None` when it has no nulls.
    fn validity(&self, key: &str) -> Option<&[bool]>;
    /// The declared row-reference targets, as `(column, target)`.
    fn targets(&self) -> Vec<(&str, &str)>;
}

impl BlockAccess for Block {
    fn column<'a>(&'a self, key: &str) -> Option<ColumnView<'a>> {
        self.get(key).map(ColumnView::from)
    }

    fn n_rows(&self) -> Option<usize> {
        Block::n_rows(self)
    }

    fn len(&self) -> usize {
        Block::len(self)
    }

    fn is_empty(&self) -> bool {
        Block::is_empty(self)
    }

    fn contains_key(&self, key: &str) -> bool {
        Block::contains_key(self, key)
    }

    fn column_keys(&self) -> Vec<&str> {
        self.keys().collect()
    }

    fn column_dtype(&self, key: &str) -> Option<DType> {
        self.get(key).map(|col| col.dtype())
    }

    fn column_shape(&self, key: &str) -> Option<Vec<usize>> {
        self.get(key).map(|col| col.shape().to_vec())
    }

    fn validity(&self, key: &str) -> Option<&[bool]> {
        Block::validity(self, key)
    }
    fn targets(&self) -> Vec<(&str, &str)> {
        Block::targets(self).collect()
    }
}

impl BlockAccess for BlockView<'_> {
    fn column<'a>(&'a self, key: &str) -> Option<ColumnView<'a>> {
        self.get(key).map(ColumnView::reborrow)
    }

    fn n_rows(&self) -> Option<usize> {
        BlockView::n_rows(self)
    }

    fn len(&self) -> usize {
        BlockView::len(self)
    }

    fn is_empty(&self) -> bool {
        BlockView::is_empty(self)
    }

    fn contains_key(&self, key: &str) -> bool {
        BlockView::contains_key(self, key)
    }

    fn column_keys(&self) -> Vec<&str> {
        self.keys().copied().collect()
    }

    fn column_dtype(&self, key: &str) -> Option<DType> {
        BlockView::dtype(self, key)
    }

    fn column_shape(&self, key: &str) -> Option<Vec<usize>> {
        self.get(key).map(|col_view| col_view.shape().to_vec())
    }

    fn validity(&self, key: &str) -> Option<&[bool]> {
        BlockView::validity(self, key)
    }

    fn targets(&self) -> Vec<(&str, &str)> {
        BlockView::targets(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Column;
    use crate::op::{F, Idx};
    use ndarray::Array1;

    fn make_block() -> Block {
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
    }

    #[test]
    fn test_column_projection_on_column() {
        let col = Column::from(Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn());
        assert!(col.get::<f64>().is_some());
        assert!(col.get::<i32>().is_none());
        assert_eq!(col.n_rows(), Some(2));
        assert_eq!(col.dtype(), DType::Float);
        assert_eq!(col.shape(), &[2]);
    }

    #[test]
    fn test_column_projection_on_column_view() {
        let col = Column::from(Array1::from_vec(vec![1, 2, 3]).into_dyn());
        let view = ColumnView::from(&col);
        assert!(view.get::<i32>().is_some());
        assert!(view.get::<f64>().is_none());
        assert_eq!(view.n_rows(), Some(3));
        assert_eq!(view.dtype(), DType::Int);
    }

    #[test]
    fn test_block_access_on_block() {
        let block = make_block();
        assert!(block.column("x").and_then(|c| c.get::<f64>()).is_some());
        assert!(block.column("id").and_then(|c| c.get::<u64>()).is_some());
        assert_eq!(BlockAccess::n_rows(&block), Some(3));
        assert_eq!(BlockAccess::len(&block), 2);
        assert!(!BlockAccess::is_empty(&block));
        assert!(BlockAccess::contains_key(&block, "x"));
        assert!(!BlockAccess::contains_key(&block, "missing"));
    }

    #[test]
    fn test_block_access_on_block_view() {
        let block = make_block();
        let view = BlockView::from(&block);
        assert!(view.column("x").and_then(|c| c.get::<f64>()).is_some());
        assert!(view.column("id").and_then(|c| c.get::<u64>()).is_some());
        assert_eq!(BlockAccess::n_rows(&view), Some(3));
        assert_eq!(BlockAccess::len(&view), 2);
        assert!(!BlockAccess::is_empty(&view));
        assert!(BlockAccess::contains_key(&view, "x"));
    }

    #[test]
    fn test_generic_function_with_block_access() {
        fn count_float_columns(b: &impl BlockAccess) -> usize {
            b.column_keys()
                .iter()
                .filter(|k| b.column(k).and_then(|c| c.get::<f64>()).is_some())
                .count()
        }

        let block = make_block();
        assert_eq!(count_float_columns(&block), 1);

        let view = BlockView::from(&block);
        assert_eq!(count_float_columns(&view), 1);
    }
}

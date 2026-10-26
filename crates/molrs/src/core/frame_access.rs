//! Unified access trait for owned and borrowed frame types: [`FrameAccess`].

use crate::core::BlockAccess;
use crate::core::Frame;
use crate::core::FrameView;
use crate::core::MetaMap;
use crate::core::SimBox;

/// Unified read-only access for [`Frame`] and [`FrameView`], enabling generic
/// code that works with either.
///
/// A frame read is two steps: [`visit_block`](FrameAccess::visit_block), then
/// the column. Metadata, the simulation box, and one column read as two keys.
/// Project a dtype from the returned [`ColumnView`](crate::core::ColumnView).
pub trait FrameAccess {
    /// The column `col_key` inside block `block_key`, or `None` when either key is absent.
    ///
    /// A missing block and a missing column are both `None`. A present column of
    /// another dtype is still `Some`; `get::<T>` on that column is the dtype check.
    fn column<'a>(&'a self, block_key: &str, col_key: &str) -> Option<crate::core::ColumnView<'a>>;
    /// Returns a reference to the simulation box, if present.
    fn simbox_ref(&self) -> Option<&SimBox>;
    /// Returns a reference to the metadata map.
    fn meta_ref(&self) -> &MetaMap;
    /// Returns block keys as a `Vec`.
    fn block_keys(&self) -> Vec<&str>;
    /// Returns `true` if the frame contains the specified block key.
    fn contains_block(&self, key: &str) -> bool;
    /// Number of blocks.
    fn n_blocks(&self) -> usize;
    /// Returns `true` if the frame contains no blocks.
    fn is_empty(&self) -> bool;
    /// Visits a block by key through the [`BlockAccess`] trait, using the visitor pattern
    /// to avoid lifetime/return-type issues. Returns `None` if the block does not exist.
    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R>;
}

impl FrameAccess for Frame {
    fn column<'a>(&'a self, block_key: &str, col_key: &str) -> Option<crate::core::ColumnView<'a>> {
        self.get(block_key)?.column(col_key)
    }

    fn simbox_ref(&self) -> Option<&SimBox> {
        self.simbox.as_ref()
    }

    fn meta_ref(&self) -> &MetaMap {
        &self.meta
    }

    fn block_keys(&self) -> Vec<&str> {
        self.keys().collect()
    }

    fn contains_block(&self, key: &str) -> bool {
        self.contains_key(key)
    }

    fn n_blocks(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        Frame::is_empty(self)
    }

    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R> {
        self.get(key).map(|block| f(block))
    }
}

impl FrameAccess for FrameView<'_> {
    fn column<'a>(&'a self, block_key: &str, col_key: &str) -> Option<crate::core::ColumnView<'a>> {
        self.get(block_key)?.column(col_key)
    }

    fn simbox_ref(&self) -> Option<&SimBox> {
        self.simbox
    }

    fn meta_ref(&self) -> &MetaMap {
        self.meta
    }

    fn block_keys(&self) -> Vec<&str> {
        self.keys().copied().collect()
    }

    fn contains_block(&self, key: &str) -> bool {
        self.contains_key(key)
    }

    fn n_blocks(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        FrameView::is_empty(self)
    }

    fn visit_block<R>(&self, key: &str, f: impl FnOnce(&dyn BlockAccess) -> R) -> Option<R> {
        self.get(key).map(|block_view| f(block_view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::op::{F, Idx};
    use ndarray::Array1;

    fn make_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "id",
                Array1::from_vec(vec![10 as Idx, 20, 30]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert("title", "Test");
        frame
    }

    #[test]
    fn test_frame_access_on_frame() {
        let frame = make_frame();
        assert!(column_is_float(&frame, "atoms", "x"));
        assert!(column_is_uint(&frame, "atoms", "id"));
        assert!(!column_is_float(&frame, "atoms", "missing"));
        assert!(!column_is_float(&frame, "missing", "x"));
        assert_eq!(FrameAccess::n_blocks(&frame), 1);
        assert!(FrameAccess::contains_block(&frame, "atoms"));
        assert!(!FrameAccess::is_empty(&frame));
        assert_eq!(
            FrameAccess::meta_ref(&frame).get("title").unwrap().as_str(),
            Some("Test")
        );
        assert!(FrameAccess::simbox_ref(&frame).is_none());
    }

    #[test]
    fn test_frame_access_on_frame_view() {
        let frame = make_frame();
        let view = FrameView::from(&frame);
        assert!(column_is_float(&view, "atoms", "x"));
        assert!(column_is_uint(&view, "atoms", "id"));
        assert!(!column_is_float(&view, "atoms", "missing"));
        assert_eq!(FrameAccess::n_blocks(&view), 1);
        assert!(FrameAccess::contains_block(&view, "atoms"));
        assert!(!FrameAccess::is_empty(&view));
        assert_eq!(
            FrameAccess::meta_ref(&view).get("title").unwrap().as_str(),
            Some("Test")
        );
    }

    fn column_is_float(f: &impl FrameAccess, block: &str, col: &str) -> bool {
        f.visit_block(block, |b| {
            b.column(col).and_then(|c| c.get::<f64>()).is_some()
        })
        .unwrap_or(false)
    }

    fn column_is_uint(f: &impl FrameAccess, block: &str, col: &str) -> bool {
        f.visit_block(block, |b| {
            b.column(col).and_then(|c| c.get::<u64>()).is_some()
        })
        .unwrap_or(false)
    }

    #[test]
    fn test_generic_function_with_frame_access() {
        fn get_x_data(f: &impl FrameAccess) -> Option<Vec<F>> {
            f.visit_block("atoms", |b| {
                b.column("x")
                    .and_then(|c| c.get::<f64>())
                    .map(|a| a.iter().copied().collect())
            })
            .flatten()
        }

        let frame = make_frame();
        assert_eq!(get_x_data(&frame), Some(vec![1.0, 2.0, 3.0]));

        let view = FrameView::from(&frame);
        assert_eq!(get_x_data(&view), Some(vec![1.0, 2.0, 3.0]));
    }
}

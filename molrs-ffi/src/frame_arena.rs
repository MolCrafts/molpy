//! [`FrameArena`]: owns frames and mediates access to them through handles.
//!
//! Column access goes through [`crate::BlockRef`]; the arena hands out and
//! validates the [`FrameId`] / [`BlockHandle`] keys.

use crate::error::FfiError;
use crate::handle::{BlockHandle, FrameId};
use molrs::core::MetaMap;
use molrs::core::SimBox;
use molrs::core::{Block, Frame};
use slotmap::SlotMap;
use std::collections::{HashMap, HashSet};

/// A frame and its invalidation tracking.
struct TrackedFrame {
    frame: Frame,
    /// Version counter for each block key. Increments on remove/replace.
    block_versions: HashMap<String, u64>,
}

/// FrameArena owns all frames and mediates access via handles.
pub struct FrameArena {
    frames: SlotMap<FrameId, TrackedFrame>,
}

impl FrameArena {
    /// Creates a new empty arena.
    pub fn new() -> Self {
        Self {
            frames: SlotMap::with_key(),
        }
    }

    /// Creates a new frame and returns its stable ID.
    pub fn frame_new(&mut self) -> FrameId {
        let entry = TrackedFrame {
            frame: Frame::new(),
            block_versions: HashMap::new(),
        };
        self.frames.insert(entry)
    }

    /// Drops a frame from the arena, invalidating all handles to it.
    pub fn drop_frame(&mut self, id: FrameId) -> Result<(), FfiError> {
        self.frames
            .remove(id)
            .ok_or(FfiError::InvalidFrameId)
            .map(|_| ())
    }

    /// Sets a block in a frame. If the key exists, increments version to invalidate old handles.
    pub fn set_block(&mut self, id: FrameId, key: &str, block: Block) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // If key exists, increment version to invalidate old handles
        if entry.frame.contains_key(key) {
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
        }

        entry.frame.insert(key, block);
        Ok(())
    }

    /// Removes a block from a frame, incrementing version to invalidate handles.
    pub fn del_block(&mut self, id: FrameId, key: &str) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        if entry.frame.remove(key).is_some() {
            let version = entry.block_versions.entry(key.to_string()).or_insert(0);
            *version += 1;
            Ok(())
        } else {
            Err(FfiError::KeyNotFound {
                key: key.to_string(),
            })
        }
    }

    /// Gets a block handle for a key in a frame.
    pub fn get_block(&self, id: FrameId, key: &str) -> Result<BlockHandle, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;

        if entry.frame.contains_key(key) {
            let version = entry.block_versions.get(key).copied().unwrap_or(0);
            Ok(BlockHandle::new(id, key.to_string(), version))
        } else {
            Err(FfiError::KeyNotFound {
                key: key.to_string(),
            })
        }
    }

    /// Clears all blocks from a frame, invalidating all block handles.
    pub fn clear(&mut self, id: FrameId) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        for key in entry.frame.keys() {
            if let Some(v) = entry.block_versions.get_mut(key) {
                *v += 1;
            } else {
                entry.block_versions.insert(key.to_string(), 1);
            }
        }

        entry.frame.clear();
        Ok(())
    }

    /// Returns an owned clone of a frame.
    pub fn clone_frame(&self, id: FrameId) -> Result<Frame, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;
        Ok(entry.frame.clone())
    }

    /// Borrows a frame immutably and runs a closure on it.
    pub fn with_frame<R>(&self, id: FrameId, f: impl FnOnce(&Frame) -> R) -> Result<R, FfiError> {
        let entry = self.frames.get(id).ok_or(FfiError::InvalidFrameId)?;
        Ok(f(&entry.frame))
    }

    /// Borrows a frame mutably and runs a closure on it.
    ///
    /// Conservatively invalidates all block handles after the closure runs,
    /// since `&mut Frame` allows arbitrary block modifications.
    pub fn with_frame_mut<R>(
        &mut self,
        id: FrameId,
        f: impl FnOnce(&mut Frame) -> R,
    ) -> Result<R, FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // Snapshot keys before closure to detect removals
        let keys_before: HashSet<String> = entry.frame.keys().map(|k| k.to_string()).collect();

        let result = f(&mut entry.frame);

        // Collect all unique keys (before + after + tracked) and bump once each
        let mut all_keys = keys_before;
        all_keys.extend(entry.frame.keys().map(|k| k.to_string()));
        all_keys.extend(entry.block_versions.keys().cloned());

        for key in all_keys {
            let version = entry.block_versions.entry(key).or_insert(0);
            *version += 1;
        }

        Ok(result)
    }

    /// Borrows a frame's metadata mutably and runs a closure on it.
    ///
    /// Metadata holds no block, so unlike [`with_frame_mut`](Self::with_frame_mut)
    /// this invalidates no block handle.
    pub fn with_frame_meta_mut<R>(
        &mut self,
        id: FrameId,
        f: impl FnOnce(&mut MetaMap) -> R,
    ) -> Result<R, FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;
        Ok(f(&mut entry.frame.meta))
    }

    /// Sets an entire frame, replacing the existing one.
    ///
    /// Invalidates handles for both old and new frame's keys.
    pub fn set_frame(&mut self, id: FrameId, frame: Frame) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

        // Collect all unique keys and bump once each
        let mut all_keys: HashSet<String> = entry.frame.keys().map(|k| k.to_string()).collect();
        all_keys.extend(frame.keys().map(|k| k.to_string()));
        all_keys.extend(entry.block_versions.keys().cloned());

        for key in all_keys {
            let version = entry.block_versions.entry(key).or_insert(0);
            *version += 1;
        }

        entry.frame = frame;
        Ok(())
    }

    /// Sets the simbox of a frame without invalidating block handles.
    pub fn set_frame_box(&mut self, id: FrameId, simbox: Option<SimBox>) -> Result<(), FfiError> {
        let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;
        entry.frame.simbox = simbox;
        Ok(())
    }

    /// Returns an owned clone of a block.
    pub fn clone_block(&self, handle: &BlockHandle) -> Result<Block, FfiError> {
        let (_, block) = self.validated_block(handle)?;
        Ok(block.clone())
    }

    /// Borrows a block immutably and runs a closure on it.
    pub fn with_block<R>(
        &self,
        handle: &BlockHandle,
        f: impl FnOnce(&Block) -> R,
    ) -> Result<R, FfiError> {
        let (_, block) = self.validated_block(handle)?;
        Ok(f(block))
    }

    /// Borrows a block mutably, runs a closure, then bumps the version and updates the handle.
    pub fn with_block_mut<R>(
        &mut self,
        handle: &mut BlockHandle,
        f: impl FnOnce(&mut Block) -> R,
    ) -> Result<R, FfiError> {
        self.validate_block_handle(handle)?;
        let entry = self
            .frames
            .get_mut(handle.frame_id)
            .ok_or(FfiError::InvalidBlockHandle)?;
        let block = entry
            .frame
            .get_mut(&handle.key)
            .ok_or(FfiError::InvalidBlockHandle)?;
        let result = f(block);

        // Bump version to invalidate other handles to this block
        let version = entry.block_versions.entry(handle.key.clone()).or_insert(0);
        *version += 1;
        handle.version = *version;

        Ok(result)
    }

    // ---- Private helpers ----

    /// Validate handle and return (&TrackedFrame, &Block) in one lookup pass.
    fn validated_block<'a>(
        &'a self,
        handle: &BlockHandle,
    ) -> Result<(&'a TrackedFrame, &'a Block), FfiError> {
        let entry = self
            .frames
            .get(handle.frame_id)
            .ok_or(FfiError::InvalidBlockHandle)?;

        let current_version = entry.block_versions.get(&handle.key).copied().unwrap_or(0);
        if current_version != handle.version {
            return Err(FfiError::InvalidBlockHandle);
        }

        let block = entry
            .frame
            .get(&handle.key)
            .ok_or(FfiError::InvalidBlockHandle)?;

        Ok((entry, block))
    }

    /// Validates a block handle (for mut paths where we can't return &Block).
    fn validate_block_handle(&self, handle: &BlockHandle) -> Result<(), FfiError> {
        let entry = self
            .frames
            .get(handle.frame_id)
            .ok_or(FfiError::InvalidBlockHandle)?;

        if !entry.frame.contains_key(&handle.key) {
            return Err(FfiError::InvalidBlockHandle);
        }

        let current_version = entry.block_versions.get(&handle.key).copied().unwrap_or(0);
        if current_version != handle.version {
            return Err(FfiError::InvalidBlockHandle);
        }

        Ok(())
    }
}

impl Default for FrameArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::op::F;
    use ndarray::Array1;

    #[test]
    fn test_frame_lifecycle() {
        let mut arena = FrameArena::new();

        let id = arena.frame_new();
        assert!(arena.clone_frame(id).is_ok());

        arena.drop_frame(id).unwrap();
        assert!(arena.clone_frame(id).is_err());
    }

    #[test]
    fn test_block_operations() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();

        arena.set_block(id, "atoms", block).unwrap();

        let handle = arena.get_block(id, "atoms").unwrap();
        assert!(arena.clone_block(&handle).is_ok());
    }

    #[test]
    fn test_block_invalidation_on_remove() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();

        arena.set_block(id, "atoms", block).unwrap();
        let handle = arena.get_block(id, "atoms").unwrap();

        arena.del_block(id, "atoms").unwrap();

        assert!(matches!(
            arena.clone_block(&handle),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_block_invalidation_on_replace() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();

        arena.set_block(id, "atoms", block1).unwrap();
        let handle = arena.get_block(id, "atoms").unwrap();

        let mut block2 = Block::new();
        block2
            .insert("x", Array1::from_vec(vec![3.0 as F, 4.0]).into_dyn(), None)
            .unwrap();
        arena.set_block(id, "atoms", block2).unwrap();

        assert!(matches!(
            arena.clone_block(&handle),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_reinsert_does_not_resurrect() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();

        arena.set_block(id, "atoms", block1).unwrap();
        let old_handle = arena.get_block(id, "atoms").unwrap();

        arena.del_block(id, "atoms").unwrap();

        let mut block2 = Block::new();
        block2
            .insert("x", Array1::from_vec(vec![3.0 as F, 4.0]).into_dyn(), None)
            .unwrap();
        arena.set_block(id, "atoms", block2).unwrap();

        assert!(matches!(
            arena.clone_block(&old_handle),
            Err(FfiError::InvalidBlockHandle)
        ));

        let new_handle = arena.get_block(id, "atoms").unwrap();
        assert!(arena.clone_block(&new_handle).is_ok());
    }

    #[test]
    fn test_clear_invalidates_all_blocks() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block1 = Block::new();
        block1
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        let mut block2 = Block::new();
        block2
            .insert("y", Array1::from_vec(vec![2.0 as F]).into_dyn(), None)
            .unwrap();

        arena.set_block(id, "atoms", block1).unwrap();
        arena.set_block(id, "bonds", block2).unwrap();

        let handle1 = arena.get_block(id, "atoms").unwrap();
        let handle2 = arena.get_block(id, "bonds").unwrap();

        arena.clear(id).unwrap();

        assert!(matches!(
            arena.clone_block(&handle1),
            Err(FfiError::InvalidBlockHandle)
        ));
        assert!(matches!(
            arena.clone_block(&handle2),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_with_block_read() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        arena.set_block(id, "atoms", block).unwrap();

        let handle = arena.get_block(id, "atoms").unwrap();
        let len = arena.with_block(&handle, |b| b.len()).unwrap();
        assert_eq!(len, 1);

        let nrows = arena.with_block(&handle, |b| b.n_rows()).unwrap();
        assert_eq!(nrows, Some(3));
    }

    #[test]
    fn test_with_block_mut_insert() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let block = Block::new();
        arena.set_block(id, "atoms", block).unwrap();
        let mut handle = arena.get_block(id, "atoms").unwrap();
        let old_version = handle.version;

        arena
            .with_block_mut(&mut handle, |b| {
                b.insert("y", Array1::from_vec(vec![4.0 as F, 5.0]).into_dyn(), None)
                    .unwrap();
            })
            .unwrap();

        assert!(handle.version > old_version);

        let nrows = arena.with_block(&handle, |b| b.n_rows()).unwrap();
        assert_eq!(nrows, Some(2));
    }

    #[test]
    fn test_with_block_mut_invalidates_old_handles() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let block = Block::new();
        arena.set_block(id, "atoms", block).unwrap();
        let old_handle = arena.get_block(id, "atoms").unwrap();
        let mut handle = old_handle.clone();

        arena
            .with_block_mut(&mut handle, |b| {
                b.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
                    .unwrap();
            })
            .unwrap();

        assert!(matches!(
            arena.with_block(&old_handle, |_| ()),
            Err(FfiError::InvalidBlockHandle)
        ));
    }

    #[test]
    fn test_with_frame_mut() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        arena.set_block(id, "atoms", block).unwrap();

        let renamed = arena
            .with_frame_mut(id, |f| f.rename_block("atoms", "particles"))
            .unwrap();
        assert!(renamed);

        assert!(arena.get_block(id, "atoms").is_err());
        assert!(arena.get_block(id, "particles").is_ok());
    }

    #[test]
    fn test_with_frame_meta_mut_keeps_block_handles_valid() {
        let mut arena = FrameArena::new();
        let id = arena.frame_new();

        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        arena.set_block(id, "atoms", block).unwrap();
        let handle = arena.get_block(id, "atoms").unwrap();

        arena
            .with_frame_meta_mut(id, |meta| meta.insert("title", "water"))
            .unwrap();

        // The meta write landed, and the block handle taken before it still
        // resolves at the same version.
        let title = arena
            .with_frame(id, |f| f.meta.get("title").cloned())
            .unwrap();
        assert!(title.is_some());
        assert!(arena.clone_block(&handle).is_ok());
        assert_eq!(
            arena.get_block(id, "atoms").unwrap().version,
            handle.version
        );
    }
}

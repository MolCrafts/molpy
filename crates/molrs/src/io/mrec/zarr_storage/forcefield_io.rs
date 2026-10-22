//! The `forcefield/` group of a record: the force-field document as the
//! group's attributes, one block group per style table.
//!
//! ```text
//! forcefield/                 attributes = the document, verbatim
//! └── <category>.<style>/     one block group per table (frame_io's layout)
//!     ├── <column>            array
//!     └── _validity/<column>  bool[T]   absent parameters
//! ```
//!
//! Contract: molrec `docs/spec/forcefield.md`. The document is plain JSON (no
//! `_meta_types`), written and read back key for key; the tables are blocks
//! written by [`write_block_group`] and read by [`read_block_group`], the one
//! description of a block every frame-shaped section shares. No unit is
//! converted in either direction.

use std::sync::Arc;

use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage,
    WritableStorageTraits,
};

use crate::io::mrec::zarr_storage::frame_io::{
    join_path, node_prefix, read_block_group, write_block_group,
};
use molrs::core::MolRsError;
use molrs::io::mrec::ForceFieldSection;

/// The record's root group holding its force field.
pub(crate) const FORCEFIELD_GROUP: &str = "forcefield";

/// Write `section` as the group at `prefix`, replacing whatever was there.
///
/// # Errors
///
/// Whatever [`ForceFieldSection::validate`] refuses — checked before anything
/// is erased — and the store errors of writing a block group.
pub(crate) fn write_forcefield_group(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    section: &ForceFieldSection,
) -> Result<(), MolRsError> {
    section.validate()?;
    store.erase_prefix(&node_prefix(prefix)?)?;
    GroupBuilder::new()
        .attributes(section.document.clone())
        .build(store.clone(), prefix)?
        .store_metadata()?;
    for (name, table) in &section.tables {
        write_block_group(store, &join_path(prefix, name), table)?;
    }
    Ok(())
}

/// Read the group at `prefix` back into a [`ForceFieldSection`] — the
/// attribute map is the document, every child group a table — then validated.
///
/// # Errors
///
/// A block group that fails to decode, or a section
/// [`ForceFieldSection::validate`] refuses.
pub(crate) fn read_forcefield_group<S>(
    store: &Arc<S>,
    prefix: &str,
) -> Result<ForceFieldSection, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let section = read_stored_forcefield(store, prefix)?;
    section.validate()?;
    Ok(section)
}

fn read_stored_forcefield<S>(store: &Arc<S>, prefix: &str) -> Result<ForceFieldSection, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut section = ForceFieldSection {
        document: zarrs::group::Group::open(store.clone(), prefix)?
            .attributes()
            .clone(),
        ..ForceFieldSection::default()
    };
    for child in Node::open(store, prefix)?.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let path = child.path().as_str();
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let table = read_block_group(store, path, name)?;
        section.tables.insert(name.to_owned(), table);
    }
    Ok(section)
}

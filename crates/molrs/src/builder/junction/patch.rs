//! [`Retype::Patch`](super::Retype::Patch): atom types the caller declares
//! per linked anchor.

use std::collections::{BTreeMap, HashMap};

use super::JunctionError;
use crate::core::keys;
use crate::core::{Atomistic, Junction, MolRsError, NodeId};

/// The atom types a link gives a copy of a template, declared per anchor.
///
/// An entry is keyed by a template name and the template row of an anchor;
/// its value maps template rows to the types those atoms **of the same copy**
/// take when that anchor gains a junction bond. Atoms are found by their
/// provenance — `frag_id` (the copy), `template` and `template_atom` (the
/// row), which the [`Assembler`](crate::builder::Assembler) stamps — so a
/// hand-built graph carries them itself.
///
/// Scope, stated: an entry is keyed per anchor only, so the same types apply
/// whatever the anchor is linked to — it fits a fixed linking chemistry and
/// cannot say "joined to A" apart from "joined to B". It changes types only:
/// no charge, atom or term, a type-only restriction of a CHARMM patch residue
/// (PRES). An anchor no entry names, and a copy none of whose anchors is
/// linked (a chain end), keep their types — a chain end is right only when the
/// unlinked template is typed as one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Patch {
    entries: HashMap<(String, usize), BTreeMap<usize, String>>,
}

/// One atom a [`Patch`] retyped: its type before (`None` when it had none)
/// and after.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Retyped {
    /// The atom.
    pub atom: NodeId,
    /// Its type before.
    pub from: Option<String>,
    /// Its type now.
    pub to: String,
}

/// An atom's provenance: copy, template, template row.
type Provenance = (i32, String, usize);

fn provenance(world: &Atomistic, atom: NodeId) -> Result<Option<Provenance>, MolRsError> {
    let props = world.get_atom(atom)?;
    let frag = props.get_int(keys::FRAG_ID);
    let template = props.get_str(keys::TEMPLATE);
    let row = props
        .get_int(keys::TEMPLATE_ATOM)
        .and_then(|r| usize::try_from(r).ok());
    Ok(match (frag, template, row) {
        (Some(frag), Some(template), Some(row)) => Some((frag, template.to_owned(), row)),
        _ => None,
    })
}

impl Patch {
    /// A patch with no entry: it changes nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// When row `anchor` of `template` is linked, give the atoms of the same
    /// copy at the rows of `types` those types. Replaces an earlier entry for
    /// the same anchor.
    pub fn insert(
        &mut self,
        template: impl Into<String>,
        anchor: usize,
        types: impl IntoIterator<Item = (usize, String)>,
    ) {
        self.entries
            .insert((template.into(), anchor), types.into_iter().collect());
    }

    /// The `(atom, type)` writes `junctions` call for, every check done.
    ///
    /// # Errors
    ///
    /// [`JunctionError::NoProvenance`], [`JunctionError::PatchedAtomMissing`],
    /// [`JunctionError::AmbiguousCopy`], [`JunctionError::PatchConflict`].
    pub(super) fn plan(
        &self,
        world: &Atomistic,
        junctions: &[Junction],
    ) -> Result<Vec<(NodeId, String)>, JunctionError> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }
        let mut copies: HashMap<Provenance, Vec<NodeId>> = HashMap::new();
        for id in world.atom_ids() {
            if let Some(key) = provenance(world, id)? {
                copies.entry(key).or_default().push(id);
            }
        }
        let mut writes: Vec<(NodeId, String)> = Vec::new();
        let mut written: HashMap<NodeId, usize> = HashMap::new();
        for (index, junction) in junctions.iter().enumerate() {
            for &anchor in &junction.anchors {
                let (frag_id, template, row) =
                    provenance(world, anchor)?.ok_or(JunctionError::NoProvenance {
                        junction: index,
                        atom: anchor,
                    })?;
                let Some(types) = self.entries.get(&(template.clone(), row)) else {
                    continue;
                };
                for (&target, ty) in types {
                    let atom = match copies
                        .get(&(frag_id, template.clone(), target))
                        .map(Vec::as_slice)
                    {
                        Some(&[atom]) => atom,
                        Some(many) if many.len() > 1 => {
                            return Err(JunctionError::AmbiguousCopy {
                                template,
                                frag_id,
                                row: target,
                                count: many.len(),
                            });
                        }
                        _ => {
                            return Err(JunctionError::PatchedAtomMissing {
                                template,
                                frag_id,
                                row: target,
                            });
                        }
                    };
                    match written.get(&atom) {
                        Some(&k) if writes[k].1 != *ty => {
                            return Err(JunctionError::PatchConflict {
                                atom,
                                types: [writes[k].1.clone(), ty.clone()],
                            });
                        }
                        Some(_) => {}
                        None => {
                            written.insert(atom, writes.len());
                            writes.push((atom, ty.clone()));
                        }
                    }
                }
            }
        }
        Ok(writes)
    }
}

/// Write each planned type; the atoms already of it are left out of the
/// record.
pub(super) fn apply(
    world: &mut Atomistic,
    writes: Vec<(NodeId, String)>,
) -> Result<Vec<Retyped>, MolRsError> {
    let mut retyped = Vec::new();
    for (atom, to) in writes {
        let from = world.get_atom(atom)?.get_str(keys::TYPE).map(str::to_owned);
        if from.as_deref() != Some(to.as_str()) {
            world.set_atom(atom, keys::TYPE, to.as_str())?;
            retyped.push(Retyped { atom, from, to });
        }
    }
    Ok(retyped)
}

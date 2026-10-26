//! Emit option flags for graph → SMILES.

use molrs::core::NodeId;

/// Options for [`SmilesIr::from_atomistic`](crate::io::smiles::SmilesIr::from_atomistic).
///
/// Every science or representation choice is an explicit field — no silent
/// policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmilesEmitOptions {
    /// Use WL [`canonical_order`](molrs::core::Atomistic::canonical_order)
    /// for root selection and branch ordering.
    pub canonical: bool,
    /// Override root atom; when set, root selection ignores `canonical` root pick
    /// (branch order may still use canonical colors).
    pub root: Option<NodeId>,
    /// Aromatic emission style.
    pub aromatic: AromaticEmit,
    /// How hydrogens appear in the string.
    pub hydrogens: HydrogenEmit,
    /// Emit tetrahedral and double-bond stereo when the graph states it.
    ///
    /// Tetrahedral `"CW"` / `"CCW"` is the clock sense of the reference
    /// substituent order. The written `@` / `@@` is that sense in this
    /// string's OpenSMILES neighbor order, and the glyph is omitted when the
    /// substituents this string writes are not those four. One bare protium
    /// this hydrogen policy skips is that centre's bracket `H`. Bond `"E"` /
    /// `"Z"` is double-bond geometry; `/` and `\` are written for the
    /// priority substituent on each end. `"up"` / `"down"` is a slash that is
    /// not that geometry (a ring digit, or a cumulene end): `/` / `\` from
    /// the bond's first endpoint toward its second, flipped when this string
    /// traverses the bond the other way.
    pub include_stereo: bool,
    /// Multi-component graph policy.
    pub multi_component: MultiComponentEmit,
    /// Prefer Daylight organic-subset letters when legal.
    pub organic_subset: bool,
}

/// Aromatic write policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AromaticEmit {
    /// Honour `is_aromatic` / aromatic bond markers on the graph.
    AsMarked,
    /// Ignore aromatic markers; require integer bond numbers.
    KekuleOnly,
}

/// Hydrogen write policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrogenEmit {
    /// Daylight omission for an atom that states no count and has no omitted
    /// protium neighbor. A stored count is written in a bracket. Each bare
    /// protium neighbor this policy does not draw is added to that count, so
    /// drawing the hydrogens and writing them does not drop them. A stored
    /// zero with nothing omitted is the bare bracket `[C]`.
    OrganicSubset,
    /// Write every atom, including hydrogen, as its own atom. A stored count
    /// is written as stated; omitted protium is not added on top.
    ExplicitAll,
    /// The same count as [`OrganicSubset`](Self::OrganicSubset): the stored
    /// count plus each bare protium neighbor that is not drawn as an atom.
    AsStored,
}

/// Multi-component policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiComponentEmit {
    /// Error if more than one connected component (safe default for systems).
    ErrorIfMultiple,
    /// Join components with `'.'`.
    JoinDot,
    /// Emit only the component containing `root`, or the first canonical component.
    FirstOnly,
}

impl Default for SmilesEmitOptions {
    fn default() -> Self {
        Self {
            canonical: true,
            root: None,
            aromatic: AromaticEmit::AsMarked,
            hydrogens: HydrogenEmit::OrganicSubset,
            include_stereo: false,
            multi_component: MultiComponentEmit::ErrorIfMultiple,
            organic_subset: true,
        }
    }
}

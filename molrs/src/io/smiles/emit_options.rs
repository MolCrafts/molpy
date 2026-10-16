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
    /// Emit tetrahedral / double-bond stereo markers when present on the graph.
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
    /// Daylight organic-subset omission of implicit H; skip explicit H atoms.
    OrganicSubset,
    /// Write every atom including H as bracket atoms.
    ExplicitAll,
    /// Use stored `h_count` / explicit H neighbours only.
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

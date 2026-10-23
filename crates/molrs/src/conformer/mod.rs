//! 3D conformer generation for molecular graphs.
//!
//! The public API is the [`Conformer`] struct: construct it with the desired
//! [`ConformerOptions`], then call [`Conformer::generate`] to produce 3D
//! coordinates. Internally this runs a staged ETKDGv3 workflow: initial
//! coordinate build -> coarse minimization -> rotor sampling -> final
//! minimization -> stereo sanity checks.
//!
//! [`Conformer::generate`] takes an [`Atomistic`] and returns a new one, with
//! its ports ([`molrs::core::Port`]) and `frag_id` labels intact. The
//! hydrogens the pipeline adds belong to no input unit, so relabelling them is
//! the caller's own step.
//!
//! ```no_run
//! use molrs::conformer::{Conformer, ConformerOptions};
//! use molrs::core::Atomistic;
//! # fn run(mol: &Atomistic, unit: &Atomistic) -> Result<(), molrs::core::MolRsError> {
//! let conformer = Conformer::new(ConformerOptions::default());
//!
//! // An `Atomistic` in, an `Atomistic` out.
//! let (mol_3d, report) = conformer.generate(mol)?;
//! println!("{} atoms, {} stages", mol_3d.n_atoms(), report.stages.len());
//!
//! // A ported unit keeps its ports — then label the added hydrogens.
//! let (mut unit_3d, _report) = conformer.generate(unit)?;
//! let labelled = unit_3d.inherit_frag_ids();
//! println!("{labelled} of {} atoms inherited a frag_id", unit_3d.n_atoms());
//! # Ok(())
//! # }
//! ```

mod distgeom;
mod options;
mod report;

/// ETKDGv3 conformer-embedding pipeline (the active [`Conformer`] backend);
/// reached only through [`Conformer::generate`].
mod etkdg;

pub use options::ConformerOptions;
pub use report::{ConformerReport, ConformerStageReport, StageKind};

use molrs::core::Atomistic;
use molrs::core::MolRsError;

/// 3D conformer generator for all-atom molecular graphs.
///
/// Holds the [`ConformerOptions`] supplied at construction; [`generate`] runs
/// the staged ETKDGv3 pipeline against an input molecule and never mutates it.
///
/// [`generate`]: Conformer::generate
#[derive(Debug, Clone)]
pub struct Conformer {
    opts: ConformerOptions,
}

impl Conformer {
    /// Build a generator from explicit options.
    pub fn new(opts: ConformerOptions) -> Self {
        Self { opts }
    }

    /// The options this generator was constructed with.
    pub fn options(&self) -> &ConformerOptions {
        &self.opts
    }

    /// Generate 3D coordinates for an all-atom molecule.
    ///
    /// An [`Atomistic`] is what the embedding requires: every atom carries an
    /// atomic number ([`keys::ATOMIC_NUMBER`](molrs::core::keys::ATOMIC_NUMBER)),
    /// which the pipeline reads for bond-length estimation, ring geometry and
    /// force-field selection. The result is a new `Atomistic`; coordinates
    /// are written in Å, and the input molecule is never modified.
    ///
    /// # What survives
    ///
    /// Every atom, relation and property of the input molecule, including its
    /// `ports` relations and `frag_id` properties: no stage of
    /// the pipeline removes a node, a relation or a property. The embed only
    /// appends hydrogen atoms and their bonds and writes `x` / `y` / `z`. The
    /// `ports` kind also rides through the MMFF staging `Frame` as an **unread
    /// block** — that frame is emitted one block per non-empty kind, and the
    /// potential compiler selects blocks by category name, so a `ports` block
    /// is carried across and never read.
    ///
    /// # `frag_id` on the hydrogens this adds
    ///
    /// With hydrogen addition on (the default) the output carries hydrogens
    /// that existed in no input fragment and therefore carry no `frag_id`.
    /// Labelling them is the caller's visible step, not a hidden hook inside
    /// this method: call `inherit_frag_ids` on the result, as the
    /// [module example](self) shows.
    ///
    /// # Cost
    ///
    /// One clone of the molecule, made by hydrogen perception inside the
    /// pipeline, ahead of the distance-geometry solve that dominates the run
    /// and outside its retry loop.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError`] when the molecule is empty or the embedding
    /// fails.
    pub fn generate(&self, mol: &Atomistic) -> Result<(Atomistic, ConformerReport), MolRsError> {
        etkdg::generate_3d_impl(mol, &self.opts)
    }
}

impl Default for Conformer {
    fn default() -> Self {
        Self::new(ConformerOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::{Conformer, ConformerOptions};
    use crate::props;
    use molrs::core::Atomistic;
    use molrs::core::BondNumber;
    use molrs::core::Element;
    use molrs::core::NodeId;
    use molrs::core::PortKind;
    use molrs::core::keys;

    /// `H–C–C–H` with a `$` port on each C–H valence and a `frag_id` on every
    /// atom: four atoms, three bonds, two ports, two distinct fragment labels
    /// (so a per-atom check cannot pass by broadcasting a single label).
    fn ported_pair() -> (Atomistic, [NodeId; 4]) {
        let mut frag = Atomistic::new();
        let c0 = frag
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c1 = frag
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let h0 = frag
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        let h1 = frag
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        frag.add_bond(c0, c1, props! {})
            .expect("both carbons are live");
        frag.add_bond(c0, h0, props! {})
            .expect("both endpoints are live");
        frag.add_bond(c1, h1, props! {})
            .expect("both endpoints are live");
        frag.add_port(c0, h0, PortKind::Symmetric, "A", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");
        frag.add_port(c1, h1, PortKind::Symmetric, "B", BondNumber::Single)
            .expect("a bonded H handle on its anchor is a legal port");
        for (atom, id) in [(c0, 7), (h0, 7), (c1, 9), (h1, 9)] {
            frag.set_atom(atom, keys::FRAG_ID, id)
                .expect("the atom is live");
        }
        (frag, [c0, c1, h0, h1])
    }

    fn conformer() -> Conformer {
        Conformer::new(ConformerOptions {
            rng_seed: Some(42),
            ..ConformerOptions::default()
        })
    }

    #[test]
    fn generate_keeps_ports_bonds_and_frag_ids_and_adds_hydrogens() {
        let (frag, atoms) = ported_pair();
        let (out, _report) = conformer().generate(&frag).expect("ethane embeds");

        // Each carbon gains two hydrogens: 4 + 4 atoms, 3 + 4 bonds.
        assert_eq!(out.n_atoms(), 8);
        assert_eq!(out.n_bonds(), 7, "a port is never counted as a bond");
        assert_eq!(out.n_ports(), 2, "the ports survive the embed");
        for (atom, want) in atoms.into_iter().zip([7, 9, 7, 9]) {
            let a = out.get_atom(atom).expect("an input atom stays live");
            assert_eq!(
                a.get_int(keys::FRAG_ID),
                Some(want),
                "frag_id of {atom:?} changed across the embed"
            );
            assert!(a.position().is_some(), "{atom:?} has coordinates");
        }
        assert_eq!(
            out.atoms()
                .filter(|(_, a)| a.get_int(keys::FRAG_ID).is_none())
                .count(),
            4,
            "the added hydrogens carry no frag_id until inherit_frag_ids"
        );
        assert_eq!(frag.n_atoms(), 4, "the input is never modified");
    }

    #[test]
    fn generate_refuses_an_empty_molecule() {
        let err = conformer()
            .generate(&Atomistic::new())
            .expect_err("nothing to embed");
        assert!(err.to_string().contains("empty"), "{err}");
    }

    /// Aromatic SMILES declares its ring through bond classes alone (no
    /// Kekulé numbers); the embed must still treat it as aromatic, which
    /// distance geometry turns into a flat ring.
    #[cfg(feature = "smiles")]
    #[test]
    fn generate_embeds_a_smiles_benzene_as_a_flat_ring() {
        use molrs::io::smiles::SmilesIr;
        use molrs::op::vec3::{cross, dot, normalize, sub};

        let mol = SmilesIr::parse("c1ccccc1")
            .expect("valid SMILES")
            .to_atomistic()
            .expect("benzene");
        let (out, _report) = conformer().generate(&mol).expect("benzene embeds");
        assert_eq!(out.n_atoms(), 12, "six hydrogens are added");

        let ring: Vec<[f64; 3]> = out
            .atoms()
            .filter(|(_, a)| a.atomic_number() == Some(Element::C.z()))
            .map(|(_, a)| a.position().expect("xyz"))
            .collect();
        assert_eq!(ring.len(), 6);
        let normal = normalize(cross(sub(ring[1], ring[0]), sub(ring[2], ring[0])))
            .expect("three ring carbons span a plane");
        for p in &ring {
            let off = dot(sub(*p, ring[0]), normal).abs();
            assert!(off < 0.05, "ring carbon {off} Å off the ring plane");
        }
    }
}

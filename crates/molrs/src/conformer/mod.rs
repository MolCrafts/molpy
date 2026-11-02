//! 3D conformer generation for molecular graphs.
//!
//! The public API is the [`Conformer`] struct: construct it with the
//! embedding parameters (RDKit `EmbedParameters` names), then call
//! [`Conformer::generate`]. It is ETKDGv3 distance-geometry embedding and
//! nothing else — initial coordinates, the distance-geometry minimization, the
//! experimental-torsion / basic-knowledge refinement and a stereo check. It
//! runs no force field: an MMFF94 (or any other) relaxation is the caller's
//! next step, composed from typing, `ff::compile::PotentialCompiler` and
//! `optimize::Lbfgs`, as RDKit's `EmbedMolecule` is followed by a separate
//! `MMFFOptimizeMolecule`.
//!
//! [`Conformer::generate`] takes an [`Atomistic`] and returns a new one, with
//! its ports ([`molrs::core::Port`]) and `frag_id` labels intact. The
//! hydrogens the pipeline adds belong to no input unit, so relabelling them is
//! the caller's own step.
//!
//! ```no_run
//! use molrs::conformer::Conformer;
//! use molrs::core::Atomistic;
//! # fn run(mol: &Atomistic, unit: &Atomistic) -> Result<(), molrs::core::MolRsError> {
//! let conformer = Conformer::new(true, 0, Some(42), false, true, true);
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
mod report;

/// ETKDGv3 conformer-embedding pipeline (the active [`Conformer`] backend);
/// reached only through [`Conformer::generate`].
mod etkdg;

pub use report::{ConformerReport, ConformerStageReport, StageKind};

use molrs::core::Atomistic;
use molrs::core::MolRsError;

/// ETKDGv3 3D embedding of an all-atom molecular graph, RDKit's
/// `EmbedMolecule`.
///
/// The parameters are fixed at construction; [`generate`] embeds an input
/// molecule and never mutates it. With a `seed` the result is reproducible.
///
/// [`generate`]: Conformer::generate
#[derive(Debug, Clone)]
pub struct Conformer {
    add_hydrogens: bool,
    max_iterations: usize,
    seed: Option<u64>,
    use_random_coords: bool,
    use_exp_torsion_angle_prefs: bool,
    use_basic_knowledge: bool,
}

impl Conformer {
    /// An embedder with RDKit's `EmbedParameters` of the same names.
    ///
    /// * `add_hydrogens` — add the implicit hydrogens as explicit atoms first.
    /// * `max_iterations` — embedding attempts before giving up (RDKit
    ///   `maxIterations`); `0` is RDKit's `10 × n_atoms`.
    /// * `seed` — the random seed; `None` draws one (reported as a warning).
    /// * `use_random_coords` — start every attempt from random coordinates
    ///   instead of the eigenvector embedding of a random distance matrix.
    /// * `use_exp_torsion_angle_prefs` — the ETKDG experimental-torsion terms
    ///   (the "ET").
    /// * `use_basic_knowledge` — flat aromatic rings and planar sp2 centres
    ///   (the "K"). With both torsion switches off the embedding is plain
    ///   distance geometry: no 3D refinement runs.
    pub fn new(
        add_hydrogens: bool,
        max_iterations: usize,
        seed: Option<u64>,
        use_random_coords: bool,
        use_exp_torsion_angle_prefs: bool,
        use_basic_knowledge: bool,
    ) -> Self {
        Self {
            add_hydrogens,
            max_iterations,
            seed,
            use_random_coords,
            use_exp_torsion_angle_prefs,
            use_basic_knowledge,
        }
    }

    /// Embed an all-atom molecule in 3D.
    ///
    /// Every atom carries an atomic number
    /// ([`keys::ATOMIC_NUMBER`](molrs::core::keys::ATOMIC_NUMBER)); a dummy
    /// atom (`0`, SMILES `*`) is embedded as RDKit embeds it — radii 0, no
    /// UFF type, so its bonds take the covalent-radius fallback (C–`*`
    /// 0.76 Å). The result is a new `Atomistic` with `x` / `y` / `z` in Å.
    ///
    /// # What survives
    ///
    /// Every atom, relation and property of the input molecule, including its
    /// `ports` relations and `frag_id` properties: the embedding only appends
    /// hydrogen atoms and their bonds and writes `x` / `y` / `z`.
    ///
    /// # `frag_id` on the hydrogens this adds
    ///
    /// With `add_hydrogens` the output carries hydrogens that existed in no
    /// input fragment and therefore carry no `frag_id`. Labelling them is the
    /// caller's visible step: call `inherit_frag_ids` on the result, as the
    /// [module example](self) shows.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError`] when the molecule is empty or the embedding
    /// fails.
    pub fn generate(&self, mol: &Atomistic) -> Result<(Atomistic, ConformerReport), MolRsError> {
        etkdg::generate_3d_impl(mol, self)
    }
}

impl Default for Conformer {
    /// RDKit's `ETKDGv3()` defaults: hydrogens added, `10 × n_atoms` attempts,
    /// a drawn seed, eigenvector start, ET and K on.
    fn default() -> Self {
        Self::new(true, 0, None, false, true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::Conformer;
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
        Conformer::new(true, 0, Some(42), false, true, true)
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

    fn coords(mol: &Atomistic) -> Vec<[f64; 3]> {
        mol.atoms()
            .map(|(_, a)| a.position().expect("xyz"))
            .collect()
    }

    /// A seed fixes the embedding: two runs with it are the same conformer,
    /// another seed is another one.
    #[test]
    fn a_seed_reproduces_the_embedding() {
        let (frag, _) = ported_pair();
        let one = coords(&conformer().generate(&frag).expect("embeds").0);
        let two = coords(&conformer().generate(&frag).expect("embeds").0);
        assert_eq!(one, two);
        let other = Conformer::new(true, 0, Some(43), false, true, true);
        assert_ne!(one, coords(&other.generate(&frag).expect("embeds").0));
    }

    /// The dummy atoms of a polymer repeat unit are embedded, RDKit-style,
    /// inside RDKit's C–`*` bounds (0.684..0.836 Å); no hydrogen lands on them.
    #[cfg(feature = "smiles")]
    #[test]
    fn a_repeat_unit_embeds_with_its_dummy_atoms() {
        use molrs::io::smiles::SmilesIr;

        let mol = SmilesIr::parse("[*]CC([*])C(=O)OC")
            .expect("valid SMILES")
            .to_atomistic()
            .expect("methyl acrylate repeat unit");
        let (out, _report) = conformer().generate(&mol).expect("a repeat unit embeds");
        let stars: Vec<_> = out
            .atoms()
            .filter(|(_, a)| a.atomic_number() == Some(0))
            .map(|(id, _)| id)
            .collect();
        assert_eq!(stars.len(), 2);
        for star in stars {
            let neighbours: Vec<_> = out
                .neighbors(star, molrs::core::NeighborKinds::BONDS)
                .map(|(n, _)| n)
                .collect();
            assert_eq!(neighbours.len(), 1, "a dummy atom keeps its one bond");
            let p = out.get_atom(star).expect("live").position().expect("xyz");
            let q = out
                .get_atom(neighbours[0])
                .expect("live")
                .position()
                .expect("xyz");
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            assert!(
                (0.684..=0.836).contains(&d),
                "C-* {d} Å outside RDKit's bounds"
            );
        }
    }

    /// The torsion refinement keeps the bonded skeleton: across seeds, every
    /// bond and every 1-3 distance of an ester and an anilide stays within
    /// 0.05 Å of its bounds (an under-weighted stage once pulled O=C-O down
    /// to 1.8 Å).
    #[cfg(feature = "smiles")]
    #[test]
    fn the_refinement_keeps_bonds_and_angles_in_their_bounds() {
        use crate::conformer::distgeom::DgConstraints;
        use molrs::io::smiles::SmilesIr;

        for smiles in ["c1ccccc1C(=O)OC", "CC(=O)Nc1ccccc1"] {
            let mol = SmilesIr::parse(smiles)
                .expect("valid SMILES")
                .to_atomistic()
                .expect("graph");
            for seed in 0..5 {
                let embedder = Conformer::new(true, 0, Some(seed), false, true, true);
                let (out, _report) = embedder.generate(&mol).expect("embeds");
                let c = DgConstraints::from_graph(&out).expect("constraints");
                let p = coords(&out);
                let dist = |i: usize, j: usize| {
                    ((p[i][0] - p[j][0]).powi(2)
                        + (p[i][1] - p[j][1]).powi(2)
                        + (p[i][2] - p[j][2]).powi(2))
                    .sqrt()
                };
                let pairs = c
                    .bonds
                    .iter()
                    .map(|&[i, j]| (i, j))
                    .chain(c.angles.iter().map(|a| (a.atoms[0], a.atoms[2])));
                for (i, j) in pairs {
                    let (d, lo, hi) = (dist(i, j), c.bounds.lower(i, j), c.bounds.upper(i, j));
                    assert!(
                        d > lo - 0.05 && d < hi + 0.05,
                        "{smiles} seed {seed}: {i}-{j} at {d:.3} Å, bounds {lo:.3}..{hi:.3}"
                    );
                }
            }
        }
    }

    /// A PEO heptamer's 3D refinement can relieve its torsion strain by
    /// straightening a C–O–C to 180° (seed 7 does), where the torsions through
    /// it are undefined; that attempt is rejected and the embedding keeps every
    /// angle of a centre that is not sp bent.
    #[cfg(feature = "smiles")]
    #[test]
    fn a_long_ether_chain_keeps_every_angle_bent() {
        use crate::conformer::distgeom::DgConstraints;
        use molrs::io::smiles::SmilesIr;

        let smiles = format!("COCC{}OCCOC", "OCC".repeat(5));
        let mol = SmilesIr::parse(&smiles)
            .expect("valid SMILES")
            .to_atomistic()
            .expect("PEO heptamer");
        let embedder = Conformer::new(true, 0, Some(7), false, true, true);
        let (out, _report) = embedder.generate(&mol).expect("the chain embeds");
        let c = DgConstraints::from_graph(&out).expect("constraints");
        let p = coords(&out);
        for a in c.angles.iter().filter(|a| !a.linear) {
            let [i, j, k] = a.atoms;
            let theta = molrs::op::vec3::angle(p[i], p[j], p[k]).to_degrees();
            assert!(theta <= 170.0, "{i}-{j}-{k} at {theta:.1}°");
        }
    }

    /// Random-coordinate starts and plain distance geometry (no ET, no K)
    /// both embed.
    #[test]
    fn random_coordinates_and_plain_distance_geometry_embed() {
        let (frag, _) = ported_pair();
        for embedder in [
            Conformer::new(true, 0, Some(7), true, true, true),
            Conformer::new(true, 0, Some(7), false, false, false),
        ] {
            let (out, _report) = embedder.generate(&frag).expect("ethane embeds");
            assert_eq!(out.n_atoms(), 8);
        }
    }
}

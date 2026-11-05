//! Expansion of the lowest `CGsmiles` level into one [`Atomistic`] graph.
//!
//! Each node of that level is one instance of its fragment definition: the
//! body is converted once per definition, and each run of consecutive nodes
//! naming the same definition is stamped into one accumulating graph by a
//! single [`Atomistic::replicate`], every copy carrying its node's index as
//! `frag_id`.
//! Every [`ResolvedPair`](crate::io::cgsmiles::ResolvedPair) of the level then
//! becomes one real bond, written with the class the pairing resolved rather
//! than left to the single bond [`Atomistic::add_bond`] defaults to.
//!
//! What comes out is **topology only** — the heavy atoms the bodies wrote, the
//! bonds they and the pairing imply, and the properties the notation declared.
//! A line notation states no geometry, so no atom carries a position; no
//! hydrogen is added and no perception is run. Those are steps a caller
//! composes.
//!
//! Only the lowest level becomes atoms. The pairs of an intermediate level
//! stay on the IR for a reader that wants the coarse graph itself.

use crate::io::cgsmiles::CgSmilesIr;
use crate::io::cgsmiles::templates::cg_build;
use crate::io::smiles::SmilesError;
use molrs::core::{Atomistic, GraphAccess, LinkRequest, RelationId};
use molrs::op::I;
use molrs::op::Rigid;

impl CgSmilesIr {
    /// Expand the lowest coarse-grained level into one [`Atomistic`] graph.
    ///
    /// Every node of that level becomes a disjoint copy of its fragment body,
    /// and every pair the reader resolved becomes one bond between two of
    /// those copies, carrying the class the pairing decided: the bond symbol
    /// the notation wrote next to either descriptor if it wrote one, otherwise
    /// aromatic if both port atoms are written aromatic, otherwise single.
    /// Unpaired single ports become implicit hydrogen caps. An unpaired
    /// non-single port requires an explicit cap fragment and is refused.
    ///
    /// # Per-atom instance membership
    ///
    /// Every atom carries the key **`frag_id`**, a [`PropValue::Int`](molrs::core::PropValue::Int) holding
    /// the index of the lowest-level node it came from, so a caller can
    /// partition the result by instance without re-deriving the grouping. It
    /// is not `mol_id` (a molecule id — coarse instances are sub-molecular)
    /// and not `res_id` (a biopolymer residue); the name mirrors the
    /// `CGsmiles` reference implementation's `fragid`.
    ///
    /// # Topology only
    ///
    /// No coordinates, no hydrogen repletion, no aromaticity perception and no
    /// kekulization: `to_atomistic` builds the graph the notation states and
    /// stops. Explicit atoms written in the bodies are retained; temporary
    /// port hydrogens are removed after linking, preserving declared stereo.
    ///
    /// # Indices, and what is guaranteed about them
    ///
    /// The pairing walks edges in **parse order**, while the reference
    /// implementation walks its graph library's adjacency order — for
    /// `{[#A]1[#B][#C]1}` that is (0,1), (1,2), (0,2) here against (0,1),
    /// (0,2), (1,2) there — so a ring-bearing string can pair different ports
    /// and lay its atoms out in a different order. What is contracted is the
    /// **connectivity**: the expansion is isomorphic to the reference's, and to
    /// what the molpy builder builds from the same string, as an unlabelled
    /// graph. Atom indices are not part of that contract; `frag_id` and the
    /// element are.
    ///
    /// Bond classes are not part of it either. The reference sets order 1.5
    /// whenever both endpoint atoms are aromatic, over a written symbol as
    /// well: biphenyl's `-[$]` bond is 1.5 there and
    /// [`BondKind::Single`](crate::io::smiles::BondKind) here,
    /// and a `=`-on-`=` pair is 1.5 there and `Double` here. molrs follows the
    /// Daylight Theory Manual, *SMILES* § 3.2.2 — a written symbol states the
    /// bond's order — and reads aromaticity only where neither side wrote one.
    ///
    /// A method, not a free function: the data it reads is this value's, and
    /// the free name
    /// [`SmilesIr::to_atomistic`](crate::io::smiles::SmilesIr::to_atomistic) already belongs to
    /// the SMILES pipeline, where it converts a [`SmilesIr`](crate::io::smiles::SmilesIr).
    /// A second free function of the same name in the same public module would
    /// be told apart only by its argument type.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable) when there is no atomistic body to
    /// expand: a base-only string (payload `base-only string (no fragment
    /// table)`) or a lowest level whose bodies are coarse graphs (payload the
    /// fragment's name).
    ///
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild) when a structural write to the graph
    /// fails — replicating a run of instances (which stamps `frag_id`),
    /// adding a bond, or classing it — or when the IR names a fragment its own
    /// table does not define, or a port the converted body does not hold. No
    /// fallible call on this path is discarded.
    ///
    /// Whatever converting a fragment body raises — an unmatched ring closure
    /// inside it, say — propagates with the converter's **own** kind, stamped
    /// [`Notation::CgSmiles`](crate::io::smiles::Notation::CgSmiles) and with
    /// no input text: a value returned by
    /// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse) has bodies that
    /// already converted once, so only a hand-built IR reaches this.
    ///
    /// # Examples
    ///
    /// An OH-capped PEO trimer: five beads over two definitions, eleven heavy
    /// atoms and ten bonds.
    ///
    /// ```
    /// use molrs::io::cgsmiles::CgSmilesIr;
    ///
    /// let ir = CgSmilesIr::parse("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}").unwrap();
    /// let mol = ir.to_atomistic().unwrap();
    ///
    /// assert_eq!(mol.n_atoms(), 11);
    /// assert_eq!(mol.n_bonds(), 10);
    /// ```
    pub fn to_atomistic(&self) -> Result<Atomistic, SmilesError> {
        self.expand_lowest_level()
    }

    /// Expand the lowest level into one graph.
    ///
    /// The expansion behind [`to_atomistic`](Self::to_atomistic): every node
    /// becomes a copy of its converted body, and every [`ResolvedPair`] of the
    /// level becomes one bond of the class it resolved. Each run of
    /// consecutive nodes naming one definition is placed by a single
    /// [`Atomistic::replicate`] under identity transforms, so node order — and
    /// hence `frag_id` — follows the level. Removing cap atoms may reorder rows.
    ///
    /// **`frag_id` is the lowest-level node index.** Every atom of node `i`'s
    /// copy carries `frag_id = i`.
    ///
    /// # Errors
    ///
    /// Those of [`to_atomistic`](Self::to_atomistic).
    fn expand_lowest_level(&self) -> Result<Atomistic, SmilesError> {
        let templates = self.templates()?;
        let assembly = self.lowest_assembly(&templates)?;
        let names = &assembly.names;
        let build = |reason| cg_build(self.span, reason);
        let mut mol = Atomistic::new();
        // Per instance, its ports in template port order.
        let mut ports: Vec<Vec<RelationId>> = Vec::with_capacity(names.len());
        let mut first = 0;
        while first < names.len() {
            let name = names[first];
            let last = names[first..]
                .iter()
                .position(|other| *other != name)
                .map_or(names.len(), |len| first + len);
            let template = &templates[name];
            let frag_ids = (first..last)
                .map(|instance| {
                    I::try_from(instance)
                        .map_err(|e| build(format!("instance {instance} is not an Int: {e}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let run = |e| build(format!("instances {first}..{last} of '{name}': {e}"));
            let handles = mol
                .replicate(template, &vec![Rigid::IDENTITY; frag_ids.len()], &frag_ids)
                .map_err(run)?;
            ports.extend(
                mol.graph()
                    .copied_ports(template.graph(), &handles)
                    .map_err(run)?,
            );
            first = last;
        }
        let requests = assembly
            .connections
            .iter()
            .map(|connection| {
                let port = |(instance, port): (usize, usize)| {
                    ports
                        .get(instance)
                        .and_then(|list| list.get(port))
                        .copied()
                        .ok_or_else(|| build(format!("no port {port} on instance {instance}")))
                };
                Ok(LinkRequest {
                    a: port(connection.ends[0])?,
                    b: port(connection.ends[1])?,
                    class: Some((connection.kind.bond_type(), connection.kind.bond_number())),
                })
            })
            .collect::<Result<Vec<_>, SmilesError>>()?;
        mol.link_many(&requests)
            .map_err(|e| build(format!("bond between two fragments: {e}")))?;

        // Only generated unpaired handles become implicit hydrogens; explicit
        // atoms written in a fragment body are never removed here. Every
        // handle left is a single port's: `lowest_assembly` refused any other.
        let caps = mol
            .port_ids()
            .map(|id| {
                mol.get_port(id)
                    .map(|port| port.handle)
                    .map_err(|e| build(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::perceive::implicitize_hydrogens(&mut mol, &caps)
            .map_err(|e| build(e.to_string()))?;
        Ok(mol)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::io::cgsmiles::fixtures::{body, fragment, pair};
    use crate::io::cgsmiles::{
        CgBondOrder, CgEdge, CgFragmentDef, CgGraph, CgNode, CgSmilesIr, EdgeOrigin, FragmentBody,
    };
    use crate::io::smiles::{BondKind, SmilesErrorKind, Span};
    use molrs::core::Atomistic;
    use molrs::core::Element;
    use molrs::core::NeighborKinds;
    use molrs::core::NodeId;
    use molrs::core::PropValue;
    use molrs::core::keys;
    use molrs::core::{BondNumber, BondOrder};

    // Every count below is hand-derived from the fixtures of § Domain basis of
    // `.claude/specs/cgsmiles-01d-resolve.md`: heavy atoms are counted off the
    // written bodies (no hydrogen is added here), intra-fragment bonds off the
    // same bodies, and one inter-fragment bond is added per resolved pair.
    // R4.14 fixes the class of an inter-fragment bond, R4.17 drops unpaired
    // ports, and R5.1 stamps each atom with its instance as `frag_id`. No
    // external program produced any value here.

    // -- helpers ------------------------------------------------------------

    /// The atomistic expansion of a `CGsmiles` string that must parse.
    fn expanded(text: &str) -> Atomistic {
        CgSmilesIr::parse(text)
            .unwrap_or_else(|e| panic!("CgSmilesIr::parse({text:?}) failed: {e}"))
            .to_atomistic()
            .unwrap_or_else(|e| panic!("to_atomistic({text:?}) failed: {e}"))
    }

    /// The instance an expanded atom belongs to, as the `frag_id` stamp states
    /// it. Panics unless the stamp is an [`PropValue::Int`] — the variant the
    /// contract names, and the one `mol_id` already uses.
    fn frag_id(mol: &Atomistic, id: NodeId) -> i32 {
        let atom = mol
            .get_atom(id)
            .unwrap_or_else(|e| panic!("atom {id:?} is missing: {e}"));
        match atom.get("frag_id") {
            Some(PropValue::Int(value)) => *value,
            other => panic!("atom {id:?} carries frag_id {other:?}"),
        }
    }

    /// `frag_id` of every atom, sorted: how many atoms each instance
    /// contributed, and that every atom carries the stamp.
    fn frag_ids(mol: &Atomistic) -> Vec<i32> {
        let mut ids: Vec<i32> = mol.atoms().map(|(id, _)| frag_id(mol, id)).collect();
        ids.sort();
        ids
    }

    /// The class of every bond whose two atoms belong to **different**
    /// instances — the bonds resolution created, told from the bodies' own by
    /// the stamp rather than by index arithmetic.
    fn inter_fragment_bonds(mol: &Atomistic) -> Vec<(BondOrder, BondNumber)> {
        let mut classes = Vec::new();
        for (bid, bond) in mol.bonds() {
            let (i, j) = (bond.nodes[0], bond.nodes[1]);
            if frag_id(mol, i) != frag_id(mol, j) {
                classes.push((mol.bond_type(bid).unwrap(), mol.bond_number(bid).unwrap()));
            }
        }
        classes
    }

    /// The number of bonds at each atom of instance `instance`, sorted.
    fn degrees_of(mol: &Atomistic, instance: i32) -> Vec<usize> {
        let mut degrees: Vec<usize> = mol
            .atoms()
            .filter(|(id, _)| frag_id(mol, *id) == instance)
            .map(|(id, _)| mol.neighbors(id, NeighborKinds::BONDS).count())
            .collect();
        degrees.sort();
        degrees
    }

    /// A coarse node named `name`, with nothing else on it: expansion reads a
    /// node's name and its index, never its descriptors.
    fn node(name: &str) -> CgNode {
        CgNode {
            name: name.to_owned(),
            charge: None,
            annotations: Vec::new(),
            descriptors: Vec::new(),
            parent: None,
            span: Span::new(0, 0),
        }
    }

    /// A written single edge between `i` and `j`.
    fn edge(i: usize, j: usize) -> CgEdge {
        CgEdge {
            i,
            j,
            order: CgBondOrder::Single,
            span: Span::new(0, 0),
            origin: EdgeOrigin::Written,
        }
    }

    /// One fragment table of atomistic bodies, `name` → parsed body.
    fn table(entries: &[(&str, &str)]) -> BTreeMap<String, CgFragmentDef> {
        entries
            .iter()
            .map(|(name, text)| {
                let body = fragment(text);
                (
                    (*name).to_owned(),
                    CgFragmentDef {
                        name: (*name).to_owned(),
                        body: FragmentBody::Smiles(body),
                        span: Span::new(0, 0),
                    },
                )
            })
            .collect()
    }

    // -- F5 and F7 from a hand-built IR -------------------------------------

    /// F5, polystyrene, as a resolved IR: BB0(PH1), BB2(PH3) with edges
    /// (0,1), (2,3), (0,2) and the three pairs resolution derives from them —
    /// BB's `$` to each phenyl's `$`, and BB0's `>` to BB2's `<`. Built by
    /// hand so expansion is exercised as its own unit.
    fn f5_ir() -> CgSmilesIr {
        CgSmilesIr {
            levels: vec![CgGraph {
                nodes: vec![node("BB"), node("PH"), node("BB"), node("PH")],
                edges: vec![edge(0, 1), edge(2, 3), edge(0, 2)],
            }],
            fragments: vec![table(&[("BB", "[>]CC[<][$]"), ("PH", "[$]c1ccccc1")])],
            pairs: vec![vec![
                pair(0, 0, body(0, 2), body(1, 0), BondKind::Single),
                pair(1, 0, body(2, 2), body(3, 0), BondKind::Single),
                pair(2, 0, body(0, 0), body(2, 1), BondKind::Single),
            ]],
            span: Span::new(0, 0),
        }
    }

    /// Two backbone carbons and six ring carbons twice over: 16 atoms. Bonds:
    /// one per backbone pair (2), six per ring (12), and three inter-fragment
    /// bonds — 17. The two unpaired ports add nothing (R4.17).
    #[test]
    fn test_f5_hand_built_ir_expands_to_sixteen_atoms_and_seventeen_bonds() {
        let mol = f5_ir().to_atomistic().expect("F5 must expand");
        assert_eq!(mol.n_atoms(), 16);
        assert_eq!(mol.n_bonds(), 17);
    }

    /// BB2's second carbon carries two resolved pairs — the `<` that bound it
    /// to BB0 and the `$` that bound it to its phenyl — on top of its own
    /// backbone bond, so it has degree 3, while BB2's first carbon keeps its
    /// unpaired `>` and stays at degree 1.
    #[test]
    fn test_f5_backbone_carbon_of_two_pairs_has_degree_three() {
        let mol = f5_ir().to_atomistic().expect("F5 must expand");
        assert_eq!(degrees_of(&mol, 2), vec![1, 3]);
    }

    /// F7, dichlorotoluene over three beads, as a resolved IR: the labels
    /// `$a`, `$b` and the unlabelled `$` leave the three edges no freedom, and
    /// every port atom is written aromatic, so every pair is aromatic.
    fn f7_ir() -> CgSmilesIr {
        CgSmilesIr {
            levels: vec![CgGraph {
                nodes: vec![node("SC4"), node("SX3"), node("SX3A")],
                edges: vec![edge(0, 1), edge(1, 2), edge(0, 2)],
            }],
            fragments: vec![table(&[
                ("SC4", "Cc[$a]c[$]"),
                ("SX3", "Clc[$a]c[$b]"),
                ("SX3A", "Clc[$b]c[$]"),
            ])],
            pairs: vec![vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Aromatic),
                pair(1, 0, body(1, 1), body(2, 0), BondKind::Aromatic),
                pair(2, 0, body(0, 1), body(2, 1), BondKind::Aromatic),
            ]],
            span: Span::new(0, 0),
        }
    }

    /// Three heavy atoms per body (`Cl` is one atom): 9. Two bonds per body
    /// plus three inter-bead bonds: 9, one ring.
    #[test]
    fn test_f7_hand_built_ir_expands_to_nine_atoms_and_nine_bonds() {
        let mol = f7_ir().to_atomistic().expect("F7 must expand");
        assert_eq!(mol.n_atoms(), 9);
        assert_eq!(mol.n_bonds(), 9);
    }

    // -- F2: counts, stamps and inter-fragment classes ----------------------

    /// F2 — an OH-capped PEO trimer.
    const F2: &str = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}";

    /// One oxygen per cap and three heavy atoms per `[$]COC[$]`: 11 atoms.
    /// Two bonds per PEO body (6) and four inter-fragment bonds: 10 — a tree.
    #[test]
    fn test_f2_expands_to_eleven_atoms_and_ten_bonds() {
        let mol = expanded(F2);
        assert_eq!(mol.n_atoms(), 11);
        assert_eq!(mol.n_bonds(), 10);
    }

    /// Every atom carries the index of the lowest-level node it came from, as
    /// an `Int`: one atom for each cap, three for each of the three PEO
    /// instances.
    #[test]
    fn test_f2_stamps_every_atom_with_its_instance_index() {
        let mol = expanded(F2);
        assert_eq!(frag_ids(&mol), vec![0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]);
    }

    /// The four bonds resolution created are plain single bonds — no symbol is
    /// written on any F2 descriptor and no port atom is aromatic — and they
    /// are given that class explicitly rather than left to `add_bond`'s
    /// default.
    #[test]
    fn test_f2_inter_fragment_bonds_are_single() {
        let mol = expanded(F2);
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondOrder::Single, BondNumber::Single); 4]
        );
    }

    // -- F3: an aromatic ring of beads --------------------------------------

    /// Martini benzene: two carbons per bead (6) and one bond per bead plus
    /// three inter-bead bonds (6), closing one ring.
    #[test]
    fn test_f3_expands_to_six_atoms_and_six_bonds() {
        let mol = expanded("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    /// The notation declares delocalization, not a Kekulé phase: the three
    /// inter-bead bonds are of the aromatic class with no localized number.
    #[test]
    fn test_f3_inter_bead_bonds_are_aromatic_with_an_unknown_number() {
        let mol = expanded("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}");
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondOrder::Aromatic, BondNumber::Unknown); 3]
        );
    }

    // -- F4: a double coarse edge is two single bonds -----------------------

    /// Martini cyclohexane: three carbons per bead (6), two bonds per bead
    /// plus the two bonds the one `=` edge stands for (6), closing one ring.
    #[test]
    fn test_f4_expands_to_six_atoms_and_six_bonds() {
        let mol = expanded("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    /// Both of them are single bonds. A single double bond here would be the
    /// coarse multiplicity read as chemistry.
    #[test]
    fn test_f4_inter_bead_bonds_are_both_single() {
        let mol = expanded("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}");
        assert_eq!(
            inter_fragment_bonds(&mol),
            vec![(BondOrder::Single, BondNumber::Single); 2]
        );
    }

    // -- F6: a tripeptide ---------------------------------------------------

    /// `[>]NCC(=O)[<]` is four heavy atoms and three bonds, `[>]NC(C)C(=O)[<]`
    /// five and four: 13 atoms, 10 intra-fragment bonds and 2 inter-fragment
    /// bonds.
    #[test]
    fn test_f6_expands_to_thirteen_atoms_and_twelve_bonds() {
        let mol = expanded("{[#GLY][#ALA][#GLY]}.{#GLY=[>]NCC(=O)[<],#ALA=[>]NC(C)C(=O)[<]}");
        assert_eq!(mol.n_atoms(), 13);
        assert_eq!(mol.n_bonds(), 12);
    }

    // -- the shared heavy-atom golden (assembly-06 ac-009) ------------------

    /// `{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}` is the heavy-atom skeleton of
    /// prop-1-en-1-ol, C–C=C–O: the `=` belongs to each descriptor, so the
    /// one resolved pair is a double bond. Hand-derived (spec assembly-06
    /// § Domain basis): heavy atoms C, C, C, O in chain order, bond numbers
    /// (1, 2, 1) along the chain, `frag_id` [0, 0, 1, 1].
    #[test]
    fn test_shared_golden_expands_to_the_c_c_eq_c_o_chain() {
        let mol = expanded("{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3, "an unbranched chain of four atoms");

        // Walk the chain from the end atom of instance 0.
        let start = mol
            .atoms()
            .map(|(id, _)| id)
            .find(|&id| {
                frag_id(&mol, id) == 0 && mol.neighbors(id, NeighborKinds::BONDS).count() == 1
            })
            .expect("instance 0 holds a chain end");
        let element = |id: NodeId| -> Element {
            let z = mol
                .get_atom(id)
                .unwrap_or_else(|e| panic!("atom {id:?} is missing: {e}"))
                .atomic_number()
                .unwrap_or_else(|| panic!("atom {id:?} carries no atomic number"));
            Element::by_number(z).unwrap_or_else(|| panic!("atom {id:?} is the wildcard"))
        };
        let mut elements = vec![element(start)];
        let mut numbers = Vec::new();
        let mut frags = vec![frag_id(&mol, start)];
        let (mut prev, mut here) = (None, start);
        while let Some((next, bid)) = mol
            .neighbors(here, NeighborKinds::BONDS)
            .find(|&(other, _)| Some(other) != prev)
        {
            elements.push(element(next));
            numbers.push(mol.bond_number(bid).unwrap());
            frags.push(frag_id(&mol, next));
            prev = Some(here);
            here = next;
        }

        assert_eq!(
            elements,
            vec![Element::C, Element::C, Element::C, Element::O]
        );
        assert_eq!(
            numbers,
            vec![BondNumber::Single, BondNumber::Double, BondNumber::Single]
        );
        assert_eq!(frags, vec![0, 0, 1, 1]);
    }

    // -- what expansion does not do -----------------------------------------

    /// Topology only: a notation string states no geometry, so no atom is
    /// given a position for a later step to mistake for one.
    #[test]
    fn test_expansion_writes_no_atom_positions() {
        let mol = expanded(F2);
        for (id, atom) in mol.atoms() {
            for key in [keys::X, keys::Y, keys::Z] {
                assert!(
                    !atom.contains_key(key),
                    "atom {id:?} carries {key:?}: {atom:?}"
                );
            }
        }
    }

    /// Hydrogen repletion is a step the caller composes: the expansion holds
    /// exactly the heavy atoms the bodies wrote — one per cap, three per PEO.
    #[test]
    fn test_expansion_adds_no_hydrogen_atoms() {
        let mol = expanded(F2);
        assert_eq!(mol.n_atoms(), 11);
        for (id, atom) in mol.atoms() {
            assert_ne!(
                atom.atomic_number(),
                Some(Element::H.z()),
                "atom {id:?} is a hydrogen"
            );
        }
    }

    // -- what expansion refuses ---------------------------------------------

    /// A lowest level whose bodies are coarse graphs has no atoms to build,
    /// and the refusal names the fragment it stopped at.
    #[test]
    fn test_graph_body_at_the_lowest_level_is_not_expandable() {
        let ir = CgSmilesIr {
            levels: vec![CgGraph {
                nodes: vec![node("A")],
                edges: Vec::new(),
            }],
            fragments: vec![BTreeMap::from([(
                "A".to_owned(),
                CgFragmentDef {
                    name: "A".to_owned(),
                    body: FragmentBody::Graph(CgGraph {
                        nodes: vec![node("X")],
                        edges: Vec::new(),
                    }),
                    span: Span::new(0, 0),
                },
            )])],
            pairs: vec![Vec::new()],
            span: Span::new(0, 0),
        };
        let err = ir
            .to_atomistic()
            .expect_err("a coarse lowest level cannot be expanded");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(ref name) if name == "A"),
            "kind was {:?}",
            err.kind
        );
    }

    /// One pair list per level is what the reader builds. A hand-built IR that
    /// drops the list — F5 with its `pairs` emptied — would otherwise expand
    /// to a bondless molecule, so the misalignment is named rather than
    /// honoured.
    #[test]
    fn test_ir_without_a_pair_list_per_level_is_refused() {
        let mut ir = f5_ir();
        ir.pairs = vec![];
        let err = ir
            .to_atomistic()
            .expect_err("an IR with no pair list cannot be expanded");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgBuild(ref msg) if msg.contains("pairs/levels misaligned")
            ),
            "kind was {:?}",
            err.kind
        );
    }

    /// A base-only string names no fragment at all, so the refusal has no
    /// fragment to name and says so in the phrase `templates` reuses.
    #[test]
    fn test_base_only_ir_is_not_expandable() {
        let ir = CgSmilesIr::parse("{[#PEO][#PEO][#PEO]}").expect("base-only string must parse");
        let err = ir
            .to_atomistic()
            .expect_err("a base-only string has no body to expand");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgNotExpandable(ref payload)
                    if payload == "base-only string (no fragment table)"
            ),
            "kind was {:?}",
            err.kind
        );
    }
}

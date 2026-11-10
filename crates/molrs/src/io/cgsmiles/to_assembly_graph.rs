//! The deepest resolved coarse level as an explicit assembly graph.

use crate::core::{CoarseGrain, NodeId, Props, keys};
use crate::io::cgsmiles::CgSmilesIr;
use crate::io::cgsmiles::templates::cg_build;
use crate::io::smiles::SmilesError;

impl CgSmilesIr {
    /// Build the deepest coarse level as an explicit assembly graph.
    ///
    /// Each bead names one template returned by [`Self::templates`]. Each
    /// resolved descriptor pair becomes one CG bond with `src_port` and
    /// `dst_port` indexing those templates' port tables, and `bond_type` /
    /// `bond_number` recording the resolved atomistic bond chemistry. Parallel
    /// connections and connections derived from upper levels are preserved.
    /// The graph carries no coordinates; generate template conformers before
    /// passing it to the builder's assembler.
    ///
    /// Unlike [`Self::to_coarsegrain`], this conversion needs the fragment
    /// definitions and does not project the base level or discard pairings.
    /// Port ordinals refer to the unchanged template definitions: generating
    /// coordinates is allowed, changing the template's port table is not.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable)
    /// for base-only input. Template conversion errors propagate. The
    /// pairing is validated as for [`Self::to_atomistic`] (a missing
    /// template, invalid pair end, reused port, incomplete edge pairing, or
    /// an unpaired non-single port) and refused with
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild);
    /// no partial graph is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::cgsmiles::CgSmilesIr;
    /// let ir = CgSmilesIr::parse("{[#A]=[#A]}.{#A=[$]CCC[$]}")?;
    /// let sites = ir.to_assembly_graph()?;
    /// assert_eq!(sites.n_beads(), 2);
    /// assert_eq!(sites.n_bonds(), 2); // two single connections, not one double bond
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_assembly_graph(&self) -> Result<CoarseGrain, SmilesError> {
        let templates = self.templates()?;
        let assembly = self.lowest_assembly(&templates)?;
        let mut cg = CoarseGrain::new();
        let beads = assembly
            .names
            .iter()
            .enumerate()
            .map(|(instance, name)| {
                cg.add_bead(Props::new().with(keys::BEAD_TYPE, *name))
                    .map_err(|e| cg_build(self.span, format!("instance {instance}: {e}")))
            })
            .collect::<Result<Vec<NodeId>, _>>()?;
        for (index, connection) in assembly.connections.iter().enumerate() {
            let build = |reason| cg_build(self.span, format!("resolved pair {index}: {reason}"));
            let [(src, src_port), (dst, dst_port)] = connection.ends;
            cg.add_site_bond(
                beads[src],
                beads[dst],
                (src_port, dst_port),
                connection.kind.bond_type(),
                connection.kind.bond_number(),
            )
            .map_err(|e| build(format!("adding the CG bond: {e}")))?;
        }
        Ok(cg)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "builder")]
    use std::collections::{BTreeMap, HashMap};

    #[cfg(feature = "builder")]
    use crate::builder::{Assembler, SitePlacer};
    #[cfg(feature = "builder")]
    use crate::core::{Atomistic, Element, NeighborKinds};
    use crate::core::{CoarseGrain, keys};
    use crate::io::cgsmiles::{CgSmilesIr, PairEnd};
    use crate::io::smiles::SmilesErrorKind;
    #[cfg(feature = "builder")]
    use crate::perceive::add_hydrogens;

    // Coordinates are deliberately just distinct finite points. These tests
    // exercise chemistry, not placement or a conformer's stochastic search.
    #[cfg(feature = "builder")]
    fn assembled(text: &str) -> Atomistic {
        let ir = CgSmilesIr::parse(text).unwrap();
        let mut library = HashMap::new();
        for (name, template) in ir.templates().unwrap() {
            let mut template = add_hydrogens(&template).unwrap();
            for (row, id) in template
                .atom_ids()
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
            {
                template.set_atom(id, keys::X, row as f64).unwrap();
                template.set_atom(id, keys::Y, (row % 2) as f64).unwrap();
                template.set_atom(id, keys::Z, 0.0).unwrap();
            }
            library.insert(name, template);
        }
        let mut sites = ir.to_assembly_graph().unwrap();
        for (row, id) in sites.bead_ids().collect::<Vec<_>>().into_iter().enumerate() {
            sites.set_bead(id, keys::X, 4.0 * row as f64).unwrap();
            sites.set_bead(id, keys::Y, 0.0).unwrap();
            sites.set_bead(id, keys::Z, 0.0).unwrap();
        }
        Assembler::new(library, Box::new(SitePlacer::new()), None)
            .assemble(&sites)
            .unwrap()
    }

    #[cfg(feature = "builder")]
    fn composition(mol: &Atomistic) -> BTreeMap<u8, usize> {
        let mut counts = BTreeMap::new();
        for (_, atom) in mol.atoms() {
            *counts.entry(atom.atomic_number().unwrap()).or_default() += 1;
        }
        counts
    }

    #[cfg(feature = "builder")]
    use crate::core::stereo::permuted_label;

    #[test]
    fn direct_refuses_unpaired_non_single_port_without_explicit_cap() {
        for text in ["{[#A]}.{#A=C=[>]}", "{[#A]}.{#A=[CH]#[>]}"] {
            let ir = CgSmilesIr::parse(text).unwrap();
            let error = ir.to_atomistic().unwrap_err().to_string();
            assert!(error.contains("instance 0"), "{error}");
            assert!(error.contains("non-single port 0"), "{error}");
            assert!(error.contains("explicit cap fragment"), "{error}");
        }
    }

    #[test]
    #[cfg(feature = "builder")]
    fn direct_and_assembled_chemistry_match_handwritten_compositions() {
        // Cyclohexane, benzene, prop-1-en-1-ol and dimethylammonium,
        // respectively: both paths are checked against independent formulae.
        for (text, formula, charge, n_heavy_bonds, n_aromatic, n_double) in [
            (
                "{[#A]=[#A]}.{#A=[$]CCC[$]}",
                vec![(1, 12), (6, 6)],
                0,
                6,
                0,
                0,
            ),
            (
                "{[#A]1[#A][#A]1}.{#A=[$]cc[$]}",
                vec![(1, 6), (6, 6)],
                0,
                6,
                6,
                0,
            ),
            (
                "{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}",
                vec![(1, 6), (6, 3), (8, 1)],
                0,
                3,
                0,
                1,
            ),
            (
                "{[#A][#B]}.{#A=[>]C,#B=[<][NH2+]C}",
                vec![(1, 8), (6, 2), (7, 1)],
                1,
                2,
                0,
                0,
            ),
        ] {
            let direct =
                add_hydrogens(&CgSmilesIr::parse(text).unwrap().to_atomistic().unwrap()).unwrap();
            for mol in [direct, assembled(text)] {
                assert_eq!(
                    composition(&mol),
                    formula.iter().copied().collect(),
                    "{text}"
                );
                assert_eq!(
                    mol.atoms().map(|(_, a)| a.formal_charge()).sum::<i32>(),
                    charge
                );
                let heavy = mol
                    .bonds()
                    .filter(|(_, bond)| {
                        bond.nodes.iter().all(|&id| {
                            mol.get_atom(id).unwrap().atomic_number() != Some(Element::H.z())
                        })
                    })
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                assert_eq!(heavy.len(), n_heavy_bonds);
                assert_eq!(
                    heavy
                        .iter()
                        .filter(|&&id| mol.bond_type(id).unwrap().is_aromatic())
                        .count(),
                    n_aromatic
                );
                assert_eq!(
                    heavy
                        .iter()
                        .filter(|&&id| mol.bond_number(id).unwrap().count() == 2)
                        .count(),
                    n_double
                );
            }
        }

        // Atom-map labels make positional isomers distinguishable. These
        // edge lists are read directly from the four written structures.
        for (text, mut expected) in [
            (
                "{[#A]=[#A]}.{#A=[$][CH2:1][CH2:2][CH2:3][$]}",
                vec![
                    ((0, 1), (0, 2), 1, 1),
                    ((0, 2), (0, 3), 1, 1),
                    ((1, 1), (1, 2), 1, 1),
                    ((1, 2), (1, 3), 1, 1),
                    ((0, 1), (1, 1), 1, 1),
                    ((0, 3), (1, 3), 1, 1),
                ],
            ),
            (
                "{[#A]1[#A][#A]1}.{#A=[$][cH:1][cH:2][$]}",
                vec![
                    ((0, 1), (0, 2), 4, 0),
                    ((1, 1), (1, 2), 4, 0),
                    ((2, 1), (2, 2), 4, 0),
                    ((0, 1), (1, 1), 4, 0),
                    ((1, 2), (2, 1), 4, 0),
                    ((0, 2), (2, 2), 4, 0),
                ],
            ),
            (
                "{[#A][#B]}.{#A=[CH3:1][CH:2]=[>],#B=[<]=[CH:3][OH:4]}",
                vec![
                    ((0, 1), (0, 2), 1, 1),
                    ((0, 2), (1, 3), 2, 2),
                    ((1, 3), (1, 4), 1, 1),
                ],
            ),
            (
                "{[#A][#B]}.{#A=[>][CH3:1],#B=[<][NH2+:2][CH3:3]}",
                vec![((0, 1), (1, 2), 1, 1), ((1, 2), (1, 3), 1, 1)],
            ),
        ] {
            expected.sort_unstable();
            for mol in [
                CgSmilesIr::parse(text).unwrap().to_atomistic().unwrap(),
                assembled(text),
            ] {
                let mut edges = Vec::new();
                for (id, bond) in mol.bonds() {
                    let a = mol.get_atom(bond.nodes[0]).unwrap();
                    let b = mol.get_atom(bond.nodes[1]).unwrap();
                    if a.atomic_number() == Some(1) || b.atomic_number() == Some(1) {
                        continue;
                    }
                    let a = (
                        a.get_int("frag_id").unwrap(),
                        a.get_int(keys::ATOM_MAP).unwrap(),
                    );
                    let b = (
                        b.get_int("frag_id").unwrap(),
                        b.get_int(keys::ATOM_MAP).unwrap(),
                    );
                    edges.push((
                        a.min(b),
                        a.max(b),
                        mol.bond_type(id).unwrap().code(),
                        mol.bond_number(id).unwrap().code(),
                    ));
                }
                edges.sort_unstable();
                assert_eq!(edges, expected, "{text}");
            }
        }
    }

    #[test]
    #[cfg(feature = "builder")]
    fn direct_and_assembled_keep_declared_tetrahedral_configuration() {
        let text = "{[#A][#B]}.{#A=F[C@](Cl)(Br)C[>],#B=[<]O}";
        for mol in [
            CgSmilesIr::parse(text).unwrap().to_atomistic().unwrap(),
            assembled(text),
        ] {
            let (center, atom) = mol
                .atoms()
                .find(|(_, atom)| atom.contains_key(keys::STEREO))
                .unwrap();
            let order: Vec<_> = mol
                .neighbors(center, NeighborKinds::BONDS)
                .map(|(id, _)| mol.get_atom(id).unwrap().atomic_number().unwrap())
                .collect();
            let reference = [9, 17, 35, 6]; // F, Cl, Br, C in the written @ order.
            let permutation: Vec<_> = order
                .iter()
                .map(|z| reference.iter().position(|value| value == z).unwrap())
                .collect();
            // Written `@` is anticlockwise in the F, Cl, Br, C order.
            assert_eq!(
                atom.get_str(keys::STEREO),
                Some(permuted_label(false, &permutation))
            );
        }
    }

    #[test]
    #[cfg(feature = "builder")]
    fn descriptor_position_preserves_anchor_chirality_in_both_paths() {
        for (body, expected) in [
            ("[C@H](F)(Cl)[>]", "CCW"),
            ("[C@H](F)([>])Cl", "CW"),
            ("[>][C@H](F)Cl", "CW"),
            ("[C@@H](F)(Cl)[>]", "CW"),
        ] {
            let text = format!("{{[#A][#B]}}.{{#A={body},#B=[<]C}}");
            for mol in [
                CgSmilesIr::parse(&text).unwrap().to_atomistic().unwrap(),
                assembled(&text),
            ] {
                let (center, atom) = mol
                    .atoms()
                    .find(|(_, atom)| matches!(atom.get_str(keys::STEREO), Some("CW" | "CCW")))
                    .expect("declared port-anchor stereo must survive");
                let mut order: Vec<_> = mol
                    .neighbors(center, NeighborKinds::BONDS)
                    .map(|(id, _)| mol.get_atom(id).unwrap().atomic_number().unwrap())
                    .collect();
                if atom.get_int(keys::H_COUNT) == Some(1) {
                    order.insert(0, 1);
                }
                let reference = [1, 9, 17, 6]; // H, F, Cl, the new methyl partner.
                assert_eq!(order.len(), 4);
                let permutation: Vec<_> = reference
                    .iter()
                    .map(|z| order.iter().position(|value| value == z).unwrap())
                    .collect();
                let normalized =
                    permuted_label(atom.get_str(keys::STEREO) == Some("CW"), &permutation);
                assert_eq!(normalized, expected, "{text}");
            }
        }
    }

    #[test]
    #[cfg(feature = "builder")]
    fn descriptor_substitution_preserves_alkene_geometry_in_both_paths() {
        let text = "{[#A][#B]}.{#A=F/C=C(/Cl)[>],#B=[<]Br}";
        for mol in [
            CgSmilesIr::parse(text).unwrap().to_atomistic().unwrap(),
            assembled(text),
        ] {
            let labels: Vec<_> = mol
                .bonds()
                .filter_map(|(_, bond)| bond.props.get_str(keys::STEREO).map(str::to_owned))
                .collect();
            // F and Cl are opposite in the body. Br replaces the opposite
            // cap and outranks Cl, so F and Br are on the same side: Z.
            assert_eq!(labels, ["Z"]);
        }
    }

    #[test]
    #[cfg(feature = "builder")]
    fn open_single_cap_keeps_stereo_when_made_implicit() {
        let text = "{[#A]}.{#A=F[C@](Cl)(Br)[>]}";
        for mol in [
            CgSmilesIr::parse(text).unwrap().to_atomistic().unwrap(),
            assembled(text),
        ] {
            let (center, atom) = mol
                .atoms()
                .find(|(_, atom)| atom.contains_key(keys::STEREO))
                .unwrap();
            let mut order: Vec<_> = mol
                .neighbors(center, NeighborKinds::BONDS)
                .map(|(id, _)| mol.get_atom(id).unwrap().atomic_number().unwrap())
                .collect();
            if atom.get_int(keys::H_COUNT) == Some(1) {
                order.insert(0, 1);
            }
            let reference = [1, 9, 17, 35];
            let permutation: Vec<_> = reference
                .iter()
                .map(|z| order.iter().position(|value| value == z).unwrap())
                .collect();
            assert_eq!(
                permuted_label(atom.get_str(keys::STEREO) == Some("CW"), &permutation),
                "CW"
            );
        }
    }

    fn connections(cg: &CoarseGrain) -> Vec<(usize, usize, i32, i32, i32, i32)> {
        let table = cg.bead_table();
        cg.bonds()
            .map(|(_, bond)| {
                (
                    table.row(bond.nodes[0]).unwrap(),
                    table.row(bond.nodes[1]).unwrap(),
                    bond.props.get_int(keys::SRC_PORT).unwrap(),
                    bond.props.get_int(keys::DST_PORT).unwrap(),
                    bond.props.get_int(keys::BOND_TYPE).unwrap(),
                    bond.props.get_int(keys::BOND_NUMBER).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn multiplicity_becomes_two_distinct_single_connections() {
        let ir = CgSmilesIr::parse("{[#A]=[#A]}.{#A=[$]CCC[$]}").unwrap();
        let cg = ir.to_assembly_graph().unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(connections(&cg), [(0, 1, 0, 0, 1, 1), (0, 1, 1, 1, 1, 1)]);
    }

    #[test]
    fn preserves_double_and_aromatic_bond_chemistry() {
        for (text, bond_type, bond_number) in [
            ("{[#A][#B]}.{#A=CC=[>],#B=[<]=CO}", 2, 2),
            ("{[#A][#A]}.{#A=[$]cc[$]}", 4, 0),
            ("{[#A][#A]}.{#A=[$]-c1ccccc1-[$]}", 1, 1),
        ] {
            let cg = CgSmilesIr::parse(text)
                .unwrap()
                .to_assembly_graph()
                .unwrap();
            assert_eq!(connections(&cg), [(0, 1, 0, 0, bond_type, bond_number)]);
        }
    }

    #[test]
    fn deepest_level_includes_written_and_derived_connections() {
        let ir = CgSmilesIr::parse(
            "{[#B1][#B2][#B1]}.\
             {#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
             {#PEO=[>]COC[<],#PE=[>]CC[<]}",
        )
        .unwrap();
        let cg = ir.to_assembly_graph().unwrap();
        let names: Vec<_> = cg
            .beads()
            .map(|(_, props)| props.get_str(keys::BEAD_TYPE).unwrap().to_owned())
            .collect();
        assert_eq!(names, ["PEO", "PEO", "PE", "PE", "PEO", "PEO"]);
        assert_eq!(
            connections(&cg),
            [
                (0, 1, 0, 1, 1, 1),
                (2, 3, 0, 1, 1, 1),
                (4, 5, 0, 1, 1, 1),
                (0, 3, 1, 0, 1, 1),
                (2, 5, 1, 0, 1, 1),
            ]
        );
    }

    #[test]
    fn upper_level_multiplicity_keeps_parallel_derived_connections() {
        let cg = CgSmilesIr::parse("{[#A]=[#A]}.{#A=[$][#B][$]}.{#B=[$]CCC[$]}")
            .unwrap()
            .to_assembly_graph()
            .unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(connections(&cg), [(0, 1, 0, 0, 1, 1), (0, 1, 1, 1, 1, 1)]);
    }

    #[test]
    fn isolated_defined_fragment_is_an_assembly_site() {
        let cg = CgSmilesIr::parse("{[#A]}.{#A=[$]CC[$]}")
            .unwrap()
            .to_assembly_graph()
            .unwrap();
        assert_eq!((cg.n_beads(), cg.n_bonds()), (1, 0));
    }

    #[test]
    fn base_only_cannot_be_an_assembly_graph() {
        let err = CgSmilesIr::parse("{[#A][#A]}")
            .unwrap()
            .to_assembly_graph()
            .unwrap_err();
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(ref message) if message.contains("base-only"))
        );
    }

    /// Both doors read the pairing through one validation: a hand-edited IR
    /// that drops or corrupts a pair is refused by direct expansion as by
    /// the assembly graph, never expanded with the lost bond capped by H.
    #[test]
    fn malformed_pairs_refuse_instead_of_losing_connections() {
        let ir = CgSmilesIr::parse("{[#A]=[#A]}.{#A=[$]CCC[$]}").unwrap();
        let corruptions: [fn(&mut CgSmilesIr); 6] = [
            |bad| {
                bad.pairs[0].pop();
            },
            |bad| bad.pairs[0][0].src = PairEnd::Sub { node: 0, port: 0 },
            |bad| {
                bad.pairs[0][0].src = PairEnd::Body {
                    instance: 0,
                    port: 9,
                }
            },
            |bad| {
                bad.pairs[0][0].src = PairEnd::Body {
                    instance: 9,
                    port: 0,
                }
            },
            |bad| {
                bad.pairs[0][1].src = PairEnd::Body {
                    instance: 0,
                    port: 0,
                }
            },
            |bad| bad.pairs.clear(),
        ];
        for corrupt in corruptions {
            let mut bad = ir.clone();
            corrupt(&mut bad);
            let graph = bad.to_assembly_graph().unwrap_err();
            let direct = bad.to_atomistic().unwrap_err();
            assert!(matches!(graph.kind, SmilesErrorKind::CgBuild(_)), "{graph}");
            assert!(
                matches!(direct.kind, SmilesErrorKind::CgBuild(_)),
                "{direct}"
            );
        }
    }
}

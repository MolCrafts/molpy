//! Tests for MMFF94 typifier.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::props;
    use std::collections::{BTreeMap, BTreeSet};

    use crate::ff::forcefield::ForceField;
    use crate::ff::typifier::mmff::Mmff94Typifier;
    use molrs::core::{Atomistic, BondNumber, BondOrder, Element, NodeId, keys};
    use molrs::core::{PropValue, Props};

    fn atom(element: Element) -> Props {
        props! { keys::ATOMIC_NUMBER => element }
    }

    fn bond_order(mol: &mut Atomistic, a: NodeId, b: NodeId, order: f64) {
        if let Ok(bid) = mol.add_bond(a, b, props! {}) {
            // The fixture's float order, split into class (1.5 = aromatic) and count.
            let (class, number) = if (order - 1.5).abs() < 1e-6 {
                (BondOrder::Aromatic, BondNumber::Unknown)
            } else {
                match order as u32 {
                    1 => (BondOrder::Single, BondNumber::Single),
                    2 => (BondOrder::Double, BondNumber::Double),
                    3 => (BondOrder::Triple, BondNumber::Triple),
                    other => panic!("fixture bond order {other}"),
                }
            };
            let _ = mol.set_bond_class(bid, class, number);
        }
    }

    fn test_typifier() -> Mmff94Typifier {
        Mmff94Typifier::new()
    }

    // -----------------------------------------------------------------------
    // XML loading tests
    // -----------------------------------------------------------------------

    /// MMFF types drawn hydrogens only: a molecule whose hydrogens are
    /// implicit is refused with the atom named, not typed as a radical.
    #[test]
    fn implicit_hydrogens_are_refused() {
        let mut mol = Atomistic::new();
        mol.add_atom(atom(Element::C)).unwrap();
        let mut typing = crate::ff::typifier::Typing::new(test_typifier());
        let err = typing.typify(&mol).unwrap_err().to_string();
        assert!(
            err.contains("atom 0") && err.contains("does not draw"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------
    // Atom typing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ethane_atom_types() {
        // CH3-CH3 through the live typify path (RDKit-validated front-end): both
        // C are MMFF type 1 (CR), all H are type 5 (HC). Atom rows follow
        // insertion order — c1, c2, then the six H — so rows 0..2 are carbons.
        let typifier = test_typifier();
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom(atom(Element::C)).unwrap();
        let c2 = mol.add_atom(atom(Element::C)).unwrap();
        bond_order(&mut mol, c1, c2, 1.0);
        for _ in 0..3 {
            let h = mol.add_atom(atom(Element::H)).unwrap();
            bond_order(&mut mol, c1, h, 1.0);
        }
        for _ in 0..3 {
            let h = mol.add_atom(atom(Element::H)).unwrap();
            bond_order(&mut mol, c2, h, 1.0);
        }

        let frame = crate::ff::typifier::Typing::new(typifier)
            .typify(&mol)
            .expect("typify ethane")
            .to_frame()
            .expect("a schema-conforming graph converts");
        let types = frame
            .get("atoms")
            .unwrap()
            .get("type")
            .and_then(|c| c.get::<String>())
            .expect("atoms.type column");
        assert_eq!(types[0], "1", "C1 should be MMFF type 1 (CR)");
        assert_eq!(types[1], "1", "C2 should be MMFF type 1 (CR)");
        for (i, t) in types.iter().enumerate().skip(2) {
            assert_eq!(t, "5", "H at row {i} should be type 5 (HC)");
        }
    }

    // -----------------------------------------------------------------------
    // Bond / angle / torsion type classification
    // -----------------------------------------------------------------------
    //
    // The seven unit tests that lived here drove `Mmff94Typifier::typify_bond` /
    // `typify_angle` / `typify_dihedral` — three front-door methods over
    // `typifier/mmff/classify.rs`, a second implementation of MMFF's context
    // rules that `ff/mmff/params.rs` already implements correctly. All three are
    // deleted (`mmff-orthogonal-02`), and so are the tests, because the values
    // they pinned were WRONG:
    //
    //   * `typify_bond(37, 37, 1.5) == 1` — an aromatic bond is bond type **0**.
    //     `getMMFFBondType` returns 1 only for a bond that is SINGLE and joins two
    //     sbmb/arom types; after MMFF aromaticity perception a ring bond is
    //     AROMATIC, never SINGLE. Backwards.
    //   * `typify_angle(bt_ij, bt_jk)` — the signature cannot express the rule.
    //     A C-C-C angle in cyclopropane is angle type **3**, and no function of
    //     two bond types can say so: ring membership is not among its arguments.
    //
    // The replacements assert against RDKit's answers, on molecules rather than on
    // bare integers: `tests/ff/typifier/mmff_labels.rs`.

    // -----------------------------------------------------------------------
    // Typing<Mmff94Typifier> output (system-forcefield-07)
    // -----------------------------------------------------------------------

    /// Explicit-H 1,3-butadiene `H2C=CH-CH=CH2`, hand-built: carbons
    /// `C0=C1-C2=C3` are atoms 0..=3; hydrogens follow (two on C0, one on C1,
    /// one on C2, two on C3).
    fn butadiene() -> (Atomistic, [NodeId; 4]) {
        let mut mol = Atomistic::new();
        let c: [NodeId; 4] = std::array::from_fn(|_| mol.add_atom(atom(Element::C)).unwrap());
        bond_order(&mut mol, c[0], c[1], 2.0);
        bond_order(&mut mol, c[1], c[2], 1.0);
        bond_order(&mut mol, c[2], c[3], 2.0);
        for (carbon, n_h) in [(c[0], 2), (c[1], 1), (c[2], 1), (c[3], 2)] {
            for _ in 0..n_h {
                let h = mol.add_atom(atom(Element::H)).unwrap();
                bond_order(&mut mol, carbon, h, 1.0);
            }
        }
        (mol, c)
    }

    fn str_prop(props: &Props, key: &str) -> Option<String> {
        match props.get(key) {
            Some(PropValue::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Per `(category, style)` of `ff`, the set of type names; styles holding
    /// no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    /// Butadiene types through the base without a conflict, and its two
    /// C-C-C angles (angle type 1: one C=C bond, bond type 0, and the central
    /// C-C single bond between two sbmb carbons, bond type 1) carry different
    /// `stbn_type`s of the form `{sbt}_{i}_{j}_{k}`.
    ///
    /// Hand-derived from MMFF's stretch-bend classes (Halgren 1996; the angle
    /// read in its own node order i-j-k): angle type 1 with BT_ij = 1,
    /// BT_jk = 0 is SBT 1, with BT_ij = 0, BT_jk = 1 is SBT 2. Both carbons'
    /// MMFF type is 2 (`C=C`), so the label is `{sbt}_2_2_2`.
    #[test]
    fn typing_butadiene_gives_its_two_stbn_orientations_distinct_names() {
        let (mol, c) = butadiene();
        let mut typing = crate::ff::typifier::Typing::new(Mmff94Typifier::new());
        let typed = typing
            .typify(&mol)
            .expect("butadiene types without a TypeConflict");

        let is_carbon = |id: &NodeId| c.contains(id);
        let ccc: Vec<(Vec<NodeId>, String)> = typed
            .angles()
            .filter(|(_, a)| a.nodes.iter().all(is_carbon))
            .map(|(_, a)| {
                let stbn = str_prop(&a.props, "stbn_type").expect("every angle has a stbn_type");
                (a.nodes.to_vec(), stbn)
            })
            .collect();
        assert_eq!(ccc.len(), 2, "butadiene has two C-C-C angles");

        for (nodes, stbn) in &ccc {
            // i-j is the terminal C=C (BT 0) exactly when i is C0 or C3.
            let ij_is_double = nodes[0] == c[0] || nodes[0] == c[3];
            let sbt = if ij_is_double { 2 } else { 1 };
            assert_eq!(stbn, &format!("{sbt}_2_2_2"), "angle {nodes:?}");
        }
        assert_ne!(ccc[0].1, ccc[1].1, "the two orientations are two names");
    }

    /// The output holds exactly the stamped labels, per `(category, style)`:
    /// bond `type` under `mmff_bond`, angle `type` under `mmff_angle`, angle
    /// `stbn_type` under `mmff_stbn`, dihedral `type` under `mmff_torsion`,
    /// improper `type` under `mmff_oop`, and the atom `type`s used as the
    /// `mmff_vdw` pair rows. No other style holds a type.
    #[test]
    fn typing_butadiene_output_names_equal_the_stamped_labels() {
        let (mol, _) = butadiene();
        let mut typing = crate::ff::typifier::Typing::new(Mmff94Typifier::new());
        let typed = typing.typify(&mol).expect("butadiene types");

        let atoms: BTreeSet<String> = typed
            .atoms()
            .map(|(_, a)| a.get_str("type").expect("every atom typed").to_owned())
            .collect();
        let labels = |rows: Vec<Props>, key: &str| -> BTreeSet<String> {
            rows.iter()
                .map(|p| str_prop(p, key).unwrap_or_else(|| panic!("every row has {key}")))
                .collect()
        };
        let bonds: Vec<_> = typed.bonds().map(|(_, b)| b.props).collect();
        let angles: Vec<_> = typed.angles().map(|(_, a)| a.props).collect();
        let dihedrals: Vec<_> = typed.dihedrals().map(|(_, d)| d.props).collect();
        let impropers: Vec<_> = typed.impropers().map(|(_, i)| i.props).collect();
        assert!(!impropers.is_empty(), "the four sp2 carbons are trigonal");

        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let expected = BTreeMap::from([
            (key("bond", "mmff_bond"), labels(bonds, "type")),
            (key("angle", "mmff_angle"), labels(angles.clone(), "type")),
            (key("angle", "mmff_stbn"), labels(angles, "stbn_type")),
            (key("dihedral", "mmff_torsion"), labels(dihedrals, "type")),
            (key("improper", "mmff_oop"), labels(impropers, "type")),
            (key("pair", "mmff_vdw"), atoms),
        ]);
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    /// Explicit-H 2H-azet-2-one-like four-membered ring `O=C1-N=C-C1`,
    /// hand-built. Atom order is load-bearing (see the test below):
    /// `c_co` (the carbonyl C, MMFF type 3) is atom 0, `c_im` (the imine C of
    /// the ring `C=N`, type 3) atom 1, `c_h2` (the sp3 ring CH2, type 20 `CR4R`)
    /// atom 2, `n` (the ring imine N, type 9 `N=C`) atom 3, `o` (type 7) atom 4;
    /// the three hydrogens follow.
    fn azetone() -> (Atomistic, [NodeId; 4]) {
        let mut mol = Atomistic::new();
        let c_co = mol.add_atom(atom(Element::C)).unwrap();
        let c_im = mol.add_atom(atom(Element::C)).unwrap();
        let c_h2 = mol.add_atom(atom(Element::C)).unwrap();
        let n = mol.add_atom(atom(Element::N)).unwrap();
        let o = mol.add_atom(atom(Element::O)).unwrap();
        bond_order(&mut mol, c_co, o, 2.0);
        bond_order(&mut mol, c_co, n, 1.0);
        bond_order(&mut mol, n, c_im, 2.0);
        bond_order(&mut mol, c_im, c_h2, 1.0);
        bond_order(&mut mol, c_h2, c_co, 1.0);
        for (heavy, n_h) in [(c_im, 1), (c_h2, 2)] {
            for _ in 0..n_h {
                let h = mol.add_atom(atom(Element::H)).unwrap();
                bond_order(&mut mol, heavy, h, 1.0);
            }
        }
        (mol, [c_co, c_im, c_h2, n])
    }

    fn f64_prop(props: &Props, key: &str) -> f64 {
        match props.get(key) {
            Some(PropValue::F64(v)) => *v,
            other => panic!("{key} is not an f64: {other:?}"),
        }
    }

    /// ac-004: one MMFF torsion name names one parameter set — the torsion
    /// label must carry every input of the resolver, not only the principal
    /// torsion type and the four atom types.
    ///
    /// `resolve::torsion_params` also reads the **secondary** torsion type
    /// (`torsion_type(..).1`): when all four equivalence levels of the
    /// principal type miss, `torsion_lookup` restarts on the secondary type.
    /// Two ring torsions of `O=C1-N=C-C1` read from the CH2 end are both
    /// `c_h2-C-N-C`, MMFF types `20-3-9-3`, principal type 4 (all four atoms in
    /// the 4-ring, no 1-3 bond across it):
    ///
    /// * `c_h2-c_co-n-c_im`: the j-k bond `C(=O)-N` is SINGLE between two
    ///   sbmb types (3, 9) → bond type 1 → secondary type **1**;
    /// * `c_h2-c_im=n-c_co`: the j-k bond `C=N` is DOUBLE → bond type 0 →
    ///   secondary type **0**.
    ///
    /// The principal-4 table has no row reaching `(3, 9)` at any level, so the
    /// first resolves off the type-1 row `1 0 3 9 0` (v2 = 1.8) and the second
    /// off the type-0 row `0 0 3 9 0` (v2 = 16.0) — different parameters under
    /// one `{tt}_{types}` label, `4_20_3_9_3`. Typing must still succeed, with
    /// each parameter set under its own name.
    ///
    /// Orientation: dihedrals are enumerated `i-j-k-l` with `j < k` in atom
    /// order; with `n` after both ring carbons, both torsions come out read
    /// from the CH2 end, so their labels coincide under today's grammar.
    #[test]
    fn typing_names_ring_torsions_with_different_secondary_types_apart() {
        let (mol, [c_co, c_im, c_h2, n]) = azetone();
        let mut typing = crate::ff::typifier::Typing::new(Mmff94Typifier::new());
        let typed = typing
            .typify(&mol)
            .expect("one torsion label names one parameter set, so no TypeConflict");

        let find = |path: [NodeId; 4]| {
            let (_, d) = typed
                .dihedrals()
                .find(|(_, d)| {
                    let nodes = d.nodes.to_vec();
                    nodes == path || nodes.iter().rev().copied().eq(path)
                })
                .unwrap_or_else(|| panic!("dihedral {path:?} enumerated"));
            let name = str_prop(&d.props, "type").expect("every dihedral has a type");
            let v = ["v1", "v2", "v3"].map(|k| f64_prop(&d.props, k));
            (name, v)
        };
        let (single_name, single_v) = find([c_h2, c_co, n, c_im]);
        let (double_name, double_v) = find([c_h2, c_im, n, c_co]);

        assert_ne!(
            single_v, double_v,
            "premise: the secondary torsion type changes the resolved (v1, v2, v3)"
        );
        assert_ne!(
            single_name, double_name,
            "two parameter sets, two names ({single_name} / {double_name})"
        );

        // Every name the output defines holds exactly the params stamped under it.
        let torsions = typing
            .forcefield()
            .get_style("dihedral", "mmff_torsion")
            .expect("mmff_torsion declared");
        for (name, v) in [(&single_name, single_v), (&double_name, double_v)] {
            let (_, params) = torsions
                .defs()
                .collect_type_params()
                .into_iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} defined in the output"));
            let defined = ["v1", "v2", "v3"].map(|k| {
                params
                    .get(k)
                    .and_then(|v| v.as_num())
                    .expect("v1..v3 defined")
            });
            assert_eq!(defined, v, "{name} defines the params stamped under it");
        }
    }
}

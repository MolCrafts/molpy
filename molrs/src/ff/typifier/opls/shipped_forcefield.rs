//! The shipped OPLS-AA parameter set, assembled from the compiled tables.
//!
//! [`OplsAaTypifier::oplsaa`](super::OplsAaTypifier::oplsaa) parses no XML:
//! the potential [`ForceField`] comes from [`crate::ff::params::oplsaa`] (generated from
//! GROMACS `oplsaa.ff`, in molrs units), the typing metadata from the molrs-owned
//! rules of [`crate::ff::params::oplsaa_typing`], joined to the atom rows by
//! name.
//!
//! A caller's own OPLS / CL&P / CL&Pol file, layers and all, is read —
//! by `molrs::io` — and handed to
//! [`OplsAaTypifier::new`](super::OplsAaTypifier::new). molrs does not
//! re-parse *its own* parameter set at runtime.

use std::collections::{HashMap, HashSet};

use crate::core::constants::{OPLS_COULOMB_14, OPLS_LJ_14, VACUUM_DIELECTRIC};
use crate::ff::forcefield::{DefError, ForceField};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::params::OPLSAA_TYPING;
use crate::ff::params::{
    OPLSAA_ANGLES, OPLSAA_ATOMS, OPLSAA_BONDS, OPLSAA_DIHEDRALS, OPLSAA_MIXING, OPLSAA_NAME,
};
use crate::ff::params::{OplsAtomRow, OplsRuleRow};
use molrs::core::TypeName;
use molrs::core::constants::COULOMB_REAL;

use super::typing_metadata::{OplsTypeRow, OplsTypingMetadata};

/// Build the shipped [`ForceField`].
///
/// An input-free constructor over a compiled table: the one definition result
/// it can meet is the table's own, and
/// `tests::force_field_defines_without_conflict` proves it `Ok`.
pub(super) fn force_field() -> ForceField {
    try_force_field().expect(
        "OPLS-AA table defines without conflict — proved by \
         ff::typifier::opls::shipped_forcefield::tests::force_field_defines_without_conflict",
    )
}

/// The fallible body of [`force_field`].
///
/// Style order is the source file's section order, and it is load-bearing:
/// `ForceField` lookups scan and take the first match.
fn try_force_field() -> Result<ForceField, DefError> {
    let mut ff = ForceField::new(OPLSAA_NAME);

    let bonds = ff.def_style("bond", "harmonic", Params::new())?;
    for row in OPLSAA_BONDS {
        let ends = [row.i, row.j];
        bonds.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter([("k", row.force_constant), ("r0", row.r0)]),
        )?;
    }

    let angles = ff.def_style("angle", "harmonic", Params::new())?;
    for row in OPLSAA_ANGLES {
        let ends = [row.i, row.j, row.k];
        angles.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter([("k", row.force_constant), ("theta0", row.theta0)]),
        )?;
    }

    let dihedrals = ff.def_style("dihedral", "opls", Params::new())?;
    for row in OPLSAA_DIHEDRALS {
        let ends = [row.i, row.j, row.k, row.l];
        dihedrals.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter([
                ("k1", row.f1),
                ("k2", row.f2),
                ("k3", row.f3),
                ("k4", row.f4),
            ]),
        )?;
    }

    // Atoms carry mass + charge; the LJ pair style carries ε / σ and declares
    // the combining rule (geometric, GROMACS comb-rule 3). Charges are per-atom
    // at evaluation time, so `coul/cut` has no rows at all.
    //
    // But its CONSTANTS are not the kernel's job. `coul/cut` is the buffered Coulomb
    // `E = k·qᵢqⱼ/(D·(r + δ))`; OPLS is the unbuffered case (δ = 0, the semantic
    // default) in vacuum (D = 1.0) with CODATA's k, and the style states them
    // rather than leaning on a constant the kernel holds.
    let atoms = ff.def_style("atom", "full", Params::new())?;
    for row in OPLSAA_ATOMS {
        atoms.def_type(
            row.name,
            &[],
            Params::from_iter([("mass", row.mass), ("charge", row.charge)]),
        )?;
    }

    let mut lj_params = Params::new();
    lj_params.set("mixing", OPLSAA_MIXING);
    let lj = ff.def_style("pair", "lj/cut", lj_params)?;
    for row in OPLSAA_ATOMS {
        lj.def_type(
            row.name,
            &[row.name],
            Params::from_iter([("epsilon", row.epsilon), ("sigma", row.sigma)]),
        )?;
    }
    ff.def_style(
        "pair",
        "coul/cut",
        Params::from_iter([("coulomb", COULOMB_REAL), ("dielectric", VACUUM_DIELECTRIC)]),
    )?;

    // OPLS excludes 1-2 / 1-3 (molrs omits them from the neighbour list) and
    // scales 1-4 by the source's own weights.
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, OPLS_LJ_14],
        coul: [0.0, 0.0, OPLS_COULOMB_14],
    });
    Ok(ff)
}

/// Build the shipped typing metadata (SMARTS `def`, `overrides`, `priority`,
/// `layer`) — the half of the parameter set no energy test can see, and the half
/// that decides what gets typed at all.
///
/// An input-free constructor over compiled tables, like [`force_field`]:
/// `tests::typing_meta_joins_every_rule` proves the join `Ok`.
pub(super) fn typing_meta() -> OplsTypingMetadata {
    try_typing_meta().expect(
        "OPLS-AA typing rules join their atom rows — proved by \
         ff::typifier::opls::shipped_forcefield::tests::typing_meta_joins_every_rule",
    )
}

/// The fallible body of [`typing_meta`]: the shipped rules joined to the
/// shipped atom rows.
fn try_typing_meta() -> Result<OplsTypingMetadata, String> {
    try_typing_meta_from(OPLSAA_TYPING, OPLSAA_ATOMS)
}

/// Join `rules` to `atoms` by name: one [`OplsTypeRow`] per rule, its class
/// taken from the atom row of the same name, no explicit priority, layer 0.
///
/// `Err` when a rule names no atom row, when two rules name the same type, or
/// when an override names a type that has no rule.
fn try_typing_meta_from(
    rules: &[OplsRuleRow],
    atoms: &[OplsAtomRow],
) -> Result<OplsTypingMetadata, String> {
    let classes: HashMap<&str, &str> = atoms.iter().map(|row| (row.name, row.class)).collect();
    let ruled: HashSet<&str> = rules.iter().map(|rule| rule.name).collect();
    let mut meta = OplsTypingMetadata::new();
    for rule in rules {
        let class = classes
            .get(rule.name)
            .ok_or_else(|| format!("OPLS typing rule {} names no atom row", rule.name))?;
        if let Some(missing) = rule.overrides.iter().find(|name| !ruled.contains(**name)) {
            return Err(format!(
                "OPLS typing rule {} overrides {missing}, which has no typing rule",
                rule.name
            ));
        }
        if meta.get(rule.name).is_some() {
            return Err(format!("OPLS typing rule {} is declared twice", rule.name));
        }
        meta.insert(
            rule.name,
            OplsTypeRow {
                class: (*class).to_owned(),
                def: Some(rule.def.to_owned()),
                overrides: rule.overrides.iter().map(|s| (*s).to_owned()).collect(),
                priority: None,
                layer: 0,
            },
        );
    }
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    /// The shipped OPLS-AA table defines every style and type without a
    /// conflict. `force_field` `expect`s this result and names this test.
    #[test]
    fn force_field_defines_without_conflict() {
        assert_eq!(try_force_field().err(), None);
    }

    use crate::ff::forcefield::{Style, StyleDefs};
    use crate::ff::params::OPLSAA_TYPING;
    use crate::ff::params::{OplsAtomRow, OplsRuleRow};
    use crate::ff::typifier::OplsAaTypifier;
    use crate::ff::typifier::{Typifier, Typing};
    use molrs::core::{Atomistic, NodeId};
    use molrs::core::{BondNumber, BondOrder};

    /// Relative tolerance of the source-equivalence pins. A zero expectation
    /// therefore demands an exact zero (`-0.0` compares equal).
    const REL_TOL: f64 = 1e-12;

    fn assert_rel(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= REL_TOL * expected.abs(),
            "{what}: got {actual:?}, want {expected:?} (relative tolerance {REL_TOL:e})"
        );
    }

    fn style<'f>(ff: &'f ForceField, category: &str, name: &str) -> &'f Style {
        ff.get_style(category, name)
            .unwrap_or_else(|| panic!("embedded OPLS-AA has no {category}/{name} style"))
    }

    fn param(params: &Params, key: &str, what: &str) -> f64 {
        params
            .get(key)
            .and_then(|v| v.as_num())
            .unwrap_or_else(|| panic!("{what} has no `{key}` param"))
    }

    fn atom_row(name: &str) -> &'static OplsAtomRow {
        OPLSAA_ATOMS
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("OPLSAA_ATOMS has no {name} row"))
    }

    /// Pin one GROMACS `[ atomtypes ]` line, hand-converted, against the
    /// assembled `atom/full` row, the `lj/cut` self row and the table's class.
    fn assert_atom_pin(
        ff: &ForceField,
        name: &str,
        class: &str,
        mass: f64,
        charge: f64,
        sigma: f64,
        epsilon: f64,
    ) {
        assert_eq!(atom_row(name).class, class, "{name} class");
        let atom = style(ff, "atom", "full")
            .get_atomtype(name)
            .unwrap_or_else(|| panic!("atom/full has no {name}"));
        assert_rel(
            param(&atom.params, "mass", name),
            mass,
            &format!("{name} mass"),
        );
        assert_rel(
            param(&atom.params, "charge", name),
            charge,
            &format!("{name} charge"),
        );
        let lj = style(ff, "pair", "lj/cut")
            .get_pairtype(name, None)
            .unwrap_or_else(|| panic!("lj/cut has no {name} self row"));
        assert_rel(
            param(&lj.params, "sigma", name),
            sigma,
            &format!("{name} sigma"),
        );
        assert_rel(
            param(&lj.params, "epsilon", name),
            epsilon,
            &format!("{name} epsilon"),
        );
    }

    /// The embedded library's `lj/cut` declares OPLS-AA's combining rule —
    /// geometric in σ and ε (Jorgensen et al. 1996; GROMACS `forcefield.itp`
    /// `[ defaults ] 1 3 yes 0.5 0.5`, comb-rule 3). Undeclared, the kernel
    /// mixes arithmetically and CT–HC σ comes out 2.958 Å instead of 3.000 Å.
    #[test]
    fn library_lj_cut_declares_geometric_mixing() {
        let typifier = OplsAaTypifier::oplsaa();
        let lj = style(typifier.source_forcefield(), "pair", "lj/cut");
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    /// GROMACS v2026.3 (`42105e4672205b4aa951962b8e3cdb4c27890da1`),
    /// `share/top/oplsaa.ff/ffnonbonded.itp`, hand-copied:
    ///
    /// ```text
    /// ; name  bond_type    mass    charge   ptype          sigma      epsilon
    ///  opls_135   CT  6     12.01100    -0.180       A    3.50000e-01  2.76144e-01
    ///  opls_150   C=  6     12.01100    -0.115       A    3.55000e-01  3.17984e-01
    ///  opls_155   HO    1      1.00800     0.418       A    0.00000e+00  0.00000e+00
    /// ```
    ///
    /// Hand conversion (σ nm × 10 → Å; ε kJ/mol ÷ 4.184 → kcal/mol):
    /// - opls_135: σ 0.35 × 10 = 3.5; ε 0.276144 / 4.184 = 0.066;
    /// - opls_150: σ 0.355 × 10 = 3.55; ε 0.317984 / 4.184 = 0.076;
    /// - opls_155: σ 0 and ε 0 — GROMACS's own σ, not the 10 Å foyer wrote.
    #[test]
    fn atom_types_match_hand_converted_gromacs_rows() {
        let ff = try_force_field().expect("embedded OPLS-AA defines");
        assert_atom_pin(&ff, "opls_135", "CT", 12.011, -0.18, 3.5, 0.066);
        assert_atom_pin(&ff, "opls_150", "C=", 12.011, -0.115, 3.55, 0.076);
        assert_atom_pin(&ff, "opls_155", "HO", 1.008, 0.418, 0.0, 0.0);
    }

    /// GROMACS v2026.3 `ffbonded.itp` `[ bondtypes ]` (funct 1,
    /// `½k(r − b₀)²`), hand-copied:
    ///
    /// ```text
    ///   C=    C=      1    0.14600   322168.0   ; wlj 1,3-diene 3/97
    ///   CT    HC      1    0.10900   284512.0   ; CHARMM 22 parameter file
    /// ```
    ///
    /// Hand conversion (b₀ nm × 10 → Å; GROMACS ½k_b in kJ/mol/nm² → LAMMPS
    /// K = k_b / 418.4 / 2 kcal/mol/Å²):
    /// - CT-HC: r0 1.09, k 284512.0 / 418.4 / 2 = 340.0;
    /// - C=-C=: r0 1.46, k 322168.0 / 418.4 / 2 = 385.0.
    #[test]
    fn bond_types_match_hand_converted_gromacs_rows() {
        let ff = try_force_field().expect("embedded OPLS-AA defines");
        let bonds = style(&ff, "bond", "harmonic");
        for (i, j, r0, k) in [("CT", "HC", 1.09, 340.0), ("C=", "C=", 1.46, 385.0)] {
            let what = format!("bond {i}-{j}");
            let bond = bonds
                .get_bondtype(i, j)
                .unwrap_or_else(|| panic!("no {what}"));
            assert_rel(param(&bond.params, "r0", &what), r0, &format!("{what} r0"));
            assert_rel(param(&bond.params, "k", &what), k, &format!("{what} k"));
        }
    }

    /// GROMACS v2026.3 `ffbonded.itp` `[ angletypes ]` (funct 1,
    /// `½k(θ − θ₀)²`), hand-copied:
    ///
    /// ```text
    ///   CM     C=     C=      1   124.000    585.760   ; wlj
    /// ```
    ///
    /// Hand conversion: θ₀ = 124° as written; GROMACS ½k_θ → LAMMPS
    /// K = 585.760 / 4.184 / 2 = 70.0 kcal/mol/rad².
    #[test]
    fn angle_type_matches_hand_converted_gromacs_row() {
        let ff = try_force_field().expect("embedded OPLS-AA defines");
        let StyleDefs::Angle(angles) = style(&ff, "angle", "harmonic").defs() else {
            panic!("angle/harmonic holds no angle defs");
        };
        let angle = angles
            .iter()
            .find(|t| {
                t.jtom == "C="
                    && ((t.itom == "CM" && t.ktom == "C=") || (t.itom == "C=" && t.ktom == "CM"))
            })
            .expect("no angle CM-C=-C=");
        let what = "angle CM-C=-C=";
        assert_rel(
            param(&angle.params, "theta0", what),
            124.0,
            "angle CM-C=-C= theta0",
        );
        assert_rel(param(&angle.params, "k", what), 70.0, "angle CM-C=-C= k");
    }

    /// GROMACS v2026.3 `ffbonded.itp` `[ dihedraltypes ]` (funct 3,
    /// Ryckaert–Bellemans, kJ/mol), hand-copied:
    ///
    /// ```text
    ///   HC     CT     CT     HC      3      0.62760   1.88280   0.00000  -2.51040   0.00000   0.00000 ; hydrocarbon *new* 11/99
    /// ```
    ///
    /// Hand conversion to the OPLS Fourier form (GROMACS manual Eqs. 200–201;
    /// ÷ 4.184 → kcal/mol):
    /// - F1 = −2C1 − 3C3/2 = −3.7656 + 3.7656 = 0;
    /// - F2 = −C2 − C4 = 0;
    /// - F3 = −C3/2 = 1.2552 kJ/mol = 0.3 kcal/mol;
    /// - F4 = −C4/4 = 0.
    #[test]
    fn dihedral_type_matches_hand_converted_gromacs_row() {
        let ff = try_force_field().expect("embedded OPLS-AA defines");
        let StyleDefs::Dihedral(dihedrals) = style(&ff, "dihedral", "opls").defs() else {
            panic!("dihedral/opls holds no dihedral defs");
        };
        let dihedral = dihedrals
            .iter()
            .find(|t| t.itom == "HC" && t.jtom == "CT" && t.ktom == "CT" && t.ltom == "HC")
            .expect("no dihedral HC-CT-CT-HC");
        let what = "dihedral HC-CT-CT-HC";
        for (key, expected) in [("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)] {
            assert_rel(
                param(&dihedral.params, key, what),
                expected,
                &format!("{what} {key}"),
            );
        }
    }

    /// Every atom row's class is a GROMACS `bond_type` — the vocabulary the
    /// bonded tables key on. GROMACS never uses an `opls_NNN` name as a
    /// `bond_type`; foyer gave 651 of the 813 types their own name as class,
    /// which left their bonded rows unmatchable.
    #[test]
    fn atom_class_is_gromacs_bond_type() {
        let self_classed: Vec<&str> = OPLSAA_ATOMS
            .iter()
            .filter(|row| row.class.starts_with("opls_"))
            .map(|row| row.name)
            .collect();
        assert!(
            self_classed.is_empty(),
            "{} atom rows carry an opls_NNN class, e.g. {:?}",
            self_classed.len(),
            &self_classed[..self_classed.len().min(5)]
        );
    }

    fn atom(name: &'static str, class: &'static str) -> OplsAtomRow {
        OplsAtomRow {
            name,
            class,
            mass: 12.011,
            charge: 0.0,
            sigma: 3.5,
            epsilon: 0.066,
        }
    }

    fn rule(
        name: &'static str,
        def: &'static str,
        overrides: &'static [&'static str],
    ) -> OplsRuleRow {
        OplsRuleRow {
            name,
            def,
            overrides,
        }
    }

    /// `typing_meta` `expect`s this result and names this test. The shipped
    /// join is `Ok`, holds one row per rule, and takes each class from the
    /// GROMACS-classed atom row (opls_135 → `CT`). Test-local tables with a
    /// rule naming no atom row, or an override naming no rule, are `Err`.
    #[test]
    fn typing_meta_joins_every_rule() {
        let meta = try_typing_meta().expect("shipped OPLS-AA typing rules join");
        assert_eq!(meta.len(), OPLSAA_TYPING.len(), "one typing row per rule");
        for r in OPLSAA_TYPING {
            let row = meta
                .get(r.name)
                .unwrap_or_else(|| panic!("rule {} missing from typing meta", r.name));
            assert_eq!(row.def.as_deref(), Some(r.def), "{} def", r.name);
            assert_eq!(row.overrides, r.overrides, "{} overrides", r.name);
            assert_eq!(row.priority, None, "{} priority", r.name);
            assert_eq!(row.layer, 0, "{} layer", r.name);
        }
        let ct = meta.get("opls_135").expect("opls_135 has a typing rule");
        assert_eq!(ct.class, "CT");

        // A consistent test-local table joins, class taken from the atom row.
        let atoms = [atom("t_a", "CA"), atom("t_b", "CB")];
        let rules = [rule("t_a", "[C]", &[]), rule("t_b", "[C]C", &["t_a"])];
        let local = try_typing_meta_from(&rules, &atoms).expect("consistent tables join");
        assert_eq!(local.len(), 2);
        assert_eq!(local.get("t_b").map(|row| row.class.as_str()), Some("CB"));

        // A rule naming no atom row.
        let dangling_rule = [rule("t_a", "[C]", &[]), rule("t_missing", "[N]", &[])];
        assert!(
            try_typing_meta_from(&dangling_rule, &atoms).is_err(),
            "a rule naming no atom row must be Err"
        );

        // An override naming no rule (t_b is an atom row, but has no rule).
        let dangling_override = [rule("t_a", "[C]", &["t_b"])];
        assert!(
            try_typing_meta_from(&dangling_override, &atoms).is_err(),
            "an override naming no rule must be Err"
        );
    }

    // -- Golden typing (opls-gromacs-03 Testing strategy) ----------------------
    //
    // Every molecule is hand-built with explicit hydrogens and typed by the
    // shipped rules through `Typing<OplsAaTypifier::oplsaa().with_strict(false)>`.
    // Every atom's expected type is the spec's golden table, checked against the
    // GROMACS v2026.3 `atomtypes.atp` description of that type; each molecule is
    // neutral, so the `OPLSAA_ATOMS` charges of its types sum to 0.

    const S: BondOrder = BondOrder::Single;
    const D: BondOrder = BondOrder::Double;
    const T: BondOrder = BondOrder::Triple;
    const A: BondOrder = BondOrder::Aromatic;
    /// A Kekulé six-ring: `r0=r1-r2=r3-r4=r5-r0`.
    const KEKULE6: [BondOrder; 6] = [D, S, D, S, D, S];

    /// A hand-built golden molecule: the graph plus the OPLS-AA type each of
    /// its atoms must receive.
    struct Golden {
        graph: Atomistic,
        /// `(atom, element, expected type)`, in insertion order.
        expected: Vec<(NodeId, &'static str, &'static str)>,
    }

    impl Golden {
        fn new() -> Self {
            Self {
                graph: Atomistic::new(),
                expected: Vec::new(),
            }
        }

        /// Add an atom the input writes aliphatic, expected to type `ty`.
        fn atom(&mut self, element: &'static str, ty: &'static str) -> NodeId {
            let x = 1.2 * self.expected.len() as f64;
            let id = self
                .graph
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::by_symbol(element).unwrap(),
                    keys::X => x,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap();
            self.expected.push((id, element, ty));
            id
        }

        /// Add an atom the input declares aromatic (`is_aromatic = 1`, as a
        /// lowercase SMILES atom reads), expected to type `ty`.
        fn aromatic_atom(&mut self, element: &'static str, ty: &'static str) -> NodeId {
            let id = self.atom(element, ty);
            self.graph.set_atom(id, keys::IS_AROMATIC, true).unwrap();
            id
        }

        fn bond(&mut self, a: NodeId, b: NodeId, order: BondOrder) {
            let bond = self.graph.add_bond(a, b, props! {}).unwrap();
            // An aromatic bond is written unkekulized (no bond number), as a
            // lowercase SMILES bond reads.
            let number = order.implied_number().unwrap_or(BondNumber::Unknown);
            self.graph.set_bond_class(bond, order, number).unwrap();
        }

        /// Close `atoms` into a ring, bond `k` joining `atoms[k]` and
        /// `atoms[k + 1]` (wrapping) with `orders[k]`.
        fn ring(&mut self, atoms: &[NodeId], orders: &[BondOrder]) {
            assert_eq!(atoms.len(), orders.len(), "one order per ring bond");
            for (k, &order) in orders.iter().enumerate() {
                self.bond(atoms[k], atoms[(k + 1) % atoms.len()], order);
            }
        }

        /// Add `n` hydrogens singly bonded to `heavy`, each expected `ty`.
        fn hydrogens(&mut self, heavy: NodeId, n: usize, ty: &'static str) {
            for _ in 0..n {
                let h = self.atom("H", ty);
                self.bond(heavy, h, S);
            }
        }

        /// Type the molecule with the shipped rules (non-strict); assert every
        /// atom's type and that the charges of the assigned types sum to 0.
        fn assert_typed(&self) {
            let typed = Typing::new(OplsAaTypifier::oplsaa().with_strict(false))
                .typify(&self.graph)
                .expect("non-strict OPLS-AA typing is Ok");

            let mut wrong = Vec::new();
            let mut net = 0.0;
            for (i, &(id, element, want)) in self.expected.iter().enumerate() {
                let got = typed
                    .get_atom(id)
                    .expect("the typed copy keeps every atom id")
                    .get_str("type")
                    .map(str::to_owned);
                if got.as_deref() != Some(want) {
                    wrong.push(format!("atom {i} ({element}): got {got:?}, want {want}"));
                }
                if let Some(name) = got.as_deref() {
                    net += atom_row(name).charge;
                }
            }
            assert!(wrong.is_empty(), "mistyped atoms:\n{}", wrong.join("\n"));
            assert!(net.abs() < 1e-9, "net charge {net:e}, want 0");
        }
    }

    /// N-methylacetamide `CH3-C(=O)-NH-CH3`.
    #[test]
    fn golden_n_methylacetamide() {
        let mut m = Golden::new();
        let ca = m.atom("C", "opls_135");
        let c = m.atom("C", "opls_235");
        let o = m.atom("O", "opls_236");
        let n = m.atom("N", "opls_238");
        let cn = m.atom("C", "opls_242");
        m.bond(ca, c, S);
        m.bond(c, o, D);
        m.bond(c, n, S);
        m.bond(n, cn, S);
        m.hydrogens(ca, 3, "opls_140");
        m.hydrogens(n, 1, "opls_241");
        m.hydrogens(cn, 3, "opls_140");
        m.assert_typed();
    }

    /// 1,3-butadiene `H2C=CH-CH=CH2`: terminal C opls_143, inner C opls_150.
    #[test]
    fn golden_butadiene() {
        let mut m = Golden::new();
        let c1 = m.atom("C", "opls_143");
        let c2 = m.atom("C", "opls_150");
        let c3 = m.atom("C", "opls_150");
        let c4 = m.atom("C", "opls_143");
        m.bond(c1, c2, D);
        m.bond(c2, c3, S);
        m.bond(c3, c4, D);
        m.hydrogens(c1, 2, "opls_144");
        m.hydrogens(c2, 1, "opls_144");
        m.hydrogens(c3, 1, "opls_144");
        m.hydrogens(c4, 2, "opls_144");
        m.assert_typed();
    }

    /// Ethanol `CH3-CH2-OH`.
    #[test]
    fn golden_ethanol() {
        let mut m = Golden::new();
        let c1 = m.atom("C", "opls_135");
        let c2 = m.atom("C", "opls_157");
        let o = m.atom("O", "opls_154");
        m.bond(c1, c2, S);
        m.bond(c2, o, S);
        m.hydrogens(c1, 3, "opls_140");
        m.hydrogens(c2, 2, "opls_140");
        m.hydrogens(o, 1, "opls_155");
        m.assert_typed();
    }

    /// Benzene written aromatic: `is_aromatic` atoms, `Aromatic` bonds.
    #[test]
    fn golden_benzene_aromatic_input() {
        let mut m = Golden::new();
        let r: Vec<NodeId> = (0..6).map(|_| m.aromatic_atom("C", "opls_145")).collect();
        m.ring(&r, &[A; 6]);
        for &c in &r {
            m.hydrogens(c, 1, "opls_146");
        }
        m.assert_typed();
    }

    /// Benzene written Kekulé: alternating `Double` / `Single`, no aromatic flag.
    #[test]
    fn golden_benzene_kekule_input() {
        let mut m = Golden::new();
        let r: Vec<NodeId> = (0..6).map(|_| m.atom("C", "opls_145")).collect();
        m.ring(&r, &KEKULE6);
        for &c in &r {
            m.hydrogens(c, 1, "opls_146");
        }
        m.assert_typed();
    }

    /// Propylene carbonate (4-methyl-1,3-dioxolan-2-one): ring
    /// `O1-C2(=O)-O3-C4H2-C5H(CH3)-O1`.
    #[test]
    fn golden_propylene_carbonate() {
        let mut m = Golden::new();
        let o1 = m.atom("O", "opls_773");
        let c2 = m.atom("C", "opls_772");
        let o3 = m.atom("O", "opls_773");
        let c4 = m.atom("C", "opls_774");
        let c5 = m.atom("C", "opls_775");
        let exo = m.atom("O", "opls_771");
        let me = m.atom("C", "opls_776");
        m.ring(&[o1, c2, o3, c4, c5], &[S; 5]);
        m.bond(c2, exo, D);
        m.bond(c5, me, S);
        m.hydrogens(c4, 2, "opls_777");
        m.hydrogens(c5, 1, "opls_778");
        m.hydrogens(me, 3, "opls_779");
        m.assert_typed();
    }

    /// Methyl methacrylate `CH2=C(CH3)-C(=O)-O-CH3`. The α-carbon is not a
    /// diene carbon (its other neighbours are sp3 and carbonyl), so opls_141.
    #[test]
    fn golden_methyl_methacrylate() {
        let mut m = Golden::new();
        let cm = m.atom("C", "opls_143");
        let ca = m.atom("C", "opls_141");
        let me = m.atom("C", "opls_135");
        let cc = m.atom("C", "opls_465");
        let od = m.atom("O", "opls_466");
        let os = m.atom("O", "opls_467");
        let co = m.atom("C", "opls_468");
        m.bond(cm, ca, D);
        m.bond(ca, me, S);
        m.bond(ca, cc, S);
        m.bond(cc, od, D);
        m.bond(cc, os, S);
        m.bond(os, co, S);
        m.hydrogens(cm, 2, "opls_144");
        m.hydrogens(me, 3, "opls_140");
        m.hydrogens(co, 3, "opls_469");
        m.assert_typed();
    }

    /// Methyl formate `H-C(=O)-O-CH3`.
    #[test]
    fn golden_methyl_formate() {
        let mut m = Golden::new();
        let cf = m.atom("C", "opls_465");
        let od = m.atom("O", "opls_466");
        let os = m.atom("O", "opls_467");
        let co = m.atom("C", "opls_468");
        m.bond(cf, od, D);
        m.bond(cf, os, S);
        m.bond(os, co, S);
        m.hydrogens(cf, 1, "opls_279");
        m.hydrogens(co, 3, "opls_469");
        m.assert_typed();
    }

    /// Benzonitrile, Kekulé ring; `r0` is the ipso carbon.
    #[test]
    fn golden_benzonitrile() {
        let mut m = Golden::new();
        let ipso = m.atom("C", "opls_260");
        let rest: Vec<NodeId> = (0..5).map(|_| m.atom("C", "opls_145")).collect();
        let cn = m.atom("C", "opls_261");
        let n = m.atom("N", "opls_262");
        let ring = [&[ipso][..], &rest].concat();
        m.ring(&ring, &KEKULE6);
        m.bond(ipso, cn, S);
        m.bond(cn, n, T);
        for &c in &rest {
            m.hydrogens(c, 1, "opls_146");
        }
        m.assert_typed();
    }

    /// Chlorobenzene, Kekulé ring; `r0` carries the Cl.
    #[test]
    fn golden_chlorobenzene() {
        let mut m = Golden::new();
        let ipso = m.atom("C", "opls_263");
        let rest: Vec<NodeId> = (0..5).map(|_| m.atom("C", "opls_145")).collect();
        let cl = m.atom("Cl", "opls_264");
        let ring = [&[ipso][..], &rest].concat();
        m.ring(&ring, &KEKULE6);
        m.bond(ipso, cl, S);
        for &c in &rest {
            m.hydrogens(c, 1, "opls_146");
        }
        m.assert_typed();
    }

    /// Chloroethane `CH3-CH2-Cl`.
    #[test]
    fn golden_chloroethane() {
        let mut m = Golden::new();
        let c1 = m.atom("C", "opls_135");
        let c2 = m.atom("C", "opls_152");
        let cl = m.atom("Cl", "opls_151");
        m.bond(c1, c2, S);
        m.bond(c2, cl, S);
        m.hydrogens(c1, 3, "opls_140");
        m.hydrogens(c2, 2, "opls_153");
        m.assert_typed();
    }

    /// Fluorobenzene, Kekulé ring; `r0` carries the F.
    #[test]
    fn golden_fluorobenzene() {
        let mut m = Golden::new();
        let ipso = m.atom("C", "opls_718");
        let rest: Vec<NodeId> = (0..5).map(|_| m.atom("C", "opls_145")).collect();
        let f = m.atom("F", "opls_719");
        let ring = [&[ipso][..], &rest].concat();
        m.ring(&ring, &KEKULE6);
        m.bond(ipso, f, S);
        for &c in &rest {
            m.hydrogens(c, 1, "opls_146");
        }
        m.assert_typed();
    }

    /// Pyridine, Kekulé ring `N1=C2-C3=C4-C5=C6-N1`: C2/C6 α (opls_521, H
    /// opls_524), C3/C5 β (opls_522, H opls_525), C4 γ (opls_523, H opls_526).
    #[test]
    fn golden_pyridine() {
        let mut m = Golden::new();
        let n1 = m.atom("N", "opls_520");
        let c2 = m.atom("C", "opls_521");
        let c3 = m.atom("C", "opls_522");
        let c4 = m.atom("C", "opls_523");
        let c5 = m.atom("C", "opls_522");
        let c6 = m.atom("C", "opls_521");
        m.ring(&[n1, c2, c3, c4, c5, c6], &KEKULE6);
        m.hydrogens(c2, 1, "opls_524");
        m.hydrogens(c3, 1, "opls_525");
        m.hydrogens(c4, 1, "opls_526");
        m.hydrogens(c5, 1, "opls_525");
        m.hydrogens(c6, 1, "opls_524");
        m.assert_typed();
    }

    /// Pyrimidine, Kekulé ring `N1=C2-N3=C4-C5=C6-N1`.
    #[test]
    fn golden_pyrimidine() {
        let mut m = Golden::new();
        let n1 = m.atom("N", "opls_530");
        let c2 = m.atom("C", "opls_531");
        let n3 = m.atom("N", "opls_530");
        let c4 = m.atom("C", "opls_532");
        let c5 = m.atom("C", "opls_533");
        let c6 = m.atom("C", "opls_532");
        m.ring(&[n1, c2, n3, c4, c5, c6], &KEKULE6);
        m.hydrogens(c2, 1, "opls_534");
        m.hydrogens(c4, 1, "opls_535");
        m.hydrogens(c5, 1, "opls_536");
        m.hydrogens(c6, 1, "opls_535");
        m.assert_typed();
    }

    /// Pyrrole, Kekulé ring `N1(H)-C2=C3-C4=C5-N1`: C2/C5 α, C3/C4 β.
    #[test]
    fn golden_pyrrole() {
        let mut m = Golden::new();
        let n1 = m.atom("N", "opls_542");
        let c2 = m.atom("C", "opls_543");
        let c3 = m.atom("C", "opls_544");
        let c4 = m.atom("C", "opls_544");
        let c5 = m.atom("C", "opls_543");
        m.ring(&[n1, c2, c3, c4, c5], &[S, D, S, D, S]);
        m.hydrogens(n1, 1, "opls_545");
        m.hydrogens(c2, 1, "opls_546");
        m.hydrogens(c3, 1, "opls_547");
        m.hydrogens(c4, 1, "opls_547");
        m.hydrogens(c5, 1, "opls_546");
        m.assert_typed();
    }
}

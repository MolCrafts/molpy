//! Antechamber's `ATD` / `WILDATOM` atom-type engine — one engine, N tables.

mod conjugate;
mod facts;
mod pattern;
mod rules;

pub(crate) use facts::antechamber_bond_type;

use std::sync::OnceLock;

use molrs::core::PropValue;
use molrs::core::keys;
use molrs::core::{Atomistic, MolRsError, NodeId};
use molrs::perceive::{BondOrderSource, assign_bcc_bond_types};

use self::facts::MolFacts;
use crate::ff::forcefield::ForceField;
use crate::ff::params::atomtype_abcg2::ATOMTYPE_ABCG2;
use crate::ff::params::atomtype_amber::ATOMTYPE_AMBER;
use crate::ff::params::atomtype_bcc::ATOMTYPE_BCC;
use crate::ff::params::atomtype_gas::ATOMTYPE_GAS;
use crate::ff::params::atomtype_gff::ATOMTYPE_GFF;
use crate::ff::params::atomtype_gff2::ATOMTYPE_GFF2;
use crate::ff::params::atomtype_sybyl::ATOMTYPE_SYBYL;
use crate::ff::params::{AtdRule, AtdTable};
use crate::ff::typifier::{Annotation, TypeAssignment, Typifier, TypingError, species_label};

/// Which `ATOMTYPE_*.DEF` table an [`AtdTypifier`] walks.
///
/// This is the **atom-type** axis, and it is wider than the BCC-correction
/// axis: `ATOMTYPE_GAS.DEF` exists but there is no
/// `BCCPARM_GAS.DAT`, so GAS is a set of atom types with no correction family.
/// Only [`BccParameterSet`](crate::ff::charge::BccParameterSet) — `Bcc` and `Abcg2`
/// — names both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtdParameterSet {
    /// `ATOMTYPE_BCC.DEF` — AM1-BCC atom types (`antechamber -at bcc`).
    Bcc,
    /// `ATOMTYPE_ABCG2.DEF` — ABCG2 atom types (`antechamber -at abcg2`).
    Abcg2,
    /// `ATOMTYPE_GAS.DEF` — Gasteiger atom types (`antechamber -at gas`).
    Gas,
    /// `ATOMTYPE_GFF.DEF` — GAFF atom types (`antechamber -at gaff`).
    Gff,
    /// `ATOMTYPE_GFF2.DEF` — GAFF2 atom types (`antechamber -at gaff2`).
    Gff2,
    /// `ATOMTYPE_AMBER.DEF` — AMBER atom types (`antechamber -at amber`).
    Amber,
    /// `ATOMTYPE_SYBYL.DEF` — SYBYL atom types (`antechamber -at sybyl`).
    Sybyl,
}

impl AtdParameterSet {
    /// Every atom-type table, in antechamber's `-at` order.
    pub const ALL: [AtdParameterSet; 7] = [
        Self::Bcc,
        Self::Abcg2,
        Self::Gas,
        Self::Gff,
        Self::Gff2,
        Self::Amber,
        Self::Sybyl,
    ];

    /// The antechamber `-at` flag naming this table (`"bcc"`, `"gaff2"`, …).
    pub fn name(self) -> &'static str {
        match self {
            Self::Bcc => "bcc",
            Self::Abcg2 => "abcg2",
            Self::Gas => "gas",
            Self::Gff => "gaff",
            Self::Gff2 => "gaff2",
            Self::Amber => "amber",
            Self::Sybyl => "sybyl",
        }
    }

    /// The table an antechamber `-at` flag names — the inverse of
    /// [`name`](Self::name).
    ///
    /// # Errors
    ///
    /// An unknown name. Never a fallback to a default table: an atom type
    /// from the wrong table is a plausible-looking answer.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|set| set.name() == name)
            .ok_or_else(|| {
                let known: Vec<&str> = Self::ALL.iter().map(|set| set.name()).collect();
                format!(
                    "unknown atom-type parameter set {name:?}; expected one of {}",
                    known.join(", ")
                )
            })
    }

    /// The compile-time table this set names.
    pub fn table(self) -> AtdTable {
        match self {
            Self::Bcc => ATOMTYPE_BCC,
            Self::Abcg2 => ATOMTYPE_ABCG2,
            Self::Gas => ATOMTYPE_GAS,
            Self::Gff => ATOMTYPE_GFF,
            Self::Gff2 => ATOMTYPE_GFF2,
            Self::Amber => ATOMTYPE_AMBER,
            Self::Sybyl => ATOMTYPE_SYBYL,
        }
    }
}

/// The type every `.DEF` file's terminal catch-all row assigns: "no rule matched".
///
/// Antechamber's dummy type. Six of the seven tables end with an unconditional
/// `ATD DU &` row, so an atom no rule covers is not *unlabelled* — it is labelled
/// `DU`, and the label carries no chemistry. In the AMBER column that is genuinely
/// antechamber's answer (nitromethane's nitro oxygens, DMSO's sulfur), so the engine
/// must keep emitting it; but a consumer that needs *parameters* for the type — the
/// BCC correction tables have no `DU` row — has to read it as the refusal it is.
pub(crate) const DUMMY_TYPE: &str = "DU";

/// Why the engine could not type a molecule.
///
/// The typed twin of the `String` that [`AtdTypifier`]'s
/// `assign` ([`Typifier`]) returns. A charge model
/// has to tell "no rule covers this atom" (a permanent property of the table — boron
/// and bare sulfur are the real cases) apart from "this graph is malformed", because
/// the C++ and Python bridges have to report them differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AtdError {
    /// No rule of `table` matched the atom.
    MissingAtomType {
        /// The `.DEF` file that was walked.
        table: &'static str,
        /// The atom's 0-based index in graph atom order.
        atom: usize,
        /// The atom's atomic number (`0` for a wildcard).
        atomic_number: u8,
    },
    /// The molecule's facts could not be derived — a defect in the input graph.
    Malformed {
        /// What the facts layer said.
        detail: String,
    },
}

impl From<AtdError> for TypingError {
    fn from(err: AtdError) -> Self {
        match err {
            AtdError::MissingAtomType {
                table,
                atom,
                atomic_number,
            } => TypingError::missing_atom(table, atom, atomic_number),
            AtdError::Malformed { detail } => TypingError::Malformed { detail },
        }
    }
}

impl std::fmt::Display for AtdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAtomType {
                table,
                atom,
                atomic_number,
            } => write!(
                f,
                "missing {table} atom type for atom {atom} ({})",
                species_label(*atomic_number)
            ),
            Self::Malformed { detail } => write!(f, "{detail}"),
        }
    }
}

/// The ATD rule engine, bound to one atom-type table.
///
/// Its `assign` ([`Typifier`]) perceives antechamber bond types, derives
/// the facts each rule can ask about, and labels every atom with the first rule
/// of the table that matches it. An atom no rule matches is an **error**, not an
/// untyped or defaulted atom.
///
/// All seven `ATOMTYPE_*.DEF` files share one rule language, so they share one
/// interpreter: [`AtdTypifier`] is that interpreter, and [`AtdParameterSet`]
/// chooses the table it walks. The tables are `&'static` Rust data generated
/// from the upstream `.DEF` files (see [`crate::ff::params`]), so a typifier
/// carries no state beyond which table it names, and matching parses nothing.
///
/// The engine holds **no** per-table knowledge. That is a testable claim rather
/// than a stylistic one: the three tables disagree exactly where typing is hard
/// (imidazole's pyridine-type N is `24` under BCC, `28` under ABCG2 and `n2`
/// under GAS), so a table-specific special case that satisfied one column would
/// break another.
///
/// ```no_run
/// use molrs::core::Atomistic;
/// use molrs::ff::typifier::Typing;
/// use molrs::ff::typifier::{AtdParameterSet, AtdTypifier};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mol = Atomistic::new();
/// let mut typing = Typing::new(AtdTypifier::new(AtdParameterSet::Bcc));
/// let typed = typing.typify(&mol)?;       // every atom's `type` stamped
/// assert!(typing.forcefield().styles().is_empty()); // ATD defines no type
/// # Ok(())
/// # }
/// ```
///
/// # Layering
///
/// Bond types are always **perceived** here, never read off the input: the rules
/// count `sb` / `db` / `ab` / `DL` bonds, which need the delocalized (9) and
/// aromatic (7/8) types that a bond *order* cannot express.
///
/// # Which bond orders the types follow
///
/// antechamber (its default `-j 4`) discards the bond orders of its input and
/// judges new ones from the connectivity (`bondtype -j full`), and the atom
/// types follow that structure: on a molecule with two Kekulé structures
/// (azulene, cyclooctatetraene) the `cc` / `cd` colouring is the one its search
/// settles on, not the one the input drew. [`BondOrderSource::Connectivity`],
/// the default, does the same, so a molecule read from the file antechamber
/// reads types as antechamber types it, whatever orders the molrs graph
/// carries. [`BondOrderSource::Stated`] keeps the graph's own orders instead
/// (aromatic bonds without one are kekulized), for a caller whose bond orders
/// are the chemistry it wants typed. Every hydrogen must be drawn for
/// `Connectivity`; where no valence state closes, the graph's own orders are
/// used (antechamber keeps its file's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdTypifier {
    set: AtdParameterSet,
    bond_orders: BondOrderSource,
}

impl AtdTypifier {
    /// Bind the engine to the table `set` names, perceiving bond orders as
    /// antechamber does ([`BondOrderSource::Connectivity`]).
    pub fn new(set: AtdParameterSet) -> Self {
        Self {
            set,
            bond_orders: BondOrderSource::Connectivity,
        }
    }

    /// The same engine, perceiving bond types from `bond_orders`.
    pub fn with_bond_orders(self, bond_orders: BondOrderSource) -> Self {
        Self {
            bond_orders,
            ..self
        }
    }

    /// The parameter set this typifier walks.
    pub fn parameter_set(&self) -> AtdParameterSet {
        self.set
    }

    /// Which bond orders the bond types are perceived from.
    pub fn bond_orders(&self) -> BondOrderSource {
        self.bond_orders
    }

    /// Perceive antechamber bond types onto every bond of `mol`, from the bond
    /// orders [`bond_orders`](Self::bond_orders) names — the input
    /// [`types_of`](Self::types_of) wants.
    ///
    /// # Errors
    ///
    /// [`assign_bcc_bond_types`]'s refusal; `mol` must then be discarded.
    pub(crate) fn perceive_bond_types(&self, mol: &mut Atomistic) -> Result<(), MolRsError> {
        assign_bcc_bond_types(mol, self.bond_orders)
    }

    /// The types this table assigns, in graph atom order — **computed, not written**.
    ///
    /// The half of `assign` ([`Typifier`]) that a charge model wants: the atom
    /// types come back as a `Vec`, so the model can look its corrections up without
    /// ever putting a BCC code into the caller's [`keys::TYPE`] column (where their
    /// GAFF / OPLS force-field types live).
    ///
    /// # Arguments
    ///
    /// * `perceived` — a molecule whose bonds already carry perceived antechamber
    ///   bond types, i.e. the output of
    ///   [`perceive_bond_types`](Self::perceive_bond_types).
    ///   The rules count `sb` / `db` / `ab` / `DL` bonds, so they cannot run on bond
    ///   *orders*.
    ///
    /// # Errors
    ///
    /// [`AtdError::MissingAtomType`] when no rule matches an atom;
    /// [`AtdError::Malformed`] when the graph's facts cannot be derived.
    pub(crate) fn types_of(&self, perceived: &Atomistic) -> Result<Vec<&'static str>, AtdError> {
        let table = self.set.table();
        let bcc = matches!(self.set, AtdParameterSet::Bcc | AtdParameterSet::Abcg2);
        let facts =
            MolFacts::new(perceived, bcc).map_err(|detail| AtdError::Malformed { detail })?;
        let atom_ids: Vec<NodeId> = perceived.atoms().map(|(aid, _)| aid).collect();

        // Pass 1 — the table: the first rule that matches each atom.
        let assigned: Vec<&'static AtdRule> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, aid)| {
                rules::assign_rule(&table, *aid, &facts).ok_or_else(|| AtdError::MissingAtomType {
                    table: table.name,
                    atom: i,
                    atomic_number: facts.atomic_number[i],
                })
            })
            .collect::<Result<_, _>>()?;

        // Pass 2 — the conjugated 2-colouring, which is a fact about the whole
        // molecule and so cannot be folded into the per-atom loop above: half of
        // each conjugated system is renamed to the matched rule's `alternate`.
        Ok(conjugate::resolve_types(&atom_ids, &assigned, &facts))
    }
}

impl Typifier for AtdTypifier {
    /// Perceive antechamber bond types onto `graph` (from the bond orders
    /// [`bond_orders`](Self::bond_orders) names), then label every atom from
    /// the table's rules: `type` → a plain value on every atom. Defines
    /// no style, type or pair, so the typing output stays empty.
    ///
    /// # Errors
    ///
    /// A message naming the atom no rule of the table matched — including
    /// one the table's catch-all row labels `DU`: that label is the
    /// table's "no rule matched", never a type to hand back.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, TypingError> {
        self.perceive_bond_types(graph)
            .map_err(TypingError::malformed)?;
        let types = self.types_of(graph)?;
        if let Some(atom) = types.iter().position(|t| *t == DUMMY_TYPE) {
            let atomic_number = graph
                .atom_ids()
                .nth(atom)
                .and_then(|id| graph.get_atom(id).ok())
                .and_then(|a| a.atomic_number())
                .expect("Atomistic invariant: every typed atom carries an atomic number");
            return Err(TypingError::missing_atom(
                self.set.table().name,
                atom,
                atomic_number,
            ));
        }
        Ok(TypeAssignment {
            nodes: types
                .into_iter()
                .map(|t| {
                    vec![(
                        keys::TYPE.to_owned(),
                        Annotation::Value(PropValue::Str(t.to_owned())),
                    )]
                })
                .collect(),
            ..TypeAssignment::default()
        })
    }

    /// An empty force field named `ATD`: the ATD rules assign labels, not
    /// parameters, so there is nothing to match against. Built once.
    fn source_forcefield(&self) -> &ForceField {
        static LIBRARY: OnceLock<ForceField> = OnceLock::new();
        LIBRARY.get_or_init(|| {
            let mut ff = ForceField::new("ATD");
            ff.set_units("real");
            ff
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::Typing;
    use crate::props;
    use molrs::core::BondOrder;
    use molrs::core::Element;
    use molrs::core::keys;

    /// Methane, hand-built: C is atom 0, the four hydrogens follow.
    fn methane() -> Atomistic {
        let mut m = Atomistic::new();
        let c = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        for _ in 0..4 {
            let h = m
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                .unwrap();
            m.add_bond(c, h, props! {}).unwrap();
        }
        m
    }

    /// Typing through the base stamps a non-empty `type` on every atom and
    /// defines nothing: the match is stamp-only, so the output holds no type.
    #[test]
    fn typing_stamps_every_atom_and_defines_no_type() {
        let mut typing = Typing::new(AtdTypifier::new(AtdParameterSet::Bcc));
        let typed = typing.typify(&methane()).expect("methane types");

        assert_eq!(typed.atoms().count(), 5);
        for (id, atom) in typed.atoms() {
            let t = atom.get_str(keys::TYPE);
            assert!(t.is_some_and(|t| !t.is_empty()), "atom {id:?}: {t:?}");
        }
        assert!(
            typing
                .forcefield()
                .styles()
                .iter()
                .all(|s| s.defs().collect_type_params().is_empty()),
            "output holds no type: {:?}",
            typing.forcefield()
        );
    }

    /// The library an ATD typifier matches against is empty: no styles.
    #[test]
    fn library_is_an_empty_forcefield() {
        let typing = Typing::new(AtdTypifier::new(AtdParameterSet::Bcc));
        assert!(typing.typifier().source_forcefield().styles().is_empty());
    }

    /// A molecule as a mol2 file lists it — atoms by element, bonds in file
    /// order — with `stated` bond types (single where it says nothing).
    fn mol2(elements: &[&str], bonds: &[(usize, usize)], stated: &[BondOrder]) -> Atomistic {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = elements
            .iter()
            .map(|e| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::by_symbol(e).unwrap() })
                    .unwrap()
            })
            .collect();
        for (k, (i, j)) in bonds.iter().enumerate() {
            let b = mol.add_bond(ids[*i], ids[*j], props! {}).unwrap();
            let order = stated.get(k).copied().unwrap_or(BondOrder::Single);
            let number = order.implied_number().expect("a Kekulé fixture bond");
            mol.set_bond_class(b, order, number).unwrap();
        }
        mol
    }

    /// Heavy atoms then hydrogens, as the benchmark's mol2 files order them.
    fn elements(heavy: &[&'static str], hydrogens: usize) -> Vec<&'static str> {
        let mut e = heavy.to_vec();
        e.extend(std::iter::repeat_n("H", hydrogens));
        e
    }

    fn types(set: AtdParameterSet, orders: BondOrderSource, mol: &Atomistic) -> Vec<String> {
        let typed = Typing::new(AtdTypifier::new(set).with_bond_orders(orders))
            .typify(mol)
            .expect("types");
        typed
            .atoms()
            .map(|(_, a)| a.get_str(keys::TYPE).unwrap().to_owned())
            .collect()
    }

    fn azulene() -> Atomistic {
        mol2(
            &elements(&["C"; 10], 8),
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (3, 7),
                (7, 8),
                (8, 9),
                (0, 9),
                (0, 10),
                (1, 11),
                (2, 12),
                (4, 13),
                (5, 14),
                (6, 15),
                (8, 16),
                (9, 17),
            ],
            &[],
        )
    }

    fn cyclooctatetraene(stated: &[BondOrder]) -> Atomistic {
        mol2(
            &elements(&["C"; 8], 8),
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (0, 7),
                (0, 8),
                (1, 9),
                (2, 10),
                (3, 11),
                (4, 12),
                (5, 13),
                (6, 14),
                (7, 15),
            ],
            stated,
        )
    }

    /// `antechamber -at gaff` / `-at gaff2` (AmberTools 26.1) on the
    /// benchmark's azulene: the Kekulé structure its `bondtype` judges, and
    /// the colouring `atadjust` sweeps over it (the perimeter's parity is odd,
    /// so the sweep order is the answer).
    #[test]
    fn azulene_types_as_antechamber_types_it() {
        let want = "cc cc cd cd cc cc cd cd cc cc ha ha ha ha ha ha ha ha";
        for set in [AtdParameterSet::Gff, AtdParameterSet::Gff2] {
            assert_eq!(
                types(set, BondOrderSource::Connectivity, &azulene()).join(" "),
                want
            );
        }
    }

    /// Cyclooctatetraene, likewise — whichever Kekulé structure the input
    /// states, because antechamber ignores it. Keeping the input's orders
    /// types the structure the input drew instead.
    #[test]
    fn cyclooctatetraene_types_as_antechamber_types_it() {
        use BondOrder::{Double, Single};
        let want = "cc cc cd cd cc cc cd cd ha ha ha ha ha ha ha ha";
        let drawn = [
            Double, Single, Double, Single, Double, Single, Double, Single,
        ];
        for set in [AtdParameterSet::Gff, AtdParameterSet::Gff2] {
            for mol in [cyclooctatetraene(&[]), cyclooctatetraene(&drawn)] {
                assert_eq!(
                    types(set, BondOrderSource::Connectivity, &mol).join(" "),
                    want
                );
            }
            assert_eq!(
                types(set, BondOrderSource::Stated, &cyclooctatetraene(&drawn)).join(" "),
                "cc cd cd cc cc cd cd cc ha ha ha ha ha ha ha ha"
            );
        }
    }

    /// o-Terphenyl's middle ring holds two bridge carbons joined by an
    /// aromatic bond, which `cpadjust` flips: `cp … cq cq`.
    #[test]
    fn o_terphenyl_colours_its_bridge_carbons() {
        let mut bonds = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (0, 5),
            (3, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (6, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (12, 17),
        ];
        bonds.extend(
            [0, 1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17]
                .iter()
                .enumerate()
                .map(|(h, c)| (*c, 18 + h)),
        );
        let mol = mol2(&elements(&["C"; 18], 14), &bonds, &[]);
        let got = types(AtdParameterSet::Gff, BondOrderSource::Connectivity, &mol);
        assert_eq!(
            got[..18].join(" "),
            "ca ca ca cp ca ca cp ca ca ca ca cq cq ca ca ca ca ca"
        );
    }

    /// Carbazole's N is BCC type 23 through `XX<a1>, XB(XB(XX<a2>)) a1:a2:any`
    /// — matched along its second carbon path, so the labelled-bond check
    /// has to take part in the backtracking.
    #[test]
    fn carbazole_n_is_bcc_type_23() {
        let mut heavy = vec!["C"; 13];
        heavy[6] = "N";
        let mol = mol2(
            &elements(&heavy, 9),
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (0, 5),
                (4, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (7, 12),
                (3, 12),
                (0, 13),
                (1, 14),
                (2, 15),
                (5, 16),
                (6, 17),
                (8, 18),
                (9, 19),
                (10, 20),
                (11, 21),
            ],
            &[],
        );
        for set in [AtdParameterSet::Bcc, AtdParameterSet::Abcg2] {
            assert_eq!(types(set, BondOrderSource::Connectivity, &mol)[6], "23");
        }
        assert_eq!(
            types(AtdParameterSet::Gff, BondOrderSource::Connectivity, &mol)[..13].join(" "),
            "ca ca ca cp ca ca na ca ca ca ca ca cp"
        );
    }

    /// Methyl azide only closes in a raised valence state (C–N–N≡N), and
    /// tropylium in none, where antechamber keeps the file's single bonds.
    #[test]
    fn valence_states_and_their_failure_type_as_antechamber() {
        let azide = mol2(
            &["C", "N", "N", "N", "H", "H", "H"],
            &[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (0, 6)],
            &[],
        );
        assert_eq!(
            types(AtdParameterSet::Gff, BondOrderSource::Connectivity, &azide).join(" "),
            "c3 n2 n1 n1 h1 h1 h1"
        );
        let mut bonds: Vec<(usize, usize)> = (0..6).map(|i| (i, i + 1)).collect();
        bonds.push((0, 6));
        bonds.extend((0..7).map(|i| (i, i + 7)));
        let tropylium = mol2(&elements(&["C"; 7], 7), &bonds, &[]);
        assert_eq!(
            types(
                AtdParameterSet::Gff,
                BondOrderSource::Connectivity,
                &tropylium
            )[..7]
                .join(" "),
            "c2 c2 c2 c2 c2 c2 c2"
        );
    }
}

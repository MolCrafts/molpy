//! Convert [`SmilesIr`] and [`FragmentIr`] into [`Atomistic`] molecular
//! graphs.

use std::collections::{HashMap, HashSet};

use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::ast::*;
use crate::line_notation::atom_keys::{H_COUNT, IS_AROMATIC, ISOTOPE, STEREO, WILDCARD_Z};
use crate::line_notation::canonical_element_symbol;
use molrs::core::Atomistic;
use molrs::core::Element;
use molrs::core::MolRsError;
use molrs::core::NodeId;
use molrs::core::Props;
use molrs::core::keys;
use molrs::op::I;

impl SmilesIr {
    /// Convert a parsed SMILES IR into an [`Atomistic`] molecular graph.
    ///
    /// This resolves ring closures into bonds, sets atom properties (charge,
    /// isotope, chirality), and records bond orders. Every atom carries its
    /// species as `atomic_number` (`0` for the wildcard `*`); no `element`
    /// symbol is written. Implicit hydrogens are **not** added — call
    /// [`add_hydrogens`](crate::perceive::add_hydrogens) separately if needed.
    ///
    /// # Aromaticity
    ///
    /// Lowercase symbols are *notation*, not element symbols: `c` is carbon,
    /// `atomic_number = 6`, and the declared aromaticity is carried by the
    /// project-wide markers instead —
    /// `is_aromatic = true` on the atom, and `bond_type = Aromatic` on every bond the
    /// notation declares aromatic. Its localized `bond_number` is left `Unknown`:
    /// the notation declares delocalization, not a Kekulé phase, and kekulization
    /// is what picks one. A bond written *without* a
    /// symbol between two aromatic atoms is aromatic (the Daylight rule); an
    /// explicit `-` between two aromatic atoms (biphenyl) is not.
    ///
    /// # Hydrogen counts
    ///
    /// A bracket atom states its hydrogen count exactly, so every bracket atom
    /// gets an `h_count` component — `0` when the notation omits it. Organic-subset
    /// atoms get none and are left to valence-based
    /// [`add_hydrogens`](crate::perceive::add_hydrogens).
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::UnmatchedRingClosure`] for a ring digit that is
    /// never closed, [`SmilesErrorKind::InvalidQueryPrimitive`] for SMARTS query
    /// atoms and bond queries (which have no single atomistic interpretation), and
    /// [`SmilesErrorKind::DescriptorsUnconvertible`] for a node carrying a bonding
    /// descriptor — this is the plain conversion, and dropping the descriptor
    /// silently is the one outcome it will not produce; a fragment body is
    /// [`FragmentIr::to_atomistic`]'s.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::smiles::SmilesIr;
    ///
    /// let ir = SmilesIr::parse("C(=O)O").unwrap();
    /// let mol = ir.to_atomistic().unwrap();
    /// assert_eq!(mol.n_atoms(), 3);
    /// assert_eq!(mol.n_bonds(), 2);
    /// ```
    pub fn to_atomistic(&self) -> Result<Atomistic, SmilesError> {
        let mut builder = Builder::new(self, /*collect_descriptors*/ false);

        for component in &self.components {
            builder.build_chain(component, None)?;
        }

        builder.close_rings()?;

        Ok(builder.mol)
    }
}

/// A parsed SMILES **fragment** body: SMILES plus `CGsmiles` / `BigSMILES`
/// bonding descriptors (`[<]OCC[>]`, `[$]COC[$]`), as
/// [`SmilesIr::from_fragment`] parses it.
///
/// Its own type, because its conversion is not [`SmilesIr::to_atomistic`]:
/// the descriptors are data the graph does not hold, so
/// [`to_atomistic`](Self::to_atomistic) returns them beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentIr {
    ir: SmilesIr,
}

/// The fragment body as text — what [`SmilesIr::from_fragment`] parses back
/// to the same IR. Descriptors are written in the canonical trailing form
/// (`C=[$]CC`); see the fragment writer for the conventions. Formatting fails
/// for an IR no fragment parser produces (an empty one, a SMARTS query atom,
/// a descriptor order outside single … quadruple).
///
/// ```
/// use molrs::io::smiles::SmilesIr;
///
/// let body = SmilesIr::from_fragment("[<]OCC[>]")?;
/// let again = SmilesIr::from_fragment(&body.to_string())?;
/// assert_eq!(again.to_string(), body.to_string());
/// assert_eq!(again.to_template()?.n_ports(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl std::fmt::Display for FragmentIr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = crate::line_notation::writer::fragment_smiles_text(&self.ir)
            .map_err(|_| std::fmt::Error)?;
        f.write_str(&text)
    }
}

impl FragmentIr {
    /// Wrap a fragment-dialect IR (what the fragment parser produced).
    pub(crate) fn new(ir: SmilesIr) -> Self {
        Self { ir }
    }

    /// The syntax tree.
    pub fn ir(&self) -> &SmilesIr {
        &self.ir
    }

    /// Convert the fragment into an [`Atomistic`] graph plus the descriptor
    /// map.
    ///
    /// The graph is built by the same walk as [`SmilesIr::to_atomistic`], so
    /// descriptor-free input gives the same atoms, bonds and properties either
    /// way.
    ///
    /// The second return value pairs each descriptor with the [`NodeId`] of the
    /// atom it binds to. Its order is a contract, not an accident: the later
    /// stages that turn descriptors into *ports* — the named joining sites a
    /// fragment offers — index this vector, so the order is fixed as atom-visit
    /// order first, then each atom's own descriptor-list order. That is not text
    /// order: in `C(N[<])[>]` the `>` the parser folded onto the head carbon
    /// precedes the `<` on the branch nitrogen, because the walk reaches the head
    /// carbon before it descends into the branch. Inputs with several
    /// `.`-separated components are walked component by component, in written
    /// order.
    ///
    /// Descriptors are data, not atoms: no hydrogens are added, and an unpaired
    /// descriptor creates neither an atom nor a bond here. Filling open sites — by
    /// pairing them with another fragment's, or by capping them with hydrogen — is
    /// a later step that reads this map.
    ///
    /// Descriptors are passed through, not re-checked: the parser validated every
    /// descriptor at its single construction site, so a hand-built IR carrying a
    /// label or order no parser would have produced is returned as given.
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::UnmatchedRingClosure`] for a ring digit that is
    /// never closed and [`SmilesErrorKind::InvalidQueryPrimitive`] for SMARTS
    /// query atoms and bond queries (which have no single atomistic
    /// interpretation).
    pub fn to_atomistic(
        &self,
    ) -> Result<(Atomistic, Vec<(NodeId, BondingDescriptor)>), SmilesError> {
        fragment_to_atomistic(&self.ir)
    }
}

/// The fragment walk: the graph plus each descriptor against its atom.
fn fragment_to_atomistic(
    ir: &SmilesIr,
) -> Result<(Atomistic, Vec<(NodeId, BondingDescriptor)>), SmilesError> {
    let mut builder = Builder::new(ir, /*collect_descriptors*/ true);

    for component in &ir.components {
        builder.build_chain(component, None)?;
    }

    builder.close_rings()?;

    let Builder {
        mol, descriptors, ..
    } = builder;
    Ok((mol, descriptors))
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Pending ring closure: the atom that opened it and the optional bond kind.
struct PendingRing {
    atom: NodeId,
    bond: Option<BondKind>,
    span: Span,
}

/// Reduce a SMARTS-style [`BondQuery`] back to a single [`BondKind`]. The
/// SMILES → atomistic pipeline cannot represent SMARTS logical bond
/// operators (there's no single concrete order for `!=` or `-,=`), so a
/// query that isn't a plain `Kind(_)` is rejected with a clean error.
fn bond_query_to_kind(q: Option<&BondQuery>) -> Result<Option<BondKind>, SmilesError> {
    match q {
        None => Ok(None),
        Some(BondQuery::Kind(k)) => Ok(Some(*k)),
        Some(_) => Err(SmilesError::new(
            SmilesErrorKind::InvalidQueryPrimitive(
                "SMARTS bond query cannot be atomized".to_owned(),
            ),
            Span::new(0, 0),
            "",
            Notation::Smarts,
        )),
    }
}

struct Builder<'a> {
    mol: Atomistic,
    /// Maps ring numbers to pending (unmatched) ring-closure openers.
    open_rings: HashMap<u16, PendingRing>,
    /// Atoms the notation declared aromatic (lowercase symbol). Needed while
    /// building because a symbol-less bond between two of them is aromatic.
    aromatic_atoms: HashSet<NodeId>,
    /// Reference to the original IR for error messages.
    ir: &'a SmilesIr,
    /// Which entry point this walk serves: `true` on the fragment path, which
    /// records every descriptor against its atom; `false` on the plain path,
    /// where a node carrying one is an error rather than a silent loss.
    collect_descriptors: bool,
    /// Each descriptor against the atom it binds to, in visit order. Empty on
    /// the plain path, which never gets past the guard in
    /// [`Builder::add_atom_node`]. Owned, not borrowed: the map is returned by
    /// value.
    descriptors: Vec<(NodeId, BondingDescriptor)>,
}

impl<'a> Builder<'a> {
    fn new(ir: &'a SmilesIr, collect_descriptors: bool) -> Self {
        Self {
            mol: Atomistic::new(),
            open_rings: HashMap::new(),
            aromatic_atoms: HashSet::new(),
            ir,
            collect_descriptors,
            descriptors: Vec::new(),
        }
    }

    /// Build a chain, returning the [`NodeId`] of the head atom.
    ///
    /// `prev` is the atom to bond the head to (if any — `None` for top-level).
    fn build_chain(
        &mut self,
        chain: &Chain,
        prev: Option<(NodeId, Option<BondKind>)>,
    ) -> Result<NodeId, SmilesError> {
        let head_id = self.add_atom_node(&chain.head)?;

        // Bond head to the previous atom (if coming from a branch or sequence).
        if let Some((prev_id, bond)) = prev {
            self.add_bond(prev_id, head_id, bond)?;
        }

        let mut current = head_id;

        for elem in &chain.tail {
            match elem {
                ChainElement::BondedAtom { bond, atom } => {
                    let atom_id = self.add_atom_node(atom)?;
                    self.add_bond(current, atom_id, bond_query_to_kind(bond.as_ref())?)?;
                    current = atom_id;
                }
                ChainElement::Branch { bond, chain, .. } => {
                    // Branch: build sub-chain rooted at `current`.
                    self.build_chain(chain, Some((current, bond_query_to_kind(bond.as_ref())?)))?;
                    // `current` does NOT change — branches don't advance the main chain.
                }
                ChainElement::RingClosure { bond, rnum, span } => {
                    self.handle_ring_closure(
                        current,
                        *rnum,
                        bond_query_to_kind(bond.as_ref())?,
                        *span,
                    )?;
                }
            }
        }

        Ok(head_id)
    }

    /// Create an atom from an [`AtomNode`] and return its id.
    ///
    /// Every atom of every chain, branch and component flows through here, so
    /// this is also where bonding descriptors are either refused (plain path)
    /// or recorded against their atom (fragment path). The atom is written as
    /// one property bag — species, mass, aromatic marker and the bracket
    /// facts together — so a refused write leaves no half-built atom behind.
    ///
    /// # Species
    ///
    /// Every atom carries [`keys::ATOMIC_NUMBER`]: the element's Z, or `0` for
    /// the wildcard (`*`, and the bracket `[*]`, `[A]`, `[a]`). No `element`
    /// symbol is written; a caller that wants one converts explicitly with
    /// [`Element::fill_symbols`].
    ///
    /// # Mass
    ///
    /// Every atom that names an element carries `mass` (amu, `f64`): the
    /// standard atomic weight [`Element::atomic_mass`] states, the same table
    /// hydrogen repletion reads. An atom written with an explicit isotope
    /// (`[13C]`) carries its **mass number** A instead — an approximation,
    /// since molrs has no isotope-mass table; the error is below 0.02 amu for
    /// the light elements. The wildcard names no element and carries no mass,
    /// isotope or not: an isotope on a wildcard is an attachment label, not a
    /// nucleus.
    ///
    /// # Bracket facts
    ///
    /// Integer facts are stored as integers: `isotope`, `h_count`,
    /// [`keys::FORMAL_CHARGE`] and the atom class as [`keys::ATOM_MAP`]. The
    /// chirality is the string `stereo` (`"CW"` / `"CCW"`), and a written
    /// aromatic atom carries `is_aromatic = true`.
    fn add_atom_node(&mut self, node: &AtomNode) -> Result<NodeId, SmilesError> {
        if !self.collect_descriptors && !node.descriptors.is_empty() {
            return Err(SmilesError::new(
                SmilesErrorKind::DescriptorsUnconvertible,
                node.span,
                "", // input not available here; span is enough
                Notation::Smiles,
            ));
        }

        let aromatic = node.spec.written_aromatic();
        let props = match &node.spec {
            AtomSpec::Organic { symbol, .. } => {
                let element = self.element(symbol, node.span)?;
                species_props(Some(element), None, aromatic)
            }
            AtomSpec::Bracket {
                isotope,
                symbol,
                chirality,
                hcount,
                charge,
                atom_class,
            } => {
                let element = match symbol {
                    BracketSymbol::Element { symbol, .. } => Some(self.element(symbol, node.span)?),
                    BracketSymbol::Any | BracketSymbol::Aliphatic | BracketSymbol::Aromatic => None,
                };
                let mut props = species_props(element, *isotope, aromatic);
                if let Some(iso) = isotope {
                    props.set(ISOTOPE, I::from(*iso));
                }
                if let Some(ch) = chirality {
                    let s = match ch {
                        Chirality::CounterClockwise => "CCW",
                        Chirality::Clockwise => "CW",
                    };
                    props.set(STEREO, s);
                }
                // A bracket atom states its hydrogen count exactly; an omitted
                // count means zero, not "fill the valence".
                props.set(H_COUNT, I::from(hcount.unwrap_or(0)));
                if let Some(c) = charge {
                    props.set(keys::FORMAL_CHARGE, I::from(*c));
                }
                if let Some(cls) = atom_class {
                    props.set(keys::ATOM_MAP, I::from(*cls));
                }
                props
            }
            AtomSpec::Wildcard => species_props(None, None, false),
            AtomSpec::Query(_) => {
                return Err(SmilesError::new(
                    SmilesErrorKind::InvalidQueryPrimitive(
                        "SMARTS query atoms cannot be converted to Atomistic".into(),
                    ),
                    node.span,
                    "", // input not available here; span is enough
                    Notation::Smarts,
                ));
            }
        };

        let id = self
            .mol
            .add_atom(props)
            .map_err(|e| self.build_error(&e, node.span))?;
        if aromatic {
            self.aromatic_atoms.insert(id);
        }

        if self.collect_descriptors {
            self.descriptors
                .extend(node.descriptors.iter().map(|desc| (id, desc.clone())));
        }

        Ok(id)
    }

    /// The element a SMILES atom symbol names — the aromatic lowercase form
    /// (`c`, `se`) included.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::Build`] at `span` when `symbol` names no element: the
    /// parser refuses such a symbol, so only a hand-built IR reaches this.
    fn element(&self, symbol: &str, span: Span) -> Result<Element, SmilesError> {
        Element::by_symbol(&canonical_element_symbol(symbol)).ok_or_else(|| {
            self.build_error(
                &MolRsError::validation(format!("'{symbol}' is not an element symbol")),
                span,
            )
        })
    }

    /// Bond `a`–`b` as the notation wrote it, in one `add_bond` whose props
    /// carry the class, the localized number and any directional `stereo`.
    ///
    /// A bond with no symbol between two aromatic atoms is aromatic (the
    /// Daylight rule); an unwritten bond otherwise states no class, and
    /// [`Atomistic::add_bond`] stamps it single.
    fn add_bond(
        &mut self,
        a: NodeId,
        b: NodeId,
        bond: Option<BondKind>,
    ) -> Result<(), SmilesError> {
        // A bond with no symbol between two aromatic atoms is aromatic — that
        // is what makes `c1ccccc1` a ring of aromatic bonds rather than six
        // single bonds. An explicit symbol always wins (biphenyl's `-`).
        let kind = bond.or_else(|| {
            (self.aromatic_atoms.contains(&a) && self.aromatic_atoms.contains(&b))
                .then_some(BondKind::Aromatic)
        });

        let mut props = Props::new();
        if let Some(kind) = kind {
            props.set(keys::BOND_TYPE, kind.bond_type());
            props.set(keys::BOND_NUMBER, kind.bond_number());
            match kind {
                BondKind::Up => props.set(STEREO, "up"),
                BondKind::Down => props.set(STEREO, "down"),
                _ => {}
            }
        }

        self.mol
            .add_bond(a, b, props)
            .map_err(|e| self.build_error(&e, self.ir.span))?;
        Ok(())
    }

    /// Wrap a graph-write failure as [`SmilesErrorKind::Build`] at `span`.
    ///
    /// Every write this builder makes is to a node or relation it has just
    /// created, with a key and a value the grammar fixes, so a failure is an
    /// internal invariant of the builder rather than anything the input said.
    /// Reporting it by value is what keeps the walk free of discarded
    /// `Result`s: a silently dropped write would leave an `Atomistic` missing
    /// a component the conversion promised to set, and no caller would learn.
    /// The input text is not available at this stage, so the error carries the
    /// span alone.
    fn build_error(&self, e: &MolRsError, span: Span) -> SmilesError {
        SmilesError::new(
            SmilesErrorKind::Build(e.to_string()),
            span,
            "",
            Notation::Smiles,
        )
    }

    fn handle_ring_closure(
        &mut self,
        current: NodeId,
        rnum: u16,
        bond: Option<BondKind>,
        span: Span,
    ) -> Result<(), SmilesError> {
        if let Some(pending) = self.open_rings.remove(&rnum) {
            // Close the ring: bond `pending.atom` ↔ `current`.
            // Use the bond type from whichever side specified one
            // (the opener or the closer). If both specified, they must agree.
            let effective_bond = match (pending.bond, bond) {
                (Some(a), Some(b)) if a != b => {
                    return Err(SmilesError::new(
                        SmilesErrorKind::RingBondConflict { rnum },
                        span,
                        "",
                        Notation::Smiles,
                    ));
                }
                (Some(a), _) => Some(a),
                (_, Some(b)) => Some(b),
                (None, None) => None,
            };
            self.add_bond(pending.atom, current, effective_bond)?;
        } else {
            // Open a new ring closure.
            self.open_rings.insert(
                rnum,
                PendingRing {
                    atom: current,
                    bond,
                    span,
                },
            );
        }
        Ok(())
    }

    fn close_rings(&self) -> Result<(), SmilesError> {
        if let Some((&rnum, pending)) = self.open_rings.iter().next() {
            return Err(SmilesError::new(
                SmilesErrorKind::UnmatchedRingClosure(rnum),
                pending.span,
                "",
                Notation::Smiles,
            ));
        }
        Ok(())
    }
}

/// The species half of an atom's bag: the atomic number (`0` for the wildcard,
/// `element == None`), the mass [`Builder::add_atom_node`] states for it, and
/// the aromatic marker when the notation wrote the atom aromatic.
fn species_props(element: Option<Element>, isotope: Option<u16>, aromatic: bool) -> Props {
    let mut props = match element {
        Some(element) => Props::new().with(keys::ATOMIC_NUMBER, element).with(
            keys::MASS,
            isotope.map_or(f64::from(element.atomic_mass()), f64::from),
        ),
        None => Props::new().with(keys::ATOMIC_NUMBER, WILDCARD_Z),
    };
    if aromatic {
        props.set(IS_AROMATIC, true);
    }
    props
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::SmilesIr;
    use molrs::core::PropValue;

    fn smiles_to_mol(input: &str) -> Atomistic {
        let ir = SmilesIr::parse(input).unwrap();
        ir.to_atomistic()
            .unwrap_or_else(|e| panic!("to_atomistic({input:?}) failed: {e}"))
    }

    // -- basic molecules ----------------------------------------------------

    #[test]
    fn test_single_atom() {
        let mol = smiles_to_mol("C");
        assert_eq!(mol.n_atoms(), 1);
        assert_eq!(mol.n_bonds(), 0);
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.atomic_number(), Some(6));
        assert!(
            !atom.contains_key(keys::ELEMENT),
            "the builder writes the species as Z, never as a symbol"
        );
    }

    #[test]
    fn test_ethane() {
        let mol = smiles_to_mol("CC");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
    }

    #[test]
    fn test_ethanol() {
        let mol = smiles_to_mol("CCO");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_bonds(), 2);
    }

    // -- bond orders --------------------------------------------------------

    #[test]
    fn test_double_bond() {
        let mol = smiles_to_mol("C=O");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        let (_, bond) = mol.bonds().next().unwrap();
        assert_eq!(bond.props.get(keys::BOND_TYPE), Some(&PropValue::Int(2)));
        assert_eq!(bond.props.get(keys::BOND_NUMBER), Some(&PropValue::Int(2)));
    }

    #[test]
    fn test_triple_bond() {
        let mol = smiles_to_mol("C#N");
        let (_, bond) = mol.bonds().next().unwrap();
        assert_eq!(bond.props.get(keys::BOND_TYPE), Some(&PropValue::Int(3)));
        assert_eq!(bond.props.get(keys::BOND_NUMBER), Some(&PropValue::Int(3)));
    }

    // -- branches -----------------------------------------------------------

    #[test]
    fn test_branch_isobutane() {
        // isobutane: CC(C)C
        let mol = smiles_to_mol("CC(C)C");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    #[test]
    fn test_acetic_acid() {
        // CC(=O)O
        let mol = smiles_to_mol("CC(=O)O");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    // -- ring closures ------------------------------------------------------

    #[test]
    fn test_cyclohexane() {
        let mol = smiles_to_mol("C1CCCCC1");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6); // 5 chain + 1 ring closure
    }

    #[test]
    fn test_benzene() {
        let mol = smiles_to_mol("c1ccccc1");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
        // Check aromatic flag
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get(IS_AROMATIC), Some(&PropValue::Bool(true)));
    }

    #[test]
    fn test_aromatic_element_symbol_is_canonical() {
        // The lowercase `c` of the organic aromatic subset is *notation*, not an
        // element symbol: it is carbon, Z = 6.
        let mol = smiles_to_mol("c1ccccc1");
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.atomic_number(), Some(6));
        }
    }

    #[test]
    fn test_aromatic_bracket_element_symbol_is_canonical() {
        // Pyrrole — the bracket `[nH]` must land as nitrogen, flagged aromatic.
        let mol = smiles_to_mol("c1cc[nH]c1");
        let species: Vec<Option<u8>> = mol.atoms().map(|(_, a)| a.atomic_number()).collect();
        assert_eq!(species, [Some(6), Some(6), Some(6), Some(7), Some(6)]);
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get(IS_AROMATIC), Some(&PropValue::Bool(true)));
        }
    }

    #[test]
    fn test_implicit_aromatic_bond_gets_the_aromatic_class() {
        // A bond written without a symbol between two aromatic atoms *is* an
        // aromatic bond (Daylight rule) — order 1.5, not the single-bond default.
        let mol = smiles_to_mol("c1ccccc1");
        for (_, bond) in mol.bonds() {
            // The notation declares the class; it declares no Kekulé phase, so
            // the localized number stays unknown until standardization.
            assert_eq!(bond.props.get(keys::BOND_TYPE), Some(&PropValue::Int(4)));
            assert_eq!(bond.props.get(keys::BOND_NUMBER), Some(&PropValue::Int(0)));
        }
    }

    #[test]
    fn test_aromatic_smarts_discriminates_ring_from_chain() {
        // The markers exist so SMARTS can tell an aromatic ring from a
        // saturated one: `c` must match benzene and `C` must not, and the
        // reverse for cyclohexane.
        use crate::perceive::smarts::{MatchOptions, SmartsPattern};

        let n_matches = |pattern: &str, smiles: &str| {
            SmartsPattern::parse(pattern)
                .expect("parse SMARTS")
                .find(&smiles_to_mol(smiles), MatchOptions::default())
                .len()
        };

        assert_eq!(n_matches("[c]", "c1ccccc1"), 6, "benzene is aromatic");
        assert_eq!(n_matches("[C]", "c1ccccc1"), 0, "benzene is not aliphatic");
        assert_eq!(n_matches("[C]", "C1CCCCC1"), 6, "cyclohexane is aliphatic");
        assert_eq!(
            n_matches("[c]", "C1CCCCC1"),
            0,
            "cyclohexane is not aromatic"
        );
    }

    #[test]
    fn test_bond_from_aromatic_to_aliphatic_is_not_aromatic() {
        // Toluene: the ring→methyl bond has one aliphatic end, so it stays single.
        let mol = smiles_to_mol("Cc1ccccc1");
        let n_aromatic = mol
            .bonds()
            .filter(|(_, b)| b.props.get(keys::BOND_TYPE) == Some(&PropValue::Int(4)))
            .count();
        assert_eq!(n_aromatic, 6, "only the 6 ring bonds are aromatic");
    }

    #[test]
    fn test_aliphatic_bond_does_not_get_the_aromatic_class() {
        let mol = smiles_to_mol("CC");
        let (_, bond) = mol.bonds().next().unwrap();
        assert_ne!(bond.props.get(keys::BOND_TYPE), Some(&PropValue::Int(4)));
    }

    #[test]
    fn test_bracket_atom_records_exact_hydrogen_count() {
        // In a bracket atom the H count is *exact* — an omitted count means zero,
        // it does not mean "fill the valence".
        let mol = smiles_to_mol("[C]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get(H_COUNT), Some(&PropValue::Int(0)));
    }

    #[test]
    fn test_two_digit_ring() {
        let mol = smiles_to_mol("C%12CCCCC%12");
        assert_eq!(mol.n_atoms(), 6);
        assert_eq!(mol.n_bonds(), 6);
    }

    // -- bracket atoms with properties --------------------------------------

    #[test]
    fn test_isotope() {
        let mol = smiles_to_mol("[13CH4]");
        assert_eq!(mol.n_atoms(), 1);
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get(ISOTOPE), Some(&PropValue::Int(13)));
        assert_eq!(atom.get(H_COUNT), Some(&PropValue::Int(4)));
    }

    #[test]
    fn test_charge() {
        let mol = smiles_to_mol("[Fe+2]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.atomic_number(), Some(26));
        assert_eq!(atom.get(keys::FORMAL_CHARGE), Some(&PropValue::Int(2)));
    }

    #[test]
    fn test_negative_charge() {
        let mol = smiles_to_mol("[O-]");
        let (_, atom) = mol.atoms().next().unwrap();
        assert_eq!(atom.get(keys::FORMAL_CHARGE), Some(&PropValue::Int(-1)));
    }

    #[test]
    fn test_chirality() {
        let mol = smiles_to_mol("[C@@H](F)(Cl)Br");
        assert_eq!(mol.n_atoms(), 4); // C, F, Cl, Br (H is in h_count)
        let atoms: Vec<_> = mol.atoms().collect();
        let c_atom = &atoms
            .iter()
            .find(|(_, a)| a.atomic_number() == Some(6))
            .unwrap()
            .1;
        assert_eq!(c_atom.get_str(STEREO), Some("CW"));
    }

    #[test]
    fn test_atom_class() {
        let mol = smiles_to_mol("[CH3:1]");
        let (_, atom) = mol.atoms().next().unwrap();
        // The atom class is the schema's atom map, an unsigned integer.
        assert_eq!(atom.get(keys::ATOM_MAP), Some(&PropValue::Int(1)));
    }

    // -- mass ---------------------------------------------------------------
    //
    // The expected mass of an element atom is read from molrs's own `Element`
    // table (the source of truth, `.claude/specs/assembly-03-template.md`
    // § Domain basis); an explicit isotope writes its mass number A. No
    // external program produced any value here.

    /// The standard atomic weight of `symbol` from the `Element` table, as
    /// the `f64` a mass prop stores.
    fn table_mass(symbol: &str) -> f64 {
        f64::from(
            molrs::core::Element::by_symbol(symbol)
                .unwrap_or_else(|| panic!("{symbol:?} is not in the Element table"))
                .atomic_mass(),
        )
    }

    #[test]
    fn test_element_atoms_carry_their_table_mass() {
        let mol = smiles_to_mol("CCO");
        let masses: Vec<(Option<u8>, Option<f64>)> = mol
            .atoms()
            .map(|(_, a)| (a.atomic_number(), a.get_f64(molrs::core::keys::MASS)))
            .collect();
        let expected = [("C", 6), ("C", 6), ("O", 8)];
        assert_eq!(masses.len(), expected.len());
        for ((z, mass), (element, want_z)) in masses.iter().zip(expected) {
            assert_eq!(*z, Some(want_z), "{element}");
            let mass = mass.unwrap_or_else(|| panic!("{element} atom carries no mass"));
            assert!(
                (mass - table_mass(element)).abs() < 1e-12,
                "{element}: mass {mass} != table {}",
                table_mass(element)
            );
        }
    }

    #[test]
    fn test_isotope_atom_carries_its_mass_number() {
        let mol = smiles_to_mol("[13CH4]");
        let (_, atom) = mol.atoms().next().unwrap();
        let mass = atom
            .get_f64(molrs::core::keys::MASS)
            .expect("an isotope atom carries a mass");
        assert!((mass - 13.0).abs() < 1e-12, "mass {mass} != 13.0");
    }

    #[test]
    fn test_wildcard_atom_carries_no_mass() {
        let mol = smiles_to_mol("*C");
        let atoms: Vec<_> = mol.atoms().map(|(_, a)| a).collect();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].atomic_number(), Some(WILDCARD_Z));
        assert!(
            !atoms[0].contains_key(molrs::core::keys::MASS),
            "the wildcard has no element, hence no mass"
        );
        assert_eq!(atoms[1].atomic_number(), Some(6));
        let mass = atoms[1]
            .get_f64(molrs::core::keys::MASS)
            .expect("the carbon carries a mass");
        assert!((mass - table_mass("C")).abs() < 1e-12);
    }

    // -- disconnected components --------------------------------------------

    #[test]
    fn test_salt() {
        let mol = smiles_to_mol("[Na+].[Cl-]");
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 0); // disconnected
    }

    // -- directional bonds --------------------------------------------------

    #[test]
    fn test_cis_trans() {
        let mol = smiles_to_mol("F/C=C/F");
        assert_eq!(mol.n_atoms(), 4);
        assert_eq!(mol.n_bonds(), 3);
    }

    // -- real molecules -----------------------------------------------------

    #[test]
    fn test_caffeine() {
        let mol = smiles_to_mol("Cn1cnc2c1c(=O)n(c(=O)n2C)C");
        assert!(mol.n_atoms() >= 14);
    }

    #[test]
    fn test_aspirin() {
        let mol = smiles_to_mol("CC(=O)Oc1ccccc1C(=O)O");
        assert!(mol.n_atoms() >= 13);
    }

    // -- error cases --------------------------------------------------------

    /// Hand-built IR for `CCCC1` — a chain of four carbons whose last element
    /// is a ring digit that never closes.
    ///
    /// The parser refuses an unmatched ring closure itself, so this unit can
    /// only be reached with an IR built directly.
    fn unmatched_ring_ir() -> SmilesIr {
        fn carbon(start: usize) -> AtomNode {
            AtomNode {
                spec: AtomSpec::Organic {
                    symbol: "C".to_owned(),
                    aromatic: false,
                },
                span: Span::new(start, start + 1),
                descriptors: Vec::new(),
            }
        }

        SmilesIr {
            components: vec![Chain {
                head: carbon(0),
                tail: vec![
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(1),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(2),
                    },
                    ChainElement::BondedAtom {
                        bond: None,
                        atom: carbon(3),
                    },
                    ChainElement::RingClosure {
                        bond: None,
                        rnum: 1,
                        span: Span::new(4, 5),
                    },
                ],
            }],
            span: Span::new(0, 5),
        }
    }

    #[test]
    fn test_unmatched_ring() {
        let ir = unmatched_ring_ir();
        let err = ir.to_atomistic().unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnmatchedRingClosure(1)));
    }

    #[test]
    fn test_smarts_query_rejected() {
        let ir = crate::line_notation::parser::parse_smarts("[!C]").unwrap();
        let err = ir.to_atomistic();
        assert!(err.is_err());
    }

    // -- fragment dialect: bonding descriptors ------------------------------

    fn fragment_to_mol(input: &str) -> (Atomistic, Vec<(NodeId, BondingDescriptor)>) {
        let ir =
            FragmentIr::new(crate::line_notation::parser::parse_fragment_smiles(input).unwrap());
        ir.to_atomistic()
            .unwrap_or_else(|e| panic!("FragmentIr::to_atomistic({input:?}) failed: {e}"))
    }

    /// Atom ids in the order the walker created them.
    fn atom_ids(mol: &Atomistic) -> Vec<NodeId> {
        mol.atoms().map(|(id, _)| id).collect()
    }

    /// Atomic numbers in creation order.
    fn atomic_numbers(mol: &Atomistic) -> Vec<Option<u8>> {
        mol.atoms().map(|(_, a)| a.atomic_number()).collect()
    }

    /// Declared bond class (`bond_type`, `bond_number`) in creation order.
    fn bond_classes(mol: &Atomistic) -> Vec<(Option<PropValue>, Option<PropValue>)> {
        mol.bonds()
            .map(|(_, b)| {
                (
                    b.props.get(keys::BOND_TYPE).cloned(),
                    b.props.get(keys::BOND_NUMBER).cloned(),
                )
            })
            .collect()
    }

    #[test]
    fn test_to_atomistic_rejects_a_descriptor_bearing_ir() {
        // The plain entry point routes the caller instead of silently dropping
        // the descriptors the fragment dialect parsed.
        let fragment = SmilesIr::from_fragment("[$]COC[$]").unwrap();
        let err = fragment.ir().to_atomistic().unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::DescriptorsUnconvertible
        ));
    }

    #[test]
    fn test_fragment_to_atomistic_builds_the_heavy_atom_graph() {
        // `[$]COC[$]` is three heavy atoms and two bonds; descriptors are not
        // atoms and add neither.
        let (mol, _) = fragment_to_mol("[$]COC[$]");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_bonds(), 2);
    }

    #[test]
    fn test_fragment_to_atomistic_maps_each_descriptor_to_its_atom() {
        // R4.2: the leading `$` binds to the first carbon, the trailing one to
        // the last, and the map lists them in that (visit) order.
        let (mol, ports) = fragment_to_mol("[$]COC[$]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Symmetric)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[2], DescriptorKind::Symmetric)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_keeps_per_atom_descriptor_order() {
        // R4.3: atom 0 carries `[>]` then `[$1]` in written order, so the map
        // emits both before moving on to the next atom's descriptors.
        let (mol, ports) = fragment_to_mol("[>][$1]COC[<]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 3);
        assert_eq!(
            (ports[0].0, ports[0].1.kind, ports[0].1.label.as_str()),
            (ids[0], DescriptorKind::Right, "")
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind, ports[1].1.label.as_str()),
            (ids[0], DescriptorKind::Symmetric, "1")
        );
        assert_eq!(
            (ports[2].0, ports[2].1.kind, ports[2].1.label.as_str()),
            (ids[2], DescriptorKind::Left, "")
        );
    }

    #[test]
    fn test_fragment_to_atomistic_map_follows_the_atom_descriptor_list() {
        // `C([$]O)[>]`: `(` is not a node, so the branch's leading `$` and the
        // trailing `>` both anchor on the first carbon; the map keeps that
        // atom's list order, `$` before `>`.
        let (mol, ports) = fragment_to_mol("C([$]O)[>]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Symmetric)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[0], DescriptorKind::Right)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_map_is_visit_order_not_text_position() {
        // `C(N[<])[>]`: `<` anchors on the branch nitrogen (atom 1), `>` on the
        // head carbon (atom 0), which the parser folds on after the branch. The
        // walker visits the head before the branch, so the `>` entry comes
        // first — the reverse of the text, where `[<]` precedes `[>]`.
        let (mol, ports) = fragment_to_mol("C(N[<])[>]");
        let ids = atom_ids(&mol);
        assert_eq!(ports.len(), 2);
        assert_eq!(
            (ports[0].0, ports[0].1.kind),
            (ids[0], DescriptorKind::Right)
        );
        assert_eq!(
            (ports[1].0, ports[1].1.kind),
            (ids[1], DescriptorKind::Left)
        );
    }

    #[test]
    fn test_fragment_to_atomistic_matches_to_atomistic_without_descriptors() {
        // One walker behind two entry points: descriptor-free input must give
        // the same graph through either, and an empty map.
        let plain = (SmilesIr::parse("CC(=O)O").unwrap())
            .to_atomistic()
            .unwrap();
        let (fragment, ports) = fragment_to_mol("CC(=O)O");
        assert!(ports.is_empty());
        assert_eq!(fragment.n_atoms(), plain.n_atoms());
        assert_eq!(fragment.n_bonds(), plain.n_bonds());
        assert_eq!(atomic_numbers(&fragment), atomic_numbers(&plain));
        assert_eq!(bond_classes(&fragment), bond_classes(&plain));
    }
}

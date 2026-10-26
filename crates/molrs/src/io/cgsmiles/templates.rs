//! Atomistic fragment templates: one ported [`Atomistic`] per definition of the last
//! `CGsmiles` fragment table.
//!
//! A template sits beside the expansion
//! [`CgSmilesIr::to_atomistic`](crate::io::cgsmiles::CgSmilesIr::to_atomistic)
//! builds, and answers the other half of the question a `CGsmiles` string
//! poses. Expansion says what the *whole molecule* is; a template says what
//! **one named piece** is — the instance-free graph of `#PEO` with its
//! unsatisfied valences made explicit, which is what a builder places a
//! thousand times without re-reading the string, and what a coarse-graining
//! mapping matches against.
//!
//! **The unit is the table, not one definition.** A `CGsmiles` string writes
//! its atomistic fragments as one block, `{#OH=[$]O,#PEO=[$]COC[$]}`, and a
//! caller that wants `#PEO` almost always wants the library it was written in;
//! returning one value per call would make the caller re-walk the same string
//! once per name. The returned [`BTreeMap`] is keyed by the name written after
//! `#`, and its key set is exactly the last table's — including a definition
//! the level above never references, which is a fragment all the same (R4.1: a
//! fragment is a graph plus attachment points, whether or not anything uses
//! it).
//!
//! **One walker, once per definition.** Every body is converted by
//! [`FragmentIr::to_atomistic`](crate::io::smiles::FragmentIr::to_atomistic), the
//! single walker over a fragment body, entered exactly once for each entry of
//! the table. 01d's per-call `FragmentCache` amortizes cloning a body across
//! *instances*; a template is definition-level and has no instances, so there
//! is nothing to amortize and no cache here.
//!
//! **One piece without a table.** A unit written on its own — the body
//! `[<]OCC[>]`, parsed by
//! [`SmilesIr::from_fragment`](crate::io::smiles::SmilesIr::from_fragment) — is
//! built by [`FragmentIr::to_template`], the same conversion a table entry goes
//! through; a caller does not wrap the body in a one-bead `CGsmiles` string
//! only to index the one template back out.

use std::collections::BTreeMap;

use crate::io::cgsmiles::{CgSmilesIr, FragmentBody};
use crate::io::smiles::{BondingDescriptor, DescriptorKind, FragmentIr, Span};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use molrs::core::Atomistic;
use molrs::core::BondNumber;
use molrs::core::Element;
use molrs::core::MolRsError;
use molrs::core::NodeId;
use molrs::core::PortKind;
use molrs::core::keys;

impl CgSmilesIr {
    /// Build one ported [`Atomistic`] template per definition of the last
    /// fragment table.
    ///
    /// Each body is converted to its heavy-atom graph, every bonding
    /// descriptor it wrote becomes a **capping hydrogen** bonded to its anchor,
    /// and each `(anchor, handle)` pair
    /// is recorded as a port carrying the descriptor's kind (R4.3), label
    /// (R4.4) and bond order (R4.5). Keys are the names written after `#`, in
    /// name order.
    ///
    /// # What a handle is
    ///
    /// A port names two atoms, and the second one is a **real hydrogen atom**,
    /// not a dummy: an unconsumed descriptor is discarded and the valence it
    /// freed is filled with hydrogen (R4.17), so the hydrogen is what the
    /// notation makes of an open site, and a later pairing *replaces* it rather
    /// than adding to it. That is why a template of `[$]COC[$]` is five atoms
    /// and four bonds, not three and two.
    ///
    /// # No valence-bearing property is written
    ///
    /// Capping adds an atom and a bond and writes **no valence-bearing
    /// property**: no `h_count`, no `formal_charge`, no `is_aromatic` on
    /// either anchor or handle. The handle carries its `atomic_number` and the
    /// hydrogen `mass` of the [`Element`] table — mass is not
    /// valence-bearing, so it changes nothing here. That single rule keeps
    /// two opposite valence paths both correct, and any "adjust the anchor's
    /// `h_count` by one" breaks both:
    ///
    /// * **Organic-subset anchor** (`[$]COC[$]`): the SMILES builder declares
    ///   no `h_count`, so hydrogen repletion sums the atom's real incident
    ///   bonds — the handle bond among them. C0 counts O + handle = 2 and a
    ///   later
    ///   [`add_hydrogens`](crate::perceive::add_hydrogens) adds
    ///   exactly the remaining 2. The handle is *subtracted* from the budget.
    /// * **Bracket anchor** (`[$][CH3]`, `[$][O-]`): the builder always writes
    ///   `h_count`, and repletion then short-circuits on it, bond-blind. The
    ///   handle is *additive*. molrs follows the Daylight bracket rule (a
    ///   bracket atom states its hydrogen count exactly); the reference
    ///   implementation instead recomputes hydrogens from valence at
    ///   repletion, which agrees for neutral bracket atoms and can differ
    ///   for a declared charge.
    ///
    /// # A handle is told from a repletion hydrogen only by the port
    ///
    /// Nothing marks a handle on the atom itself; the `ports` relation is the
    /// whole record. Do not round-trip a template through
    /// [`remove_hydrogens`](crate::perceive::remove_hydrogens):
    /// it decides by a kind-blind neighbour count, so a handle (one `bonds`
    /// relation plus one `ports` relation to the same anchor) counts as
    /// degree 2 and is kept while every repletion hydrogen is stripped — a
    /// port relation is being counted as a bond, which is a routed defect,
    /// not a contract. Adding hydrogens is safe: repletion skips atoms that
    /// are already hydrogen and never touches a handle.
    ///
    /// # A template is instance-free
    ///
    /// No atom carries `frag_id` (R5.1: instance membership belongs to an
    /// instance, and `to_atomistic` is what stamps it) and no atom carries
    /// `x` / `y` / `z` — a line notation states no geometry. No perception, no
    /// kekulization and no inter-fragment bond is formed here.
    ///
    /// # Port indices
    ///
    /// Ports are added in the descriptor order
    /// [`FragmentIr::to_atomistic`](crate::io::smiles::FragmentIr::to_atomistic)
    /// returns, which is its walk order — the same index
    /// [`PairEnd::Body::port`](crate::io::cgsmiles::PairEnd::Body::port) uses
    /// against the same body. The *n*-th descriptor of a definition is
    /// therefore the *n*-th port added for it.
    /// [`port_ids`](molrs::core::Atomistic::port_ids) promises no iteration order, so read a port
    /// back by its [`Port`](molrs::core::Port) rather than by position.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`] when there is no atomistic body to
    /// build from: a base-only string, spanned at the whole string and carrying
    /// the payload `base-only string (no fragment table)`, or a definition
    /// whose body is a coarse graph, spanned at that definition and carrying
    /// its name. An empty map is never returned in place of either — it would
    /// be indistinguishable from a table that was dropped.
    ///
    /// [`SmilesErrorKind::InvalidDescriptorOrder`] when a descriptor writes a
    /// bond symbol that states no multiplicity (`:` aromatic, the directional
    /// `/` and `\`), naming the kind that was written. The parser already
    /// refuses these, so only a hand-built [`BondingDescriptor`] reaches it.
    ///
    /// [`SmilesErrorKind::CgBuild`] when a structural write fails — bonding a
    /// handle, promoting the graph, or recording a port — naming the fragment
    /// and the underlying error. No fallible call on this path is discarded.
    ///
    /// Whatever converting a body raises propagates **unchanged**, with the
    /// converter's own kind: re-wrapping it would hide which rule the body
    /// broke.
    ///
    /// # Examples
    ///
    /// The PEO template of an OH-capped trimer, and the repletion its handles
    /// leave room for.
    ///
    /// ```
    /// use molrs::io::cgsmiles::CgSmilesIr;
    /// use molrs::perceive::add_hydrogens;
    /// use molrs::core::{Atomistic, NeighborKinds};
    ///
    /// let ir = CgSmilesIr::parse("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")?;
    /// let mut templates = ir.templates()?;
    /// let peo = templates.remove("PEO").expect("the table defines #PEO");
    ///
    /// // `[$]COC[$]`: three heavy atoms and two bonds, plus one capping
    /// // hydrogen and one bond per descriptor.
    /// assert_eq!(peo.n_atoms(), 5);
    /// assert_eq!(peo.n_bonds(), 4);
    /// assert_eq!(peo.n_ports(), 2);
    ///
    /// // The handles are real bonds, so repletion completes each carbon to
    /// // four: C0 has O + handle and gains 2 H, C2 likewise, the ether oxygen
    /// // is already satisfied — 5 + 4 = 9 atoms.
    /// let rebuilt = add_hydrogens(&peo)?;
    /// assert_eq!(rebuilt.n_atoms(), 9);
    ///
    /// // The ports survive repletion, and every handle is still terminal.
    /// assert_eq!(rebuilt.n_ports(), 2);
    /// for id in rebuilt.port_ids() {
    ///     let port = rebuilt.get_port(id)?;
    ///     assert_eq!(rebuilt.neighbors(port.handle, NeighborKinds::BONDS).count(), 1);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn templates(&self) -> Result<BTreeMap<String, Atomistic>, SmilesError> {
        let Some(table) = self.fragments.last() else {
            return Err(SmilesError::new(
                SmilesErrorKind::CgNotExpandable("base-only string (no fragment table)".to_owned()),
                self.span,
                "",
                Notation::CgSmiles,
            ));
        };

        let mut templates = BTreeMap::new();
        for (name, def) in table {
            // Positional dispatch (01c) makes the last table's bodies
            // atomistic, so the coarse arm is unreachable through the reader
            // and exists for totality.
            let FragmentBody::Smiles(ir) = &def.body else {
                return Err(SmilesError::new(
                    SmilesErrorKind::CgNotExpandable(name.clone()),
                    def.span,
                    "",
                    Notation::CgSmiles,
                ));
            };
            let context = format!("fragment '{name}'");
            templates.insert(name.clone(), build_template(ir, &context, def.span)?);
        }
        Ok(templates)
    }
}

impl FragmentIr {
    /// Build the ported [`Atomistic`] template of one fragment body.
    ///
    /// The single-unit form of [`CgSmilesIr::templates`]: `self` is a body as
    /// [`SmilesIr::from_fragment`](crate::io::smiles::SmilesIr::from_fragment)
    /// reads it (`[<]OCC[>]`), and the result is exactly the template a table
    /// entry with that body builds — heavy atoms, one capping-hydrogen handle
    /// and one port per bonding descriptor, no `frag_id`, no coordinates, no
    /// valence-bearing property written. See [`CgSmilesIr::templates`] for
    /// what a handle is and why nothing else is written.
    ///
    /// A body without descriptors (`CCO`) builds a template with no ports.
    ///
    /// # Errors
    ///
    /// Whatever converting the body raises, unchanged;
    /// [`SmilesErrorKind::InvalidDescriptorOrder`] for a hand-built descriptor
    /// whose bond symbol states no multiplicity; [`SmilesErrorKind::CgBuild`],
    /// spanned at the whole body, when a handle, its bond or a port cannot be
    /// written.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::smiles::SmilesIr;
    ///
    /// let eo = SmilesIr::from_fragment("[<]OCC[>]")?.to_template()?;
    /// // Three heavy atoms plus one handle per descriptor.
    /// assert_eq!(eo.n_atoms(), 5);
    /// assert_eq!(eo.n_ports(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_template(&self) -> Result<Atomistic, SmilesError> {
        build_template(self, "fragment", self.ir().span)
    }
}

/// The one template conversion: walk the body once, then cap every
/// descriptor it wrote. `context` and `span` locate a build failure.
fn build_template(ir: &FragmentIr, context: &str, span: Span) -> Result<Atomistic, SmilesError> {
    let (atomistic, descriptors) = ir.to_atomistic()?;
    let sites: Vec<OpenSite<'_>> = descriptors
        .iter()
        .map(|(anchor, descriptor)| OpenSite {
            anchor: *anchor,
            descriptor,
        })
        .collect();
    cap_open_sites(atomistic, &sites, context, span)
}

/// One open valence to cap: the anchor atom and the descriptor written on it.
struct OpenSite<'a> {
    /// The atom the descriptor was written on.
    anchor: NodeId,
    /// The descriptor itself: kind, label and written bond order.
    descriptor: &'a BondingDescriptor,
}

/// Cap every open site of `atomistic` with a hydrogen **handle** and record
/// one port per site, in `sites` order.
///
/// Each handle is a real hydrogen carrying its atomic number and the
/// [`Element`] table's H mass, bonded to the anchor by a single bond; handles are
/// appended in `sites` order, so they follow every atom already present. Each
/// port records the descriptor's kind (R4.3), label (R4.4) and order (R4.5).
/// Handles and their bonds are written first: `Atomistic::add_bond` stamps
/// both bond facts in one call, and `add_port` re-checks that the handle is
/// bonded to its anchor (it accepts a handle of any element), so the bond
/// must exist by then. The graph is edited in place, never copied atom by
/// atom — a copy would drop `h_count`, `formal_charge`, `isotope`,
/// `is_aromatic` and the stereo the SMILES builder declared.
///
/// # Errors
///
/// [`SmilesErrorKind::InvalidDescriptorOrder`] (see [`port_order`]) before
/// anything is written; otherwise [`SmilesErrorKind::CgBuild`], spanned at
/// `span` and prefixed by `context`, when a handle, its bond or a port cannot
/// be written.
fn cap_open_sites(
    mut atomistic: Atomistic,
    sites: &[OpenSite<'_>],
    context: &str,
    span: Span,
) -> Result<Atomistic, SmilesError> {
    let orders = sites
        .iter()
        .map(|site| port_order(site.descriptor, span))
        .collect::<Result<Vec<BondNumber>, SmilesError>>()?;
    let build = |e: MolRsError| cg_build(span, format!("{context}: {e}"));
    let h_mass = f64::from(Element::H.atomic_mass());
    let mut handles = Vec::with_capacity(sites.len());
    for site in sites {
        let handle = atomistic
            .add_atom(crate::props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::MASS => h_mass,
            })
            .map_err(build)?;
        atomistic
            .add_bond(site.anchor, handle, crate::props! {})
            .map_err(build)?;
        handles.push(handle);
    }
    for ((site, handle), order) in sites.iter().zip(handles).zip(orders) {
        let desc = site.descriptor;
        atomistic
            .add_port(
                site.anchor,
                handle,
                port_kind(desc.kind),
                &desc.label,
                order,
            )
            .map_err(build)?;
    }
    Ok(atomistic)
}

/// The stored port role a written descriptor operator denotes.
///
/// R4.3: the vocabulary is closed and the mapping is 1:1, so the match is
/// exhaustive without a catch-all arm — a fifth operator would have to be
/// handled here rather than silently falling through.
/// [`DescriptorKind::Shared`] (`!`) cannot arrive through
/// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse), which refuses the
/// squash operator while reading; its arm is totality, not support.
fn port_kind(kind: DescriptorKind) -> PortKind {
    match kind {
        DescriptorKind::Symmetric => PortKind::Symmetric,
        DescriptorKind::Left => PortKind::Left,
        DescriptorKind::Right => PortKind::Right,
        DescriptorKind::Shared => PortKind::Shared,
    }
}

/// The multiplicity a descriptor states for the bond its pairing will form.
///
/// R4.5: a descriptor with no bond symbol beside it means a single bond, and a
/// written symbol means what it says.
///
/// # Errors
///
/// Returns [`SmilesErrorKind::InvalidDescriptorOrder`], spanned at `span` and
/// naming the written kind, for a symbol whose [`BondNumber`] is `Unknown` —
/// today only the aromatic `:`; the directional `/` and `\` and the query
/// kinds map to `Single`, as Daylight reads them. `Ok(BondNumber::Unknown)`
/// is not reachable:
/// an indefinite order must never reach a port, where it would later read as a
/// valence of zero. The parser validates every descriptor it builds, but
/// [`BondingDescriptor`] has public fields and the body converter does not
/// re-validate, so a hand-built value reaches this guard.
fn port_order(desc: &BondingDescriptor, span: Span) -> Result<BondNumber, SmilesError> {
    let Some(written) = desc.order else {
        return Ok(BondNumber::Single);
    };
    let order = written.bond_number();
    if order == BondNumber::Unknown {
        return Err(SmilesError::new(
            SmilesErrorKind::InvalidDescriptorOrder(written),
            span,
            "",
            Notation::CgSmiles,
        ));
    }
    Ok(order)
}

/// A reader invariant violated while building from an IR — expanding the
/// lowest level, or building a template — spanned at `span`: the whole string
/// for the expansion, the definition for a template. The one constructor of a
/// [`SmilesErrorKind::CgBuild`] on those paths.
///
/// The input text is not carried: 01b froze the IR without its source string,
/// so by the time it is built from, the string is the caller's. The notation
/// is stamped [`Notation::CgSmiles`] so the refusal can never render as a
/// SMILES one.
pub(super) fn cg_build(span: Span, reason: String) -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::CgBuild(reason),
        span,
        "",
        Notation::CgSmiles,
    )
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{port_kind, port_order};
    use crate::io::cgsmiles::fixtures::fragment;
    use crate::io::cgsmiles::{CgFragmentDef, CgGraph, CgNode, CgSmilesIr, FragmentBody};
    use crate::io::smiles::{
        BondKind, BondingDescriptor, DescriptorKind, FragmentIr, Notation, SmilesErrorKind, Span,
    };
    use crate::line_notation::atom_keys::{H_COUNT, IS_AROMATIC, ISOTOPE};
    use crate::line_notation::parser::parse_fragment_smiles;
    use molrs::core::Atomistic;
    use molrs::core::Element;
    use molrs::core::NeighborKinds;
    use molrs::core::NodeId;
    use molrs::core::PropValue;
    use molrs::core::keys;
    use molrs::core::{BondNumber, BondOrder};
    use molrs::core::{Port, PortKind};

    // Every count below is hand-derived from the fixtures of § Domain basis /
    // § Testing strategy of `.claude/specs/cgsmiles-02b-to-fragment.md`: heavy
    // atoms are counted off the written body, one capping hydrogen (a handle)
    // is added per bonding descriptor (R4.17), and one port is recorded per
    // descriptor carrying its kind (R4.3), its label (R4.4) and its bond order
    // (R4.5). A template is instance-free (R5.1): no `frag_id`, no
    // coordinates, no repletion hydrogen. No external program produced any
    // value here.

    /// F2 — an OH-capped PEO trimer, the string 01d's expansion test uses.
    const F2: &str = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}";

    /// F5's backbone bead, referenced by a one-node level so the table is the
    /// last one: `>` on C0, `<` and `$` on C1.
    const F5_BB: &str = "{[#BB]}.{#BB=[>]CC[<][$]}";

    /// F6's glycine residue, referenced by a one-node level.
    const F6_GLY: &str = "{[#GLY]}.{#GLY=[>]NCC(=O)[<]}";

    /// F7's dichlorotoluene bead: two labelled descriptors on an aromatic
    /// body.
    const F7_SX3: &str = "{[#SX3]}.{#SX3=Clc[$a]c[$b]}";

    /// A span no fixture shares with the IR's own, so "spanned at the
    /// definition" is a statement a test can fail.
    const DEF_SPAN: Span = Span { start: 13, end: 27 };

    // -- helpers ------------------------------------------------------------

    /// The whole template table of a `CGsmiles` string that must convert.
    fn templates(text: &str) -> BTreeMap<String, Atomistic> {
        CgSmilesIr::parse(text)
            .unwrap_or_else(|e| panic!("CgSmilesIr::parse({text:?}) failed: {e}"))
            .templates()
            .unwrap_or_else(|e| panic!("templates({text:?}) failed: {e}"))
    }

    /// One named template of a string that must convert.
    fn template(text: &str, name: &str) -> Atomistic {
        templates(text)
            .remove(name)
            .unwrap_or_else(|| panic!("{text:?} builds no template named {name:?}"))
    }

    /// Every port of a template, read back through `Atomistic::get_port`.
    ///
    /// `Atomistic::port_ids()` promises no iteration order (02a), so every
    /// assertion below is over the port *set*, never over an index.
    fn ports_of(frag: &Atomistic) -> Vec<Port> {
        frag.port_ids()
            .map(|id| {
                frag.get_port(id)
                    .unwrap_or_else(|e| panic!("port {id:?} does not read back: {e}"))
            })
            .collect()
    }

    /// The element of one atom of a template, read from its atomic number.
    fn element(frag: &Atomistic, atom: NodeId) -> Element {
        let z = frag
            .get_atom(atom)
            .unwrap_or_else(|e| panic!("atom {atom:?} is missing: {e}"))
            .atomic_number()
            .unwrap_or_else(|| panic!("atom {atom:?} carries no atomic number"));
        Element::by_number(z).unwrap_or_else(|| panic!("atom {atom:?} is the wildcard"))
    }

    /// The atoms **bonded** to `atom`: its bonds alone, never a port (an
    /// arity-2 relation too, which a generic adjacency would report as a
    /// second neighbour).
    fn bonded(frag: &Atomistic, atom: NodeId) -> Vec<NodeId> {
        frag.neighbors(atom, NeighborKinds::BONDS)
            .map(|(other, _)| other)
            .collect()
    }

    /// The `(BondOrder, BondNumber)` of the bond joining a port's anchor to its
    /// handle.
    fn handle_bond_class(frag: &Atomistic, port: &Port) -> (BondOrder, BondNumber) {
        let rid = frag
            .bond_between(port.handle, port.anchor)
            .unwrap_or_else(|| {
                panic!(
                    "no bond joins handle {:?} to anchor {:?}",
                    port.handle, port.anchor
                )
            });
        let bond = frag
            .get_bond(rid)
            .unwrap_or_else(|e| panic!("bond {rid:?} does not read back: {e}"));
        (
            BondOrder::from_prop(bond.props.get(keys::BOND_TYPE)),
            BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER)),
        )
    }

    /// An unlabelled symmetric descriptor carrying `order`.
    fn descriptor(order: Option<BondKind>) -> BondingDescriptor {
        BondingDescriptor {
            kind: DescriptorKind::Symmetric,
            label: String::new(),
            order,
        }
    }

    /// A coarse node named `name`: a template is built from a *definition*, so
    /// nothing else on a node is read.
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

    /// One fragment definition named `name`, spanned at [`DEF_SPAN`].
    fn def(name: &str, body: FragmentBody) -> CgFragmentDef {
        CgFragmentDef {
            name: name.to_owned(),
            body,
            span: DEF_SPAN,
        }
    }

    /// A hand-built IR whose single (and therefore last) fragment table holds
    /// `defs`. `templates` reads `fragments.last()` and `span`, so the level
    /// and the pair list are present only to make the value well-formed.
    fn ir_over(defs: &[CgFragmentDef]) -> CgSmilesIr {
        CgSmilesIr {
            levels: vec![CgGraph {
                nodes: vec![node(&defs[0].name)],
                edges: Vec::new(),
            }],
            fragments: vec![
                defs.iter()
                    .map(|d| (d.name.clone(), d.clone()))
                    .collect::<BTreeMap<String, CgFragmentDef>>(),
            ],
            pairs: vec![Vec::new()],
            span: Span::new(0, 0),
        }
    }

    // -- port_kind: the closed vocabulary, 1:1 (R4.3) -----------------------

    /// The four written operators and the four stored roles are the same four
    /// things; the mapping is total and has no catch-all arm.
    #[test]
    fn map_each_descriptor_kind_to_its_port_kind() {
        assert_eq!(port_kind(DescriptorKind::Symmetric), PortKind::Symmetric);
        assert_eq!(port_kind(DescriptorKind::Left), PortKind::Left);
        assert_eq!(port_kind(DescriptorKind::Right), PortKind::Right);
        assert_eq!(port_kind(DescriptorKind::Shared), PortKind::Shared);
    }

    // -- port_order: the descriptor's bond order (R4.5) ---------------------

    /// A descriptor with no bond symbol beside it forms a single bond — the
    /// same default `BondKind::Single` the pairing step uses.
    #[test]
    fn defaults_to_single_when_no_order_written() {
        let order = port_order(&descriptor(None), DEF_SPAN)
            .expect("an unannotated descriptor has a definite order");
        assert_eq!(order, BondNumber::Single);
    }

    /// `=`, `#` and `$` written beside the bracket are the multiplicities 2, 3
    /// and 4, and the port stores them as written.
    #[test]
    fn reads_the_written_bond_number() {
        for (written, stored) in [
            (BondKind::Double, BondNumber::Double),
            (BondKind::Triple, BondNumber::Triple),
            (BondKind::Quadruple, BondNumber::Quadruple),
        ] {
            let order = port_order(&descriptor(Some(written)), DEF_SPAN)
                .unwrap_or_else(|e| panic!("{written:?} is a definite order: {e}"));
            assert_eq!(order, stored, "{written:?}");
        }
    }

    /// An aromatic descriptor order states delocalization, not a
    /// multiplicity, so it has no port order at all: the guard names the
    /// written kind rather than handing `BondNumber::Unknown` on.
    #[test]
    fn rejects_an_aromatic_descriptor_order() {
        let err = port_order(&descriptor(Some(BondKind::Aromatic)), DEF_SPAN)
            .expect_err("an aromatic order is never Ok(Unknown)");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.span, DEF_SPAN);
        assert_eq!(err.notation, Notation::CgSmiles);
    }

    // -- templates: the table it returns ----------------------------------

    /// One template per definition of the last table, no more and no fewer —
    /// the key set is the table's own.
    #[test]
    fn returns_one_template_per_definition() {
        let ir = CgSmilesIr::parse(F2).expect("F2 must parse");
        let defined: Vec<String> = ir
            .fragments
            .last()
            .expect("F2 writes a fragment table")
            .keys()
            .cloned()
            .collect();
        let built: Vec<String> = ir
            .templates()
            .expect("F2 must build its templates")
            .keys()
            .cloned()
            .collect();
        assert_eq!(built, vec!["OH".to_owned(), "PEO".to_owned()]);
        assert_eq!(built, defined);
    }

    // -- FragmentIr::to_template: one body, no table ---------------------

    /// `[<]OCC[>]` alone: three heavy atoms, two handles, one `<` and one `>`
    /// port — the same template the table entry `#EO=[<]OCC[>]` builds.
    #[test]
    fn to_template_builds_one_unit_from_a_body() {
        let eo = fragment("[<]OCC[>]")
            .to_template()
            .expect("a fragment body must build its template");
        assert_eq!(eo.n_atoms(), 5);
        assert_eq!(eo.n_bonds(), 4);
        let mut kinds: Vec<PortKind> = ports_of(&eo).iter().map(|p| p.kind).collect();
        kinds.sort_by_key(|k| format!("{k:?}"));
        assert_eq!(kinds, vec![PortKind::Left, PortKind::Right]);

        let from_table = template("{[#EO]}.{#EO=[<]OCC[>]}", "EO");
        assert_eq!(eo.n_atoms(), from_table.n_atoms());
        assert_eq!(eo.n_ports(), from_table.n_ports());
    }

    /// A body with no descriptor builds a template with no port.
    #[test]
    fn to_template_of_a_plain_body_has_no_ports() {
        let mol = fragment("CCO").to_template().expect("a plain body builds");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(mol.n_ports(), 0);
    }

    // -- templates: template shape ----------------------------------------

    /// `[$]COC[$]` is three heavy atoms and two bonds; each descriptor adds
    /// one capping hydrogen and one bond — 5 atoms, 4 bonds, 2 ports. Both
    /// anchors are the ether's carbons, both descriptors are bare `$`.
    #[test]
    fn builds_the_peo_template() {
        let peo = template(F2, "PEO");
        assert_eq!(peo.n_atoms(), 5);
        assert_eq!(peo.n_bonds(), 4);
        assert_eq!(peo.n_ports(), 2);

        let ports = ports_of(&peo);
        let oxygens: Vec<NodeId> = peo
            .atom_ids()
            .filter(|&atom| element(&peo, atom) == Element::O)
            .collect();
        assert_eq!(oxygens.len(), 1, "the body writes one ether oxygen");
        for port in &ports {
            assert_eq!(port.kind, PortKind::Symmetric);
            assert_eq!(port.label, "", "a bare `[$]` is unnamed");
            assert_eq!(port.order, BondNumber::Single);
            assert_eq!(
                element(&peo, port.handle),
                Element::H,
                "a handle is a hydrogen"
            );
            assert_eq!(element(&peo, port.anchor), Element::C);
            assert!(
                bonded(&peo, port.anchor).contains(&oxygens[0]),
                "an anchor of `[$]COC[$]` is a carbon bonded to the oxygen"
            );
        }
        assert_ne!(
            ports[0].anchor, ports[1].anchor,
            "the two descriptors sit on the two different carbons"
        );
    }

    /// `[$]O` is one heavy atom and no bond; its one descriptor adds the
    /// hydrogen and the bond that carries the port.
    #[test]
    fn builds_the_hydroxyl_template() {
        let oh = template(F2, "OH");
        assert_eq!(oh.n_atoms(), 2);
        assert_eq!(oh.n_bonds(), 1);
        assert_eq!(oh.n_ports(), 1);

        let ports = ports_of(&oh);
        assert_eq!(element(&oh, ports[0].anchor), Element::O);
        assert_eq!(element(&oh, ports[0].handle), Element::H);
        assert_eq!(ports[0].kind, PortKind::Symmetric);
        assert_eq!(ports[0].order, BondNumber::Single);
    }

    /// A multivalent anchor gets one handle per descriptor, never one shared
    /// handle: `[>]CC[<][$]` is 2 heavy atoms + 3 handles = 5 atoms and 4
    /// bonds, with C1 carrying two of the three ports.
    #[test]
    fn puts_two_ports_on_one_anchor() {
        let bb = template(F5_BB, "BB");
        assert_eq!(bb.n_atoms(), 5);
        assert_eq!(bb.n_bonds(), 4);
        assert_eq!(bb.n_ports(), 3);

        let ports = ports_of(&bb);
        let count_on = |anchor: NodeId| ports.iter().filter(|p| p.anchor == anchor).count();
        let lone: Vec<&Port> = ports.iter().filter(|p| count_on(p.anchor) == 1).collect();
        let shared: Vec<&Port> = ports.iter().filter(|p| count_on(p.anchor) == 2).collect();
        assert_eq!(lone.len(), 1, "C0 carries the one `>`");
        assert_eq!(shared.len(), 2, "C1 carries both `<` and `$`");

        assert_eq!(lone[0].kind, PortKind::Right);
        assert_eq!(element(&bb, lone[0].anchor), Element::C);
        assert_eq!(element(&bb, shared[0].anchor), Element::C);
        assert!(
            bonded(&bb, lone[0].anchor).contains(&shared[0].anchor),
            "the two anchors are the body's two bonded carbons"
        );
        let kinds: Vec<PortKind> = shared.iter().map(|p| p.kind).collect();
        assert!(
            kinds.contains(&PortKind::Left) && kinds.contains(&PortKind::Symmetric),
            "C1's two ports are `<` and `$`, got {kinds:?}"
        );
        assert_ne!(
            shared[0].handle, shared[1].handle,
            "two descriptors on one anchor are two distinct capping hydrogens"
        );
    }

    /// `[>]NCC(=O)[<]` is four heavy atoms and three bonds; two descriptors
    /// add two handles and two bonds — 6 atoms, 5 bonds. The `<` written after
    /// the branch anchors on the carbonyl carbon, because a branch does not
    /// advance the chain's current atom.
    #[test]
    fn anchors_glycine_ports() {
        let gly = template(F6_GLY, "GLY");
        assert_eq!(gly.n_atoms(), 6);
        assert_eq!(gly.n_bonds(), 5);
        assert_eq!(gly.n_ports(), 2);

        let ports = ports_of(&gly);
        let right = ports
            .iter()
            .find(|p| p.kind == PortKind::Right)
            .expect("the `>` written before N");
        let left = ports
            .iter()
            .find(|p| p.kind == PortKind::Left)
            .expect("the `<` written after the carbonyl branch");
        assert_eq!(element(&gly, right.anchor), Element::N);
        assert_eq!(element(&gly, left.anchor), Element::C);
        assert!(
            bonded(&gly, left.anchor)
                .iter()
                .any(|&other| element(&gly, other) == Element::O),
            "the `<` anchor is the carbonyl carbon, not the methylene"
        );
    }

    /// `$a` and `$b` are two descriptor classes of one kind, so the label the
    /// notation wrote reaches the port verbatim.
    #[test]
    fn keeps_descriptor_labels() {
        let sx3 = template(F7_SX3, "SX3");
        let mut labels: Vec<String> = ports_of(&sx3).iter().map(|p| p.label.clone()).collect();
        labels.sort();
        assert_eq!(labels, vec!["a".to_owned(), "b".to_owned()]);
    }

    /// A fragment is a graph plus attachment points whether or not anything
    /// uses it, so a table that defines more than the level above it names
    /// still yields a template for each definition.
    #[test]
    fn keeps_a_definition_the_level_above_never_references() {
        let built = templates("{[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}");
        let names: Vec<String> = built.keys().cloned().collect();
        assert_eq!(names, vec!["OH".to_owned(), "PEO".to_owned()]);
        let peo = built.get("PEO").expect("the unreferenced definition");
        assert_eq!(peo.n_atoms(), 5);
        assert_eq!(peo.n_ports(), 2);
    }

    // -- templates: property discipline -----------------------------------

    /// A handle is added with its atomic number and the hydrogen mass, and nothing
    /// else: no coordinates, no bead stamps, and none of the
    /// valence-bearing properties whose presence would change what a later
    /// `add_hydrogens` computes. Mass is not valence-bearing
    /// (`.claude/specs/assembly-03-template.md` § Design 1, reworked from the
    /// earlier "no mass literal" contract).
    #[test]
    fn writes_only_element_and_mass_on_a_handle() {
        let peo = template(F2, "PEO");
        let h_mass = f64::from(Element::H.atomic_mass());
        for port in ports_of(&peo) {
            let handle = peo
                .get_atom(port.handle)
                .unwrap_or_else(|e| panic!("handle {:?} is missing: {e}", port.handle));
            assert_eq!(handle.atomic_number(), Some(Element::H.z()));
            let mass = handle
                .get_f64(keys::MASS)
                .unwrap_or_else(|| panic!("handle {:?} carries no mass", port.handle));
            assert!(
                (mass - h_mass).abs() < 1e-12,
                "handle {:?}: mass {mass} != H mass {h_mass}",
                port.handle
            );
            for key in [
                keys::X,
                keys::Y,
                keys::Z,
                H_COUNT,
                IS_AROMATIC,
                keys::FORMAL_CHARGE,
                keys::FRAG_ID,
            ] {
                assert!(
                    !handle.contains_key(key),
                    "handle {:?} carries '{key}'",
                    port.handle
                );
            }
            let mut written: Vec<&str> = handle.keys().collect();
            written.sort_unstable();
            let mut expected = vec![keys::ATOMIC_NUMBER, keys::MASS];
            expected.sort_unstable();
            assert_eq!(
                written, expected,
                "a handle carries the atomic number and mass and nothing else"
            );
        }
    }

    /// Every capping hydrogen of every template carries the `Element` table's
    /// H mass (the source of truth), so a mass-weighted bead centroid exists.
    #[test]
    fn writes_the_hydrogen_mass_on_every_handle() {
        let h_mass = f64::from(Element::H.atomic_mass());
        for text in [F2, F5_BB, F6_GLY] {
            for (name, frag) in templates(text) {
                for port in ports_of(&frag) {
                    let handle = frag
                        .get_atom(port.handle)
                        .unwrap_or_else(|e| panic!("{name}: handle is missing: {e}"));
                    let mass = handle
                        .get_f64(keys::MASS)
                        .unwrap_or_else(|| panic!("{name}: handle {:?} has no mass", port.handle));
                    assert!(
                        (mass - h_mass).abs() < 1e-12,
                        "{name}: handle mass {mass} != H mass {h_mass}"
                    );
                }
            }
        }
    }

    /// An organic-subset anchor states no hydrogen count, and adding a handle
    /// must not invent one: the handle is a real incident bond, so a later
    /// repletion step subtracts it and completes C0 to exactly four.
    #[test]
    fn leaves_an_organic_anchor_untouched() {
        let peo = template(F2, "PEO");
        for port in ports_of(&peo) {
            let anchor = peo
                .get_atom(port.anchor)
                .unwrap_or_else(|e| panic!("anchor {:?} is missing: {e}", port.anchor));
            assert_eq!(anchor.atomic_number(), Some(Element::C.z()));
            assert!(
                !anchor.contains_key(H_COUNT),
                "an organic anchor declares no hydrogen count"
            );
            assert!(
                !anchor.contains_key(keys::FORMAL_CHARGE),
                "an organic anchor declares no formal charge"
            );
        }
    }

    /// Promotion re-wraps the graph the SMILES builder wrote; it does not copy
    /// atoms. A bracket atom therefore keeps every property the builder
    /// declared, and a body written aromatic keeps its aromatic stamp.
    #[test]
    fn preserves_bracket_atom_properties() {
        let methyl = template("{[#M]}.{#M=[$][13CH3-]}", "M");
        assert_eq!(methyl.n_atoms(), 2);
        assert_eq!(methyl.n_ports(), 1);
        let ports = ports_of(&methyl);
        let anchor = methyl
            .get_atom(ports[0].anchor)
            .unwrap_or_else(|e| panic!("anchor is missing: {e}"));
        assert_eq!(anchor.atomic_number(), Some(Element::C.z()));
        assert_eq!(anchor.get(ISOTOPE), Some(&PropValue::Int(13)));
        assert_eq!(anchor.get(H_COUNT), Some(&PropValue::Int(3)));
        assert_eq!(anchor.get(keys::FORMAL_CHARGE), Some(&PropValue::Int(-1)));

        let sx3 = template(F7_SX3, "SX3");
        for port in ports_of(&sx3) {
            let anchor = sx3
                .get_atom(port.anchor)
                .unwrap_or_else(|e| panic!("anchor is missing: {e}"));
            assert_eq!(
                anchor.get(IS_AROMATIC),
                Some(&PropValue::Bool(true)),
                "an anchor written `c` stays aromatic through promotion"
            );
        }
    }

    /// A handle bond is a fully stated single bond: both the class and the
    /// localized number, never a silent `Unknown` that a valence count would
    /// read as zero.
    #[test]
    fn sets_single_class_on_every_handle_bond() {
        let peo = template(F2, "PEO");
        for port in ports_of(&peo) {
            assert_eq!(
                handle_bond_class(&peo, &port),
                (BondOrder::Single, BondNumber::Single),
                "handle bond of port {port:?}"
            );
        }
    }

    /// A template is instance-free and geometry-free: a line notation states
    /// no coordinates, and membership belongs to an instance that a template
    /// does not have.
    #[test]
    fn adds_no_coordinates_and_no_frag_id() {
        for (name, frag) in templates(F2) {
            for atom in frag.atom_ids() {
                let props = frag
                    .get_atom(atom)
                    .unwrap_or_else(|e| panic!("{name}: atom {atom:?} is missing: {e}"));
                for key in [keys::X, keys::Y, keys::Z] {
                    assert!(
                        !props.contains_key(key),
                        "{name}: atom {atom:?} carries '{key}'"
                    );
                }
                assert!(
                    !props.contains_key(keys::FRAG_ID),
                    "{name}: atom {atom:?} is stamped with an instance"
                );
            }
        }
    }

    // -- templates: refusals ----------------------------------------------

    /// A string that never says what its beads are made of has no template to
    /// give, and an empty map would be indistinguishable from a dropped table.
    #[test]
    fn rejects_a_base_only_ir() {
        let ir = CgSmilesIr::parse("{[#EO]|5}").expect("a base-only string must parse");
        let err = ir
            .templates()
            .expect_err("a base-only string defines no fragment body");
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgNotExpandable(ref payload)
                    if payload == "base-only string (no fragment table)"
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.span, ir.span);
        assert_eq!(
            err.notation,
            Notation::CgSmiles,
            "a CGsmiles refusal never renders as a SMILES one"
        );
    }

    /// The last table's bodies are atomistic by position, so a coarse body
    /// there has no atoms to build; the refusal names the definition it
    /// stopped at.
    #[test]
    fn rejects_a_graph_body_in_the_last_table() {
        let ir = ir_over(&[def(
            "A",
            FragmentBody::Graph(CgGraph {
                nodes: vec![node("X")],
                edges: Vec::new(),
            }),
        )]);
        let err = ir
            .templates()
            .expect_err("a coarse body builds no atomistic template");
        assert!(
            matches!(err.kind, SmilesErrorKind::CgNotExpandable(ref name) if name == "A"),
            "kind was {:?}",
            err.kind
        );
    }

    /// `BondingDescriptor` has public fields and the body converter does not
    /// re-validate, so a hand-built aromatic order reaches this method; it is
    /// named here rather than stored as an indefinite port order.
    #[test]
    fn rejects_an_invalid_descriptor_order() {
        let mut body = parse_fragment_smiles("[$]O").expect("the body must parse");
        body.components[0].head.descriptors[0] = descriptor(Some(BondKind::Aromatic));
        let ir = ir_over(&[def("A", FragmentBody::Smiles(FragmentIr::new(body)))]);
        let err = ir
            .templates()
            .expect_err("an aromatic descriptor order is not a port order");
        assert!(
            matches!(err.kind, SmilesErrorKind::InvalidDescriptorOrder(_)),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(
            err.span, DEF_SPAN,
            "the refusal is spanned at the definition that carries the descriptor"
        );
    }
}

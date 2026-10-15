//! Typing: a molecular graph in, the force-field types it carries (and the
//! per-instance parameters some force fields resolve) out. Charges are
//! [`crate::ff::charge`]'s; kernels are [`crate::ff::potential`]'s.
//!
//! Typing is a template method. A [`Typifier`] implements only
//! `assign` ([`Typifier`]): it reads a molecular graph and produces a
//! [`TypeAssignment`] — positional per-atom / per-link [`Annotation`]s, the styles they
//! are defined under, and the pair rows among the atom types used. The base,
//! [`Typing`], runs the match: [`TypeAssignment::apply_to`] stamps the annotations onto
//! a private copy of the input and defines the matched types in the output
//! force field the base owns. No implementor writes the output itself.
//!
//! Materializing a typed graph into a [`Frame`](molrs::core::Frame) for
//! `PotentialCompiler::compile` is the graph's `to_frame` job; typifiers stay on
//! the graph boundary.

pub(crate) mod atd;
pub mod cmap;
mod element;
pub(crate) mod estimate;
mod gaff;
pub mod mmff;
mod opls;
pub(crate) mod topology;
pub(crate) mod uff;

pub use atd::{AtdParameterSet, AtdTypifier};
pub use element::ElementTypifier;
pub use estimate::{
    BondedTerm, EmpiricalSet, Estimate, EstimateMethod, EstimationInputs, ParameterInterpolator,
    Parmchk2Estimator, PenaltyTier, Provenance,
};
pub use gaff::{GaffParameterSet, GaffTypifier};
pub use opls::{OplsAaTypifier, OplsTypeRow, OplsTypingMetadata};
pub use uff::UffTypifier;

use crate::ff::forcefield::Category;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;
use molrs::core::Atomistic;
use molrs::core::{MolRsError, NodeId, PropValue, Relation, RelationId};

use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::Params;
use crate::ff::ir::{Arity, EndpointOrder};
use estimate::candidate::is_wildcard;

/// The symbol an error message spells for atomic number `z`: the element's
/// symbol, or the wildcard `"*"` for `z = 0` (the spelling
/// [`fill_symbols`](molrs::core::Element::fill_symbols) writes). Only for
/// messages — never read back.
pub(crate) fn species_label(z: u8) -> &'static str {
    molrs::core::Element::by_number(z).map_or("*", molrs::core::Element::symbol)
}

/// A graph typifier: what matching a molecule against a force field produces.
///
/// An implementor provides [`assign`](Self::assign) and
/// [`source_forcefield`](Self::source_forcefield) and nothing else; it cannot
/// type a molecule by itself. [`Typing`] owns the implementor and the output
/// force field and is the only caller of `assign`. The Python hook has the
/// same name.
pub trait Typifier {
    /// Type `graph` and return what it assigns.
    ///
    /// `graph` is the base's private working copy of the input: an
    /// implementation may write intermediate results onto it (generated
    /// topology, perceived bond types), and those stay on the graph
    /// [`Typing::typify`] returns. Types and parameters are not stamped here;
    /// they travel in the [`TypeAssignment`], positional against this same graph.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String>;

    /// The force field this typifier matches against. [`Typing::new`] seeds
    /// its output from it with [`ForceField::empty_like`].
    fn source_forcefield(&self) -> &ForceField;
}

impl Typifier for Box<dyn Typifier + Send + Sync> {
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
        (**self).assign(graph)
    }

    fn source_forcefield(&self) -> &ForceField {
        (**self).source_forcefield()
    }
}

/// The value a [`TypeAssignment`] writes under one key of one graph element.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// Stamp `key = value`; defines nothing.
    Value(PropValue),
    /// Stamp `key = name` and every param under its own key (numeric as
    /// [`PropValue::F64`], string as [`PropValue::Str`]), and define the type
    /// `(category, style, name)` with `params`. The category follows where
    /// the annotation sits: `nodes` → `atom`, a [`TypeAssignment::links`] kind → the
    /// category whose Frame block that kind is ([`link_category`]: `bonds` →
    /// `bond`, `urey_bradleys` → `urey_bradley`). `endpoints` are the type's
    /// endpoint atom types, given to [`Style::def_type`] as they are (none
    /// for an atom type); `name` is never read for them.
    Type {
        style: String,
        name: String,
        endpoints: Vec<String>,
        params: Params,
    },
}

/// The annotations of one graph element, `key → annotation`, in write order.
pub type Annotations = Vec<(String, Annotation)>;

/// What a [`Typifier`] assigns to one graph.
///
/// `nodes` is positional against the graph's atoms in row order
/// (`graph.atoms()`). `links` maps a relation block of the graph — the Frame
/// block of its category: `bonds`, `angles`, `dihedrals`, `impropers`, or a
/// term block (`cmaps`, a custom `urey_bradleys`; see
/// [`Atomistic::term_blocks`]) — to rows positional against **that block's
/// own** rows (the order `graph.bond_ids()`, `graph.term_ids(block)` etc.
/// yield): an improper never shifts a dihedral position. Every relation
/// block, built-in or a term block, takes this one path; `ports` carry no
/// annotations. An absent block or an empty vector means "nothing for this
/// block"; an empty per-element list means "this element gets nothing".
///
/// `styles` are `(category, style, style params)` to declare, in output
/// order; `pairs` are `(style, name, endpoints, params)` pair rows.
#[derive(Debug, Clone, Default)]
pub struct TypeAssignment {
    pub nodes: Vec<Annotations>,
    pub links: IndexMap<String, Vec<Annotations>>,
    pub styles: Vec<(String, String, Params)>,
    pub pairs: Vec<(String, String, Vec<String>, Params)>,
}

/// The category whose Frame block is the relation kind `kind`, and how its
/// type rows' endpoints match a term's atoms.
///
/// A category the process-wide force-field IR registry declares with that
/// block (`bonds` → `bond`, a registered `urey_bradleys` →
/// `urey_bradley`); otherwise molrec's rule for a category outside its
/// table, whose block is its name followed by `s`, matched
/// [`EndpointOrder::Reversible`]. `None` for a kind not spelled that way and
/// for `atoms`, the block of the atom and pair categories, which name no
/// relation.
pub fn link_category(kind: &str) -> Option<(String, EndpointOrder)> {
    // `Some(_)` when registered categories own the block: the relation
    // among them, if any.
    let owned = crate::ff::style_registry::with_global_registry(|r| {
        let owners: Vec<_> = r.categories().filter(|c| c.block == kind).collect();
        (!owners.is_empty()).then(|| {
            owners
                .into_iter()
                .find(|c| matches!(c.arity, Arity::Exact(n) if n > 0))
                .map(|c| (c.name.clone().into_owned(), c.order))
        })
    });
    match owned {
        Some(relation) => relation,
        None => kind
            .strip_suffix('s')
            .filter(|c| !c.is_empty())
            .map(|c| (c.to_owned(), EndpointOrder::Reversible)),
    }
}

/// Whether the type-row endpoints `pattern` name a term over atoms of the
/// types `atoms` under `order`: slot by slot, a wildcard
/// ([`is_wildcard`]) matching any type. [`EndpointOrder::Reversible`] also
/// tries `pattern` reversed; [`EndpointOrder::Unordered`] any permutation.
fn endpoints_match(pattern: &[&str], atoms: &[&str], order: EndpointOrder) -> bool {
    fn slot(pattern: &str, atom: &str) -> bool {
        is_wildcard(pattern) || pattern == atom
    }
    /// Whether `pattern[i..]` can be assigned to the atoms `used` leaves.
    fn permuted(pattern: &[&str], atoms: &[&str], used: &mut [bool]) -> bool {
        let Some((first, rest)) = pattern.split_first() else {
            return true;
        };
        for j in 0..atoms.len() {
            if !used[j] && slot(first, atoms[j]) {
                used[j] = true;
                let found = permuted(rest, atoms, used);
                used[j] = false;
                if found {
                    return true;
                }
            }
        }
        false
    }
    if pattern.len() != atoms.len() {
        return false;
    }
    let forward = || pattern.iter().zip(atoms).all(|(p, a)| slot(p, a));
    match order {
        EndpointOrder::Ordered => forward(),
        EndpointOrder::Reversible => {
            forward()
                || pattern
                    .iter()
                    .zip(atoms.iter().rev())
                    .all(|(p, a)| slot(p, a))
        }
        EndpointOrder::Unordered => permuted(pattern, atoms, &mut vec![false; atoms.len()]),
    }
}

/// One graph element a [`TypeAssignment`] writes to.
#[derive(Debug, Clone)]
enum Element {
    Node(NodeId),
    Link(Rc<str>, RelationId),
}

/// The relation blocks with typed `Atomistic` methods that take annotations,
/// in the order an `Atomistic` registers them; term blocks follow.
const TYPED_LINK_BLOCKS: [&str; 4] = ["bonds", "angles", "dihedrals", "impropers"];

/// The relation blocks `graph` holds, in registration order: the four typed
/// ones, then its term blocks.
fn link_blocks(graph: &Atomistic) -> Vec<String> {
    TYPED_LINK_BLOCKS
        .iter()
        .copied()
        .chain(graph.term_blocks())
        .map(str::to_owned)
        .collect()
}

/// The rows of relation block `block` of `graph`, in row order; `None` when
/// the graph holds no such block.
///
/// # Errors
///
/// `ports`, which take no annotations.
fn link_rows(
    graph: &Atomistic,
    block: &str,
) -> Result<Option<Vec<(RelationId, Relation)>>, String> {
    Ok(Some(match block {
        "bonds" => graph.bonds().collect(),
        "angles" => graph.angles().collect(),
        "dihedrals" => graph.dihedrals().collect(),
        "impropers" => graph.impropers().collect(),
        "ports" => {
            return Err(
                "relation block 'ports' takes no annotations: a port is a valence \
                 descriptor, not a force-field term"
                    .to_owned(),
            );
        }
        term if graph.term_blocks().any(|b| b == term) => graph
            .term_ids(term)
            .and_then(|ids| ids.map(|id| Ok((id, graph.get_term(term, id)?))).collect())
            .map_err(|e: MolRsError| e.to_string())?,
        _ => return Ok(None),
    }))
}

/// Set `key = value` on row `id` of relation block `block`.
fn set_link(
    graph: &mut Atomistic,
    block: &str,
    id: RelationId,
    key: &str,
    value: PropValue,
) -> Result<(), MolRsError> {
    match block {
        "bonds" => graph.set_bond(id, key, value),
        "angles" => graph.set_angle(id, key, value),
        "dihedrals" => graph.set_dihedral(id, key, value),
        "impropers" => graph.set_improper(id, key, value),
        term => graph.set_term(term, id, key, value),
    }
}

/// One positional vector of a [`TypeAssignment`], resolved against the graph.
struct Vector {
    /// The category a type annotated here defines; `None` for a relation
    /// kind no category names ([`link_category`]).
    category: Option<String>,
    /// The graph kind, `atoms` for the nodes.
    kind: String,
    elements: Vec<Element>,
    rows: Vec<Annotations>,
}

/// The keys one element receives, each written once.
struct Stamp {
    element: Element,
    /// `"{category} {position}"`, for error messages.
    label: String,
    writes: Vec<(String, PropValue)>,
}

impl Stamp {
    /// Record `key = value`. The same value twice is one write; a different
    /// value under a key already written is an error.
    fn write(&mut self, key: String, value: PropValue) -> Result<(), String> {
        match self.writes.iter().find(|(k, _)| *k == key) {
            Some((_, old)) if *old == value => Ok(()),
            Some((_, old)) => Err(format!(
                "{}: key '{key}' written twice with different values ({old:?}, {value:?})",
                self.label
            )),
            None => {
                self.writes.push((key, value));
                Ok(())
            }
        }
    }

    fn apply(self, graph: &mut Atomistic) -> Result<(), String> {
        for (key, value) in self.writes {
            match &self.element {
                Element::Node(id) => graph.set_atom(*id, &key, value),
                Element::Link(block, id) => set_link(graph, block, *id, &key, value),
            }
            .map_err(|e| format!("{}: {e}", self.label))?;
        }
        Ok(())
    }
}

impl TypeAssignment {
    /// Run this match: stamp it onto `graph` and define it in `forcefield`.
    ///
    /// The one execution path of typing, in three phases:
    ///
    /// 1. **Validate.** A non-empty positional vector whose length differs
    ///    from the graph's count of that kind is an error naming the kind and
    ///    both counts (positions are the kind's own rows, see [`TypeAssignment`]), as
    ///    is a non-empty one for a relation kind the graph does not have, and
    ///    a type under a kind no category names ([`link_category`]).
    ///    Every style is checked against `forcefield` and the other styles,
    ///    and every [`Annotation::Type`] and pair row against `forcefield` and
    ///    the rest of the match, by the rule [`ForceField::def_style`] /
    ///    [`Style::def_type`] apply. A type or pair row whose style is neither
    ///    in `forcefield` nor in `styles` is an error, as is one element
    ///    writing one key twice with different values.
    /// 2. **Stamp** every annotation onto `graph`. A write the graph refuses
    ///    (a value contradicting the dtype of a declared key) is an error.
    /// 3. **Commit** through [`ForceField::def_style`] (in `styles` order) and
    ///    [`Style::def_type`] (types in element order — nodes, then the
    ///    relation kinds in the graph's registration order: bonds, angles,
    ///    dihedrals, impropers, then any kind registered later — then pair
    ///    rows). Phase 1 makes this infallible.
    ///
    /// # Errors
    ///
    /// On `Err`, `forcefield` is unchanged, and `graph` may be partly stamped
    /// and must be discarded. The force field is never cloned.
    pub fn apply_to(
        self,
        graph: &mut Atomistic,
        forcefield: &mut ForceField,
    ) -> Result<(), String> {
        let TypeAssignment {
            nodes,
            mut links,
            styles,
            pairs,
        } = self;

        // Phase 1: validate into a batch force field of this match's
        // definitions alone, and a list of stamps.
        let mut batch = ForceField::new(&forcefield.name);
        for (category, name, params) in styles {
            forcefield
                .check_style(&category, &name, &params)
                .map_err(|e| e.to_string())?;
            batch
                .def_style(&category, &name, params)
                .map_err(|e| format!("match styles: {e}"))?;
        }

        // Nodes first, then the relation kinds in the graph's registration
        // order (bonds, angles, dihedrals, impropers, then any kind added
        // later), whatever order `links` lists them in.
        let mut vectors: Vec<Vector> = Vec::new();
        if !nodes.is_empty() {
            let elements = graph.atom_ids().map(Element::Node).collect();
            vectors.push(Vector {
                category: Some("atom".to_owned()),
                kind: "atoms".to_owned(),
                elements,
                rows: nodes,
            });
        }
        for kind in link_blocks(graph) {
            let Some(rows) = links.shift_remove(&kind) else {
                continue;
            };
            if rows.is_empty() {
                continue;
            }
            let block: Rc<str> = Rc::from(kind.as_str());
            let elements = link_rows(graph, &kind)?
                .into_iter()
                .flatten()
                .map(|(r, _)| Element::Link(Rc::clone(&block), r))
                .collect();
            let category = link_category(&kind).map(|(c, _)| c);
            vectors.push(Vector {
                category,
                kind,
                elements,
                rows,
            });
        }
        if let Some((kind, _)) = links.iter().find(|(_, rows)| !rows.is_empty()) {
            link_rows(graph, kind)?;
            return Err(format!("graph has no '{kind}' relation kind"));
        }

        let mut stamps = Vec::new();
        for Vector {
            category,
            kind,
            elements,
            rows,
        } in vectors
        {
            // A kind no category names still takes stamped values; only a
            // type needs its category.
            let label = category.as_deref().unwrap_or(&kind);
            if rows.len() != elements.len() {
                return Err(format!(
                    "match has {} {label} annotation rows, the graph has {} {label} rows",
                    rows.len(),
                    elements.len()
                ));
            }
            for (position, (element, annotations)) in elements.into_iter().zip(rows).enumerate() {
                let mut stamp = Stamp {
                    element,
                    label: format!("{label} {position}"),
                    writes: Vec::new(),
                };
                for (key, annotation) in annotations {
                    match annotation {
                        Annotation::Value(value) => stamp.write(key, value)?,
                        Annotation::Type {
                            style,
                            name,
                            endpoints,
                            params,
                        } => {
                            for (k, v) in params
                                .iter()
                                .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                            {
                                stamp.write(k.to_owned(), PropValue::F64(v))?;
                            }
                            for (k, v) in params
                                .iter()
                                .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
                            {
                                stamp.write(k.to_owned(), PropValue::Str(v.to_owned()))?;
                            }
                            let category = category.as_deref().ok_or_else(|| {
                                format!(
                                    "{}: relation kind '{kind}' is the block of no category",
                                    stamp.label
                                )
                            })?;
                            // An array param (a cmap `grid`) has no property
                            // form; it stays on the type `name` defines below.
                            let target =
                                Self::batch_style(&mut batch, forcefield, category, &style)?;
                            let e: Vec<&str> = endpoints.iter().map(String::as_str).collect();
                            target
                                .def_type(&name, &e, params)
                                .map_err(|e| format!("{}: {e}", stamp.label))?;
                            stamp.write(key, PropValue::Str(name))?;
                        }
                    }
                }
                stamps.push(stamp);
            }
        }

        for (style, name, endpoints, params) in pairs {
            let e: Vec<&str> = endpoints.iter().map(String::as_str).collect();
            Self::batch_style(&mut batch, forcefield, "pair", &style)?
                .def_type(&name, &e, params)
                .map_err(|e| format!("match pairs: {e}"))?;
        }

        for style in batch.styles() {
            if let Some(existing) = forcefield.get_style(style.category(), style.name()) {
                for (name, endpoints, params) in style.type_rows() {
                    existing
                        .check_type(name, &endpoints, params)
                        .map_err(|e| e.to_string())?;
                }
            }
        }

        // Phase 2: stamp.
        for stamp in stamps {
            stamp.apply(graph)?;
        }

        // Phase 3: commit. Styles come first in the batch in `styles` order;
        // a style added to the batch only for a type is already declared in
        // `forcefield`, so `def_style` returns it in place.
        for style in batch.styles() {
            let target = forcefield
                .def_style(
                    Category::new(style.category()).arity(style.arity()),
                    style.name(),
                    style.params().clone(),
                )
                .map_err(|e| format!("commit after validation: {e}"))?;
            for (name, endpoints, params) in style.type_rows() {
                target
                    .def_type(name, &endpoints, params.clone())
                    .map_err(|e| format!("commit after validation: {e}"))?;
            }
        }
        Ok(())
    }

    /// The batch's `(category, style)` style, declared into the batch from
    /// `forcefield` when only the force field declares it; an error when
    /// neither does.
    fn batch_style<'b>(
        batch: &'b mut ForceField,
        forcefield: &ForceField,
        category: &str,
        style: &str,
    ) -> Result<&'b mut Style, String> {
        // The declared style's arity: a category no registry declares (read
        // from a record) is known to the batch only through it.
        let (arity, params) = match batch
            .get_style(category, style)
            .or_else(|| forcefield.get_style(category, style))
        {
            Some(declared) => (declared.arity(), declared.params().clone()),
            None => {
                return Err(format!(
                    "{category} style '{style}' is declared neither in the force field nor in \
                     the match's styles"
                ));
            }
        };
        batch
            .def_style(Category::new(category).arity(arity), style, params)
            .map_err(|e| format!("match: {e}"))
    }

    /// Declare every style of `library`, in library order and with its style
    /// params, whether or not the match defines a type under it.
    pub(crate) fn declare_styles_of(&mut self, library: &ForceField) {
        self.styles.extend(library.styles().iter().map(|s| {
            (
                s.category().to_owned(),
                s.name().to_owned(),
                s.params().clone(),
            )
        }));
    }

    /// Add every pair row of `library` whose endpoints are all atom types in
    /// `used`: the self rows of the types used and the cross rows between
    /// them. Rows are added in library order.
    pub(crate) fn add_pairs_among(&mut self, library: &ForceField, used: &HashSet<&str>) {
        for style in library.styles().iter().filter(|s| s.category() == "pair") {
            for (name, endpoints, params) in style.type_rows() {
                if endpoints.iter().all(|e| used.contains(e)) {
                    self.pairs.push((
                        style.name().to_owned(),
                        name.to_owned(),
                        endpoints.iter().map(|e| (*e).to_owned()).collect(),
                        params.clone(),
                    ));
                }
            }
        }
    }

    /// The rows of relation kind `kind`, created empty when the match has
    /// none.
    pub fn link_mut(&mut self, kind: &str) -> &mut Vec<Annotations> {
        self.links.entry(kind.to_owned()).or_default()
    }

    /// Type every row of relation kind `kind` of `graph` by its atoms' types
    /// against the type rows `library` holds in the kind's category
    /// ([`link_category`]): the one generic endpoint matcher, for a built-in
    /// kind and a registered one alike.
    ///
    /// An atom's type is its `key` annotation in [`TypeAssignment::nodes`] (a
    /// [`Annotation::Type`]'s name or a string [`Annotation::Value`]), else
    /// its string `key` property on `graph`. A type row matches a term when
    /// its endpoints equal the atoms' types slot by slot, a wildcard (`""`,
    /// `*`, `X`) matching any type, in the orders the category's
    /// [`EndpointOrder`] allows: as listed or reversed
    /// ([`Reversible`](EndpointOrder::Reversible): bonds, angles, proper
    /// dihedrals, a custom category by default), as listed only
    /// ([`Ordered`](EndpointOrder::Ordered): impropers), or in any order
    /// ([`Unordered`](EndpointOrder::Unordered)). Among the rows that match,
    /// the one with the fewest wildcards wins, and among those the first in
    /// table order (the library's styles in order, each style's rows in
    /// definition order) — molrec's rule.
    ///
    /// The winner is added to the term's annotations as
    /// `key = (style, name, endpoints, params)` exactly as the library holds
    /// it, and its style, with the library's style params, is added to
    /// [`TypeAssignment::styles`] unless already there.
    ///
    /// Returns the positions (in the kind's own row order) no row matched;
    /// they gain nothing. A graph without the kind assigns nothing.
    ///
    /// # Errors
    ///
    /// A `kind` no category names, an atom of a term without a type, and an
    /// existing non-empty vector for `kind` of another length than the
    /// graph's count of it.
    pub fn assign_terms(
        &mut self,
        graph: &Atomistic,
        kind: &str,
        library: &ForceField,
        key: &str,
    ) -> Result<Vec<usize>, String> {
        let Some(terms) = link_rows(graph, kind)? else {
            return Ok(Vec::new());
        };
        let (category, order) = link_category(kind)
            .ok_or_else(|| format!("relation kind '{kind}' is the block of no category"))?;

        let candidates: Vec<(&Style, &str, Vec<&str>, &Params)> = library
            .get_styles(&category)
            .into_iter()
            .flat_map(|style| {
                style
                    .type_rows()
                    .into_iter()
                    .map(move |(name, endpoints, params)| (style, name, endpoints, params))
            })
            .collect();

        let row_of: HashMap<NodeId, usize> = graph
            .atom_ids()
            .enumerate()
            .map(|(row, id)| (id, row))
            .collect();
        let atom_type = |node: NodeId| -> Option<String> {
            let row = *row_of.get(&node)?;
            let annotated = self.nodes.get(row).and_then(|annotations| {
                annotations
                    .iter()
                    .find(|(k, _)| k == key)
                    .and_then(|(_, a)| match a {
                        Annotation::Type { name, .. } => Some(name.clone()),
                        Annotation::Value(PropValue::Str(s)) => Some(s.clone()),
                        Annotation::Value(_) => None,
                    })
            });
            annotated.or_else(|| {
                graph
                    .atom_table()
                    .get::<String>(node, key)
                    .ok()
                    .map(str::to_owned)
            })
        };

        let mut assigned: Vec<Option<usize>> = Vec::new();
        let mut cache: HashMap<Vec<String>, Option<usize>> = HashMap::new();
        for (position, (_, term)) in terms.iter().enumerate() {
            let types: Vec<String> = term
                .nodes
                .iter()
                .map(|&n| {
                    atom_type(n).ok_or_else(|| {
                        format!("{category} {position}: atom {n:?} has no '{key}' type")
                    })
                })
                .collect::<Result<_, _>>()?;
            let best = *cache.entry(types).or_insert_with_key(|types| {
                let atoms: Vec<&str> = types.iter().map(String::as_str).collect();
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, _, pattern, _))| endpoints_match(pattern, &atoms, order))
                    .min_by_key(|(i, (_, _, pattern, _))| {
                        (pattern.iter().filter(|p| is_wildcard(p)).count(), *i)
                    })
                    .map(|(i, _)| i)
            });
            assigned.push(best);
        }

        let rows = self.links.entry(kind.to_owned()).or_default();
        if rows.is_empty() {
            rows.resize(assigned.len(), Vec::new());
        } else if rows.len() != assigned.len() {
            return Err(format!(
                "match has {} {category} annotation rows, the graph has {} {category} rows",
                rows.len(),
                assigned.len()
            ));
        }
        let mut unmatched = Vec::new();
        for (position, (best, annotations)) in assigned.iter().zip(rows.iter_mut()).enumerate() {
            let Some(&(style, name, ref endpoints, params)) = best.map(|i| &candidates[i]) else {
                unmatched.push(position);
                continue;
            };
            annotations.push((
                key.to_owned(),
                Annotation::Type {
                    style: style.name().to_owned(),
                    name: name.to_owned(),
                    endpoints: endpoints.iter().map(|e| (*e).to_owned()).collect(),
                    params: params.clone(),
                },
            ));
        }
        for &(style, ..) in assigned.iter().flatten().map(|&i| &candidates[i]) {
            let declared = self
                .styles
                .iter()
                .any(|(c, s, _)| *c == category && s == style.name());
            if !declared {
                self.styles.push((
                    category.clone(),
                    style.name().to_owned(),
                    style.params().clone(),
                ));
            }
        }
        Ok(unmatched)
    }
}

/// The base of every typifier: one [`Typifier`] plus the output force field
/// its typing accumulates.
///
/// A trait cannot hold the output, so the base is this struct. The output
/// starts as `typifier.source_forcefield().empty_like()` and [`typify`](Self::typify) is
/// its only writer; there is no mutable accessor.
#[derive(Debug)]
pub struct Typing<T: Typifier> {
    typifier: T,
    output: ForceField,
}

impl<T: Typifier> Typing<T> {
    /// Wrap `typifier`, with an output seeded by
    /// [`ForceField::empty_like`] of its source force field: that force field's name and
    /// declared units and special_bonds, no styles or types.
    pub fn new(typifier: T) -> Self {
        let output = typifier.source_forcefield().empty_like();
        Self { typifier, output }
    }

    /// Type `mol`: copy it, match the copy and write the match onto the copy
    /// and the output ([`TypeAssignment::apply_to`]). Returns the typed copy.
    ///
    /// `mol` is never touched. On `Err` the output is unchanged.
    pub fn typify(&mut self, mol: &Atomistic) -> Result<Atomistic, String> {
        let mut graph = mol.clone();
        let m = self.typifier.assign(&mut graph)?;
        m.apply_to(&mut graph, &mut self.output)?;
        Ok(graph)
    }

    /// The accumulated output: exactly the definitions typing has assigned.
    pub fn forcefield(&self) -> &ForceField {
        &self.output
    }

    /// The wrapped typifier.
    pub fn typifier(&self) -> &T {
        &self.typifier
    }
}

#[cfg(test)]
mod tests {
    //! `TypeAssignment::apply_to` and `Typing<T>` against hand-written stub typifiers.
    //! Every expectation is written by hand; no native typifier runs here.

    use crate::ff::potential::fixtures::OneShot;
    use crate::props;
    use indexmap::IndexMap;
    use molrs::core::Element;
    use molrs::core::keys;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use molrs::core::Atomistic;
    use molrs::core::{PropValue, Props};

    use super::*;
    use crate::ff::forcefield::ForceField;
    use crate::ff::forcefield::tests::assert_same_definitions;
    use crate::ff::ir::{Params, SpecialBonds};

    // -- stubs and fixtures ----------------------------------------------------

    /// Returns the next scripted `TypeAssignment` on each call; `Err` once the script
    /// is exhausted. The `Mutex` keeps the stub `Send + Sync`.
    struct ScriptedTypifier {
        library: ForceField,
        script: Mutex<VecDeque<TypeAssignment>>,
    }

    impl ScriptedTypifier {
        fn new(library: ForceField, script: Vec<TypeAssignment>) -> Self {
            Self {
                library,
                script: Mutex::new(script.into()),
            }
        }
    }

    impl Typifier for ScriptedTypifier {
        fn assign(&self, _graph: &mut Atomistic) -> Result<TypeAssignment, String> {
            self.script
                .lock()
                .expect("script lock")
                .pop_front()
                .ok_or_else(|| "script exhausted".to_owned())
        }

        fn source_forcefield(&self) -> &ForceField {
            &self.library
        }
    }

    /// Writes an intermediate result onto the graph it is handed (as a
    /// perception step would) and matches nothing.
    struct PerceivingTypifier {
        library: ForceField,
    }

    impl Typifier for PerceivingTypifier {
        fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
            let ids: Vec<_> = graph.atoms().map(|(id, _)| id).collect();
            for id in ids {
                graph
                    .set_atom(id, "perceived", true)
                    .map_err(|e| e.to_string())?;
            }
            Ok(TypeAssignment::default())
        }

        fn source_forcefield(&self) -> &ForceField {
            &self.library
        }
    }

    const SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.8333],
    };

    /// Three atoms in a chain: `c0-c1-c2`, two bonds, one angle.
    fn chain3() -> Atomistic {
        let mut g = Atomistic::new();
        let a = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let b = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        g.add_bond(a, b, props! {}).unwrap();
        g.add_bond(b, c, props! {}).unwrap();
        g.add_angle(a, b, c, props! {}).unwrap();
        g
    }

    /// Two atoms, no links.
    fn pair2() -> Atomistic {
        let mut g = Atomistic::new();
        g.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        g.add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        g
    }

    fn value(key: &str, v: impl Into<PropValue>) -> (String, Annotation) {
        (key.to_owned(), Annotation::Value(v.into()))
    }

    fn ty(
        key: &str,
        style: &str,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> (String, Annotation) {
        (
            key.to_owned(),
            Annotation::Type {
                style: style.to_owned(),
                name: name.to_owned(),
                endpoints: endpoints.iter().map(|s| (*s).to_owned()).collect(),
                params,
            },
        )
    }

    fn style(category: &str, name: &str, params: Params) -> (String, String, Params) {
        (category.to_owned(), name.to_owned(), params)
    }

    fn pair_row(
        style: &str,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> (String, String, Vec<String>, Params) {
        (
            style.to_owned(),
            name.to_owned(),
            endpoints.iter().map(|s| (*s).to_owned()).collect(),
            params,
        )
    }

    fn links<const N: usize>(
        kinds: [(&str, Vec<Annotations>); N],
    ) -> IndexMap<String, Vec<Annotations>> {
        kinds
            .into_iter()
            .map(|(k, rows)| (k.to_owned(), rows))
            .collect()
    }

    fn atom_full(name: &str, mass: f64) -> (String, Annotation) {
        ty(
            "type",
            "full",
            name,
            &[],
            Params::from_iter([("mass", mass)]),
        )
    }

    fn atom_full_style() -> (String, String, Params) {
        style("atom", "full", Params::new())
    }

    fn nth_atom(g: &Atomistic, i: usize) -> Props {
        g.atoms().nth(i).expect("atom index").1
    }

    fn nth_bond_props(g: &Atomistic, i: usize) -> Props {
        g.bonds().nth(i).expect("bond index").1.props
    }

    /// Type names of the `(category, style)` style, in definition order.
    fn type_names(ff: &ForceField, category: &str, style: &str) -> Vec<String> {
        ff.get_style(category, style)
            .unwrap_or_else(|| panic!("no {category}/{style} style"))
            .defs()
            .collect_type_params()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn style_keys(ff: &ForceField) -> Vec<(&str, &str)> {
        ff.styles()
            .iter()
            .map(|s| (s.category(), s.name()))
            .collect()
    }

    /// Every atom's and every link's property bag, in row order.
    type GraphProps = (Vec<Props>, Vec<Vec<Props>>);

    fn graph_props(g: &Atomistic) -> GraphProps {
        let atoms = g.atoms().map(|(_, a)| a).collect();
        let links = vec![
            g.bonds().map(|(_, r)| r.props).collect(),
            g.angles().map(|(_, r)| r.props).collect(),
            g.dihedrals().map(|(_, r)| r.props).collect(),
            g.impropers().map(|(_, r)| r.props).collect(),
        ];
        (atoms, links)
    }

    // -- TypeAssignment::apply_to: stamps ---------------------------------------------

    #[test]
    fn apply_to_stamps_a_value_on_the_atom_at_its_position_only() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            nodes: vec![
                vec![value("class", "CT")],
                vec![],
                vec![value("aromatic_flag", true), value("ring_count", 2)],
            ],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_eq!(
            nth_atom(&g, 0).get("class"),
            Some(&PropValue::Str("CT".into()))
        );
        assert_eq!(nth_atom(&g, 1).get("class"), None);
        assert_eq!(nth_atom(&g, 2).get("class"), None);
        assert_eq!(
            nth_atom(&g, 2).get("aromatic_flag"),
            Some(&PropValue::Bool(true))
        );
        assert_eq!(nth_atom(&g, 2).get("ring_count"), Some(&PropValue::Int(2)));
    }

    /// `Type` stamps `key = name` plus every param under its own key: numeric
    /// as `F64`, string as `Str`.
    #[test]
    fn apply_to_stamps_a_type_name_and_every_param_on_its_atom() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let mut params = Params::from_iter([("mass", 12.011), ("charge", -0.18)]);
        params.set("provenance", "hand");
        let m = TypeAssignment {
            nodes: vec![vec![ty("type", "full", "CT", &[], params)], vec![]],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        let a0 = nth_atom(&g, 0);
        assert_eq!(a0.get("type"), Some(&PropValue::Str("CT".into())));
        assert_eq!(a0.get("mass"), Some(&PropValue::F64(12.011)));
        assert_eq!(a0.get("charge"), Some(&PropValue::F64(-0.18)));
        assert_eq!(a0.get("provenance"), Some(&PropValue::Str("hand".into())));
        let a1 = nth_atom(&g, 1);
        assert_eq!(a1.get("type"), None);
        assert_eq!(a1.get("mass"), None);
    }

    #[test]
    fn apply_to_stamps_a_bond_type_on_the_bond_at_its_position_only() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            links: links([(
                "bonds",
                vec![
                    vec![],
                    vec![ty(
                        "type",
                        "harmonic",
                        "C-O",
                        &["C", "O"],
                        Params::from_iter([("k", 320.0), ("r0", 1.41)]),
                    )],
                ],
            )]),
            styles: vec![style("bond", "harmonic", Params::new())],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert!(!nth_bond_props(&g, 0).contains_key("type"));
        let b1 = nth_bond_props(&g, 1);
        assert_eq!(b1.get("type"), Some(&PropValue::Str("C-O".into())));
        assert_eq!(b1.get("k"), Some(&PropValue::F64(320.0)));
        assert_eq!(b1.get("r0"), Some(&PropValue::F64(1.41)));
    }

    /// Dihedral and improper vectors are positional against their own kind's
    /// rows: an improper never shifts a dihedral position, and vice versa.
    #[test]
    fn apply_to_stamps_each_link_kind_on_its_own_rows() {
        let mut g = Atomistic::new();
        let ids: Vec<_> = (0..4)
            .map(|_| {
                g.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        g.add_improper(ids[1], ids[0], ids[2], ids[3], props! {})
            .unwrap();
        g.add_dihedral(ids[0], ids[1], ids[2], ids[3], props! {})
            .unwrap();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            links: links([
                ("dihedrals", vec![vec![value("tag", "dihedral-0")]]),
                ("impropers", vec![vec![value("tag", "improper-0")]]),
            ]),
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        let dihedral = g.dihedrals().next().unwrap().1.props;
        let improper = g.impropers().next().unwrap().1.props;
        assert_eq!(
            dihedral.get("tag"),
            Some(&PropValue::Str("dihedral-0".into()))
        );
        assert_eq!(
            improper.get("tag"),
            Some(&PropValue::Str("improper-0".into()))
        );
    }

    /// An empty vector means "nothing for this kind", whatever the count.
    #[test]
    fn apply_to_accepts_an_empty_vector_for_a_kind_the_graph_has() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            nodes: vec![vec![], vec![], vec![]],
            links: links([("bonds", vec![]), ("angles", vec![])]),
            ..TypeAssignment::default()
        };

        assert_eq!(m.apply_to(&mut g, &mut ff), Ok(()));
    }

    // -- TypeAssignment::apply_to: definitions ------------------------------------------

    /// The output defines the match's `Type`s (one row per distinct
    /// definition) and pair rows, and nothing else.
    #[test]
    fn apply_to_defines_exactly_the_match_types_and_pair_rows() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let lj = |eps: f64, sigma: f64| Params::from_iter([("epsilon", eps), ("sigma", sigma)]);
        let m = TypeAssignment {
            nodes: vec![
                vec![atom_full("CT", 12.011)],
                vec![atom_full("CT", 12.011)],
                vec![atom_full("OH", 15.999)],
            ],
            links: links([(
                "bonds",
                vec![
                    vec![ty(
                        "type",
                        "harmonic",
                        "CT-CT",
                        &["CT", "CT"],
                        Params::from_iter([("k", 268.0), ("r0", 1.529)]),
                    )],
                    vec![ty(
                        "type",
                        "harmonic",
                        "CT-OH",
                        &["CT", "OH"],
                        Params::from_iter([("k", 320.0), ("r0", 1.41)]),
                    )],
                ],
            )]),
            styles: vec![
                atom_full_style(),
                style("bond", "harmonic", Params::new()),
                style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)])),
            ],
            pairs: vec![
                pair_row("lj/cut", "CT", &["CT"], lj(0.066, 3.5)),
                pair_row("lj/cut", "OH", &["OH"], lj(0.17, 3.12)),
            ],
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_eq!(ff.styles().len(), 3);
        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT", "OH"]);
        assert_eq!(type_names(&ff, "bond", "harmonic"), vec!["CT-CT", "CT-OH"]);
        assert_eq!(type_names(&ff, "pair", "lj/cut"), vec!["CT", "OH"]);
        let full = ff.get_style("atom", "full").unwrap();
        assert_eq!(
            full.get_atomtype("OH").unwrap().params,
            Params::from_iter([("mass", 15.999)])
        );
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            bond.get_bondtype("CT", "OH").unwrap().params,
            Params::from_iter([("k", 320.0), ("r0", 1.41)])
        );
        let pair = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            pair.get_pairtype("CT", None).unwrap().params,
            lj(0.066, 3.5)
        );
    }

    /// The given endpoints are the ones defined, under the name as given:
    /// `1-6` on `2`, `7` holds `2`, `7` — the name is never read.
    #[test]
    fn apply_to_defines_the_given_endpoints_and_never_reads_the_name() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let k = || Params::from_iter([("kb", 4.2)]);
        let m = TypeAssignment {
            links: links([(
                "bonds",
                vec![
                    vec![ty("type", "mmff_bond", "0_1_1", &["1", "1"], k())],
                    vec![ty("type", "mmff_bond", "1-6", &["2", "7"], k())],
                ],
            )]),
            styles: vec![style("bond", "mmff_bond", Params::new())],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        let s = ff.get_style("bond", "mmff_bond").unwrap();
        assert_eq!(
            s.type_endpoints("0_1_1"),
            Some(vec!["1".to_owned(), "1".to_owned()])
        );
        assert_eq!(
            s.type_endpoints("1-6"),
            Some(vec!["2".to_owned(), "7".to_owned()])
        );
    }

    #[test]
    fn apply_to_declares_styles_in_the_order_of_styles() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            styles: vec![
                style("pair", "lj/cut", Params::from_iter([("cutoff", 9.0)])),
                style("dihedral", "opls", Params::new()),
                atom_full_style(),
                style("bond", "harmonic", Params::new()),
            ],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_eq!(
            style_keys(&ff),
            vec![
                ("pair", "lj/cut"),
                ("dihedral", "opls"),
                ("atom", "full"),
                ("bond", "harmonic"),
            ]
        );
        assert_eq!(
            ff.get_style("pair", "lj/cut").unwrap().params(),
            &Params::from_iter([("cutoff", 9.0)])
        );
    }

    #[test]
    fn apply_to_of_a_stamp_only_match_leaves_the_forcefield_empty() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            nodes: vec![vec![value("type", "c3")], vec![value("type", "hc")]],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert!(ff.styles().is_empty(), "{:?}", ff.styles());
        assert_eq!(
            nth_atom(&g, 0).get("type"),
            Some(&PropValue::Str("c3".into()))
        );
        assert_eq!(
            nth_atom(&g, 1).get("type"),
            Some(&PropValue::Str("hc".into()))
        );
    }

    /// A `Type` whose style the force field already declares needs no entry
    /// in `styles`.
    #[test]
    fn apply_to_accepts_a_type_whose_style_the_forcefield_already_declares() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new()).unwrap();
        let m = TypeAssignment {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT"]);
    }

    /// Re-defining a type the force field holds with identical params is a
    /// no-op: still one row.
    #[test]
    fn apply_to_of_an_identical_existing_type_is_a_no_op() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap();
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_same_definitions(&ff, &before);
    }

    // -- TypeAssignment::apply_to: errors leave the force field unchanged ----------------

    /// A conflicting `Type` fails the whole match: the new style and the new
    /// type that precede it in the batch do not land either.
    #[test]
    fn apply_to_type_conflicting_with_the_forcefield_errs_and_changes_nothing() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap();
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![
                vec![atom_full("OH", 15.999)],
                vec![atom_full("CT", 12.0)],
                vec![],
            ],
            links: links([("bonds", vec![])]),
            styles: vec![style("bond", "harmonic", Params::new()), atom_full_style()],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// Two elements of one match defining one name differently is a
    /// conflict within the batch.
    #[test]
    fn apply_to_two_conflicting_types_within_one_match_err_and_change_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![atom_full("CT", 12.0)]],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn apply_to_style_conflicting_with_the_forcefield_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        let before = ff.clone();
        let m = TypeAssignment {
            styles: vec![
                atom_full_style(),
                style("pair", "lj/cut", Params::from_iter([("cutoff", 12.0)])),
            ],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn apply_to_type_under_an_undeclared_style_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![]],
            styles: vec![style("pair", "lj/cut", Params::new())],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn apply_to_pair_row_under_an_undeclared_style_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            styles: vec![atom_full_style()],
            pairs: vec![pair_row(
                "lj/cut",
                "CT",
                &["CT"],
                Params::from_iter([("epsilon", 0.066), ("sigma", 3.5)]),
            )],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// A non-empty node vector must have one entry per atom; the error names
    /// both counts.
    #[test]
    fn apply_to_node_vector_of_the_wrong_length_errs_naming_both_counts() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![
                vec![atom_full("CT", 12.011)],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        let err = m.apply_to(&mut g, &mut ff).unwrap_err();

        assert!(err.contains('5') && err.contains('2'), "{err}");
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn apply_to_bond_vector_of_the_wrong_length_errs_naming_kind_and_counts() {
        let mut g = chain3();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            links: links([("bonds", vec![vec![], vec![], vec![], vec![]])]),
            ..TypeAssignment::default()
        };

        let err = m.apply_to(&mut g, &mut ff).unwrap_err();

        assert!(err.contains("bond"), "{err}");
        assert!(err.contains('4') && err.contains('2'), "{err}");
        assert_same_definitions(&ff, &before);
    }

    /// A `Type` param and a `Value` both writing `charge` on one atom, with
    /// different values.
    #[test]
    fn apply_to_one_key_written_twice_with_different_values_errs() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![
                vec![
                    ty(
                        "type",
                        "full",
                        "CT",
                        &[],
                        Params::from_iter([("charge", 0.5)]),
                    ),
                    value("charge", -0.5),
                ],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    /// Only *different* values collide; the same value written twice is one
    /// write.
    #[test]
    fn apply_to_one_key_written_twice_with_the_same_value_is_accepted() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            nodes: vec![
                vec![
                    ty(
                        "type",
                        "full",
                        "CT",
                        &[],
                        Params::from_iter([("charge", 0.5)]),
                    ),
                    value("charge", 0.5),
                ],
                vec![],
            ],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        m.apply_to(&mut g, &mut ff).unwrap();

        assert_eq!(nth_atom(&g, 0).get("charge"), Some(&PropValue::F64(0.5)));
        assert_eq!(type_names(&ff, "atom", "full"), vec!["CT"]);
    }

    /// `mass` is declared float by the Frame schema, so a string `mass` is
    /// refused at the stamp. The valid `Type` on the other atom is not
    /// defined: stamping precedes the commit.
    #[test]
    fn apply_to_stamp_the_graph_refuses_errs_and_changes_nothing() {
        let mut g = pair2();
        let mut ff = ForceField::new("out");
        let before = ff.clone();
        let m = TypeAssignment {
            nodes: vec![vec![atom_full("CT", 12.011)], vec![value("mass", "heavy")]],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        };

        let result = m.apply_to(&mut g, &mut ff);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(&ff, &before);
    }

    // -- Typing -----------------------------------------------------------------

    fn library() -> ForceField {
        let mut lib = ForceField::new("lib");
        lib.set_units("metal");
        lib.set_special_bonds(SB);
        lib.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap()
            .def_type("UNUSED", &[], Params::from_iter([("mass", 1.0)]))
            .unwrap();
        lib
    }

    /// `CT` on atom 0, nothing on atom 1, under the declared `atom/full`.
    fn ct_match(mass: f64) -> TypeAssignment {
        TypeAssignment {
            nodes: vec![vec![atom_full("CT", mass)], vec![]],
            styles: vec![atom_full_style()],
            ..TypeAssignment::default()
        }
    }

    #[test]
    fn typing_new_seeds_the_output_with_the_library_name_units_and_special_bonds() {
        let typing = Typing::new(ScriptedTypifier::new(library(), vec![]));

        let out = typing.forcefield();

        assert_eq!(out.name, "lib");
        assert_eq!(out.declared_units(), Some("metal"));
        assert_eq!(out.declared_special_bonds(), Some(&SB));
        assert!(out.styles().is_empty(), "{:?}", out.styles());
    }

    #[test]
    fn typing_typifier_returns_the_wrapped_typifier_and_its_source_forcefield() {
        let typing = Typing::new(ScriptedTypifier::new(library(), vec![]));

        assert_same_definitions(typing.typifier().source_forcefield(), &library());
    }

    #[test]
    fn typify_returns_a_stamped_copy_and_leaves_the_input_unchanged() {
        let mol = pair2();
        let before = graph_props(&mol);
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(graph_props(&mol), before);
        assert_eq!(
            nth_atom(&typed, 0).get("type"),
            Some(&PropValue::Str("CT".into()))
        );
        assert_eq!(
            nth_atom(&typed, 0).get("mass"),
            Some(&PropValue::F64(12.011))
        );
    }

    /// `assign` receives the private working copy, and that copy is what
    /// `typify` returns: an intermediate result written by `assign` is on the
    /// returned graph and not on the input.
    #[test]
    fn typify_hands_match_the_working_copy_it_returns() {
        let mol = pair2();
        let mut typing = Typing::new(PerceivingTypifier {
            library: ForceField::new("lib"),
        });

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(
            nth_atom(&typed, 1).get("perceived"),
            Some(&PropValue::Bool(true))
        );
        assert_eq!(nth_atom(&mol, 1).get("perceived"), None);
    }

    /// The output holds the types the match assigned, not the library's
    /// unused `UNUSED` row.
    #[test]
    fn typify_defines_the_matched_types_in_the_output_only() {
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));

        typing.typify(&pair2()).unwrap();

        assert_eq!(type_names(typing.forcefield(), "atom", "full"), vec!["CT"]);
    }

    #[test]
    fn typify_of_a_stamp_only_match_leaves_the_output_empty() {
        let stamp_only = TypeAssignment {
            nodes: vec![vec![value("type", "c3")], vec![value("type", "hc")]],
            ..TypeAssignment::default()
        };
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![stamp_only]));

        let typed = typing.typify(&pair2()).unwrap();

        assert!(typing.forcefield().styles().is_empty());
        assert_eq!(
            nth_atom(&typed, 1).get("type"),
            Some(&PropValue::Str("hc".into()))
        );
    }

    #[test]
    fn typify_of_an_identical_redefinition_across_two_calls_is_a_no_op() {
        let mut typing = Typing::new(ScriptedTypifier::new(
            library(),
            vec![ct_match(12.011), ct_match(12.011)],
        ));
        typing.typify(&pair2()).unwrap();
        let after_first = typing.forcefield().clone();

        typing.typify(&pair2()).unwrap();

        assert_same_definitions(typing.forcefield(), &after_first);
    }

    #[test]
    fn typify_of_a_conflicting_type_errs_and_leaves_output_and_input_unchanged() {
        let mut typing = Typing::new(ScriptedTypifier::new(
            library(),
            vec![ct_match(12.011), ct_match(12.0)],
        ));
        typing.typify(&pair2()).unwrap();
        let after_first = typing.forcefield().clone();
        let mol = pair2();
        let mol_before = graph_props(&mol);

        let result = typing.typify(&mol);

        assert!(result.is_err(), "{result:?}");
        assert_same_definitions(typing.forcefield(), &after_first);
        assert_eq!(graph_props(&mol), mol_before);
    }

    #[test]
    fn typify_propagates_a_match_error_and_leaves_the_output_unchanged() {
        let mut typing = Typing::new(ScriptedTypifier::new(library(), vec![]));
        let before = typing.forcefield().clone();

        let result = typing.typify(&pair2());

        assert_eq!(result.err().as_deref(), Some("script exhausted"));
        assert_same_definitions(typing.forcefield(), &before);
    }

    /// `Box<dyn Typifier + Send + Sync>` is itself a `Typifier`, so one
    /// `Typing` type can hold any typifier chosen at run time.
    #[test]
    fn boxed_dyn_typifier_is_a_typifier() {
        let boxed: Box<dyn Typifier + Send + Sync> =
            Box::new(ScriptedTypifier::new(library(), vec![ct_match(12.011)]));
        let mut typing: Typing<Box<dyn Typifier + Send + Sync>> = Typing::new(boxed);

        typing.typify(&pair2()).unwrap();

        assert_eq!(typing.typifier().source_forcefield().name, "lib");
        assert_eq!(type_names(typing.forcefield(), "atom", "full"), vec!["CT"]);
    }

    // -- any relation kind: `links` and `assign_terms` ---------------------------

    /// The custom category of the force-field IR protocol's proof: 1-3
    /// Urey–Bradley terms in their own block `urey_bradleys`. Registered in
    /// the process-wide registry exactly as the Python test hook does, so
    /// the registration is idempotent across tests.
    fn register_urey_bradley() {
        crate::ff::style_registry::register_category(crate::ff::ir::CategorySpec::custom(
            "urey_bradley",
            3,
            crate::ff::ir::Coordinate::Compound,
            EndpointOrder::Reversible,
        ))
        .unwrap();
    }

    const UB_EXPRESSION: &str = "k_ub*(distance(p1,p3)-r_ub)^2";
    const UB_XYZ: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.52, 0.1, 0.05],
        [2.1, 1.45, -0.1],
        [3.55, 1.6, 0.6],
    ];

    /// Four carbon atoms in a chain with the three-atom relation kind `kind`
    /// over `(0, 1, 2)` and `(1, 2, 3)`, plus every `(name, arity)` of
    /// `extra` over the first `arity` atoms; `angles` is the Atomistic's own
    /// kind.
    ///
    /// A custom block (`urey_bradleys`, `link`) is no canonical term block,
    /// so `Atomistic::add_term` cannot introduce it: the graph is built as a
    /// [`MolGraph`](molrs::core::MolGraph) and promoted.
    fn ub_chain_with(kind: &str, extra: &[(&str, usize)]) -> Atomistic {
        let mut g = molrs::core::MolGraph::new();
        let ids: Vec<_> = UB_XYZ
            .iter()
            .map(|&[x, y, z]| {
                g.add_node(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => x,
                    keys::Y => y,
                    keys::Z => z,
                })
                .unwrap()
            })
            .collect();
        let kid = g.register_kind(kind, 3);
        for w in ids.windows(3) {
            g.add_relation(kid, w, props! {}).unwrap();
        }
        for &(name, arity) in extra {
            let kid = g.register_kind(name, arity);
            g.add_relation(kid, &ids[..arity], props! {}).unwrap();
        }
        Atomistic::try_from(g).unwrap()
    }

    /// [`ub_chain_with`] and no further kind.
    fn ub_chain(kind: &str) -> Atomistic {
        ub_chain_with(kind, &[])
    }

    /// One `category` style `style` with, in table order: a decoy of three
    /// wildcards, `t` written end-for-end (`C-B-A`), and `u` with one
    /// wildcard end (`*-C-D`). `extra` adds params every row carries.
    fn ub_library(
        category: &str,
        style: &str,
        params: Params,
        extra: &[(&str, f64)],
    ) -> ForceField {
        let mut lib = ForceField::new("lib");
        let atom = lib.def_style("atom", "full", Params::new()).unwrap();
        for name in ["A", "B", "C", "D"] {
            atom.def_type(name, &[], Params::from_iter([("mass", 12.0)]))
                .unwrap();
        }
        let s = lib.def_style(category, style, params).unwrap();
        for (name, ends, k_ub, r_ub) in [
            ("decoy", ["", "", ""], 99.0, 9.9),
            ("t", ["C", "B", "A"], 20.0, 2.45),
            ("u", ["*", "C", "D"], 11.0, 2.2),
        ] {
            let mut p = Params::from_iter([("k_ub", k_ub), ("r_ub", r_ub)]);
            for &(k, v) in extra {
                p.set(k, v);
            }
            s.def_type(name, &ends, p).unwrap();
        }
        lib
    }

    /// Types atoms `A`..`D` in row order and assigns `kind` from its library.
    struct ChainTypifier {
        library: ForceField,
        kind: &'static str,
    }

    impl Typifier for ChainTypifier {
        fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
            let mut m = TypeAssignment {
                nodes: ["A", "B", "C", "D"]
                    .iter()
                    .map(|t| {
                        vec![ty(
                            "type",
                            "full",
                            t,
                            &[],
                            Params::from_iter([("mass", 12.0)]),
                        )]
                    })
                    .collect(),
                styles: vec![atom_full_style()],
                ..TypeAssignment::default()
            };
            let missing = m.assign_terms(graph, self.kind, &self.library, "type")?;
            assert!(missing.is_empty(), "{missing:?}");
            Ok(m)
        }

        fn source_forcefield(&self) -> &ForceField {
            &self.library
        }
    }

    fn energy_forces(ff: &ForceField, typed: &Atomistic) -> (f64, Vec<f64>) {
        let frame = typed.to_frame().unwrap();
        let coords: Vec<f64> = UB_XYZ.iter().flatten().copied().collect();
        crate::ff::compile::PotentialCompiler::new(ff)
            .compile(&frame)
            .unwrap()
            .energy_forces_at(&coords)
    }

    /// A registered custom category is typified exactly like `angles`: the
    /// same matcher assigns its terms, they land in the block `urey_bradleys`
    /// (`atomi`, `atomj`, `atomk`, `type`), and the typed output prices them
    /// as LAMMPS `angle charmm` with K = 0 prices the same two terms.
    #[test]
    fn a_custom_relation_kind_is_typified_compiled_and_priced_like_angle_charmm() {
        register_urey_bradley();
        let mut expr = Params::new();
        expr.set("expression", UB_EXPRESSION);
        let mut ub = Typing::new(ChainTypifier {
            library: ub_library("urey_bradley", "spring", expr, &[]),
            kind: "urey_bradleys",
        });
        let typed = ub.typify(&ub_chain("urey_bradleys")).unwrap();

        let frame = typed.to_frame().unwrap();
        let block = frame.get("urey_bradleys").expect("urey_bradleys block");
        for key in ["atomi", "atomj", "atomk", "type"] {
            assert!(block.contains_key(key), "urey_bradleys has no '{key}'");
        }
        let types: Vec<_> = typed
            .term_ids("urey_bradleys")
            .unwrap()
            .map(|id| typed.get_term("urey_bradleys", id).unwrap().props["type"].clone())
            .collect();
        assert_eq!(types, vec![PropValue::from("t"), PropValue::from("u")]);
        // Only the types used are defined; the decoy never is.
        assert_eq!(
            type_names(ub.forcefield(), "urey_bradley", "spring"),
            vec!["t", "u"]
        );
        assert_eq!(
            ub.forcefield()
                .get_style("urey_bradley", "spring")
                .unwrap()
                .params()
                .get("expression")
                .and_then(|v| v.as_text()),
            Some(UB_EXPRESSION)
        );

        let mut charmm = Typing::new(ChainTypifier {
            library: ub_library(
                "angle",
                "charmm",
                Params::new(),
                &[("k", 0.0), ("theta0", 109.5)],
            ),
            kind: "angles",
        });
        let reference = charmm.typify(&ub_chain("angles")).unwrap();
        assert_eq!(
            type_names(charmm.forcefield(), "angle", "charmm"),
            vec!["t", "u"]
        );

        let (e, f) = energy_forces(ub.forcefield(), &typed);
        let (e_ref, f_ref) = energy_forces(charmm.forcefield(), &reference);
        assert!(e_ref > 0.0, "a non-trivial reference: {e_ref}");
        assert!((e - e_ref).abs() <= 1e-12 * e_ref, "{e} vs {e_ref}");
        let fmax = f_ref.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
        for (i, (a, b)) in f.iter().zip(&f_ref).enumerate() {
            assert!((a - b).abs() <= 1e-12 * fmax, "force[{i}] {a} vs {b}");
        }
    }

    /// `links` takes any relation kind of the graph; a kind the graph does
    /// not have is refused by name, and a type under a kind no category
    /// names is refused while a stamped value there is not.
    #[test]
    fn apply_to_takes_any_relation_kind_and_refuses_a_kind_the_graph_lacks() {
        register_urey_bradley();
        let mut g = ub_chain_with("urey_bradleys", &[("link", 2)]);
        let mut ff = ForceField::new("out");
        let m = TypeAssignment {
            links: links([("urey_bradleys", vec![vec![value("tag", "first")], vec![]])]),
            ..TypeAssignment::default()
        };
        m.apply_to(&mut g, &mut ff).unwrap();
        let first_id = g.term_ids("urey_bradleys").unwrap().next().unwrap();
        let first = g.get_term("urey_bradleys", first_id).unwrap().props;
        assert_eq!(first.get("tag"), Some(&PropValue::Str("first".into())));

        let m = TypeAssignment {
            links: links([("cross_terms", vec![vec![]])]),
            ..TypeAssignment::default()
        };
        let err = m.apply_to(&mut g, &mut ff).unwrap_err();
        assert!(
            err.contains("graph has no 'cross_terms' relation kind"),
            "{err}"
        );

        let m = TypeAssignment {
            links: links([("ports", vec![vec![value("tag", "p")]])]),
            ..TypeAssignment::default()
        };
        let err = m.apply_to(&mut g, &mut ff).unwrap_err();
        assert!(
            err.contains("relation block 'ports' takes no annotations"),
            "{err}"
        );

        let m = TypeAssignment {
            links: links([("link", vec![vec![value("tag", "x")]])]),
            ..TypeAssignment::default()
        };
        m.apply_to(&mut g, &mut ff).unwrap();
        let m = TypeAssignment {
            links: links([(
                "link",
                vec![vec![ty("type", "s", "t", &["A", "B"], Params::new())]],
            )]),
            ..TypeAssignment::default()
        };
        let err = m.apply_to(&mut g, &mut ff).unwrap_err();
        assert!(
            err.contains("relation kind 'link' is the block of no category"),
            "{err}"
        );
        assert!(ff.styles().is_empty());
    }

    /// Fewest wildcards wins, then table order; `Reversible` matches a row
    /// written end-for-end, `Ordered` (impropers) only as written.
    #[test]
    fn assign_terms_prefers_fewest_wildcards_then_table_order_by_endpoint_order() {
        let mut lib = ForceField::new("lib");
        let bond = lib.def_style("bond", "harmonic", Params::new()).unwrap();
        for (name, ends) in [
            ("any", ["", ""]),
            ("x-b", ["X", "B"]),
            ("b-a", ["B", "A"]),
            ("a-b", ["A", "B"]),
        ] {
            bond.def_type(name, &ends, Params::from_iter([("k", 1.0), ("r0", 1.0)]))
                .unwrap();
        }
        let dihedral = lib.def_style("dihedral", "charmm", Params::new()).unwrap();
        dihedral
            .def_type(
                "dcba",
                &["D", "C", "B", "A"],
                Params::from_iter([("k", 1.0)]),
            )
            .unwrap();
        let improper = lib.def_style("improper", "cvff", Params::new()).unwrap();
        improper
            .def_type(
                "dcba",
                &["D", "C", "B", "A"],
                Params::from_iter([("k", 1.0)]),
            )
            .unwrap();
        improper
            .def_type(
                "b-wild",
                &["B", "*", "*", "*"],
                Params::from_iter([("k", 2.0)]),
            )
            .unwrap();

        let mut g = Atomistic::new();
        let ids: Vec<_> = ["A", "B", "C", "D"]
            .iter()
            .map(|t| {
                let id = g
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap();
                g.set_atom(id, "type", *t).unwrap();
                id
            })
            .collect();
        g.add_bond(ids[0], ids[1], props! {}).unwrap(); // A-B: `b-a` and `a-b` tie, `b-a` first
        g.add_bond(ids[1], ids[2], props! {}).unwrap(); // B-C: only wildcards; `x-b` reversed
        g.add_bond(ids[2], ids[3], props! {}).unwrap(); // C-D: only `any`
        g.add_dihedral(ids[0], ids[1], ids[2], ids[3], props! {})
            .unwrap();
        g.add_improper(ids[0], ids[1], ids[2], ids[3], props! {})
            .unwrap();
        g.add_improper(ids[1], ids[0], ids[2], ids[3], props! {})
            .unwrap();

        let mut m = TypeAssignment::default();
        let graph = &g;
        assert_eq!(m.assign_terms(graph, "bonds", &lib, "type"), Ok(vec![]));
        assert_eq!(m.assign_terms(graph, "dihedrals", &lib, "type"), Ok(vec![]));
        // The first improper is A-B-C-D: `dcba` only reversed, refused.
        assert_eq!(
            m.assign_terms(graph, "impropers", &lib, "type"),
            Ok(vec![0])
        );
        // No graph kind, nothing to assign.
        assert_eq!(m.assign_terms(graph, "cmaps", &lib, "type"), Ok(vec![]));

        let names = |kind: &str| -> Vec<Option<String>> {
            m.links[kind]
                .iter()
                .map(|a| match a.first() {
                    Some((_, Annotation::Type { name, .. })) => Some(name.clone()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            names("bonds"),
            [Some("b-a"), Some("x-b"), Some("any")].map(|n| n.map(str::to_owned))
        );
        assert_eq!(names("dihedrals"), vec![Some("dcba".to_owned())]);
        assert_eq!(names("impropers"), vec![None, Some("b-wild".to_owned())]);
        // The styles used, once each, with the library's style params.
        assert_eq!(
            m.styles
                .iter()
                .map(|(c, s, _)| (c.as_str(), s.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("bond", "harmonic"),
                ("dihedral", "charmm"),
                ("improper", "cvff")
            ]
        );

        let mut ff = ForceField::new("out");
        m.apply_to(&mut g, &mut ff).unwrap();
        assert_eq!(
            type_names(&ff, "bond", "harmonic"),
            vec!["b-a", "x-b", "any"]
        );
        assert_eq!(
            ff.get_style("bond", "harmonic")
                .unwrap()
                .type_endpoints("x-b"),
            Some(vec!["X".to_owned(), "B".to_owned()])
        );
    }

    /// CMAP is directional — φ is the dihedral of its first four atoms and ψ
    /// of its last four, the grid's two axes in that order — so its category
    /// is `Ordered`: a row written end-for-end names another term, and a
    /// reversed five-atom match would read φ and ψ off the wrong axes.
    #[test]
    fn a_cmap_row_is_not_matched_reversed() {
        assert_eq!(
            link_category("cmaps"),
            Some(("cmap".to_owned(), EndpointOrder::Ordered))
        );
        let mut lib = ForceField::new("lib");
        lib.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("edcba", &["E", "D", "C", "B", "A"], Params::new())
            .unwrap();
        let mut g = Atomistic::new();
        let ids: Vec<_> = ["A", "B", "C", "D", "E"]
            .iter()
            .map(|t| {
                let id = g
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap();
                g.set_atom(id, "type", *t).unwrap();
                id
            })
            .collect();
        g.add_term("cmaps", &ids, props! {}).unwrap();
        let mut m = TypeAssignment::default();
        // The only row reads E-D-C-B-A; the term is A-B-C-D-E.
        assert_eq!(m.assign_terms(&g, "cmaps", &lib, "type"), Ok(vec![0]));
    }

    /// `Unordered` matches any permutation, wildcards included.
    #[test]
    fn unordered_endpoints_match_any_permutation() {
        use EndpointOrder::{Ordered, Reversible, Unordered};
        let atoms = ["A", "B", "C"];
        assert!(endpoints_match(&["B", "C", "A"], &atoms, Unordered));
        assert!(!endpoints_match(&["B", "C", "A"], &atoms, Reversible));
        assert!(endpoints_match(&["C", "B", "A"], &atoms, Reversible));
        assert!(!endpoints_match(&["C", "B", "A"], &atoms, Ordered));
        assert!(endpoints_match(&["C", "", "B"], &atoms, Unordered));
        assert!(!endpoints_match(&["C", "C", ""], &atoms, Unordered));
        assert!(!endpoints_match(&["A", "B"], &atoms, Unordered));
    }

    /// The category of a relation kind is the registered one whose block it
    /// is, else molrec's `<name>s` rule.
    #[test]
    fn link_category_resolves_blocks_to_categories() {
        register_urey_bradley();
        assert_eq!(
            link_category("impropers"),
            Some(("improper".to_owned(), EndpointOrder::Ordered))
        );
        assert_eq!(
            link_category("urey_bradleys"),
            Some(("urey_bradley".to_owned(), EndpointOrder::Reversible))
        );
        assert_eq!(
            link_category("cross_terms"),
            Some(("cross_term".to_owned(), EndpointOrder::Reversible))
        );
        assert_eq!(link_category("link"), None);
        assert_eq!(link_category("atoms"), None);
    }
}

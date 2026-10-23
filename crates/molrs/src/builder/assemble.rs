//! [`Assembler`]: one placed, linked world graph from a site graph.
//!
//! `assemble` is a composed operation by operator ruling (notes.md
//! 2026-09-27): it supersedes the 2026-09-26 "primitives only" ruling for
//! this one concern.

use std::collections::HashMap;
use std::fmt;

use crate::builder::orient::direction_fit;
use crate::builder::{OrientError, Orienter, SiteLink, SiteView};
use crate::builder::{ParentJoin, PlaceError, PlaceSite, Placer};
use crate::core::CenterError;
use crate::core::CoarseGrain;
use crate::core::LinkManyError;
use crate::core::MolGraph;
use crate::core::MolRsError;
use crate::core::NodeId;
use crate::core::Port;
use crate::core::RelationId;
use crate::core::keys;
use crate::core::{Graph, GraphAccess};
use crate::op::I;
use crate::op::Vec3;
use crate::op::vec3::sub;
use crate::op::{Rigid, transform_point};

/// The most port assignments tried at one site before it is refused.
const MAX_ASSIGNMENTS: usize = 5040;

/// Why [`Assembler::assemble`] refused its input.
///
/// `site` is a 0-based ordinal of the site graph's beads, in
/// [`bead_ids`](CoarseGrain::bead_ids) order — the
/// unit's `frag_id`.
#[derive(Debug)]
pub enum AssembleError {
    /// The site count exceeds [`i32::MAX`], so `frag_id` or `mol_id` would
    /// not fit its `i32` node column.
    TooManyUnits {
        /// The site count.
        units: usize,
    },
    /// The site graph does not read back: a site lacks its `bead_type`, its
    /// position, or (when any site has one) its axis.
    Sites(MolRsError),
    /// Site `site` names a template the library lacks.
    UnknownName {
        /// The site.
        site: usize,
        /// The name it carries.
        name: String,
    },
    /// A port of template `name` does not read back.
    Template {
        /// The template name.
        name: String,
        /// Why the port does not read back.
        source: MolRsError,
    },
    /// No assignment of template ports to the bonds of site `site` exists:
    /// too few ports, no port accepting a partner's, or too many choices.
    Ports {
        /// The site.
        site: usize,
        /// The site's template name.
        name: String,
        /// What is missing.
        reason: String,
    },
    /// The orienter refused the copies of template `name`; `site` is the
    /// site it names, or the group's first site when the error names none.
    Orient {
        /// The template name.
        name: String,
        /// The site.
        site: usize,
        /// The orienter's refusal.
        source: OrientError,
    },
    /// The placer refused the copies of template `name`; `site` is the site
    /// it names, or the group's first site when the error names none.
    Place {
        /// The template name.
        name: String,
        /// The site.
        site: usize,
        /// The placer's refusal.
        source: PlaceError,
    },
    /// The finished world is not a valid graph of the requested output type.
    Output(MolRsError),
    /// The world refused the copies of template `name` (for example a column
    /// type that contradicts another template's), or a copy lost a port.
    Replicate {
        /// The template name.
        name: String,
        /// The world's refusal.
        source: MolRsError,
    },
    /// The batch join refused the bond between sites `site` and `partner`
    /// (for a two-pair refusal, its `first` pair).
    Link {
        /// One end of the bond.
        site: usize,
        /// The other end.
        partner: usize,
        /// The batch join's refusal; its pair indices count site bonds in
        /// the site graph's bond order.
        source: LinkManyError,
    },
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyUnits { units } => write!(
                f,
                "{units} sites exceed {}, the widest id a node column stores",
                I::MAX
            ),
            Self::Sites(e) => write!(f, "the site graph does not read back: {e}"),
            Self::UnknownName { site, name } => {
                write!(f, "site {site} names '{name}', which the library lacks")
            }
            Self::Template { name, source } => {
                write!(
                    f,
                    "a port of template '{name}' does not read back: {source}"
                )
            }
            Self::Ports { site, name, reason } => {
                write!(
                    f,
                    "site {site} ('{name}') has no port for every bond: {reason}"
                )
            }
            Self::Orient { name, site, source } => {
                write!(f, "site {site} ('{name}') cannot be oriented: {source}")
            }
            Self::Place { name, site, source } => {
                write!(f, "site {site} ('{name}') cannot be placed: {source}")
            }
            Self::Output(e) => write!(f, "the world is not a graph of the requested type: {e}"),
            Self::Replicate { name, source } => {
                write!(f, "the copies of template '{name}' were refused: {source}")
            }
            Self::Link {
                site,
                partner,
                source,
            } => write!(f, "site {site} cannot join site {partner}: {source}"),
        }
    }
}

impl std::error::Error for AssembleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sites(source)
            | Self::Output(source)
            | Self::Template { source, .. }
            | Self::Replicate { source, .. } => Some(source),
            Self::Orient { source, .. } => Some(source),
            Self::Place { source, .. } => Some(source),
            Self::Link { source, .. } => Some(source),
            Self::TooManyUnits { .. } | Self::UnknownName { .. } | Self::Ports { .. } => None,
        }
    }
}

/// One port of a template, read once.
struct TemplatePort {
    id: RelationId,
    port: Port,
    anchor_row: usize,
    handle_row: usize,
    /// Anchor and handle positions in the template (Å), when both have them.
    atoms: Option<(Vec3, Vec3)>,
    /// Centre of mass → handle (Å); `None` when the template has no centre.
    direction: Option<Vec3>,
}

/// The sites that use one template: its copies, in site order.
struct Group<'a, G> {
    name: &'a str,
    template: &'a G,
    ports: Vec<TemplatePort>,
    /// Site ordinals, one per copy.
    sites: Vec<usize>,
}

/// The site graph as the assembler reads it.
struct SiteGraph {
    names: Vec<String>,
    /// Site positions (Å), when the site graph carries them.
    positions: Option<Vec<Vec3>>,
    axes: Option<Vec<Vec3>>,
    /// Site bonds as ordinal pairs, in the site graph's bond order.
    bonds: Vec<(usize, usize)>,
    /// Per site, its `(partner, bond)` incidences.
    incident: Vec<Vec<(usize, usize)>>,
}

impl SiteGraph {
    fn read(sites: &CoarseGrain) -> Result<Self, AssembleError> {
        let ids: Vec<NodeId> = sites.bead_ids().collect();
        let n = ids.len();
        if I::try_from(n).is_err() {
            return Err(AssembleError::TooManyUnits { units: n });
        }
        let names = sites.bead_types(&ids).map_err(AssembleError::Sites)?;
        let table = sites.bead_table();
        let carried = |key: &str| ids.iter().any(|&id| table.has(id, key));
        let positions = if carried(keys::X) {
            Some(sites.positions(&ids).map_err(AssembleError::Sites)?)
        } else {
            None
        };
        let axes = if carried(keys::AXIS[0]) {
            Some(sites.axes(&ids).map_err(AssembleError::Sites)?)
        } else {
            None
        };
        let ordinal: HashMap<NodeId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut bonds = Vec::with_capacity(sites.n_bonds());
        let mut incident = vec![Vec::new(); n];
        for (_, bond) in sites.bonds() {
            let (u, v) = (ordinal[&bond.nodes[0]], ordinal[&bond.nodes[1]]);
            incident[u].push((v, bonds.len()));
            incident[v].push((u, bonds.len()));
            bonds.push((u, v));
        }
        Ok(Self {
            names,
            positions,
            axes,
            bonds,
            incident,
        })
    }

    /// Which end (0 or 1) of bond `b` site `u` is.
    fn end(&self, b: usize, u: usize) -> usize {
        usize::from(self.bonds[b].0 != u)
    }
}

/// One molecule's walk: sites in breadth-first order, each with the
/// `(parent, bond)` it was reached through.
type Walk = Vec<(usize, Option<(usize, usize)>)>;

/// Why a site's ports were not chosen.
enum PortChoice {
    /// No assignment fits the site.
    Refused(AssembleError),
    /// The forced choice index is past the last choice.
    Missing,
}

/// Builds every molecule of a site graph as one placed, linked world graph,
/// returned as the graph type the caller names.
///
/// The assembler holds a library (name → one template of the graph type `G`
/// — a [`MolGraph`], an [`Atomistic`](crate::core::Atomistic) or a
/// [`CoarseGrain`], with or without ports), a [`Placer`] and an
/// [`Orienter`] of that type. Its one verb,
/// [`assemble`](Assembler::assemble), reads a site graph (a [`CoarseGrain`]
/// whose beads are the sites and whose bonds say which sites join) and
/// builds every molecule in it in one call, as one world of the library's
/// type `G`: one template copy per site,
/// turned and placed, the copies of bonded sites joined through their ports,
/// each atom stamped with its unit (`frag_id`) and its molecule (`mol_id`).
/// Any topology is accepted: chains, branches and rings (operator,
/// 2026-09-28).
///
/// # Sites
///
/// Each bead of the site graph is one unit: its `bead_type` names the
/// library template and each bond joins two units. A position `p` (Å) and an
/// axis ([`CoarseGrain::axes`]) are optional: a site graph read from a CG
/// model carries them, one written from a CGsmiles topology
/// (`CgSmilesIr::to_coarsegrain`) does not.
///
/// # Ports
///
/// Every site bond joins one port of each end's copy, the two ports
/// accepting each other ([`Port::accepts`]: `<` with `>`, `$` with `$`,
/// equal label and order). Each connected component is walked breadth-first
/// from its lowest-degree site (lowest ordinal on a tie); at each site the
/// ports of all its bonds are chosen together, one distinct port per bond, a
/// bond whose partner already chose requiring a port that accepts the
/// partner's. When several choices remain, the one whose port directions
/// (centroid → handle) best fit the bond directions (site → partner)
/// is taken — the first in port order on a tie or without positions. When a
/// walk fails, it is retried from the first site's next choice (a chain
/// entered from its wrong end), and the first failure is reported once every
/// choice fails. Ports left without a bond stay on the world, as a chain's
/// end ports do.
///
/// # Placement
///
/// The optional orienter turns each copy about its template's centre of
/// mass (it needs positions); the placer then gives each copy its pose,
/// sites visited in walk order so a copy's parent is placed first.
/// [`SitePlacer`] puts the centroid on the site; [`GrowthPlacer`]
/// joins each copy to its parent's port and needs no position. Relax the
/// result before use.
///
/// # Ids
///
/// - `frag_id` = the site's ordinal (0-based).
/// - `mol_id` = the ordinal of the site's connected component + 1,
///   components numbered by their lowest site ordinal.
///
/// # World layout
///
/// Atoms are grouped by template name, in the order names first appear, and
/// copy-major within a name, until the batch join swap-removes the leaving
/// groups and refills their rows from the end. Read a unit's atoms by
/// `frag_id`, never by row.
///
/// **Partial columns.** Templates may differ in their optional columns; the
/// world then holds those columns for some atoms only, and `to_frame` writes
/// 0.0 in the rows without the prop
/// (routed `/mol:fix`, notes.md 2026-09-27).
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use molrs::builder::{Assembler, GrowthPlacer};
/// use molrs::core::keys;
/// use molrs::core::BondNumber;
/// use molrs::core::CoarseGrain;
/// use molrs::core::Atomistic;
/// use molrs::core::PortKind;
/// use molrs::props;
/// use molrs::core::Element;
///
/// // Joining carbons C0 (`<`, hydrogen on −x) and C1 (`>`, hydrogen on +x).
/// let mut unit = Atomistic::new();
/// let c0 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let c1 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 1.5,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let h0 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => -1.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let h1 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => 2.5,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// for (a, b) in [(c0, c1), (c0, h0), (c1, h1)] {
///     unit.add_bond(a, b, props! {}).unwrap();
/// }
/// unit.add_port(c0, h0, PortKind::Left, "", BondNumber::Single).unwrap();
/// unit.add_port(c1, h1, PortKind::Right, "", BondNumber::Single).unwrap();
///
/// // A three-site chain with no positions: grown copy by copy.
/// let mut sites = CoarseGrain::new();
/// let ids: Vec<_> = (0..3).map(|_| sites.add_bead(props! {
///     keys::BEAD_TYPE => "U",
/// }).unwrap()).collect();
/// sites.add_bond(ids[0], ids[1], props! {}).unwrap();
/// sites.add_bond(ids[1], ids[2], props! {}).unwrap();
///
/// let assembler = Assembler::new(
///     HashMap::from([("U".to_owned(), unit)]),
///     Box::new(GrowthPlacer::new()),
///     None,
/// );
/// // The world is an `Atomistic`, the type of the library's templates.
/// let world = assembler.assemble(&sites).unwrap();
///
/// // Three copies of 4 atoms, two links remove 4 hydrogens.
/// assert_eq!(world.n_atoms(), 8);
/// assert_eq!(world.n_ports(), 2);
/// ```
///
/// [`SitePlacer`]: crate::builder::SitePlacer
/// [`GrowthPlacer`]: crate::builder::GrowthPlacer
pub struct Assembler<G: Graph> {
    library: HashMap<String, G>,
    placer: Box<dyn Placer<G>>,
    orienter: Option<Box<dyn Orienter<G>>>,
}

impl<G: Graph> Assembler<G> {
    /// An assembler over `library` (name → template graph, every template of
    /// the one graph type `G` the world is built as), giving each copy its
    /// pose with `placer` after turning it with `orienter`, when given. A
    /// portless molecule is a template with no ports.
    pub fn new(
        library: HashMap<String, G>,
        placer: Box<dyn Placer<G>>,
        orienter: Option<Box<dyn Orienter<G>>>,
    ) -> Self {
        Self {
            library,
            placer,
            orienter,
        }
    }

    /// One copy of `library[bead_type]` per site of `sites`, turned, placed
    /// and joined along the site bonds; returns the world as a `G`, the
    /// library's graph type. See the type docs for the rules. An empty site
    /// graph gives an empty world.
    ///
    /// Each unit is a rigid copy of its template, so a [`CoarseGrain`]
    /// template's bead membership is not copied: it names atoms of a
    /// separate all-atom world that the copies do not own (as
    /// [`CoarseGrain::replicate`] states).
    ///
    /// # Errors
    ///
    /// Checked before the first copy is placed, the first offender in site
    /// order: [`TooManyUnits`](AssembleError::TooManyUnits),
    /// [`Sites`](AssembleError::Sites),
    /// [`UnknownName`](AssembleError::UnknownName),
    /// [`Template`](AssembleError::Template) and, walking each component,
    /// [`Ports`](AssembleError::Ports). While building:
    /// [`Orient`](AssembleError::Orient), [`Place`](AssembleError::Place),
    /// [`Replicate`](AssembleError::Replicate),
    /// [`Link`](AssembleError::Link) and, last,
    /// [`Output`](AssembleError::Output).
    ///
    /// # Complexity
    ///
    /// O(S + B + A) for S sites, B site bonds and A world atoms, times the
    /// port choices tried at a site (one along a `<`/`>` chain, at most 5040)
    /// and the walks retried per component (two for a `<`/`>` chain).
    pub fn assemble(&self, sites: &CoarseGrain) -> Result<G, AssembleError> {
        let graph = SiteGraph::read(sites)?;
        let n = graph.names.len();

        // ---- group sites by template ----
        let mut group_of_name: HashMap<&str, usize> = HashMap::new();
        let mut groups: Vec<Group<'_, G>> = Vec::new();
        let mut group_of: Vec<(usize, usize)> = Vec::with_capacity(n);
        for (site, name) in graph.names.iter().enumerate() {
            let g = match group_of_name.get(name.as_str()) {
                Some(&g) => g,
                None => {
                    let (key, template) = self.library.get_key_value(name).ok_or_else(|| {
                        AssembleError::UnknownName {
                            site,
                            name: name.clone(),
                        }
                    })?;
                    group_of_name.insert(key, groups.len());
                    groups.push(Group {
                        name: key,
                        template,
                        ports: Self::template_ports(key, template.graph())?,
                        sites: Vec::new(),
                    });
                    groups.len() - 1
                }
            };
            group_of.push((g, groups[g].sites.len()));
            groups[g].sites.push(site);
        }

        // ---- ports: one per bond end, each component walked breadth-first ----
        let mut chosen: Vec<[Option<usize>; 2]> = vec![[None, None]; graph.bonds.len()];
        let mut mol_of = vec![usize::MAX; n];
        let mut walks: Vec<Walk> = Vec::new();
        for start in 0..n {
            if mol_of[start] != usize::MAX {
                continue;
            }
            let members = Self::component(start, &graph, &mut mol_of, walks.len());
            walks.push(Self::walk(
                &members,
                &groups,
                &group_of,
                &graph,
                &mut chosen,
            )?);
        }
        let port_of = |u: usize, b: usize| -> &TemplatePort {
            let p = chosen[b][graph.end(b, u)].expect("every bond end was chosen");
            &groups[group_of[u].0].ports[p]
        };

        // ---- orient: one call per name ----
        let mut turns = vec![Rigid::IDENTITY; n];
        if let Some(orienter) = self.orienter.as_ref().filter(|_| n > 0) {
            let positions = graph.positions.as_ref().ok_or_else(|| {
                AssembleError::Sites(MolRsError::validation(
                    "the orienter needs site positions; the site graph has none",
                ))
            })?;
            for group in &groups {
                let link_lists: Vec<Vec<SiteLink>> = group
                    .sites
                    .iter()
                    .map(|&u| {
                        graph.incident[u]
                            .iter()
                            .map(|&(v, b)| SiteLink {
                                port: port_of(u, b).id,
                                toward: positions[v],
                            })
                            .collect()
                    })
                    .collect();
                let views: Vec<SiteView<'_>> = group
                    .sites
                    .iter()
                    .zip(&link_lists)
                    .map(|(&u, links)| SiteView {
                        position: positions[u],
                        axis: graph.axes.as_ref().map(|a| a[u]),
                        links,
                    })
                    .collect();
                let group_turns =
                    orienter
                        .orient_many(group.template, &views)
                        .map_err(|source| {
                            let at = match source {
                                OrientError::NoAxis { index } | OrientError::Frame { index } => {
                                    index
                                }
                                OrientError::Template(_) | OrientError::Center(_) => 0,
                            };
                            AssembleError::Orient {
                                name: group.name.to_owned(),
                                site: group.sites.get(at).copied().unwrap_or(group.sites[0]),
                                source,
                            }
                        })?;
                for (&u, turn) in group.sites.iter().zip(group_turns) {
                    turns[u] = turn;
                }
            }
        }

        // ---- place: parents before children ----
        let mut poses = vec![Rigid::IDENTITY; n];
        for walk in &walks {
            for &(u, parent) in walk {
                let group = &groups[group_of[u].0];
                let place_err = |source| AssembleError::Place {
                    name: group.name.to_owned(),
                    site: u,
                    source,
                };
                let parent = match parent {
                    None => None,
                    Some((p, b)) => {
                        let (anchor, handle) = port_of(p, b).atoms.ok_or_else(|| {
                            place_err(PlaceError::Port(format!(
                                "the port of site {p} toward site {u} has an atom without x/y/z"
                            )))
                        })?;
                        Some(ParentJoin {
                            port: port_of(u, b).id,
                            anchor: transform_point(&poses[p], anchor),
                            handle: transform_point(&poses[p], handle),
                        })
                    }
                };
                let site = PlaceSite {
                    position: graph.positions.as_ref().map(|ps| ps[u]),
                    turn: turns[u],
                    parent,
                };
                poses[u] = self
                    .placer
                    .place(group.template, &site)
                    .map_err(place_err)?;
            }
        }

        // ---- replicate and stamp mol_id: one pass per name ----
        // Each copy is a `MolGraph::replicate` of its template, so a
        // `CoarseGrain` template's bead membership is not carried: the
        // membership names foreign atoms the copies do not own (as
        // `CoarseGrain::replicate` states).
        let mut world = MolGraph::new();
        let mut copies: Vec<Vec<NodeId>> = Vec::with_capacity(groups.len());
        for group in &groups {
            let rigids: Vec<Rigid> = group.sites.iter().map(|&u| poses[u]).collect();
            let replicate_err = |source| AssembleError::Replicate {
                name: group.name.to_owned(),
                source,
            };
            let frag_ids: Vec<I> = group
                .sites
                .iter()
                .map(|&u| I::try_from(u).expect("TooManyUnits bounds every site ordinal"))
                .collect();
            let template = group.template.graph();
            let atoms = world
                .replicate(template, &rigids, &frag_ids)
                .map_err(replicate_err)?;
            let per_copy = template.n_nodes();
            for (c, &u) in group.sites.iter().enumerate() {
                let mol_id =
                    I::try_from(mol_of[u] + 1).expect("TooManyUnits bounds every component");
                for &atom in &atoms[c * per_copy..(c + 1) * per_copy] {
                    world
                        .set_node(atom, keys::MOL_ID, mol_id)
                        .map_err(replicate_err)?;
                }
            }
            copies.push(atoms);
        }

        // ---- one scan of the world's ports ----
        let mut world_ports: HashMap<(NodeId, NodeId), RelationId> =
            HashMap::with_capacity(world.n_ports());
        for id in world.port_ids() {
            if let Ok(port) = world.get_port(id) {
                world_ports.insert((port.anchor, port.handle), id);
            }
        }
        let world_port = |u: usize, b: usize| -> Result<RelationId, AssembleError> {
            let (g, c) = group_of[u];
            let group = &groups[g];
            let tp = port_of(u, b);
            let base = c * group.template.graph().n_nodes();
            let key = (
                copies[g][base + tp.anchor_row],
                copies[g][base + tp.handle_row],
            );
            world_ports
                .get(&key)
                .copied()
                .ok_or_else(|| AssembleError::Replicate {
                    name: group.name.to_owned(),
                    source: MolRsError::validation(format!("copy {c} lost a port in the world")),
                })
        };

        // ---- one batch join along the site bonds ----
        let mut pairs: Vec<(RelationId, RelationId)> = Vec::with_capacity(graph.bonds.len());
        for (b, &(u, v)) in graph.bonds.iter().enumerate() {
            pairs.push((world_port(u, b)?, world_port(v, b)?));
        }
        world.link_many(&pairs).map_err(|source| {
            let pair = match &source {
                LinkManyError::Pair { pair, .. } => *pair,
                LinkManyError::PortReused { first, .. }
                | LinkManyError::DuplicateBond { first, .. }
                | LinkManyError::BranchesOverlap { first, .. } => *first,
            };
            let (site, partner) = graph.bonds[pair];
            AssembleError::Link {
                site,
                partner,
                source,
            }
        })?;

        G::from_graph(world).map_err(AssembleError::Output)
    }

    /// Every site of `start`'s connected component, each stamped `mol` in
    /// `mol_of`.
    fn component(start: usize, graph: &SiteGraph, mol_of: &mut [usize], mol: usize) -> Vec<usize> {
        let mut members = vec![start];
        mol_of[start] = mol;
        let mut i = 0;
        while i < members.len() {
            for &(v, _) in &graph.incident[members[i]] {
                if mol_of[v] == usize::MAX {
                    mol_of[v] = mol;
                    members.push(v);
                }
            }
            i += 1;
        }
        members
    }

    /// Assign the ports of one component, walking breadth-first from its
    /// lowest-degree site. A walk that fails at a later site is retried with
    /// the first site's next choice; once those run out, or when the first
    /// site itself has no choice, the first failure is returned.
    fn walk(
        members: &[usize],
        groups: &[Group<'_, G>],
        group_of: &[(usize, usize)],
        graph: &SiteGraph,
        chosen: &mut [[Option<usize>; 2]],
    ) -> Result<Walk, AssembleError> {
        let root = members
            .iter()
            .copied()
            .min_by_key(|&s| (graph.incident[s].len(), s))
            .expect("a component holds its start");
        let mut first_error: Option<AssembleError> = None;
        for pick in 0.. {
            for &u in members {
                for &(_, b) in &graph.incident[u] {
                    chosen[b] = [None, None];
                }
            }
            match Self::try_walk(root, pick, groups, group_of, graph, chosen) {
                Ok(walk) => return Ok(walk),
                Err(Some(e)) if pick == 0 || first_error.is_none() => {
                    let later = matches!(&e, AssembleError::Ports { site, .. } if *site != root);
                    if !later {
                        return Err(first_error.unwrap_or(e));
                    }
                    first_error.get_or_insert(e);
                }
                Err(Some(_)) => {}
                // The first site has no `pick`-th choice: every choice failed.
                Err(None) => break,
            }
        }
        Err(first_error.expect("a failed walk left its error"))
    }

    /// One breadth-first walk from `root`, whose ports take their `pick`-th
    /// choice. `Err(None)` when `root` has no such choice.
    fn try_walk(
        root: usize,
        pick: usize,
        groups: &[Group<'_, G>],
        group_of: &[(usize, usize)],
        graph: &SiteGraph,
        chosen: &mut [[Option<usize>; 2]],
    ) -> Result<Walk, Option<AssembleError>> {
        let mut walk: Walk = vec![(root, None)];
        let mut seen = vec![false; graph.names.len()];
        seen[root] = true;
        let mut i = 0;
        while i < walk.len() {
            let u = walk[i].0;
            let pick = (u == root).then_some(pick);
            match Self::choose_ports(u, groups, group_of, graph, chosen, pick) {
                Ok(()) => {}
                Err(PortChoice::Missing) => return Err(None),
                Err(PortChoice::Refused(e)) => return Err(Some(e)),
            }
            for &(v, b) in &graph.incident[u] {
                if !seen[v] {
                    seen[v] = true;
                    walk.push((v, Some((u, b))));
                }
            }
            i += 1;
        }
        Ok(walk)
    }

    /// Choose one distinct port of site `u`'s template for each of its
    /// bonds, honouring the ports its partners already chose; `pick` forces
    /// the `pick`-th choice (the walk's first site on a retry).
    fn choose_ports(
        u: usize,
        groups: &[Group<'_, G>],
        group_of: &[(usize, usize)],
        graph: &SiteGraph,
        chosen: &mut [[Option<usize>; 2]],
        pick: Option<usize>,
    ) -> Result<(), PortChoice> {
        let links = &graph.incident[u];
        if links.is_empty() {
            return if pick.unwrap_or(0) == 0 {
                Ok(())
            } else {
                Err(PortChoice::Missing)
            };
        }
        let ports = &groups[group_of[u].0].ports;
        let refuse = |reason: String| {
            PortChoice::Refused(AssembleError::Ports {
                site: u,
                name: graph.names[u].clone(),
                reason,
            })
        };
        if links.len() > ports.len() {
            return Err(refuse(format!(
                "{} bonds but the template has {} ports",
                links.len(),
                ports.len()
            )));
        }
        // Per bond: the partner's chosen port, which this end must accept,
        // and the partner template's ports, one of which must accept this
        // end's port when the partner has not chosen yet.
        let partner: Vec<Partner<'_>> = links
            .iter()
            .map(|&(v, b)| {
                let theirs = &groups[group_of[v].0].ports;
                match chosen[b][graph.end(b, v)] {
                    Some(p) => Partner::Chosen(&theirs[p].port),
                    None => Partner::Open(theirs),
                }
            })
            .collect();

        let mut candidates: Vec<Vec<usize>> = Vec::new();
        let mut used = vec![false; ports.len()];
        let mut current = Vec::with_capacity(links.len());
        if !enumerate(&partner, ports, &mut used, &mut current, &mut candidates) {
            return Err(refuse(format!(
                "more than {MAX_ASSIGNMENTS} port choices; label the ports to narrow them"
            )));
        }
        let count = candidates.len();
        let best = match (count, pick) {
            (0, _) => {
                return Err(refuse(
                    "no distinct ports accept the ports its partners chose".to_owned(),
                ));
            }
            (_, Some(k)) if k >= count => return Err(PortChoice::Missing),
            (_, Some(k)) => candidates.swap_remove(k),
            (1, None) => candidates.swap_remove(0),
            (_, None) => {
                let targets: Option<Vec<Vec3>> = graph
                    .positions
                    .as_ref()
                    .map(|ps| links.iter().map(|&(v, _)| sub(ps[v], ps[u])).collect());
                let mut best: Option<(f64, usize)> = None;
                for (i, cand) in candidates.iter().enumerate() {
                    let rmsd = targets
                        .as_ref()
                        .and_then(|t| {
                            cand.iter()
                                .map(|&p| ports[p].direction)
                                .collect::<Option<Vec<Vec3>>>()
                                .and_then(|r| direction_fit(&r, t))
                        })
                        .map_or(f64::INFINITY, |(_, rmsd)| rmsd);
                    if best.is_none_or(|(b, _)| rmsd < b) {
                        best = Some((rmsd, i));
                    }
                }
                candidates.swap_remove(best.expect("two or more candidates").1)
            }
        };
        for (&(_, b), p) in links.iter().zip(best) {
            chosen[b][graph.end(b, u)] = Some(p);
        }
        Ok(())
    }

    /// Every port of `template`, with its anchor and handle rows and
    /// positions and its direction from the centroid.
    fn template_ports(name: &str, template: &MolGraph) -> Result<Vec<TemplatePort>, AssembleError> {
        let refuse = |source| AssembleError::Template {
            name: name.to_owned(),
            source,
        };
        let row = |atom: NodeId| {
            template.node_table().row(atom).ok_or_else(|| {
                refuse(MolRsError::validation(format!(
                    "a port names {atom:?}, which is no live atom"
                )))
            })
        };
        let position = |atom: NodeId| template.get_node(atom).ok().and_then(|a| a.position());
        // A template without coordinates has no geometry, so no centre.
        let center = match template.centroid(&template.node_ids().collect::<Vec<_>>()) {
            Ok(c) => Some(c),
            Err(CenterError::Empty | CenterError::BadPosition { .. }) => None,
            Err(e) => {
                return Err(refuse(MolRsError::validation(format!("its centroid: {e}"))));
            }
        };
        let mut out = Vec::new();
        for id in template.port_ids() {
            let port = template.get_port(id).map_err(refuse)?;
            let handle = position(port.handle);
            out.push(TemplatePort {
                id,
                anchor_row: row(port.anchor)?,
                handle_row: row(port.handle)?,
                atoms: position(port.anchor).zip(handle),
                direction: center.zip(handle).map(|(c, h)| sub(h, c)),
                port,
            });
        }
        Ok(out)
    }
}

/// The other end of a bond, as a site choosing its ports sees it.
enum Partner<'a> {
    /// The partner already chose this port.
    Chosen(&'a Port),
    /// The partner has not chosen; these are its template's ports.
    Open(&'a [TemplatePort]),
}

impl Partner<'_> {
    /// Whether `port` on this end can join the partner.
    fn admits(&self, port: &Port) -> bool {
        match self {
            Self::Chosen(theirs) => port.accepts(theirs),
            Self::Open(theirs) => theirs.iter().any(|t| port.accepts(&t.port)),
        }
    }
}

/// Collect every injective choice of one port per bond into `out`, bond `k`
/// taking a port that `partner[k]` admits (the chosen partner port accepts
/// it, or an open partner has a port that does). `false` when more than
/// [`MAX_ASSIGNMENTS`] choices exist.
fn enumerate(
    partner: &[Partner<'_>],
    ports: &[TemplatePort],
    used: &mut [bool],
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) -> bool {
    let k = current.len();
    if k == partner.len() {
        if out.len() == MAX_ASSIGNMENTS {
            return false;
        }
        out.push(current.clone());
        return true;
    }
    for (p, tp) in ports.iter().enumerate() {
        if used[p] || !partner[k].admits(&tp.port) {
            continue;
        }
        used[p] = true;
        current.push(p);
        let ok = enumerate(partner, ports, used, current, out);
        current.pop();
        used[p] = false;
        if !ok {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::core::Element;
    use crate::props;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{AssembleError, Assembler};
    use crate::builder::{AxisOrienter, OrientError, Orienter, SiteView};
    use crate::builder::{GrowthPlacer, PlaceError, PlaceSite, Placer, SitePlacer};
    use crate::core::Atomistic;
    use crate::core::BondNumber;
    use crate::core::CoarseGrain;
    use crate::core::NodeId;
    use crate::core::PortKind;
    use crate::core::keys;
    use crate::op::I;
    use crate::op::Vec3;
    use crate::op::{Rigid, rotation_about};

    const TOL: f64 = 1e-9;

    // ---- fixtures ----------------------------------------------------------
    //
    // Monomer M: C0 (0,0,0, m 12), C1 (1.5,0,0, m 12), H0 (−1,0,0, m 1),
    // H1 (2.5,0,0, m 1), O (0.75,1.2,0, m 16); bonds C0–C1, C0–H0, C1–H1,
    // C0–O; ports (C0, H0, `<`) and (C1, H1, `>`), unlabelled, Single.
    //
    // A link removes the two handles (−2 atoms, −2 handle bonds, −2 ports)
    // and adds one C1–C0 bond. An n-unit M chain therefore has
    // 5n − 2(n−1) atoms, 4n − 2(n−1) + (n−1) bonds and 2 ports.

    fn atom(f: &mut Atomistic, element: Element, xyz: Vec3, mass: f64) -> NodeId {
        let id = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => element,
                keys::X => xyz[0],
                keys::Y => xyz[1],
                keys::Z => xyz[2],
            })
            .unwrap();
        f.set_atom(id, keys::MASS, mass).expect("stamp mass");
        id
    }

    /// Monomer M with its `<` port labelled `left` and `>` labelled `right`.
    fn monomer_labelled(left: &str, right: &str) -> Atomistic {
        let mut m = Atomistic::new();
        let c0 = atom(&mut m, Element::C, [0.0, 0.0, 0.0], 12.0);
        let c1 = atom(&mut m, Element::C, [1.5, 0.0, 0.0], 12.0);
        let h0 = atom(&mut m, Element::H, [-1.0, 0.0, 0.0], 1.0);
        let h1 = atom(&mut m, Element::H, [2.5, 0.0, 0.0], 1.0);
        let o = atom(&mut m, Element::O, [0.75, 1.2, 0.0], 16.0);
        for (a, b) in [(c0, c1), (c0, h0), (c1, h1), (c0, o)] {
            m.add_bond(a, b, props! {}).expect("bond");
        }
        m.add_port(c0, h0, PortKind::Left, left, BondNumber::Single)
            .expect("port <");
        m.add_port(c1, h1, PortKind::Right, right, BondNumber::Single)
            .expect("port >");
        m
    }

    /// Li: one atom, no port.
    fn lithium() -> Atomistic {
        let mut li = Atomistic::new();
        atom(&mut li, Element::Li, [0.0; 3], 6.94);
        li
    }

    /// `n` `$` ports on one C, one H handle each, along ±x, ±y, …
    fn hub(n: usize) -> Atomistic {
        hub_of(n, PortKind::Symmetric)
    }

    /// `n` ports of `kind` on one C, one H handle each, along ±x, ±y, …
    fn hub_of(n: usize, kind: PortKind) -> Atomistic {
        let mut f = Atomistic::new();
        let c = atom(&mut f, Element::C, [0.0; 3], 12.0);
        let dirs = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        for d in dirs.iter().take(n) {
            let h = atom(&mut f, Element::H, *d, 1.0);
            f.add_bond(c, h, props! {}).expect("C–H");
            f.add_port(c, h, kind, "", BondNumber::Single)
                .expect("port");
        }
        f
    }

    fn library() -> HashMap<String, Atomistic> {
        HashMap::from([
            ("M".to_owned(), monomer_labelled("", "")),
            ("P".to_owned(), monomer_labelled("x", "y")),
            ("Li".to_owned(), lithium()),
            ("E".to_owned(), hub(1)),
            ("S".to_owned(), hub(2)),
            ("X".to_owned(), hub(3)),
        ])
    }

    /// A site graph: `(name, position)` per site, bonds by ordinal, the same
    /// axis on every site when given.
    fn graph(beads: &[(&str, Vec3)], bonds: &[(usize, usize)], axis: Option<Vec3>) -> CoarseGrain {
        let mut cg = CoarseGrain::new();
        let ids: Vec<_> = beads
            .iter()
            .map(|(name, p)| {
                let id = cg
                    .add_bead(props! {
                        keys::BEAD_TYPE => *name,
                        keys::X => p[0],
                        keys::Y => p[1],
                        keys::Z => p[2],
                    })
                    .unwrap();
                if let Some(a) = axis {
                    for (key, v) in keys::AXIS.into_iter().zip(a) {
                        cg.set_bead(id, key, v).expect("axis");
                    }
                }
                id
            })
            .collect();
        for &(i, j) in bonds {
            cg.add_bond(ids[i], ids[j], props! {}).expect("site bond");
        }
        cg
    }

    /// Turns nothing, so a test of ports, ids or links sees no rotation.
    struct Unturned;

    impl Orienter<Atomistic> for Unturned {
        fn orient_many(
            &self,
            _template: &Atomistic,
            sites: &[SiteView<'_>],
        ) -> Result<Vec<Rigid>, OrientError> {
            Ok(vec![Rigid::IDENTITY; sites.len()])
        }
    }

    fn assembler() -> Assembler<Atomistic> {
        Assembler::new(
            library(),
            Box::new(SitePlacer::new()),
            Some(Box::new(Unturned)),
        )
    }

    fn mol_ids(world: &Atomistic) -> Vec<u64> {
        let frame = world.to_frame().expect("the world emits a frame");
        let atoms = frame.get("atoms").expect("an atoms block");
        atoms
            .get(keys::MOL_ID)
            .and_then(|c| c.get::<u64>())
            .expect("a uint mol_id column")
            .iter()
            .copied()
            .collect()
    }

    /// Placement reads geometry and topology only: a template carrying a
    /// mass — even a nonsensical one — assembles the world it assembles
    /// without one.
    #[test]
    fn a_templates_mass_plays_no_part() {
        let world_of = |mass: Option<f64>| {
            let mut m = monomer_labelled("", "");
            if let Some(mass) = mass {
                let first = m.atoms().next().expect("an atom").0;
                m.set_atom(first, keys::MASS, mass).expect("stamp mass");
            }
            let mut lib = library();
            lib.insert("M".to_owned(), m);
            let assembler =
                Assembler::new(lib, Box::new(SitePlacer::new()), Some(Box::new(Unturned)));
            let sites = graph(&[("M", [0.0; 3]), ("M", [5.0, 0.0, 0.0])], &[(0, 1)], None);
            let world = assembler.assemble(&sites).expect("assembled");
            world
                .atoms()
                .map(|(_, a)| a.position().expect("placed"))
                .collect::<Vec<_>>()
        };
        assert_eq!(world_of(Some(-1.0)), world_of(None));
    }

    // ---- builds --------------------------------------------------------------

    #[test]
    fn assemble_links_a_three_site_chain_into_one_molecule() {
        let sites = graph(
            &[
                ("M", [0.0; 3]),
                ("M", [5.0, 0.0, 0.0]),
                ("M", [10.0, 0.0, 0.0]),
            ],
            &[(0, 1), (1, 2)],
            None,
        );
        let world = assembler().assemble(&sites).expect("a 3-site chain");

        assert_eq!(world.n_atoms(), 11);
        assert_eq!(world.n_bonds(), 10);
        assert_eq!(world.n_ports(), 2);
        let mut frag_ids: Vec<I> = world
            .atoms()
            .map(|(_, a)| a.get_int(keys::FRAG_ID).expect("frag_id"))
            .collect();
        frag_ids.sort_unstable();
        frag_ids.dedup();
        assert_eq!(frag_ids, [0, 1, 2]);
        assert!(mol_ids(&world).iter().all(|&m| m == 1));
    }

    #[test]
    fn assemble_numbers_molecules_by_connected_component() {
        let sites = graph(
            &[
                ("Li", [20.0, 0.0, 0.0]),
                ("M", [0.0; 3]),
                ("M", [5.0, 0.0, 0.0]),
            ],
            &[(1, 2)],
            None,
        );
        let world = assembler().assemble(&sites).expect("Li and an M pair");

        // Li is component 0 (lowest ordinal), the M pair component 1.
        assert_eq!(world.n_atoms(), 1 + 8);
        let ids = mol_ids(&world);
        assert_eq!(ids.iter().filter(|&&m| m == 1).count(), 1);
        assert_eq!(ids.iter().filter(|&&m| m == 2).count(), 8);
    }

    #[test]
    fn assemble_closes_a_ring() {
        let sites = graph(
            &[
                ("M", [0.0; 3]),
                ("M", [5.0, 0.0, 0.0]),
                ("M", [2.5, 4.0, 0.0]),
            ],
            &[(0, 1), (1, 2), (2, 0)],
            None,
        );
        let world = assembler().assemble(&sites).expect("a 3-ring");

        // Three links: 15 − 6 atoms, no port left.
        assert_eq!(world.n_atoms(), 9);
        assert_eq!(world.n_ports(), 0);
    }

    #[test]
    fn assemble_joins_a_branch_point_to_each_arm() {
        // X (three `$`) at the centre, one E (one `$`) on each of three arms.
        let sites = graph(
            &[
                ("E", [4.0, 0.0, 0.0]),
                ("X", [0.0; 3]),
                ("E", [-4.0, 0.0, 0.0]),
                ("E", [0.0, 4.0, 0.0]),
            ],
            &[(0, 1), (1, 2), (1, 3)],
            None,
        );
        let world = assembler().assemble(&sites).expect("a star");

        // 4 + 3·2 atoms, three links remove 6: 4 atoms, no port left.
        assert_eq!(world.n_atoms(), 4);
        assert_eq!(world.n_ports(), 0);
        assert_eq!(world.n_bonds(), 3);
    }

    #[test]
    fn assemble_picks_the_hub_ports_whose_angle_fits_the_bonds() {
        // X has `$` ports along +x, −x and +y. Its partners sit at +x and +y,
        // 90° apart, so the ±x pair (180°) is not chosen: the first 90° pair
        // in port order is (+x, +y), and the −x hydrogen stays.
        let sites = graph(
            &[
                ("E", [4.0, 0.0, 0.0]),
                ("X", [0.0; 3]),
                ("E", [0.0, 4.0, 0.0]),
            ],
            &[(0, 1), (1, 2)],
            None,
        );
        let world = assembler().assemble(&sites).expect("E–X–E");

        assert_eq!(world.n_ports(), 1);
        let kept: Vec<Vec3> = world
            .atoms()
            .filter(|(_, a)| {
                a.atomic_number() == Some(Element::H.z()) && a.get_int(keys::FRAG_ID) == Some(1)
            })
            .map(|(_, a)| a.position().expect("xyz"))
            .collect();
        assert_eq!(kept.len(), 1, "one hydrogen of X stays");
        // The placer moves X's centroid (x = 0) onto the site, so the
        // kept hydrogen stays at x = −1.
        assert!((kept[0][0] + 1.0).abs() < TOL, "{kept:?}");
    }

    /// B: M with an extra `>` port labelled `g` on C1 (a third H at y = 1).
    fn branch_labelled() -> Atomistic {
        let mut b = monomer_labelled("", "");
        let c1 = b.atom_ids().nth(1).expect("C1");
        let h = atom(&mut b, Element::H, [1.5, 1.0, 0.0], 1.0);
        b.add_bond(c1, h, props! {}).expect("C1–H");
        b.add_port(c1, h, PortKind::Right, "g", BondNumber::Single)
            .expect("port >g");
        b
    }

    /// G: one `<` port labelled `g`.
    fn graft_start() -> Atomistic {
        let mut g = Atomistic::new();
        let c = atom(&mut g, Element::C, [0.0; 3], 12.0);
        let h = atom(&mut g, Element::H, [-1.0, 0.0, 0.0], 1.0);
        g.add_bond(c, h, props! {}).expect("C–H");
        g.add_port(c, h, PortKind::Left, "g", BondNumber::Single)
            .expect("port <g");
        g
    }

    #[test]
    fn assemble_gives_a_labelled_port_only_to_the_partner_that_accepts_it() {
        // M – B(–G) – M: B's two `>` ports differ only by label, and the first
        // unconstrained choice must not hand `>g` to the unlabelled M.
        let mut lib = library();
        lib.insert("B".to_owned(), branch_labelled());
        lib.insert("G".to_owned(), graft_start());
        let sites = topology(&["M", "B", "G", "M"], &[(0, 1), (1, 2), (1, 3)]);
        let world = Assembler::new(lib, Box::new(GrowthPlacer::new()), None)
            .assemble(&sites)
            .expect("a labelled branch");

        assert_eq!(world.n_ports(), 2);
    }

    #[test]
    fn assemble_of_no_sites_is_an_empty_fragment() {
        let world = assembler()
            .assemble(&CoarseGrain::new())
            .expect("nothing to build");
        assert_eq!(world.n_atoms(), 0);
    }

    /// Counts `place` calls, then delegates to `SitePlacer`.
    struct CountingPlacer(Arc<AtomicUsize>);

    impl Placer<Atomistic> for CountingPlacer {
        fn place(&self, template: &Atomistic, site: &PlaceSite) -> Result<Rigid, PlaceError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            SitePlacer::new().place(template, site)
        }
    }

    #[test]
    fn assemble_places_each_site_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let assembler = Assembler::new(library(), Box::new(CountingPlacer(calls.clone())), None);
        let sites = graph(
            &[
                ("M", [0.0; 3]),
                ("M", [5.0, 0.0, 0.0]),
                ("Li", [20.0, 0.0, 0.0]),
            ],
            &[(0, 1)],
            None,
        );
        assembler.assemble(&sites).expect("M and Li");

        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// A site graph without positions: `(name)` per site and bonds.
    fn topology(names: &[&str], bonds: &[(usize, usize)]) -> CoarseGrain {
        let mut cg = CoarseGrain::new();
        let ids: Vec<_> = names
            .iter()
            .map(|n| cg.add_bead(props! { keys::BEAD_TYPE => *n }).unwrap())
            .collect();
        for &(i, j) in bonds {
            cg.add_bond(ids[i], ids[j], props! {}).expect("site bond");
        }
        cg
    }

    fn grower() -> Assembler<Atomistic> {
        Assembler::new(library(), Box::new(GrowthPlacer::new()), None)
    }

    #[test]
    fn a_grown_chain_joins_each_copy_at_the_parent_handle_distance() {
        let world = grower()
            .assemble(&topology(&["M", "M", "M"], &[(0, 1), (1, 2)]))
            .expect("a grown M chain");

        assert_eq!(world.n_atoms(), 3 * 5 - 4);
        // Each inter-unit C–C bond takes the parent's C–H length, 1.0 Å.
        let mut joins = Vec::new();
        for (_, bond) in world.bonds() {
            let (a, b) = (bond.nodes[0], bond.nodes[1]);
            let (pa, pb) = (world.get_atom(a).expect("a"), world.get_atom(b).expect("b"));
            if pa.get_int(keys::FRAG_ID) != pb.get_int(keys::FRAG_ID) {
                let pa = pa.position().expect("xyz");
                let pb = pb.position().expect("xyz");
                joins.push(crate::op::vec3::norm(crate::op::vec3::sub(pa, pb)));
            }
        }
        assert_eq!(joins.len(), 2);
        for d in joins {
            assert!((d - 1.0).abs() < TOL, "join length {d}");
        }
    }

    #[test]
    fn a_walk_entered_from_the_wrong_end_is_retried() {
        // K has three `<` ports; each arm is M–M. The walk starts at an arm
        // end, whose first choice (`<`) would leave `<` facing K; the retry
        // takes `>` and every arm then meets K with `>`.
        let mut lib = library();
        lib.insert("K".to_owned(), hub_of(3, PortKind::Left));
        let sites = topology(
            &["K", "M", "M", "M", "M", "M", "M"],
            &[(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)],
        );
        let world = Assembler::new(lib, Box::new(GrowthPlacer::new()), None)
            .assemble(&sites)
            .expect("a three-arm star");

        // 4 + 6·5 atoms, six links remove 12.
        assert_eq!(world.n_atoms(), 34 - 12);
        assert_eq!(world.n_ports(), 3);
    }

    /// A quarter turn about +z around the template's centroid.
    struct QuarterTurn;

    impl Orienter<Atomistic> for QuarterTurn {
        fn orient_many(
            &self,
            template: &Atomistic,
            sites: &[SiteView<'_>],
        ) -> Result<Vec<Rigid>, OrientError> {
            let c = crate::core::GraphAccess::graph(template)
                .centroid(&template.atom_ids().collect::<Vec<_>>())
                .map_err(OrientError::Center)?;
            let rz = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
            Ok(vec![rotation_about(rz, c); sites.len()])
        }
    }

    #[test]
    fn assemble_turns_each_copy_before_placing_it() {
        // Li is one atom, so its image is its site whatever the turn; an E
        // (C at 0, H at +x) turned a quarter about its centre puts its H
        // straight above (+y) its C.
        let assembler = Assembler::new(
            library(),
            Box::new(SitePlacer::new()),
            Some(Box::new(QuarterTurn)),
        );
        let world = assembler
            .assemble(&graph(&[("E", [7.0, 0.0, 0.0])], &[], None))
            .expect("one E");

        let xs: Vec<f64> = world
            .atoms()
            .map(|(_, a)| a.get_f64(keys::X).expect("x"))
            .collect();
        assert!(
            (xs[0] - xs[1]).abs() < TOL,
            "C and H share x after the turn: {xs:?}"
        );
    }

    #[test]
    fn assemble_with_the_axis_orienter_turns_chain_units_onto_their_sites() {
        // Sites along +x with axis +z: each M's joining atoms run along +x
        // and its side O (backbone → centre, +y in the template) turns to +z.
        let sites = graph(
            &[("M", [0.0; 3]), ("M", [2.5, 0.0, 0.0])],
            &[(0, 1)],
            Some([0.0, 0.0, 1.0]),
        );
        let world = Assembler::new(
            library(),
            Box::new(SitePlacer::new()),
            Some(Box::new(AxisOrienter::new())),
        )
        .assemble(&sites)
        .expect("an oriented M pair");

        for (_, a) in world.atoms() {
            if a.atomic_number() == Some(Element::O.z()) {
                assert!(
                    a.get_f64(keys::Y).expect("y").abs() < TOL,
                    "O left the xz plane"
                );
                assert!(a.get_f64(keys::Z).expect("z") > 0.5, "O points along +z");
            }
        }
    }

    #[test]
    fn assemble_builds_the_world_as_the_library_graph_type() {
        // A coarse-grained library gives a coarse-grained world: one bead
        // template, two unbonded sites, each copy's bead on its site.
        let mut bead = CoarseGrain::new();
        let b = bead
            .add_bead(props! {
                keys::BEAD_TYPE => "W",
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        bead.set_bead(b, keys::MASS, 72.0).expect("mass");
        let assembler = Assembler::new(
            HashMap::from([("W".to_owned(), bead)]),
            Box::new(SitePlacer::new()),
            None,
        );
        let world: CoarseGrain = assembler
            .assemble(&graph(
                &[("W", [1.0, 2.0, 3.0]), ("W", [4.0, 5.0, 6.0])],
                &[],
                None,
            ))
            .expect("two W beads");

        let beads: Vec<NodeId> = world.bead_ids().collect();
        assert_eq!(
            world.positions(&beads).expect("xyz"),
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        );
        assert_eq!(world.bead_types(&beads).expect("types"), ["W", "W"]);
    }

    // ---- refusals ------------------------------------------------------------

    #[test]
    fn assemble_names_an_unknown_template_by_site() {
        let err = assembler()
            .assemble(&graph(
                &[("M", [0.0; 3]), ("Q", [1.0, 0.0, 0.0])],
                &[],
                None,
            ))
            .expect_err("Q is not in the library");
        assert!(
            matches!(&err, AssembleError::UnknownName { site: 1, name } if name == "Q"),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_refuses_a_site_with_more_bonds_than_ports() {
        // E has one `$` port but two bonds to `$` hubs.
        let sites = graph(
            &[
                ("S", [-4.0, 0.0, 0.0]),
                ("E", [0.0; 3]),
                ("S", [4.0, 0.0, 0.0]),
            ],
            &[(0, 1), (1, 2)],
            None,
        );
        let err = assembler()
            .assemble(&sites)
            .expect_err("E cannot take two bonds");
        assert!(
            matches!(&err, AssembleError::Ports { site: 1, name, .. } if name == "E"),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_refuses_ports_whose_labels_never_accept() {
        // P's `>` is labelled y and its `<` x: no port of the first P can join
        // any port of the second, so the walk's first site is refused.
        let sites = graph(&[("P", [0.0; 3]), ("P", [5.0, 0.0, 0.0])], &[(0, 1)], None);
        let err = assembler().assemble(&sites).expect_err("mismatched labels");
        assert!(
            matches!(err, AssembleError::Ports { site: 0, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_maps_an_orient_refusal_to_its_site() {
        // A chain of M without axes: the AxisOrienter refuses the first
        // bonded M site.
        let sites = graph(
            &[
                ("Li", [9.0, 0.0, 0.0]),
                ("M", [0.0; 3]),
                ("M", [5.0, 0.0, 0.0]),
            ],
            &[(1, 2)],
            None,
        );
        let err = Assembler::new(
            library(),
            Box::new(SitePlacer::new()),
            Some(Box::new(AxisOrienter::new())),
        )
        .assemble(&sites)
        .expect_err("no axis");
        assert!(
            matches!(
                &err,
                AssembleError::Orient { name, site: 1, source: OrientError::NoAxis { index: 0 } }
                    if name == "M"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn assemble_maps_a_non_finite_position_to_its_site() {
        let mut sites = graph(&[("M", [0.0; 3]), ("M", [5.0, 0.0, 0.0])], &[], None);
        let second = sites.bead_ids().nth(1).expect("two sites");
        sites.set_bead(second, keys::X, f64::NAN).expect("x");
        let err = assembler().assemble(&sites).expect_err("a NaN site");
        assert!(matches!(err, AssembleError::Sites(_)), "{err:?}");
    }
}
